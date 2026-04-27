#!/usr/bin/env python3
"""Train a delta-intensity proton forecasting matrix from historical data.

This script creates a JSON matrix compatible with RELease.py --model delta-matrix.
Expected input is one or more CSV files containing at least:
- a timestamp column
- an electron flux column
- a proton flux column
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class TrainingRow:
    intensity: float
    delta: float
    target_proton: float


def _read_series(csv_path: Path, time_col: str, electron_col: str, proton_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in (time_col, electron_col, proton_col):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {csv_path}")

    out = df[[time_col, electron_col, proton_col]].copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out[electron_col] = pd.to_numeric(out[electron_col], errors="coerce")
    out[proton_col] = pd.to_numeric(out[proton_col], errors="coerce")
    out = out.dropna().sort_values(time_col).reset_index(drop=True)
    out.columns = ["time", "electron", "proton"]
    return out


def _infer_cadence_seconds(times: pd.Series) -> float:
    if len(times) < 2:
        return 300.0
    diffs = times.diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    if len(diffs) == 0:
        return 300.0
    return float(np.nanmedian(diffs))


def _remove_background_gcr(values: np.ndarray, window_points: int) -> np.ndarray:
    window_points = max(1, window_points)
    s = pd.Series(values)
    background = s.rolling(window=window_points, min_periods=1).median()
    cleaned = (s - background).clip(lower=0.0)
    return cleaned.to_numpy(dtype=float)


def _build_training_rows_for_file(
    df: pd.DataFrame,
    lead_minutes: int,
    delta_window_minutes: int,
) -> List[TrainingRow]:
    cadence_seconds = _infer_cadence_seconds(df["time"])

    bg_window_points = max(3, int(round(3600.0 / max(cadence_seconds, 1.0))))
    cleaned_electron = _remove_background_gcr(df["electron"].to_numpy(dtype=float), bg_window_points)

    delta_points = max(1, int(round((delta_window_minutes * 60.0) / max(cadence_seconds, 1.0))))
    lead_points = max(1, int(round((lead_minutes * 60.0) / max(cadence_seconds, 1.0))))

    proton = df["proton"].to_numpy(dtype=float)
    rows: List[TrainingRow] = []

    upper = len(df) - lead_points
    for i in range(delta_points, upper):
        intensity = float(cleaned_electron[i])
        delta = float(cleaned_electron[i] - cleaned_electron[i - delta_points])
        target = float(proton[i + lead_points])
        if np.isfinite(intensity) and np.isfinite(delta) and np.isfinite(target):
            rows.append(TrainingRow(intensity=intensity, delta=delta, target_proton=target))

    return rows


def _safe_quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    if len(values) == 0:
        raise ValueError("No values available for binning")

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, qs)
    edges = np.unique(edges)

    if len(edges) < 2:
        lo = float(values.min())
        hi = float(values.max())
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, n_bins + 1)

    if len(edges) - 1 < n_bins:
        lo = float(values.min())
        hi = float(values.max())
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, n_bins + 1)

    edges[0] = edges[0] - 1e-9
    edges[-1] = edges[-1] + 1e-9
    return edges


def _cell_confidence(values: Sequence[float], min_samples_per_cell: int) -> float:
    n = len(values)
    if n == 0:
        return 0.0

    n_score = min(1.0, n / float(max(1, min_samples_per_cell)))

    vals = np.asarray(values, dtype=float)
    med = float(np.median(vals))
    q25 = float(np.percentile(vals, 25))
    q75 = float(np.percentile(vals, 75))
    iqr = q75 - q25

    scale = max(abs(med), 1.0)
    stability = max(0.0, 1.0 - min(1.0, iqr / (2.0 * scale)))
    return float(max(0.0, min(1.0, n_score * stability)))


def _build_delta_matrix(
    rows: Sequence[TrainingRow],
    intensity_bins: int,
    delta_bins: int,
    min_samples_per_cell: int,
) -> List[Dict[str, float]]:
    if not rows:
        raise ValueError("No training rows were produced. Check columns and data length.")

    intensity = np.array([r.intensity for r in rows], dtype=float)
    delta = np.array([r.delta for r in rows], dtype=float)
    target = np.array([r.target_proton for r in rows], dtype=float)

    i_edges = _safe_quantile_edges(intensity, intensity_bins)
    d_edges = _safe_quantile_edges(delta, delta_bins)

    i_idx = np.digitize(intensity, i_edges[1:-1], right=False)
    d_idx = np.digitize(delta, d_edges[1:-1], right=False)

    cell_values: Dict[Tuple[int, int], List[float]] = {}
    for ii, dd, tt in zip(i_idx, d_idx, target):
        key = (int(ii), int(dd))
        cell_values.setdefault(key, []).append(float(tt))

    global_median = float(np.median(target))
    matrix: List[Dict[str, float]] = []

    for ii in range(intensity_bins):
        for dd in range(delta_bins):
            vals = cell_values.get((ii, dd), [])
            p30 = float(np.median(vals)) if vals else global_median
            conf = _cell_confidence(vals, min_samples_per_cell=min_samples_per_cell)
            matrix.append(
                {
                    "intensity_min": float(i_edges[ii]),
                    "intensity_max": float(i_edges[ii + 1]),
                    "delta_min": float(d_edges[dd]),
                    "delta_max": float(d_edges[dd + 1]),
                    "target_peak_pfu": p30,
                    "p30_peak_pfu": p30,
                    "confidence": conf,
                }
            )

    return matrix


def _instrument_tag(instrument: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9]+", "_", instrument.strip().lower()).strip("_")
    return tag or "unknown"


def _window_list_from_args(
    *,
    sliding_window_minutes: Sequence[int],
    delta_window_minutes_legacy: Optional[int],
) -> List[int]:
    windows: List[int] = []
    for w in sliding_window_minutes:
        ww = int(w)
        if ww <= 0:
            raise ValueError("All --sliding-window-minutes values must be > 0")
        windows.append(ww)

    if delta_window_minutes_legacy is not None:
        legacy_w = int(delta_window_minutes_legacy)
        if legacy_w <= 0:
            raise ValueError("--delta-window-minutes must be > 0")
        windows = [legacy_w]

    out = sorted(set(windows))
    if not out:
        raise ValueError("At least one sliding window minute value is required")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train delta-intensity matrix for RELease delta model")

    p.add_argument("csv_paths", nargs="+", help="Historical CSV file(s) used for calibration")
    p.add_argument("--time-col", type=str, default="time_tag")
    p.add_argument("--electron-col", type=str, required=True)
    p.add_argument("--proton-col", type=str, required=True)

    p.add_argument("--instrument", type=str, default="SOHO", help="Instrument label used in output naming")
    p.add_argument("--lead-minutes", type=int, default=60)
    p.add_argument(
        "--sliding-window-minutes",
        type=int,
        nargs="+",
        default=[60],
        help="One or more sliding-window lengths to train (minutes), e.g. 30 60 90",
    )
    p.add_argument(
        "--delta-window-minutes",
        type=int,
        default=None,
        help="Legacy alias for a single sliding-window length (minutes)",
    )
    p.add_argument("--intensity-bins", type=int, default=18)
    p.add_argument("--delta-bins", type=int, default=13)
    p.add_argument("--min-samples-per-cell", type=int, default=30)

    p.add_argument("--out-json", type=str, default="trained_delta_matrix.json")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for multi-matrix output. Default: parent of --out-json",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    windows = _window_list_from_args(
        sliding_window_minutes=args.sliding_window_minutes,
        delta_window_minutes_legacy=args.delta_window_minutes,
    )
    instrument = str(args.instrument)
    instrument_tag = _instrument_tag(instrument)

    out_json_path = Path(args.out_json)
    out_dir = Path(args.out_dir) if args.out_dir else out_json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep the old single-file behavior if exactly one window is requested and no out-dir override is provided.
    single_matrix_mode = len(windows) == 1 and args.out_dir is None

    index_rows: List[Dict[str, object]] = []
    for window_minutes in windows:
        all_rows: List[TrainingRow] = []
        for path_str in args.csv_paths:
            path = Path(path_str)
            df = _read_series(path, args.time_col, args.electron_col, args.proton_col)
            rows = _build_training_rows_for_file(
                df,
                lead_minutes=args.lead_minutes,
                delta_window_minutes=window_minutes,
            )
            all_rows.extend(rows)

        matrix = _build_delta_matrix(
            rows=all_rows,
            intensity_bins=args.intensity_bins,
            delta_bins=args.delta_bins,
            min_samples_per_cell=args.min_samples_per_cell,
        )

        if single_matrix_mode:
            out_path = out_json_path
        else:
            out_path = out_dir / f"matrix_{instrument_tag}_{window_minutes}min.json"

        out_path.write_text(json.dumps(matrix, indent=2))

        print(f"[{instrument}] window={window_minutes}min training rows: {len(all_rows)}")
        print(f"[{instrument}] window={window_minutes}min matrix cells: {len(matrix)}")
        print(f"Wrote: {out_path}")

        index_rows.append(
            {
                "instrument": instrument,
                "instrument_tag": instrument_tag,
                "sliding_window_minutes": int(window_minutes),
                "lead_minutes": int(args.lead_minutes),
                "matrix_path": str(out_path),
                "training_rows": int(len(all_rows)),
                "matrix_cells": int(len(matrix)),
            }
        )

    if not single_matrix_mode:
        index_path = out_dir / f"matrix_index_{instrument_tag}.json"
        index_path.write_text(json.dumps(index_rows, indent=2))
        print(f"Wrote matrix index: {index_path}")


if __name__ == "__main__":
    main()
