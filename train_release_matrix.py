#!/usr/bin/env python3
"""Train a deterministic original REleASE lookup matrix from historical data.

This script builds a 50x50 log-log matrix that maps:
- current electron intensity
- 60-minute rise parameter

to the target proton peak flux at the chosen lead time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class TrainingRow:
    log_intensity: float
    log_rise: float
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


def _safe_positive(value: float) -> float:
    if not np.isfinite(value):
        return 1.0
    return float(np.clip(value, 1.0, 1.0e5))


def _build_training_rows_for_file(
    df: pd.DataFrame,
    lead_minutes: int,
    lookback_minutes: int,
) -> List[TrainingRow]:
    cadence_seconds = _infer_cadence_seconds(df["time"])
    lookback_points = max(1, int(round((lookback_minutes * 60.0) / max(cadence_seconds, 1.0))))
    lead_points = max(1, int(round((lead_minutes * 60.0) / max(cadence_seconds, 1.0))))

    electron = df["electron"].to_numpy(dtype=float)
    proton = df["proton"].to_numpy(dtype=float)
    rows: List[TrainingRow] = []

    upper = len(df) - lead_points
    for i in range(lookback_points, upper):
        current = _safe_positive(float(electron[i]))
        prior = _safe_positive(float(electron[i - lookback_points]))
        rise = _safe_positive(current * (current / prior))
        target = float(proton[i + lead_points])

        if np.isfinite(current) and np.isfinite(rise) and np.isfinite(target):
            rows.append(
                TrainingRow(
                    log_intensity=float(np.log10(current)),
                    log_rise=float(np.log10(rise)),
                    target_proton=target,
                )
            )

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


def _build_release_matrix(
    rows: Sequence[TrainingRow],
    intensity_bins: int,
    rise_bins: int,
    min_samples_per_cell: int,
) -> List[Dict[str, float]]:
    if not rows:
        raise ValueError("No training rows were produced. Check columns and data length.")

    intensity = np.array([r.log_intensity for r in rows], dtype=float)
    rise = np.array([r.log_rise for r in rows], dtype=float)
    target = np.array([r.target_proton for r in rows], dtype=float)

    i_edges = _safe_quantile_edges(intensity, intensity_bins)
    r_edges = _safe_quantile_edges(rise, rise_bins)

    i_idx = np.digitize(intensity, i_edges[1:-1], right=False)
    r_idx = np.digitize(rise, r_edges[1:-1], right=False)

    cell_values: Dict[Tuple[int, int], List[float]] = {}
    for ii, rr, tt in zip(i_idx, r_idx, target):
        key = (int(ii), int(rr))
        cell_values.setdefault(key, []).append(float(tt))

    global_median = float(np.median(target))
    matrix: List[Dict[str, float]] = []

    for ii in range(intensity_bins):
        for rr in range(rise_bins):
            vals = cell_values.get((ii, rr), [])
            predicted = float(np.median(vals)) if vals else global_median
            conf = _cell_confidence(vals, min_samples_per_cell=min_samples_per_cell)
            matrix.append(
                {
                    "log_intensity_min": float(i_edges[ii]),
                    "log_intensity_max": float(i_edges[ii + 1]),
                    "log_rise_min": float(r_edges[rr]),
                    "log_rise_max": float(r_edges[rr + 1]),
                    "predicted_peak_pfu": predicted,
                    "target_peak_pfu": predicted,
                    "confidence": conf,
                }
            )

    return matrix


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train original REleASE release matrix")
    parser.add_argument("csv_paths", nargs="+", help="Historical CSV file(s) used for calibration")
    parser.add_argument("--time-col", type=str, default="time_tag")
    parser.add_argument("--electron-col", type=str, required=True)
    parser.add_argument("--proton-col", type=str, required=True)
    parser.add_argument("--lead-minutes", type=int, default=60)
    parser.add_argument("--lookback-minutes", type=int, default=60)
    parser.add_argument("--intensity-bins", type=int, default=50)
    parser.add_argument("--rise-bins", type=int, default=50)
    parser.add_argument("--min-samples-per-cell", type=int, default=30)
    parser.add_argument("--out-json", type=str, default="trained_release_matrix.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[TrainingRow] = []
    for path_str in args.csv_paths:
        path = Path(path_str)
        df = _read_series(path, args.time_col, args.electron_col, args.proton_col)
        all_rows.extend(
            _build_training_rows_for_file(
                df,
                lead_minutes=args.lead_minutes,
                lookback_minutes=args.lookback_minutes,
            )
        )

    matrix = _build_release_matrix(
        rows=all_rows,
        intensity_bins=args.intensity_bins,
        rise_bins=args.rise_bins,
        min_samples_per_cell=args.min_samples_per_cell,
    )

    out_path.write_text(json.dumps(matrix, indent=2))
    print(f"Training rows: {len(all_rows)}")
    print(f"Matrix cells: {len(matrix)}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
