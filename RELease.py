#!/usr/bin/env python3
"""Paper-faithful REleASE-style proton forecasting.

This script keeps only the workflow aligned to the paper-style method:
- Sliding-window electron rise parameter (5-60 min by default).
- Deterministic lookup matrix mapping rise/intensity to target proton level.
- 1-hour-ahead warning logic with paper-like filtering rules.
- Optional validation metrics (POD, FAR, AWT) against alert timestamps.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class MatrixCell:
    slope_min: float
    slope_max: float
    intensity_min: float
    intensity_max: float
    p30_peak_pfu: float
    p50_peak_pfu: float
    p100_peak_pfu: float
    confidence: float
    target_peak_pfu: Optional[float] = None


@dataclass
class DeltaMatrixCell:
    intensity_min: float
    intensity_max: float
    delta_min: float
    delta_max: float
    p30_peak_pfu: float
    confidence: float
    target_peak_pfu: Optional[float] = None


@dataclass
class ReleaseMatrixCell:
    log_intensity_min: float
    log_intensity_max: float
    log_rise_min: float
    log_rise_max: float
    predicted_peak_pfu: float
    confidence: float
    target_peak_pfu: Optional[float] = None


@dataclass
class PaperForecastOutput:
    mode: str
    model_variant: str
    peak_flux_prediction_target_pfu: float
    peak_flux_prediction_10mev_pfu: float
    confidence_level: float
    warning_times_utc: List[str]
    expected_onset_times_from_warnings_utc: List[str]
    first_expected_onset_time_target: Optional[str]
    first_expected_onset_time_10mev: Optional[str]
    forecast_lead_minutes: int
    sliding_window_min_minutes: Optional[int]
    sliding_window_max_minutes: Optional[int]
    delta_window_minutes: Optional[int]
    hazard_level_pfu: float
    warning_trigger_pfu: float
    type3_filter_applied: bool
    type3_filter_passed: Optional[bool]
    notes: List[str]
    votes: Optional[Dict[str, float]] = None
    voter_consensus: Optional[str] = None
    status: Optional[str] = None
    matrix_windows_minutes: Optional[List[int]] = None
    predicted_peak_flux_30mev: Optional[float] = None
    rise_parameter: Optional[float] = None
    alert_status: Optional[str] = None
    timestamp_utc: Optional[str] = None
    invalid_reason: Optional[str] = None


@dataclass
class DeltaMatrixPredictionBundle:
    predicted_values: pd.DataFrame
    latest_time: Optional[pd.Timestamp]
    latest_prediction_pfu: float
    peak_prediction_pfu: float
    mean_confidence: float
    onset_time: Optional[pd.Timestamp]
    type3_pass: Optional[bool]
    selected_window_minutes: int


@dataclass
class ValidationOutput:
    threshold_pfu: float
    total_alerts: int
    total_actual_events: int
    true_positives: int
    false_positives: int
    false_negatives: int
    pod: float
    far: float
    awt_minutes_mean: Optional[float]
    awt_minutes_median: Optional[float]
    pod_target_reference: str
    far_target_reference: str
    awt_target_reference: str
    notes: List[str]


def _default_matrix() -> List[MatrixCell]:
    # Placeholder bins. Replace with a historically calibrated matrix for
    # production-quality paper replication.
    return [
        MatrixCell(0.0, 0.02, 0.0, 2.0, 5.0, 2.0, 0.5, 0.25),
        MatrixCell(0.0, 0.02, 2.0, 10.0, 15.0, 6.0, 1.0, 0.40),
        MatrixCell(0.02, 0.08, 0.0, 2.0, 20.0, 8.0, 2.0, 0.50),
        MatrixCell(0.02, 0.08, 2.0, 10.0, 45.0, 18.0, 6.0, 0.68),
        MatrixCell(0.08, 0.30, 2.0, 10.0, 90.0, 35.0, 12.0, 0.78),
        MatrixCell(0.30, 9999.0, 10.0, 9999.0, 240.0, 110.0, 35.0, 0.90),
    ]


def _load_matrix(matrix_json: Optional[str]) -> List[MatrixCell]:
    if matrix_json is None:
        return _default_matrix()

    rows = json.loads(Path(matrix_json).read_text())
    return [MatrixCell(**row) for row in rows]



def _default_delta_matrix() -> List[DeltaMatrixCell]:
    # Coarse placeholder bins for (electron intensity, delta intensity).
    # Replace with historically calibrated bins from archived events.
    return [
        DeltaMatrixCell(0.0, 2_000.0, -50_000.0, 0.0, 3.0, 0.30),
        DeltaMatrixCell(0.0, 2_000.0, 0.0, 2_000.0, 10.0, 0.45),
        DeltaMatrixCell(2_000.0, 8_000.0, -50_000.0, 0.0, 8.0, 0.40),
        DeltaMatrixCell(2_000.0, 8_000.0, 0.0, 2_000.0, 30.0, 0.60),
        DeltaMatrixCell(8_000.0, 50_000.0, 0.0, 4_000.0, 90.0, 0.75),
        DeltaMatrixCell(8_000.0, 50_000.0, 4_000.0, 500_000.0, 240.0, 0.90),
    ]


def _load_delta_matrix(delta_matrix_json: Optional[str]) -> List[DeltaMatrixCell]:
    if delta_matrix_json is None:
        return _default_delta_matrix()

    rows = json.loads(Path(delta_matrix_json).read_text())
    return [DeltaMatrixCell(**row) for row in rows]


def _default_release_matrix() -> List[ReleaseMatrixCell]:
    log_intensity_edges = np.linspace(-1.0, 5.0, 51)
    log_rise_edges = np.linspace(-1.0, 5.0, 51)
    matrix: List[ReleaseMatrixCell] = []
    for i in range(50):
        for j in range(50):
            center_intensity = float((log_intensity_edges[i] + log_intensity_edges[i + 1]) / 2.0)
            center_rise = float((log_rise_edges[j] + log_rise_edges[j + 1]) / 2.0)
            predicted = float(10.0 ** max(0.0, 0.7 * center_intensity + 0.3 * center_rise + 0.5))
            matrix.append(
                ReleaseMatrixCell(
                    log_intensity_min=float(log_intensity_edges[i]),
                    log_intensity_max=float(log_intensity_edges[i + 1]),
                    log_rise_min=float(log_rise_edges[j]),
                    log_rise_max=float(log_rise_edges[j + 1]),
                    predicted_peak_pfu=predicted,
                    confidence=float(min(1.0, 0.15 + predicted / 1000.0)),
                )
            )
    return matrix


def _load_release_matrix(release_matrix_json: Optional[str]) -> List[ReleaseMatrixCell]:
    if release_matrix_json is None:
        return _default_release_matrix()

    rows = json.loads(Path(release_matrix_json).read_text())
    matrix: List[ReleaseMatrixCell] = []
    for row in rows:
        if "predicted_peak_pfu" not in row and "target_peak_pfu" in row:
            row = dict(row)
            row["predicted_peak_pfu"] = row["target_peak_pfu"]
        matrix.append(ReleaseMatrixCell(**row))
    return matrix


def _release_matrix_lookup(
    matrix: Sequence[ReleaseMatrixCell],
    log_intensity: float,
    log_rise: float,
) -> ReleaseMatrixCell:
    for cell in matrix:
        if (
            cell.log_intensity_min <= log_intensity < cell.log_intensity_max
            and cell.log_rise_min <= log_rise < cell.log_rise_max
        ):
            return cell

    conservative = [
        cell
        for cell in matrix
        if cell.log_intensity_min >= log_intensity or cell.log_rise_min >= log_rise
    ]
    if conservative:
        return min(
            conservative,
            key=lambda cell: (
                max(0.0, cell.log_intensity_min - log_intensity),
                max(0.0, cell.log_rise_min - log_rise),
            ),
        )

    return matrix[-1]


def _instrument_key(instrument: Optional[str]) -> str:
    if instrument is None:
        return ""
    return str(instrument).strip().lower()


def _load_delta_matrix_library(
    matrix_library_json: str,
) -> Dict[Tuple[str, int], List[DeltaMatrixCell]]:
    library_path = Path(matrix_library_json)
    rows = json.loads(library_path.read_text())
    if not isinstance(rows, list):
        raise ValueError("--delta-matrix-library-json must point to a JSON list")

    matrix_by_key: Dict[Tuple[str, int], List[DeltaMatrixCell]] = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Matrix library row {i} is not an object")

        inst = _instrument_key(str(row.get("instrument", "")))
        window_val = row.get("sliding_window_minutes")
        matrix_path_raw = row.get("matrix_path")

        if not inst:
            raise ValueError(f"Matrix library row {i} missing 'instrument'")
        if window_val is None:
            raise ValueError(f"Matrix library row {i} missing 'sliding_window_minutes'")
        if matrix_path_raw is None:
            raise ValueError(f"Matrix library row {i} missing 'matrix_path'")

        window_minutes = int(window_val)
        matrix_path = Path(str(matrix_path_raw))
        if not matrix_path.is_absolute():
            matrix_path = (library_path.parent / matrix_path).resolve()

        key = (inst, window_minutes)
        matrix_by_key[key] = _load_delta_matrix(str(matrix_path))

    if not matrix_by_key:
        raise ValueError("--delta-matrix-library-json did not contain any matrix entries")
    return matrix_by_key


def _select_delta_matrix(
    matrix_library: Dict[Tuple[str, int], List[DeltaMatrixCell]],
    instrument: str,
    sliding_window_minutes: int,
) -> Tuple[List[DeltaMatrixCell], int]:
    inst = _instrument_key(instrument)
    key = (inst, int(sliding_window_minutes))
    if key in matrix_library:
        return matrix_library[key], int(sliding_window_minutes)

    available = sorted(k for k in matrix_library.keys() if k[0] == inst)
    if not available:
        have = sorted({k[0] for k in matrix_library.keys()})
        raise ValueError(
            f"No delta matrix available for instrument '{instrument}'. "
            f"Available instruments: {', '.join(have)}"
        )

    nearest = min(available, key=lambda k: abs(k[1] - int(sliding_window_minutes)))
    return matrix_library[nearest], int(nearest[1])


def _read_timeseries(csv_path: str, time_col: str, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")
    if value_col not in df.columns:
        raise ValueError(f"Missing value column: {value_col}")

    out = df[[time_col, value_col]].copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna().sort_values(time_col).reset_index(drop=True)
    out.columns = ["time", "value"]
    return out


def _infer_cadence_seconds(times: pd.Series) -> float:
    if len(times) < 2:
        return 300.0
    diffs = times.diff().dropna().dt.total_seconds().values
    if len(diffs) == 0:
        return 300.0
    return float(np.nanmedian(diffs))


def _remove_background_gcr(values: np.ndarray, window_points: int) -> np.ndarray:
    window_points = max(1, window_points)
    s = pd.Series(values)
    background = s.rolling(window=window_points, min_periods=1).median()
    return (s - background).clip(lower=0.0).to_numpy(dtype=float)


def _matrix_lookup(matrix: Sequence[MatrixCell], slope: float, intensity: float) -> MatrixCell:
    for cell in matrix:
        if cell.slope_min <= slope < cell.slope_max and cell.intensity_min <= intensity < cell.intensity_max:
            return cell
    return matrix[-1]


def _delta_matrix_lookup(
    matrix: Sequence[DeltaMatrixCell],
    intensity: float,
    delta: float,
) -> DeltaMatrixCell:
    for cell in matrix:
        if cell.intensity_min <= intensity < cell.intensity_max and cell.delta_min <= delta < cell.delta_max:
            return cell
    return matrix[-1]


def _matrix_cell_target_flux(cell: MatrixCell) -> float:
    if cell.target_peak_pfu is not None:
        return float(cell.target_peak_pfu)
    return float(cell.p30_peak_pfu)


def _delta_matrix_cell_target_flux(cell: DeltaMatrixCell) -> float:
    if cell.target_peak_pfu is not None:
        return float(cell.target_peak_pfu)
    return float(cell.p30_peak_pfu)


def _max_positive_rise_parameter(
    cleaned_flux: np.ndarray,
    idx: int,
    cadence_seconds: float,
    min_window_minutes: int,
    max_window_minutes: int,
) -> float:
    # For a given timestamp idx, find the largest positive slope among all
    # valid windows ending at idx.
    min_points = max(2, int(round((min_window_minutes * 60.0) / max(cadence_seconds, 1.0))))
    max_points = max(min_points, int(round((max_window_minutes * 60.0) / max(cadence_seconds, 1.0))))

    best_slope = 0.0
    for points in range(min_points, max_points + 1):
        start = idx - points + 1
        if start < 0:
            continue
        seg = cleaned_flux[start : idx + 1]
        if len(seg) < 2:
            continue
        x_minutes = np.arange(len(seg), dtype=float) * (cadence_seconds / 60.0)
        slope, _ = np.polyfit(x_minutes, seg, deg=1)
        best_slope = max(best_slope, float(slope))

    return max(0.0, best_slope)


def _estimate_electron_onset(
    cleaned_flux: np.ndarray,
    cadence_seconds: float,
) -> Optional[int]:
    points_5min = max(1, int(round(300.0 / max(cadence_seconds, 1.0))))
    mean = pd.Series(cleaned_flux).rolling(window=points_5min, min_periods=1).mean()
    std = pd.Series(cleaned_flux).rolling(window=points_5min, min_periods=1).std(ddof=0).fillna(0.0)

    for i in range(points_5min, len(cleaned_flux)):
        baseline_mean = float(mean.iloc[max(0, i - points_5min)])
        baseline_std = float(std.iloc[max(0, i - points_5min)])
        if baseline_std <= 0:
            continue
        z = (float(cleaned_flux[i]) - baseline_mean) / baseline_std
        if z >= 3.0:
            return i
    return None


def _read_type3_times(
    csv_path: str,
    time_col: str,
    flag_col: str,
) -> List[pd.Timestamp]:
    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise ValueError(f"Missing Type III time column: {time_col}")
    if flag_col not in df.columns:
        raise ValueError(f"Missing Type III flag column: {flag_col}")

    parsed_time = pd.to_datetime(df[time_col], errors="coerce")
    flag = (
        df[flag_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y"})
    )
    out = pd.DataFrame({"time": parsed_time, "flag": flag}).dropna()
    return sorted(pd.Timestamp(t) for t in out.loc[out["flag"], "time"])


def _type3_filter_passed(
    onset_time: Optional[pd.Timestamp],
    type3_times: Sequence[pd.Timestamp],
    lookback_minutes: int,
) -> Optional[bool]:
    if onset_time is None:
        return None
    start = onset_time - pd.Timedelta(minutes=lookback_minutes)
    for t in type3_times:
        if start <= t <= onset_time:
            return True
    return False


def _has_telemetry_gap(times: pd.Series, max_gap_minutes: float = 5.0) -> bool:
    if len(times) < 2:
        return False
    diffs = times.diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    if len(diffs) == 0:
        return False
    return bool(np.nanmax(diffs) > max_gap_minutes * 60.0)


def _detect_background_onset(
    values: np.ndarray,
    cadence_seconds: float,
) -> Optional[int]:
    points_2h = max(1, int(round(7200.0 / max(cadence_seconds, 1.0))))
    if len(values) <= points_2h:
        return None

    series = pd.Series(values)
    rolling_mean = series.rolling(window=points_2h, min_periods=points_2h).mean()
    rolling_std = series.rolling(window=points_2h, min_periods=points_2h).std(ddof=0).fillna(0.0)

    for idx in range(points_2h, len(values)):
        baseline_mean = float(rolling_mean.iloc[idx - 1])
        baseline_std = float(rolling_std.iloc[idx - 1])
        if not np.isfinite(baseline_std) or baseline_std <= 0.0:
            continue
        if float(values[idx]) >= baseline_mean + (3.0 * baseline_std):
            return idx
    return None


def _build_original_release_forecast(
    electron_df: pd.DataFrame,
    proton_gt10_df: pd.DataFrame,
    matrix: Sequence[ReleaseMatrixCell],
    forecast_lead_minutes: int,
    release_warning_trigger_pfu: float,
    release_invalid_gap_minutes: float,
) -> PaperForecastOutput:
    times = electron_df["time"]
    values = electron_df["value"].to_numpy(dtype=float)
    cadence_seconds = _infer_cadence_seconds(times)

    invalid_reason: Optional[str] = None
    if _has_telemetry_gap(times, max_gap_minutes=release_invalid_gap_minutes):
        invalid_reason = "telemetry_gap_over_5_minutes"

    lookback_points = max(1, int(round((60.0 * 60.0) / max(cadence_seconds, 1.0))))
    if len(values) <= lookback_points:
        invalid_reason = invalid_reason or "insufficient_60_min_history"

    onset_idx = _detect_background_onset(values, cadence_seconds)
    start_idx = onset_idx if onset_idx is not None else 0

    predictions: List[Tuple[pd.Timestamp, float, float, float, float, float]] = []
    for idx in range(lookback_points, len(values)):
        current_time = pd.Timestamp(times.iloc[idx])
        current_intensity = float(np.clip(values[idx], 1.0, 1.0e5))
        prior_intensity = float(np.clip(values[idx - lookback_points], 1.0, 1.0e5))
        rise_parameter = float(max(1.0, current_intensity * (current_intensity / prior_intensity)))
        log_intensity = float(np.log10(current_intensity))
        log_rise = float(np.log10(rise_parameter))
        cell = _release_matrix_lookup(matrix, log_intensity=log_intensity, log_rise=log_rise)
        predicted_peak = float(cell.target_peak_pfu if cell.target_peak_pfu is not None else cell.predicted_peak_pfu)
        predictions.append((current_time, predicted_peak, rise_parameter, float(cell.confidence), log_intensity, log_rise))

    prediction_df = pd.DataFrame(
        predictions,
        columns=["time", "predicted_pfu", "rise_parameter", "cell_confidence", "log_intensity", "log_rise"],
    )

    if prediction_df.empty:
        invalid_reason = invalid_reason or "insufficient_prediction_points"

    if invalid_reason is not None:
        latest_time = pd.Timestamp(times.iloc[-1]).isoformat(sep=" ") if len(times) else None
        notes = [
            "Original REleASE mode enabled: deterministic 60-minute rise-parameter lookup.",
            f"Release matrix lookup uses conservative nearest-neighbor selection for sparse cells.",
            f"Invalid reason: {invalid_reason}.",
        ]
        return PaperForecastOutput(
            mode="original-release",
            model_variant="original-release",
            peak_flux_prediction_target_pfu=0.0,
            peak_flux_prediction_10mev_pfu=0.0,
            confidence_level=0.0,
            warning_times_utc=[],
            expected_onset_times_from_warnings_utc=[],
            first_expected_onset_time_target=None,
            first_expected_onset_time_10mev=None,
            forecast_lead_minutes=forecast_lead_minutes,
            sliding_window_min_minutes=60,
            sliding_window_max_minutes=60,
            delta_window_minutes=None,
            hazard_level_pfu=10.0,
            warning_trigger_pfu=release_warning_trigger_pfu,
            type3_filter_applied=False,
            type3_filter_passed=None,
            notes=notes,
            predicted_peak_flux_30mev=0.0,
            rise_parameter=0.0,
            alert_status="INVALID",
            timestamp_utc=latest_time,
            invalid_reason=invalid_reason,
        )

    event_predictions = prediction_df.iloc[start_idx - lookback_points if start_idx >= lookback_points else 0 :].copy()
    if event_predictions.empty:
        event_predictions = prediction_df.copy()

    peak_row = event_predictions.loc[event_predictions["predicted_pfu"].idxmax()]
    latest_row = prediction_df.iloc[-1]
    latest_time = pd.Timestamp(latest_row["time"]).isoformat(sep=" ")
    peak_pred = float(peak_row["predicted_pfu"])
    peak_rise = float(peak_row["rise_parameter"])
    confidence = float(np.clip(peak_row["cell_confidence"], 0.0, 1.0))

    if peak_pred >= 30.0:
        alert_status = "Warning"
    elif peak_pred >= release_warning_trigger_pfu:
        alert_status = "Alert"
    else:
        alert_status = "Quiet"

    proton_series = proton_gt10_df.copy().sort_values("time").reset_index(drop=True)
    warning_times = _collect_warning_times(
        predicted_values=prediction_df[["time", "predicted_pfu"]].copy(),
        proton_series=proton_series,
        hazard_level_pfu=release_warning_trigger_pfu,
        warning_trigger_pfu=release_warning_trigger_pfu,
    )

    hold_peak = float(max(prediction_df["predicted_pfu"].iloc[start_idx:])) if start_idx < len(prediction_df) else peak_pred
    peak_pred = max(peak_pred, hold_peak)

    notes = [
        "Original REleASE mode enabled: deterministic 60-minute rise-parameter lookup.",
        "Rise parameter is computed as I_e(t) * (I_e(t) / I_e(t-60m)).",
        "Inputs are clamped to [1, 1e5] before log10 matrix indexing.",
        "Empty matrix cells fall back to a conservative higher-intensity neighbor.",
    ]

    return PaperForecastOutput(
        mode="original-release",
        model_variant="original-release",
        peak_flux_prediction_target_pfu=peak_pred,
        peak_flux_prediction_10mev_pfu=peak_pred,
        confidence_level=confidence,
        warning_times_utc=[t.isoformat(sep=" ") for t in warning_times],
        expected_onset_times_from_warnings_utc=[(t + pd.Timedelta(minutes=forecast_lead_minutes)).isoformat(sep=" ") for t in warning_times],
        first_expected_onset_time_target=(warning_times[0] + pd.Timedelta(minutes=forecast_lead_minutes)).isoformat(sep=" ") if warning_times else None,
        first_expected_onset_time_10mev=(warning_times[0] + pd.Timedelta(minutes=forecast_lead_minutes)).isoformat(sep=" ") if warning_times else None,
        forecast_lead_minutes=forecast_lead_minutes,
        sliding_window_min_minutes=60,
        sliding_window_max_minutes=60,
        delta_window_minutes=None,
        hazard_level_pfu=10.0,
        warning_trigger_pfu=release_warning_trigger_pfu,
        type3_filter_applied=False,
        type3_filter_passed=None,
        notes=notes,
        predicted_peak_flux_30mev=peak_pred,
        rise_parameter=peak_rise,
        alert_status=alert_status,
        timestamp_utc=latest_time,
        invalid_reason=None,
    )


def _value_at_or_before(
    series_df: pd.DataFrame,
    time_col: str,
    value_col: str,
    at_time: pd.Timestamp,
) -> Optional[float]:
    rows = series_df[series_df[time_col] <= at_time]
    if rows.empty:
        return None
    return float(rows.iloc[-1][value_col])


def _mean_value_in_window(
    series_df: pd.DataFrame,
    time_col: str,
    value_col: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> Optional[float]:
    rows = series_df[(series_df[time_col] >= start_time) & (series_df[time_col] <= end_time)]
    if rows.empty:
        return None
    return float(rows[value_col].mean())


def _collect_warning_times(
    predicted_values: pd.DataFrame,
    proton_series: pd.DataFrame,
    hazard_level_pfu: float,
    warning_trigger_pfu: float,
    require_type3: bool = False,
    type3_pass: Optional[bool] = None,
) -> List[pd.Timestamp]:
    # Guidance: warning generation follows the same 3-rule gate in both models,
    # so this helper keeps behavior consistent between rise-matrix and delta-matrix.
    if predicted_values.empty:
        return []
    if require_type3 and type3_pass is not True:
        return []

    warning_times: List[pd.Timestamp] = []
    for _, row in predicted_values.iterrows():
        t = pd.Timestamp(row["time"])
        predicted_pfu = float(row["predicted_pfu"])

        current_actual = _value_at_or_before(proton_series, "time", "value", t)
        if current_actual is None:
            continue

        avg_2h = _mean_value_in_window(
            proton_series,
            "time",
            "value",
            start_time=t - pd.Timedelta(hours=2),
            end_time=t,
        )
        if avg_2h is None:
            continue

        pred_2h = predicted_values[
            (predicted_values["time"] >= t - pd.Timedelta(hours=2))
            & (predicted_values["time"] <= t)
        ]
        pred_2h_max = float(pred_2h["predicted_pfu"].max()) if not pred_2h.empty else predicted_pfu

        rule1 = (current_actual < hazard_level_pfu) and (avg_2h < hazard_level_pfu)
        rule2 = predicted_pfu >= warning_trigger_pfu
        rule3 = predicted_pfu >= pred_2h_max

        if rule1 and rule2 and rule3:
            warning_times.append(t)

    return sorted(set(warning_times))


def _build_forecast_output(
    model_variant: str,
    peak_prediction_target_pfu: float,
    confidence_level: float,
    warning_times: Sequence[pd.Timestamp],
    forecast_lead_minutes: int,
    sliding_window_min_minutes: Optional[int],
    sliding_window_max_minutes: Optional[int],
    delta_window_minutes: Optional[int],
    hazard_level_pfu: float,
    warning_trigger_pfu: float,
    type3_filter_applied: bool,
    type3_filter_passed: Optional[bool],
    notes: List[str],
    votes: Optional[Dict[str, float]] = None,
    voter_consensus: Optional[str] = None,
    status: Optional[str] = None,
    matrix_windows_minutes: Optional[List[int]] = None,
) -> PaperForecastOutput:
    # Guidance: "target" is the canonical naming, while 30 MeV fields are
    # retained as backward-compatible aliases for existing consumers.
    expected_onsets = [wt + pd.Timedelta(minutes=forecast_lead_minutes) for wt in warning_times]
    first_onset = expected_onsets[0].isoformat(sep=" ") if expected_onsets else None

    return PaperForecastOutput(
        mode="paper-faithful-forecast",
        model_variant=model_variant,
        peak_flux_prediction_target_pfu=peak_prediction_target_pfu,
        peak_flux_prediction_10mev_pfu=peak_prediction_target_pfu,
        confidence_level=confidence_level,
        warning_times_utc=[t.isoformat(sep=" ") for t in warning_times],
        expected_onset_times_from_warnings_utc=[t.isoformat(sep=" ") for t in expected_onsets],
        first_expected_onset_time_target=first_onset,
        first_expected_onset_time_10mev=first_onset,
        forecast_lead_minutes=forecast_lead_minutes,
        sliding_window_min_minutes=sliding_window_min_minutes,
        sliding_window_max_minutes=sliding_window_max_minutes,
        delta_window_minutes=delta_window_minutes,
        hazard_level_pfu=hazard_level_pfu,
        warning_trigger_pfu=warning_trigger_pfu,
        type3_filter_applied=type3_filter_applied,
        type3_filter_passed=type3_filter_passed,
        notes=notes,
        votes=votes,
        voter_consensus=voter_consensus,
        status=status,
        matrix_windows_minutes=matrix_windows_minutes,
    )


def _predict_delta_matrix_bundle(
    electron_df: pd.DataFrame,
    matrix: Sequence[DeltaMatrixCell],
    delta_window_minutes: int,
    require_type3: bool,
    type3_times: Sequence[pd.Timestamp],
    type3_lookback_minutes: int,
) -> DeltaMatrixPredictionBundle:
    times = electron_df["time"]
    values = electron_df["value"].to_numpy(dtype=float)
    cadence_seconds = _infer_cadence_seconds(times)

    bg_window_points = max(3, int(round(3600.0 / max(cadence_seconds, 1.0))))
    cleaned = _remove_background_gcr(values, bg_window_points)

    window_points = max(1, int(round((delta_window_minutes * 60.0) / max(cadence_seconds, 1.0))))
    onset_idx = _estimate_electron_onset(cleaned, cadence_seconds)
    onset_time = pd.Timestamp(times.iloc[onset_idx]) if onset_idx is not None else None
    type3_pass = _type3_filter_passed(onset_time, type3_times, type3_lookback_minutes)

    predictions: List[Tuple[pd.Timestamp, float, float, float]] = []
    for i in range(window_points, len(cleaned)):
        intensity = float(cleaned[i])
        delta = float(cleaned[i] - cleaned[i - window_points])
        cell = _delta_matrix_lookup(matrix, intensity=intensity, delta=delta)
        predictions.append((pd.Timestamp(times.iloc[i]), _delta_matrix_cell_target_flux(cell), delta, float(cell.confidence)))

    predicted_values = pd.DataFrame(predictions, columns=["time", "predicted_pfu", "delta", "cell_conf"])
    peak_pred = float(predicted_values["predicted_pfu"].max()) if not predicted_values.empty else 0.0
    mean_conf = float(predicted_values["cell_conf"].mean()) if not predicted_values.empty else 0.0
    latest_time = pd.Timestamp(predicted_values.iloc[-1]["time"]) if not predicted_values.empty else None
    latest_prediction = float(predicted_values.iloc[-1]["predicted_pfu"]) if not predicted_values.empty else 0.0

    if require_type3 and type3_pass is not True:
        latest_prediction = 0.0
        peak_pred = 0.0
        mean_conf = 0.0

    return DeltaMatrixPredictionBundle(
        predicted_values=predicted_values,
        latest_time=latest_time,
        latest_prediction_pfu=latest_prediction,
        peak_prediction_pfu=peak_pred,
        mean_confidence=mean_conf,
        onset_time=onset_time,
        type3_pass=type3_pass,
        selected_window_minutes=delta_window_minutes,
    )


def _build_paper_faithful_forecast(
    electron_df: pd.DataFrame,
    proton_gt10_df: pd.DataFrame,
    matrix: Sequence[MatrixCell],
    forecast_lead_minutes: int,
    min_window_minutes: int,
    max_window_minutes: int,
    hazard_level_pfu: float,
    warning_trigger_pfu: float,
) -> PaperForecastOutput:
    times = electron_df["time"]
    values = electron_df["value"].to_numpy(dtype=float)
    cadence_seconds = _infer_cadence_seconds(times)

    bg_window_points = max(3, int(round(3600.0 / max(cadence_seconds, 1.0))))
    cleaned = _remove_background_gcr(values, bg_window_points)

    proton_series = proton_gt10_df.copy().sort_values("time").reset_index(drop=True)

    predictions: List[Tuple[pd.Timestamp, float]] = []
    for i in range(len(cleaned)):
        rise_param = _max_positive_rise_parameter(
            cleaned_flux=cleaned,
            idx=i,
            cadence_seconds=cadence_seconds,
            min_window_minutes=min_window_minutes,
            max_window_minutes=max_window_minutes,
        )
        intensity = float(cleaned[i])
        cell = _matrix_lookup(matrix, slope=rise_param, intensity=intensity)
        current_time = pd.Timestamp(times.iloc[i])
        predictions.append((current_time, _matrix_cell_target_flux(cell)))

    predicted_values = pd.DataFrame(predictions, columns=["time", "predicted_pfu"])
    unique_warning_times = _collect_warning_times(
        predicted_values=predicted_values,
        proton_series=proton_series,
        hazard_level_pfu=hazard_level_pfu,
        warning_trigger_pfu=warning_trigger_pfu,
    )
    peak_pred = float(predicted_values["predicted_pfu"].max()) if not predicted_values.empty else 0.0
    confidence = min(1.0, max(0.0, 0.35 + (peak_pred / 300.0)))

    notes = [
        "Paper-faithful mode enabled: minute-by-minute sliding window rise parameter.",
        f"Sliding window bounds: {min_window_minutes}-{max_window_minutes} minutes.",
        f"Lead mapping: prediction at t corresponds to proton level at t+{forecast_lead_minutes} min.",
        f"Warning rules: actual<{hazard_level_pfu} pfu context, predicted>={warning_trigger_pfu} pfu, 2h local max.",
    ]

    return _build_forecast_output(
        model_variant="rise-matrix",
        peak_prediction_target_pfu=peak_pred,
        confidence_level=confidence,
        warning_times=unique_warning_times,
        forecast_lead_minutes=forecast_lead_minutes,
        sliding_window_min_minutes=min_window_minutes,
        sliding_window_max_minutes=max_window_minutes,
        delta_window_minutes=None,
        hazard_level_pfu=hazard_level_pfu,
        warning_trigger_pfu=warning_trigger_pfu,
        type3_filter_applied=False,
        type3_filter_passed=None,
        notes=notes,
    )


def _build_delta_matrix_forecast(
    electron_df: pd.DataFrame,
    proton_gt10_df: pd.DataFrame,
    matrix: Sequence[DeltaMatrixCell],
    instrument: str,
    forecast_lead_minutes: int,
    delta_window_minutes: int,
    matrix_window_minutes: int,
    hazard_level_pfu: float,
    warning_trigger_pfu: float,
    require_type3: bool,
    type3_times: Sequence[pd.Timestamp],
    type3_lookback_minutes: int,
) -> PaperForecastOutput:
    proton_series = proton_gt10_df.copy().sort_values("time").reset_index(drop=True)
    bundle = _predict_delta_matrix_bundle(
        electron_df=electron_df,
        matrix=matrix,
        delta_window_minutes=delta_window_minutes,
        require_type3=require_type3,
        type3_times=type3_times,
        type3_lookback_minutes=type3_lookback_minutes,
    )
    predicted_values = bundle.predicted_values
    unique_warning_times = _collect_warning_times(
        predicted_values=predicted_values,
        proton_series=proton_series,
        hazard_level_pfu=hazard_level_pfu,
        warning_trigger_pfu=warning_trigger_pfu,
        require_type3=require_type3,
        type3_pass=bundle.type3_pass,
    )
    peak_pred = bundle.peak_prediction_pfu
    confidence = min(1.0, max(0.0, bundle.mean_confidence))

    notes = [
        "Delta-matrix mode enabled: matrix axes are current intensity and delta over lookback window.",
        f"Instrument: {instrument}.",
        f"Sliding delta window for features: {delta_window_minutes} minutes.",
        f"Matrix calibration window used: {matrix_window_minutes} minutes.",
        f"Lead mapping: prediction at t corresponds to proton level at t+{forecast_lead_minutes} min.",
        f"Warning rules: actual<{hazard_level_pfu} pfu context, predicted>={warning_trigger_pfu} pfu, 2h local max.",
    ]
    if require_type3:
        notes.append(f"Type III filter enabled with {type3_lookback_minutes}-minute pre-onset lookback window.")

    return _build_forecast_output(
        model_variant="delta-matrix",
        peak_prediction_target_pfu=peak_pred,
        confidence_level=confidence,
        warning_times=unique_warning_times,
        forecast_lead_minutes=forecast_lead_minutes,
        sliding_window_min_minutes=None,
        sliding_window_max_minutes=None,
        delta_window_minutes=matrix_window_minutes,
        hazard_level_pfu=hazard_level_pfu,
        warning_trigger_pfu=warning_trigger_pfu,
        type3_filter_applied=require_type3,
        type3_filter_passed=bundle.type3_pass if require_type3 else None,
        notes=notes,
    )


def _build_three_matrix_voter_forecast(
    electron_df: pd.DataFrame,
    proton_gt10_df: pd.DataFrame,
    matrix_library: Dict[Tuple[str, int], List[DeltaMatrixCell]],
    instrument: str,
    forecast_lead_minutes: int,
    ensemble_windows_minutes: Sequence[int],
    hazard_level_pfu: float,
    warning_trigger_pfu: float,
    require_type3: bool,
    type3_times: Sequence[pd.Timestamp],
    type3_lookback_minutes: int,
) -> PaperForecastOutput:
    vote_labels = ["m30_pred", "m60_pred", "m90_pred"]
    votes: Dict[str, float] = {}
    matrix_windows_used: List[int] = []
    bundle_confidences: List[float] = []
    latest_times: List[pd.Timestamp] = []

    for label, requested_window in zip(vote_labels, ensemble_windows_minutes):
        matrix, selected_window = _select_delta_matrix(
            matrix_library=matrix_library,
            instrument=instrument,
            sliding_window_minutes=int(requested_window),
        )
        bundle = _predict_delta_matrix_bundle(
            electron_df=electron_df,
            matrix=matrix,
            delta_window_minutes=selected_window,
            require_type3=require_type3,
            type3_times=type3_times,
            type3_lookback_minutes=type3_lookback_minutes,
        )
        votes[label] = float(bundle.latest_prediction_pfu)
        matrix_windows_used.append(int(selected_window))
        bundle_confidences.append(float(bundle.mean_confidence))
        if bundle.latest_time is not None:
            latest_times.append(pd.Timestamp(bundle.latest_time))

    vote_values = [votes[label] for label in vote_labels]
    trigger_count = sum(value >= warning_trigger_pfu for value in vote_values)
    peak_pred = float(max(vote_values)) if vote_values else 0.0
    weighted_flux = float(0.2 * votes.get("m30_pred", 0.0) + 0.5 * votes.get("m60_pred", 0.0) + 0.3 * votes.get("m90_pred", 0.0))

    rising_prompt = bool(votes.get("m30_pred", 0.0) > votes.get("m60_pred", 0.0) > votes.get("m90_pred", 0.0))
    decaying = bool(votes.get("m90_pred", 0.0) > votes.get("m60_pred", 0.0) > votes.get("m30_pred", 0.0))

    base_confidence = float(np.clip(np.mean(bundle_confidences) if bundle_confidences else 0.0, 0.0, 1.0))
    if trigger_count >= 2:
        base_confidence = max(base_confidence, 0.72)
    if rising_prompt:
        base_confidence = max(base_confidence, 0.82)
    if votes.get("m30_pred", 0.0) >= warning_trigger_pfu and votes.get("m60_pred", 0.0) >= warning_trigger_pfu:
        base_confidence = max(base_confidence, 0.75)
    confidence = min(1.0, max(0.0, base_confidence))

    if trigger_count >= 2:
        status = "ALERT_ACTIVE"
    elif trigger_count == 1:
        status = "WATCH_INTERNAL"
    else:
        status = "NO_ALERT"

    if rising_prompt:
        voter_consensus = "Prompt Intensification Detected"
    elif decaying:
        voter_consensus = "Delayed or Decaying"
    elif trigger_count >= 2:
        voter_consensus = "2-of-3 Warning Consensus"
    else:
        voter_consensus = "Insufficient Consensus"

    alert_times: List[pd.Timestamp] = []
    if latest_times and trigger_count >= 1:
        alert_times = [max(latest_times)]

    notes = [
        "Ensemble-voter mode enabled: 30/60/90-minute matrix consensus.",
        f"Instrument: {instrument}.",
        f"Requested ensemble windows: {', '.join(str(int(w)) for w in ensemble_windows_minutes)} minutes.",
        f"Selected matrix windows: {', '.join(str(w) for w in matrix_windows_used)} minutes.",
        f"Weighted flux coefficients: m30=0.2, m60=0.5, m90=0.3; conservative target still uses max(votes).",
        f"Warning rules: {trigger_count}/3 matrices at or above {warning_trigger_pfu} pfu; 2-of-3 required for reportable warning.",
    ]
    if require_type3:
        notes.append(f"Type III filter enabled with {type3_lookback_minutes}-minute pre-onset lookback window.")

    result = _build_forecast_output(
        model_variant="ensemble-voter-3-matrix",
        peak_prediction_target_pfu=peak_pred,
        confidence_level=confidence,
        warning_times=alert_times,
        forecast_lead_minutes=forecast_lead_minutes,
        sliding_window_min_minutes=None,
        sliding_window_max_minutes=None,
        delta_window_minutes=None,
        hazard_level_pfu=hazard_level_pfu,
        warning_trigger_pfu=warning_trigger_pfu,
        type3_filter_applied=require_type3,
        type3_filter_passed=True if require_type3 else None,
        notes=notes,
        votes=votes,
        voter_consensus=voter_consensus,
        status=status,
        matrix_windows_minutes=matrix_windows_used,
    )
    result.mode = "HESPERIA REleASE"
    return result


def _detect_event_onsets_from_threshold(
    proton_df: pd.DataFrame,
    threshold_pfu: float,
) -> List[pd.Timestamp]:
    onsets: List[pd.Timestamp] = []
    in_event = False

    for _, row in proton_df.iterrows():
        t = pd.Timestamp(row["time"])
        v = float(row["value"])

        if not in_event and v >= threshold_pfu:
            in_event = True
            onsets.append(t)
        elif in_event and v < threshold_pfu:
            in_event = False

    return onsets


def _read_alert_times(alerts_csv: str, alert_time_col: str) -> List[pd.Timestamp]:
    df = pd.read_csv(alerts_csv)
    if alert_time_col not in df.columns:
        raise ValueError(f"Missing alert time column: {alert_time_col}")

    times = pd.to_datetime(df[alert_time_col], errors="coerce").dropna()
    return sorted(pd.Timestamp(t) for t in times)


def _match_alerts_to_events(
    alerts: Sequence[pd.Timestamp],
    events: Sequence[pd.Timestamp],
    max_lead_minutes: float,
) -> Tuple[int, int, int, List[float]]:
    tp = 0
    matched_event_idx: set[int] = set()
    awt_minutes: List[float] = []

    for alert in alerts:
        match_idx: Optional[int] = None
        match_delta_min: Optional[float] = None

        for idx, event_time in enumerate(events):
            if idx in matched_event_idx:
                continue
            delta_min = (event_time - alert).total_seconds() / 60.0
            if delta_min < 0:
                continue
            if delta_min <= max_lead_minutes:
                match_idx = idx
                match_delta_min = delta_min
                break

        if match_idx is not None and match_delta_min is not None:
            matched_event_idx.add(match_idx)
            tp += 1
            awt_minutes.append(float(match_delta_min))

    fp = max(0, len(alerts) - tp)
    fn = max(0, len(events) - len(matched_event_idx))
    return tp, fp, fn, awt_minutes


def _build_validation_output(
    proton_gt10_df: pd.DataFrame,
    alerts_csv: str,
    alert_time_col: str,
    threshold_pfu: float,
    max_lead_minutes: float,
) -> ValidationOutput:
    alerts = _read_alert_times(alerts_csv, alert_time_col)
    events = _detect_event_onsets_from_threshold(proton_gt10_df, threshold_pfu=threshold_pfu)

    tp, fp, fn, awt_minutes = _match_alerts_to_events(
        alerts=alerts,
        events=events,
        max_lead_minutes=max_lead_minutes,
    )

    pod = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    far = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0

    awt_mean = float(np.mean(awt_minutes)) if awt_minutes else None
    awt_median = float(np.median(awt_minutes)) if awt_minutes else None

    notes = [
        "Validation references from REleASE literature used for interpretation.",
        "POD target approx 0.63, FAR target approx 0.29-0.35, AWT target approx 107-123 minutes.",
        "POD and FAR are reported as fractions (0 to 1).",
    ]

    return ValidationOutput(
        threshold_pfu=float(threshold_pfu),
        total_alerts=len(alerts),
        total_actual_events=len(events),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        pod=pod,
        far=far,
        awt_minutes_mean=awt_mean,
        awt_minutes_median=awt_median,
        pod_target_reference="~0.63",
        far_target_reference="~0.29 to 0.35",
        awt_target_reference="~107 to 123 minutes",
        notes=notes,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-faithful REleASE-style proton forecast")

    p.add_argument("--electron-csv", type=str, default=None)
    p.add_argument("--electron-time-col", type=str, default="time_tag")
    p.add_argument("--electron-flux-col", type=str, default="electron_flux")

    p.add_argument("--proton-csv", type=str, required=True)
    p.add_argument("--proton-time-col", type=str, default="time_tag")
    p.add_argument("--proton-gt10-col", type=str, default=None)
    p.add_argument("--proton-gt30-col", type=str, default=None)

    p.add_argument(
        "--model",
        type=str,
        choices=["rise-matrix", "delta-matrix", "original-release", "hesperia-release"],
        default="rise-matrix",
    )
    p.add_argument("--matrix-json", type=str, default=None)
    p.add_argument("--release-matrix-json", type=str, default=None)
    p.add_argument("--delta-matrix-json", type=str, default=None)
    p.add_argument("--delta-matrix-library-json", type=str, default=None)
    p.add_argument("--instrument", type=str, default="SOHO")
    p.add_argument("--out-json", type=str, default="release_paper_output.json")
    p.add_argument("--ensemble-voter-3-matrix", action="store_true")
    p.add_argument(
        "--ensemble-window-minutes",
        type=int,
        nargs=3,
        default=[30, 60, 90],
        help="Three sliding windows to use for ensemble voting, in minutes",
    )

    p.add_argument("--forecast-lead-minutes", type=int, default=60)
    p.add_argument("--sliding-window-min-minutes", type=int, default=5)
    p.add_argument("--sliding-window-max-minutes", type=int, default=60)
    p.add_argument("--paper-hazard-pfu", type=float, default=20.0)
    p.add_argument("--paper-warning-trigger-pfu", type=float, default=30.0)
    p.add_argument("--release-warning-trigger-pfu", type=float, default=10.0)
    p.add_argument("--release-invalid-gap-minutes", type=float, default=5.0)
    p.add_argument("--delta-window-minutes", type=int, default=60)
    p.add_argument("--require-type3", action="store_true")
    p.add_argument("--type3-csv", type=str, default=None)
    p.add_argument("--type3-time-col", type=str, default="time_tag")
    p.add_argument("--type3-flag-col", type=str, default="type3")
    p.add_argument("--type3-lookback-minutes", type=int, default=250)

    p.add_argument("--validate", action="store_true")
    p.add_argument("--alerts-csv", type=str, default=None)
    p.add_argument("--alerts-time-col", type=str, default="alert_time")
    p.add_argument("--validation-threshold-pfu", type=float, default=10.0)
    p.add_argument("--max-lead-minutes", type=float, default=180.0)

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    proton_col = args.proton_gt10_col or args.proton_gt30_col
    if proton_col is None:
        raise ValueError("Provide --proton-gt10-col (preferred) or --proton-gt30-col (legacy)")

    proton_gt10_df = _read_timeseries(args.proton_csv, args.proton_time_col, proton_col)

    if args.validate:
        if not args.alerts_csv:
            raise ValueError("--validate requires --alerts-csv")
        result = _build_validation_output(
            proton_gt10_df=proton_gt10_df,
            alerts_csv=args.alerts_csv,
            alert_time_col=args.alerts_time_col,
            threshold_pfu=args.validation_threshold_pfu,
            max_lead_minutes=args.max_lead_minutes,
        )
    else:
        if not args.electron_csv:
            raise ValueError("Paper forecast mode requires --electron-csv")

        # Guidance: keep electron parsing centralized so both model branches use
        # exactly the same cleaned input stream.
        electron_df = _read_timeseries(args.electron_csv, args.electron_time_col, args.electron_flux_col)
        if args.model == "original-release":
            release_matrix = _load_release_matrix(args.release_matrix_json)
            result = _build_original_release_forecast(
                electron_df=electron_df,
                proton_gt10_df=proton_gt10_df,
                matrix=release_matrix,
                forecast_lead_minutes=args.forecast_lead_minutes,
                release_warning_trigger_pfu=args.release_warning_trigger_pfu,
                release_invalid_gap_minutes=args.release_invalid_gap_minutes,
            )
        elif args.model == "hesperia-release" or args.ensemble_voter_3_matrix:
            if not args.delta_matrix_library_json:
                raise ValueError("HESPERIA mode requires --delta-matrix-library-json")
            matrix_library = _load_delta_matrix_library(args.delta_matrix_library_json)
            type3_times: List[pd.Timestamp] = []
            if args.require_type3:
                if not args.type3_csv:
                    raise ValueError("--require-type3 needs --type3-csv")
                type3_times = _read_type3_times(
                    csv_path=args.type3_csv,
                    time_col=args.type3_time_col,
                    flag_col=args.type3_flag_col,
                )
            result = _build_three_matrix_voter_forecast(
                electron_df=electron_df,
                proton_gt10_df=proton_gt10_df,
                matrix_library=matrix_library,
                instrument=args.instrument,
                forecast_lead_minutes=args.forecast_lead_minutes,
                ensemble_windows_minutes=args.ensemble_window_minutes,
                hazard_level_pfu=args.paper_hazard_pfu,
                warning_trigger_pfu=args.paper_warning_trigger_pfu,
                require_type3=args.require_type3,
                type3_times=type3_times,
                type3_lookback_minutes=args.type3_lookback_minutes,
            )
        elif args.model == "rise-matrix":
            matrix = _load_matrix(args.matrix_json)
            result = _build_paper_faithful_forecast(
                electron_df=electron_df,
                proton_gt10_df=proton_gt10_df,
                matrix=matrix,
                forecast_lead_minutes=args.forecast_lead_minutes,
                min_window_minutes=args.sliding_window_min_minutes,
                max_window_minutes=args.sliding_window_max_minutes,
                hazard_level_pfu=args.paper_hazard_pfu,
                warning_trigger_pfu=args.paper_warning_trigger_pfu,
            )
        else:
            if args.delta_matrix_library_json:
                matrix_library = _load_delta_matrix_library(args.delta_matrix_library_json)
                delta_matrix, selected_window = _select_delta_matrix(
                    matrix_library=matrix_library,
                    instrument=args.instrument,
                    sliding_window_minutes=args.delta_window_minutes,
                )
            else:
                delta_matrix = _load_delta_matrix(args.delta_matrix_json)
                selected_window = int(args.delta_window_minutes)
            type3_times: List[pd.Timestamp] = []
            if args.require_type3:
                if not args.type3_csv:
                    raise ValueError("--require-type3 needs --type3-csv")
                type3_times = _read_type3_times(
                    csv_path=args.type3_csv,
                    time_col=args.type3_time_col,
                    flag_col=args.type3_flag_col,
                )

            result = _build_delta_matrix_forecast(
                electron_df=electron_df,
                proton_gt10_df=proton_gt10_df,
                matrix=delta_matrix,
                instrument=args.instrument,
                forecast_lead_minutes=args.forecast_lead_minutes,
                delta_window_minutes=args.delta_window_minutes,
                matrix_window_minutes=selected_window,
                hazard_level_pfu=args.paper_hazard_pfu,
                warning_trigger_pfu=args.paper_warning_trigger_pfu,
                require_type3=args.require_type3,
                type3_times=type3_times,
                type3_lookback_minutes=args.type3_lookback_minutes,
            )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), indent=2))

    print(json.dumps(asdict(result), indent=2))
    print(f"Wrote output to: {out_path}")


if __name__ == "__main__":
    main()
