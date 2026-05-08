#!/usr/bin/env python3
"""Slim-TSF: sliding-window multivariate time series forest utilities.

This module treats each sliced CSV as one sample, builds early-fusion
statistical features over sliding intervals, ranks features with Random Forest
importances, and trains a binary classifier for event/non-event prediction.

The implementation is intentionally lightweight and notebook-friendly so it can
be reused from `slimtsf_tran_personal.ipynb` and from the command line.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


TIME_COL_DEFAULT = "time_tag"


@dataclass
class SampleMetadata:
    path: str
    slice_key: str
    group_key: str
    label: int
    matched_gsep: bool


@dataclass
class RankSummary:
    feature_names: List[str]
    selected_features: List[str]
    top_k: int
    feature_counts: Dict[str, int]
    mean_importances: Dict[str, float]
    experiment_summaries: List[Dict[str, object]]


@dataclass
class FoldSummary:
    fold_index: int
    train_groups: List[str]
    test_groups: List[str]
    n_train: int
    n_test: int
    threshold: float
    tp: int
    tn: int
    fp: int
    fn: int
    tss: float
    hss: float


@dataclass
class SlimTSFArtifact:
    model_path: str
    feature_names: List[str]
    selected_features: List[str]
    threshold: float
    cv_summary: Dict[str, object]
    rank_summary: Dict[str, object]
    model_params: Dict[str, object]


def _parse_datetime_text(text: str) -> Optional[datetime]:
    value = (text or "").strip()
    if not value:
        return None
    for fmt in (
        "%Y-%m-%dT%H-%M-%S",
        "%Y-%m-%dT%H-%M",
        "%Y-%m-%d_%H-%M-%S",
        "%Y-%m-%d_%H-%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _slice_key_from_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d_%H-%M")


def _slice_key_from_path(path: Path) -> Optional[str]:
    stem = path.stem
    dt = _parse_datetime_text(stem)
    if dt is not None:
        return _slice_key_from_datetime(dt)

    match = re.search(r"(\d{4}-\d{2}-\d{2})[T_ ]?(\d{2})-(\d{2})(?:-(\d{2}))?", stem)
    if match:
        return f"{match.group(1)}_{match.group(2)}-{match.group(3)}"

    return None


def _read_slice_frame(csv_path: Path, time_col: str = TIME_COL_DEFAULT) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise ValueError(f"Missing time column '{time_col}' in {csv_path}")

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    for col in out.columns:
        if col == time_col:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return out


def _infer_cadence_seconds(times: pd.Series) -> float:
    if len(times) < 2:
        return 300.0
    diffs = times.diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    if len(diffs) == 0:
        return 300.0
    cadence = float(np.nanmedian(diffs))
    return cadence if np.isfinite(cadence) and cadence > 0 else 300.0


def _safe_positive(value: float) -> float:
    if not np.isfinite(value):
        return 1.0
    return float(np.clip(value, 1.0, 1e5))


def _remove_background_gcr(
    values: np.ndarray,
    window_points: int,
    sigma_clip: Optional[float] = None,
) -> np.ndarray:
    window_points = max(1, int(window_points))
    series = pd.Series(values)
    background = series.rolling(window=window_points, min_periods=1).median()
    cleaned = (series - background).clip(lower=0.0)
    if sigma_clip is not None:
        sigma = float(np.nanstd(cleaned.to_numpy(dtype=float), ddof=0))
        if np.isfinite(sigma) and sigma > 0:
            cap = float(sigma_clip) * sigma
            cleaned = cleaned.clip(lower=0.0, upper=cap)
    return cleaned.to_numpy(dtype=float)


def _slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    try:
        slope, _ = np.polyfit(x, values.astype(float), 1)
        return float(slope)
    except Exception:
        return 0.0


def _interval_slices(length: int, window_size: int, step_size: int) -> List[Tuple[int, int]]:
    window_size = max(1, int(window_size))
    step_size = max(1, int(step_size))
    if length <= 0:
        return []
    if length <= window_size:
        return [(0, length)]

    intervals: List[Tuple[int, int]] = []
    last_start = max(0, length - window_size)
    for start in range(0, last_start + 1, step_size):
        end = min(length, start + window_size)
        if end > start:
            intervals.append((start, end))
    if not intervals:
        intervals.append((0, length))
    return intervals


def _pooled_stat_features(stat_values: Sequence[float], pooling_width: int) -> Tuple[float, float, float]:
    arr = np.asarray(stat_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0, 0.0, 0.0

    pooling_width = max(1, int(pooling_width))
    if len(arr) <= pooling_width:
        return float(np.max(arr)), float(np.min(arr)), float(np.mean(arr))

    pooled_max: List[float] = []
    pooled_min: List[float] = []
    pooled_mean: List[float] = []
    for start in range(0, len(arr) - pooling_width + 1):
        window = arr[start : start + pooling_width]
        pooled_max.append(float(np.max(window)))
        pooled_min.append(float(np.min(window)))
        pooled_mean.append(float(np.mean(window)))

    return float(np.mean(pooled_max)), float(np.mean(pooled_min)), float(np.mean(pooled_mean))


def _build_channel_features(values: np.ndarray, window_size: int, step_size: int, pooling_width: int) -> Tuple[List[float], List[str]]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    
    # Increase the empty feature count to match the new size (22)
    if len(arr) == 0:
        features = [0.0] * 22 
        return features, [f"empty__{i}" for i in range(len(features))]

    intervals = _interval_slices(len(arr), window_size=window_size, step_size=step_size)
    interval_means, interval_stds, interval_slopes = [], [], []
    
    for start, end in intervals:
        segment = arr[start:end]
        if len(segment) == 0: continue
        interval_means.append(float(np.mean(segment)))
        interval_stds.append(float(np.std(segment, ddof=0)))
        interval_slopes.append(_slope(segment))

    # --- Existing Pooling Logic ---
    mean_max, mean_min, mean_mean = _pooled_stat_features(interval_means, pooling_width)
    std_max, std_min, std_mean = _pooled_stat_features(interval_stds, pooling_width)
    slope_max, slope_min, slope_mean = _pooled_stat_features(interval_slopes, pooling_width)

    # --- NEW: Robust Percentiles (Helps with distribution shifts) ---
    p10, p50, p90 = np.percentile(arr, [10, 50, 90])
    iqr = np.percentile(arr, 75) - np.percentile(arr, 25)

    # --- NEW: Distribution Shape (Skewness) ---
    global_std = float(np.std(arr, ddof=0))
    global_mean = float(np.mean(arr))
    skew = 0.0
    if global_std > 0:
        skew = np.mean(((arr - global_mean) / global_std) ** 3)

    # --- NEW: Acceleration (Slope of Slopes) ---
    accel = 0.0
    if len(interval_slopes) >= 2:
        accel = _slope(np.array(interval_slopes))

    features = [
        mean_max, mean_min, mean_mean,
        std_max, std_min, std_mean,
        slope_max, slope_min, slope_mean,
        float(len(intervals)),
        float(min(1.0, len(arr) / float(max(1, window_size)))), # coverage
        global_mean, global_std, _slope(arr),
        # New Features Added Below
        float(p10), float(p50), float(p90), float(iqr),
        float(skew), float(accel),
        # Coefficient of Variation (Relative Noise)
        float(global_std / (global_mean + 1e-9)),
        # Peak-to-Base Ratio
        float(np.max(arr) / (np.min(arr) + 1e-9))
    ]
    
    names = [
        "mean__pool_max", "mean__pool_min", "mean__pool_mean",
        "std__pool_max", "std__pool_min", "std__pool_mean",
        "slope__pool_max", "slope__pool_min", "slope__pool_mean",
        "interval_count", "coverage_ratio", "global_mean", "global_std", "global_slope",
        "p10", "p50", "p90", "iqr", "skew", "accel", "coeff_var", "peak_ratio"
    ]
    return features, names


def build_feature_vector(
    df: pd.DataFrame,
    *,
    time_col: str = TIME_COL_DEFAULT,
    window_size: int = 12,
    step_size: int = 1,
    pooling_width: int = 3,
    value_cols: Optional[Sequence[str]] = None,
    background_window_points: Optional[int] = None,
    background_sigma_clip: Optional[float] = None,
    only_relative_features: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    if time_col not in df.columns:
        raise ValueError(f"Missing time column '{time_col}'")

    if value_cols is None:
        value_cols = [col for col in df.columns if col != time_col and pd.api.types.is_numeric_dtype(df[col])]
    else:
        value_cols = [col for col in value_cols if col in df.columns]

    if not value_cols:
        raise ValueError("No numeric value columns available for feature generation")

    features: List[float] = []
    names: List[str] = []
    cadence_seconds = _infer_cadence_seconds(df[time_col])
    bg_points = background_window_points
    if bg_points is None:
        bg_points = max(3, int(round(3600.0 / max(cadence_seconds, 1.0))))

    for col in value_cols:
        series = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        series = series[np.isfinite(series)]
        if len(series) == 0:
            continue
        cleaned = _remove_background_gcr(series, bg_points, sigma_clip=background_sigma_clip)
        channel_features, channel_names = _build_channel_features(
            cleaned,
            window_size=window_size,
            step_size=step_size,
            pooling_width=pooling_width,
        )
        features.extend(channel_features)
        names.extend([f"{col}__{name}" for name in channel_names])

    if only_relative_features:
        relative_keywords = ["slope", "std", "accel", "skew", "coeff_var", "peak_ratio", "iqr"]
        keep = [i for i, name in enumerate(names) if any(k in name for k in relative_keywords)]
        if keep:
            features = [features[i] for i in keep]
            names = [names[i] for i in keep]

    if not features:
        raise ValueError("No features were generated from the input dataframe")

    return np.asarray(features, dtype=float), names


def _normalize_gsep_key(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    dt = _parse_datetime_text(text)
    if dt is not None:
        return _slice_key_from_datetime(dt)
    match = re.search(r"(\d{4}-\d{2}-\d{2})[T_ ]?(\d{2})[-:](\d{2})", text)
    if match:
        return f"{match.group(1)}_{match.group(2)}-{match.group(3)}"
    return None


def load_gsep_index(gsep_list_csv: Path) -> Dict[str, Dict[str, str]]:
    if not gsep_list_csv.exists():
        raise FileNotFoundError(f"GSEP list not found: {gsep_list_csv}")
    frame = pd.read_csv(gsep_list_csv)
    index: Dict[str, Dict[str, str]] = {}
    candidate_columns = ["slice_start", "slice_key", "timestamp", "event_timestamp"]
    for _, row in frame.iterrows():
        key = None
        for col in candidate_columns:
            if col in row.index:
                key = _normalize_gsep_key(row[col])
                if key is not None:
                    break
        if key is None:
            continue
        if key not in index:
            index[key] = {k: "" if pd.isna(v) else str(v) for k, v in row.items()}
    return index


def _label_from_gsep_map(slice_key: str, gsep_index: Dict[str, Dict[str, str]]) -> int:
    return 1 if slice_key in gsep_index else 0


def discover_slice_files(data_root: Path) -> List[Path]:
    return sorted(p for p in data_root.rglob("*.csv") if p.is_file())


def build_dataset_from_slices(
    slice_paths: Sequence[Path],
    *,
    gsep_index: Dict[str, Dict[str, str]],
    time_col: str = TIME_COL_DEFAULT,
    window_size: int = 12,
    step_size: int = 1,
    pooling_width: int = 3,
    value_cols: Optional[Sequence[str]] = None,
    background_window_points: Optional[int] = None,
    background_sigma_clip: Optional[float] = None,
    only_relative_features: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, List[SampleMetadata]]:
    rows: List[np.ndarray] = []
    labels: List[int] = []
    metadata: List[SampleMetadata] = []
    feature_names: Optional[List[str]] = None

    for path in slice_paths:
        try:
            df = _read_slice_frame(path, time_col=time_col)
        except Exception:
            continue

        slice_key = _slice_key_from_path(path)
        if slice_key is None and len(df) > 0:
            slice_key = _slice_key_from_datetime(pd.Timestamp(df[time_col].iloc[0]).to_pydatetime())
        if slice_key is None:
            continue

        vector, names = build_feature_vector(
            df,
            time_col=time_col,
            window_size=window_size,
            step_size=step_size,
            pooling_width=pooling_width,
            value_cols=value_cols,
            background_window_points=background_window_points,
            background_sigma_clip=background_sigma_clip,
            only_relative_features=only_relative_features,
        )
        if feature_names is None:
            feature_names = names
        elif len(feature_names) != len(names):
            raise ValueError(f"Feature name mismatch for slice {path}")

        matched = slice_key in gsep_index
        label = _label_from_gsep_map(slice_key, gsep_index)
        group_key = slice_key[:10]

        rows.append(vector)
        labels.append(label)
        metadata.append(
            SampleMetadata(
                path=str(path),
                slice_key=slice_key,
                group_key=group_key,
                label=label,
                matched_gsep=matched,
            )
        )

    if not rows or feature_names is None:
        raise ValueError("No usable slice files were found for dataset construction")

    X = pd.DataFrame(rows, columns=feature_names)
    y = pd.Series(labels, name="label", dtype=int)
    return X, y, metadata


def _safe_threshold_grid() -> np.ndarray:
    return np.linspace(0.05, 0.95, 19)


def _confusion_scores(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def _tss(tp: int, tn: int, fp: int, fn: int) -> float:
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return float(sensitivity + specificity - 1.0)


def _hss(tp: int, tn: int, fp: int, fn: int) -> float:
    numerator = 2.0 * (tp * tn - fn * fp)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _best_threshold_for_scores(y_true: Sequence[int], probabilities: Sequence[float]) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_score = -np.inf
    best_metrics: Dict[str, float] = {"tss": 0.0, "hss": 0.0}
    y_true_arr = np.asarray(y_true, dtype=int)
    prob_arr = np.asarray(probabilities, dtype=float)

    for threshold in _safe_threshold_grid():
        preds = (prob_arr >= threshold).astype(int)
        scores = _confusion_scores(y_true_arr, preds)
        tss = _tss(**scores)
        hss = _hss(**scores)
        if tss > best_score or (math.isclose(tss, best_score) and hss > best_metrics["hss"]):
            best_score = tss
            best_threshold = float(threshold)
            best_metrics = {"tss": float(tss), "hss": float(hss)}

    return best_threshold, best_metrics


def rank_features_via_rf(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_experiments: int = 8,
    random_state: int = 42,
    criterion: str = "gini",
    max_depth: Optional[int] = None,
    top_k: Optional[int] = None,
) -> RankSummary:
    if X.empty:
        raise ValueError("Cannot rank features on an empty dataset")
    if top_k is None:
        top_k = max(1, int(round(math.log2(max(2, X.shape[1])))))

    feature_counts = {name: 0 for name in X.columns}
    importances_sum = {name: 0.0 for name in X.columns}
    experiment_summaries: List[Dict[str, object]] = []

    for exp_index in range(max(1, int(n_experiments))):
        model = RandomForestClassifier(
            n_estimators=250,
            criterion=criterion,
            max_depth=max_depth,
            random_state=random_state + exp_index,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X, y)
        importances = np.asarray(model.feature_importances_, dtype=float)
        order = np.argsort(importances)[::-1]
        selected = [X.columns[idx] for idx in order[:top_k]]

        for name, importance in zip(X.columns, importances):
            importances_sum[name] += float(importance)
        for name in selected:
            feature_counts[name] += 1

        experiment_summaries.append(
            {
                "experiment": exp_index,
                "selected_features": selected,
                "selected_count": len(selected),
                "criterion": criterion,
            }
        )

    averaged_importances = {name: importances_sum[name] / float(max(1, n_experiments)) for name in X.columns}
    ranked = sorted(
        X.columns,
        key=lambda name: (-feature_counts[name], -averaged_importances[name], name),
    )
    selected_features = ranked[:top_k]

    return RankSummary(
        feature_names=list(X.columns),
        selected_features=selected_features,
        top_k=top_k,
        feature_counts=feature_counts,
        mean_importances=averaged_importances,
        experiment_summaries=experiment_summaries,
    )


def _sorted_unique(values: Iterable[str]) -> List[str]:
    return sorted({str(value) for value in values})


def time_segmented_cv(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: Sequence[SampleMetadata],
    *,
    n_splits: int = 5,
    n_experiments: int = 8,
    criterion: str = "gini",
    random_state: int = 42,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
) -> Tuple[Dict[str, object], RankSummary, float, RandomForestClassifier]:
    if len(X) != len(y) or len(X) != len(metadata):
        raise ValueError("X, y, and metadata must have the same length")

    groups = np.asarray([item.group_key for item in metadata], dtype=str)
    unique_groups = _sorted_unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Need at least two time groups for segmented cross-validation")

    n_splits = max(2, min(int(n_splits), len(unique_groups)))
    group_chunks = np.array_split(unique_groups, n_splits)

    fold_summaries: List[FoldSummary] = []
    best_fold: Optional[FoldSummary] = None
    best_model: Optional[RandomForestClassifier] = None
    best_threshold = 0.5
    rank_summary: Optional[RankSummary] = None

    for fold_index in range(1, len(group_chunks)):
        train_groups = set(g for chunk in group_chunks[:fold_index] for g in chunk.tolist())
        test_groups = set(group_chunks[fold_index].tolist())
        train_idx = [i for i, group in enumerate(groups) if group in train_groups]
        test_idx = [i for i, group in enumerate(groups) if group in test_groups]
        if not train_idx or not test_idx:
            continue

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        rank_summary = rank_features_via_rf(
            X_train,
            y_train,
            n_experiments=n_experiments,
            random_state=random_state,
            criterion=criterion,
            max_depth=max_depth,
        )
        selected_features = rank_summary.selected_features
        X_train_sel = X_train[selected_features]
        X_test_sel = X_test[selected_features]

        model = RandomForestClassifier(
            n_estimators=400,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + fold_index,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X_train_sel, y_train)
        probabilities = model.predict_proba(X_test_sel)[:, 1]
        threshold, _ = _best_threshold_for_scores(y_test, probabilities)
        predictions = (probabilities >= threshold).astype(int)
        scores = _confusion_scores(y_test, predictions)
        tss = _tss(**scores)
        hss = _hss(**scores)

        summary = FoldSummary(
            fold_index=fold_index,
            train_groups=sorted(train_groups),
            test_groups=sorted(test_groups),
            n_train=len(train_idx),
            n_test=len(test_idx),
            threshold=threshold,
            tp=scores["tp"],
            tn=scores["tn"],
            fp=scores["fp"],
            fn=scores["fn"],
            tss=tss,
            hss=hss,
        )
        fold_summaries.append(summary)

        if best_fold is None or summary.tss > best_fold.tss or (math.isclose(summary.tss, best_fold.tss) and summary.hss > best_fold.hss):
            best_fold = summary
            best_model = model
            best_threshold = threshold

    if not fold_summaries or best_model is None or rank_summary is None:
        raise ValueError("Cross-validation did not produce any valid folds")

    fold_payload = [asdict(item) for item in fold_summaries]
    summary_payload: Dict[str, object] = {
        "folds": fold_payload,
        "mean_tss": float(np.mean([item.tss for item in fold_summaries])),
        "mean_hss": float(np.mean([item.hss for item in fold_summaries])),
        "best_fold_index": int(best_fold.fold_index),
        "best_fold_threshold": float(best_threshold),
        "n_folds": len(fold_summaries),
        "selected_feature_count": len(rank_summary.selected_features),
    }
    return summary_payload, rank_summary, best_threshold, best_model


def fit_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    selected_features: Sequence[str],
    threshold: float,
    criterion: str = "gini",
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=500,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X[selected_features], y)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    selected_features: Sequence[str],
    threshold: float,
) -> Dict[str, float]:
    probs = model.predict_proba(X[selected_features])[:, 1]
    preds = (probs >= threshold).astype(int)
    scores = _confusion_scores(y, preds)
    return {
        **scores,
        "threshold": float(threshold),
        "tss": _tss(**scores),
        "hss": _hss(**scores),
        "positive_rate": float(np.mean(preds)),
        "mean_probability": float(np.mean(probs)),
    }


def save_artifact(
    *,
    model: RandomForestClassifier,
    out_dir: Path,
    feature_names: Sequence[str],
    selected_features: Sequence[str],
    threshold: float,
    cv_summary: Dict[str, object],
    rank_summary: RankSummary,
    model_params: Dict[str, object],
) -> SlimTSFArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "slimtsf_model.joblib"
    with model_path.open("wb") as fh:
        pickle.dump(model, fh)

    payload = SlimTSFArtifact(
        model_path=str(model_path),
        feature_names=list(feature_names),
        selected_features=list(selected_features),
        threshold=float(threshold),
        cv_summary=cv_summary,
        rank_summary=asdict(rank_summary),
        model_params=model_params,
    )
    (out_dir / "slimtsf_artifact.json").write_text(json.dumps(asdict(payload), indent=2))
    (out_dir / "slimtsf_rank_summary.json").write_text(json.dumps(asdict(rank_summary), indent=2))
    return payload


def load_artifact(artifact_path: Path) -> Tuple[RandomForestClassifier, SlimTSFArtifact]:
    payload = json.loads(artifact_path.read_text())
    model_path = Path(payload["model_path"])
    with model_path.open("rb") as fh:
        model = pickle.load(fh)
    artifact = SlimTSFArtifact(
        model_path=payload["model_path"],
        feature_names=list(payload["feature_names"]),
        selected_features=list(payload["selected_features"]),
        threshold=float(payload["threshold"]),
        cv_summary=payload["cv_summary"],
        rank_summary=payload["rank_summary"],
        model_params=payload["model_params"],
    )
    return model, artifact


def _warning_times_for_prediction(slice_key: str, predicted_positive: bool) -> List[str]:
    if not predicted_positive:
        return []
    dt = _parse_datetime_text(slice_key)
    if dt is None:
        return []
    return [dt.strftime("%Y-%m-%d %H:%M:%S")]


def export_release_json(
    *,
    output_dir: Path,
    slice_key: str,
    probability: float,
    threshold: float,
    source_path: Path,
    selected_features: Sequence[str],
    notes: Optional[Sequence[str]] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    predicted_positive = probability >= threshold
    payload = {
        "mode": "Slim-TSF",
        "model_variant": "slim-tsf-random-forest",
        "source_file": str(source_path),
        "slice_key": slice_key,
        "peak_flux_prediction_target_pfu": float(probability),
        "confidence_level": float(probability),
        "warning_times_utc": _warning_times_for_prediction(slice_key, predicted_positive),
        "expected_onset_times_from_warnings_utc": [],
        "alert_status": "Warning" if predicted_positive else "Quiet",
        "selected_features": list(selected_features),
        "threshold": float(threshold),
        "notes": list(notes or []),
    }
    if payload["warning_times_utc"]:
        payload["expected_onset_times_from_warnings_utc"] = payload["warning_times_utc"]

    out_path = output_dir / f"release_{slice_key}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def train_slim_tsf(
    *,
    data_root: Path,
    gsep_list_csv: Path,
    out_dir: Path,
    time_col: str = TIME_COL_DEFAULT,
    window_size: int = 12,
    step_size: int = 1,
    pooling_width: int = 3,
    value_cols: Optional[Sequence[str]] = None,
    background_window_points: Optional[int] = None,
    background_sigma_clip: Optional[float] = None,
    only_relative_features: bool = False,
    n_splits: int = 5,
    n_experiments: int = 8,
    cv_type: str = "temporal",
    criterion: str = "gini",
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, SlimTSFArtifact, pd.DataFrame, pd.Series, List[SampleMetadata], Dict[str, object]]:
    slice_paths = discover_slice_files(data_root)
    gsep_index = load_gsep_index(gsep_list_csv)
    X, y, metadata = build_dataset_from_slices(
        slice_paths,
        gsep_index=gsep_index,
        time_col=time_col,
        window_size=window_size,
        step_size=step_size,
        pooling_width=pooling_width,
        value_cols=value_cols,
        background_window_points=background_window_points,
        background_sigma_clip=background_sigma_clip,
        only_relative_features=only_relative_features,
    )
    if cv_type == "temporal":
        cv_summary, rank_summary, threshold, _ = time_segmented_cv(
            X,
            y,
            metadata,
            n_splits=n_splits,
            n_experiments=n_experiments,
            criterion=criterion,
            random_state=random_state,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
    else:
        rank_summary = rank_features_via_rf(
            X,
            y,
            n_experiments=n_experiments,
            random_state=random_state,
            criterion=criterion,
            max_depth=max_depth,
        )
        model = fit_final_model(
            X,
            y,
            selected_features=rank_summary.selected_features,
            threshold=0.5,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        probabilities = model.predict_proba(X[rank_summary.selected_features])[:, 1]
        threshold, _ = _best_threshold_for_scores(y, probabilities)
        cv_summary = {
            "folds": [],
            "mean_tss": 0.0,
            "mean_hss": 0.0,
            "best_fold_index": 0,
            "best_fold_threshold": float(threshold),
            "n_folds": 0,
            "selected_feature_count": len(rank_summary.selected_features),
        }

    final_model = fit_final_model(
        X,
        y,
        selected_features=rank_summary.selected_features,
        threshold=threshold,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    artifact = save_artifact(
        model=final_model,
        out_dir=out_dir,
        feature_names=X.columns,
        selected_features=rank_summary.selected_features,
        threshold=threshold,
        cv_summary=cv_summary,
        rank_summary=rank_summary,
        model_params={
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "n_estimators": 500,
            "random_state": random_state,
            "window_size": window_size,
            "step_size": step_size,
            "pooling_width": pooling_width,
            "background_window_points": background_window_points,
            "background_sigma_clip": background_sigma_clip,
            "only_relative_features": only_relative_features,
            "n_splits": n_splits,
            "n_experiments": n_experiments,
            "cv_type": cv_type,
        },
    )
    return final_model, artifact, X, y, metadata, cv_summary


def predict_slices(
    *,
    model: RandomForestClassifier,
    artifact: SlimTSFArtifact,
    slice_paths: Sequence[Path],
    output_dir: Optional[Path] = None,
    time_col: str = TIME_COL_DEFAULT,
    window_size: int = 12,
    step_size: int = 1,
    pooling_width: int = 3,
    value_cols: Optional[Sequence[str]] = None,
    background_window_points: Optional[int] = None,
    background_sigma_clip: Optional[float] = None,
    only_relative_features: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    output_dir = output_dir or Path(artifact.model_path).parent / "release_outputs"
    for path in slice_paths:
        try:
            df = _read_slice_frame(path, time_col=time_col)
            slice_key = _slice_key_from_path(path)
            if slice_key is None and len(df) > 0:
                slice_key = _slice_key_from_datetime(pd.Timestamp(df[time_col].iloc[0]).to_pydatetime())
            if slice_key is None:
                continue
            vector, names = build_feature_vector(
                df,
                time_col=time_col,
                window_size=window_size,
                step_size=step_size,
                pooling_width=pooling_width,
                value_cols=value_cols,
                background_window_points=background_window_points,
                background_sigma_clip=background_sigma_clip,
                only_relative_features=only_relative_features,
            )
            frame = pd.DataFrame([vector], columns=names)
            missing = [name for name in artifact.selected_features if name not in frame.columns]
            if missing:
                raise ValueError(f"Missing selected features for {path}: {missing[:5]}")
            probability = float(model.predict_proba(frame[artifact.selected_features])[:, 1][0])
            out_path = export_release_json(
                output_dir=output_dir,
                slice_key=slice_key,
                probability=probability,
                threshold=artifact.threshold,
                source_path=path,
                selected_features=artifact.selected_features,
                notes=[f"Probability {probability:.4f}", f"Threshold {artifact.threshold:.4f}"],
            )
            rows.append(
                {
                    "source_file": str(path),
                    "slice_key": slice_key,
                    "probability": probability,
                    "predicted_label": int(probability >= artifact.threshold),
                    "output_file": str(out_path),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "source_file": str(path),
                    "slice_key": "",
                    "probability": np.nan,
                    "predicted_label": -1,
                    "output_file": "",
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def build_report(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    selected_features: Sequence[str],
    threshold: float,
) -> Dict[str, float]:
    metrics = evaluate_model(model, X, y, selected_features=selected_features, threshold=threshold)
    metrics["support"] = int(len(y))
    metrics["positive_support"] = int(np.sum(y))
    metrics["negative_support"] = int(len(y) - np.sum(y))
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Slim-TSF on sliced time-series CSV files")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--gsep-list-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="oneoutput/slimtsf")
    parser.add_argument("--time-col", type=str, default=TIME_COL_DEFAULT)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--pooling-width", type=int, default=3)
    parser.add_argument("--value-cols", type=str, nargs="*", default=None)
    parser.add_argument("--background-window-points", type=int, default=None)
    parser.add_argument("--background-sigma-clip", type=float, default=None)
    parser.add_argument("--only-relative-features", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-experiments", type=int, default=8)
    parser.add_argument("--cv-type", type=str, default="temporal", choices=["temporal", "ranked"])
    parser.add_argument("--criterion", type=str, default="gini")
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--model-artifact", type=str, default=None)
    parser.add_argument("--predict-output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root)
    gsep_list_csv = Path(args.gsep_list_csv)
    out_dir = Path(args.out_dir)
    value_cols = args.value_cols if args.value_cols else None

    if args.predict_only:
        if not args.model_artifact:
            raise ValueError("--model-artifact is required with --predict-only")
        model, artifact = load_artifact(Path(args.model_artifact))
        slice_paths = discover_slice_files(data_root)
        predict_slices(
            model=model,
            artifact=artifact,
            slice_paths=slice_paths,
            output_dir=Path(args.predict_output_dir) if args.predict_output_dir else out_dir / "release_outputs",
            time_col=args.time_col,
            window_size=args.window_size,
            step_size=args.step_size,
            pooling_width=args.pooling_width,
            value_cols=value_cols,
            background_window_points=args.background_window_points,
            background_sigma_clip=args.background_sigma_clip,
            only_relative_features=args.only_relative_features,
        )
        return

    model, artifact, X, y, metadata, cv_summary = train_slim_tsf(
        data_root=data_root,
        gsep_list_csv=gsep_list_csv,
        out_dir=out_dir,
        time_col=args.time_col,
        window_size=args.window_size,
        step_size=args.step_size,
        pooling_width=args.pooling_width,
        value_cols=value_cols,
        background_window_points=args.background_window_points,
        background_sigma_clip=args.background_sigma_clip,
        only_relative_features=args.only_relative_features,
        n_splits=args.n_splits,
        n_experiments=args.n_experiments,
        cv_type=args.cv_type,
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )

    report = build_report(
        model,
        X,
        y,
        selected_features=artifact.selected_features,
        threshold=artifact.threshold,
    )
    (out_dir / "slimtsf_train_report.json").write_text(json.dumps(report, indent=2))

    slice_paths = discover_slice_files(data_root)
    predict_slices(
        model=model,
        artifact=artifact,
        slice_paths=slice_paths,
        output_dir=out_dir / "release_outputs",
        time_col=args.time_col,
        window_size=args.window_size,
        step_size=args.step_size,
        pooling_width=args.pooling_width,
        value_cols=value_cols,
        background_window_points=args.background_window_points,
        background_sigma_clip=args.background_sigma_clip,
        only_relative_features=args.only_relative_features,
    )


if __name__ == "__main__":
    main()