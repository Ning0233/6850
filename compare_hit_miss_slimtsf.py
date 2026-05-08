#!/usr/bin/env python3
"""Evaluate Slim-TSF warnings against GSEP events with event-centric logic.

Definitions:
- Actual event time: `timestamp` column in GSEP_List.csv.
- Warnings: aggregated from all Slim-TSF JSON outputs in a directory.
- Class `close`: at least one warning in [event_time - 120 min, event_time + 15 min].
- Class `not_found`: no warning in that window.

Unlike `compare_hit_miss.py` (file-to-row matching), this script is event-centric:
for each GSEP event, check whether any Slim-TSF warning exists in the timing window.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class EventComparisonRow:
    sep_index: str
    slice_start: str
    event_timestamp: str
    class_label: str
    matched_warning_time: Optional[str]
    matched_warning_delta_minutes: Optional[float]
    total_warning_candidates: int
    notes: str


def _parse_dt(text: str) -> Optional[datetime]:
    val = (text or "").strip()
    if not val:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(val)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        return None


def _read_gsep_rows(gsep_list_csv: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with gsep_list_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            event_ts = _parse_dt(row.get("timestamp", ""))
            if event_ts is None:
                continue
            rows.append(row)
    return rows


def _collect_warning_times(slimtsf_output_dir: Path, match_time_field: str) -> Tuple[List[datetime], int]:
    warning_times: List[datetime] = []
    files_scanned = 0

    for json_file in sorted(slimtsf_output_dir.glob("release_*.json")):
        files_scanned += 1
        try:
            payload = json.loads(json_file.read_text())
        except Exception:
            continue

        raw = payload.get(match_time_field, [])
        if not isinstance(raw, list):
            continue

        for item in raw:
            dt = _parse_dt(str(item))
            if dt is not None:
                warning_times.append(dt)

    warning_times.sort()
    return warning_times, files_scanned


def _collect_warning_times_from_predictions_csv(predictions_csv: Path) -> Tuple[List[datetime], int]:
    warning_times: List[datetime] = []
    rows_scanned = 0

    with predictions_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows_scanned += 1
            try:
                pred_label = int(str(row.get("predicted_label", "-1")).strip())
            except ValueError:
                continue
            if pred_label != 1:
                continue

            slice_key = str(row.get("slice_key", "")).strip()
            dt = _parse_dt(slice_key)
            if dt is not None:
                warning_times.append(dt)

    warning_times.sort()
    return warning_times, rows_scanned


def _classify_event(event_ts: datetime, warning_times: Sequence[datetime]) -> Tuple[str, Optional[datetime], Optional[float]]:
    window_start = event_ts - timedelta(minutes=120)
    window_end = event_ts + timedelta(minutes=15)

    in_window = [wt for wt in warning_times if window_start <= wt <= window_end]
    if in_window:
        closest = min(in_window, key=lambda wt: abs((wt - event_ts).total_seconds()))
        delta_min = (closest - event_ts).total_seconds() / 60.0
        return "close", closest, float(delta_min)

    return "not_found", None, None


def _build_rows(gsep_rows: Sequence[Dict[str, str]], warning_times: Sequence[datetime]) -> List[EventComparisonRow]:
    rows: List[EventComparisonRow] = []
    candidate_count = len(warning_times)

    for row in gsep_rows:
        event_ts = _parse_dt(row.get("timestamp", ""))
        if event_ts is None:
            continue

        label, matched, delta = _classify_event(event_ts=event_ts, warning_times=warning_times)
        rows.append(
            EventComparisonRow(
                sep_index=str(row.get("sep_index", "")),
                slice_start=str(row.get("slice_start", "")),
                event_timestamp=str(row.get("timestamp", "")),
                class_label=label,
                matched_warning_time=matched.strftime("%Y-%m-%d %H:%M:%S") if matched else None,
                matched_warning_delta_minutes=delta,
                total_warning_candidates=candidate_count,
                notes="",
            )
        )

    return rows


def _write_rows_csv(rows: Sequence[EventComparisonRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sep_index",
                "slice_start",
                "event_timestamp",
                "class_label",
                "matched_warning_time",
                "matched_warning_delta_minutes",
                "total_warning_candidates",
                "notes",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "sep_index": r.sep_index,
                    "slice_start": r.slice_start,
                    "event_timestamp": r.event_timestamp,
                    "class_label": r.class_label,
                    "matched_warning_time": r.matched_warning_time or "",
                    "matched_warning_delta_minutes": "" if r.matched_warning_delta_minutes is None else f"{r.matched_warning_delta_minutes:.2f}",
                    "total_warning_candidates": r.total_warning_candidates,
                    "notes": r.notes,
                }
            )


def _write_summary_json(rows: Sequence[EventComparisonRow], out_json: Path, files_scanned: int) -> None:
    total = len(rows)
    close_n = sum(1 for r in rows if r.class_label == "close")
    not_found_n = total - close_n

    matched_deltas = [r.matched_warning_delta_minutes for r in rows if r.matched_warning_delta_minutes is not None]
    mean_delta = (sum(matched_deltas) / len(matched_deltas)) if matched_deltas else None

    summary = {
        "definitions": {
            "close": "At least one warning in [event_time - 120 min, event_time + 15 min]",
            "not_found": "No warning in that window",
        },
        "counts": {
            "total": total,
            "close": close_n,
            "not_found": not_found_n,
        },
        "rates": {
            "close_rate": 0.0 if total == 0 else close_n / total,
            "not_found_rate": 0.0 if total == 0 else not_found_n / total,
        },
        "timing": {
            "matched_delta_minutes_mean": mean_delta,
        },
        "slimtsf_eval": {
            "mode": "event-centric",
            "release_json_files_scanned": files_scanned,
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Event-centric Slim-TSF hit/miss evaluation against GSEP")
    p.add_argument(
        "--gsep-list-csv",
        type=str,
        default="/Users/ran/cs/oires/data/raw/gsep_ts/GSEP_List.csv",
        help="Path to GSEP_List.csv (uses timestamp as event time)",
    )
    p.add_argument(
        "--slimtsf-output-dir",
        type=str,
        default=None,
        help="Directory containing Slim-TSF release_*.json outputs",
    )
    p.add_argument(
        "--slimtsf-predictions-csv",
        type=str,
        default=None,
        help="Slim-TSF prediction table (must include slice_key and predicted_label columns)",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="/Users/ran/cs/6850/project/oneoutput/slimtsf/slimtsf_hit_miss_comparison_historical.csv",
        help="Per-event comparison output CSV",
    )
    p.add_argument(
        "--out-summary-json",
        type=str,
        default="/Users/ran/cs/6850/project/oneoutput/slimtsf/slimtsf_hit_miss_summary_historical.json",
        help="Summary metrics JSON",
    )
    p.add_argument(
        "--match-time-field",
        type=str,
        choices=["warning_times_utc", "expected_onset_times_from_warnings_utc"],
        default="warning_times_utc",
        help="Slim-TSF JSON list field containing warning timestamps",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    gsep_csv = Path(args.gsep_list_csv)
    slimtsf_dir = Path(args.slimtsf_output_dir) if args.slimtsf_output_dir else None
    predictions_csv = Path(args.slimtsf_predictions_csv) if args.slimtsf_predictions_csv else None
    out_csv = Path(args.out_csv)
    out_summary = Path(args.out_summary_json)

    if not gsep_csv.is_file():
        raise FileNotFoundError(f"GSEP list CSV not found: {gsep_csv}")
    if predictions_csv is None and slimtsf_dir is None:
        raise ValueError("Provide either --slimtsf-predictions-csv or --slimtsf-output-dir")
    if predictions_csv is not None and not predictions_csv.is_file():
        raise FileNotFoundError(f"Slim-TSF predictions CSV not found: {predictions_csv}")
    if predictions_csv is None and slimtsf_dir is not None and not slimtsf_dir.is_dir():
        raise FileNotFoundError(f"Slim-TSF output directory not found: {slimtsf_dir}")

    gsep_rows = _read_gsep_rows(gsep_csv)
    if predictions_csv is not None:
        warning_times, files_scanned = _collect_warning_times_from_predictions_csv(predictions_csv)
    else:
        assert slimtsf_dir is not None
        warning_times, files_scanned = _collect_warning_times(slimtsf_dir, args.match_time_field)
    rows = _build_rows(gsep_rows=gsep_rows, warning_times=warning_times)

    _write_rows_csv(rows, out_csv)
    _write_summary_json(rows, out_summary, files_scanned=files_scanned)

    total = len(rows)
    close_n = sum(1 for r in rows if r.class_label == "close")
    print(f"GSEP events evaluated: {total}")
    print(f"Slim-TSF output files scanned: {files_scanned}")
    print(f"close: {close_n}")
    print(f"not_found: {total - close_n}")
    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote summary JSON: {out_summary}")


if __name__ == "__main__":
    main()
