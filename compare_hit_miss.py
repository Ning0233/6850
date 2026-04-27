#!/usr/bin/env python3
"""Compare RELease warning outputs against GSEP event start times.

Definitions used in this script:
- Actual event time: `timestamp` column in GSEP_List.csv.
- Warning times: `warning_times_utc` from each RELease JSON output.
- Class `close`: at least one warning in [event_time - 30 min, event_time + 15 min].
- Class `not_found`: no warning in that window.

Paper criteria note:
RELease warning times are assumed to already satisfy the paper warning rules
(actual proton below 20 pfu context, predicted >= 30 pfu, and 2-hour local max).
This script evaluates timing agreement only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass
class ComparisonRow:
    output_file: str
    sep_index: str
    slice_start: str
    event_timestamp: str
    class_label: str
    matched_warning_time: Optional[str]
    matched_warning_delta_minutes: Optional[float]
    warning_count: int
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
        return datetime.fromisoformat(val)
    except ValueError:
        return None


def _slice_key_from_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d_%H-%M")


def _extract_slice_key_from_release_file(path: Path) -> Optional[str]:
    # release_gsep_sc22_ts__1986-02-03_21-25.json -> 1986-02-03_21-25
    m = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})", path.stem)
    return m.group(1) if m else None


def _read_gsep_rows(gsep_list_csv: Path) -> Dict[str, Dict[str, str]]:
    """Map slice_start key -> representative GSEP row.

    If duplicates appear for the same slice_start, keep the first row.
    """
    by_slice_key: Dict[str, Dict[str, str]] = {}
    with gsep_list_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            slice_start = _parse_dt(row.get("slice_start", ""))
            event_ts = _parse_dt(row.get("timestamp", ""))
            if slice_start is None or event_ts is None:
                continue
            key = _slice_key_from_dt(slice_start)
            by_slice_key.setdefault(key, row)
    return by_slice_key


def _load_times_from_release(release_json: Path, match_time_field: str) -> List[datetime]:
    payload = json.loads(release_json.read_text())
    raw = payload.get(match_time_field, [])
    out: List[datetime] = []
    if isinstance(raw, list):
        for item in raw:
            dt = _parse_dt(str(item))
            if dt is not None:
                out.append(dt)
    out.sort()
    return out


def _classify_event(event_ts: datetime, warning_times: Sequence[datetime]) -> tuple[str, Optional[datetime], Optional[float]]:
    window_start = event_ts - timedelta(minutes=30)
    window_end = event_ts + timedelta(minutes=15)

    in_window = [wt for wt in warning_times if window_start <= wt <= window_end]
    if in_window:
        closest = min(in_window, key=lambda wt: abs((wt - event_ts).total_seconds()))
        delta_min = (closest - event_ts).total_seconds() / 60.0
        return "close", closest, float(delta_min)

    return "not_found", None, None


def _build_rows(
    gsep_map: Dict[str, Dict[str, str]],
    release_dir: Path,
    match_time_field: str,
) -> List[ComparisonRow]:
    rows: List[ComparisonRow] = []
    for release_file in sorted(release_dir.glob("release_*.json")):
        key = _extract_slice_key_from_release_file(release_file)
        if key is None:
            rows.append(
                ComparisonRow(
                    output_file=str(release_file),
                    sep_index="",
                    slice_start="",
                    event_timestamp="",
                    class_label="not_found",
                    matched_warning_time=None,
                    matched_warning_delta_minutes=None,
                    warning_count=0,
                    notes="Could not parse slice timestamp key from output filename",
                )
            )
            continue

        gsep_row = gsep_map.get(key)
        if gsep_row is None:
            rows.append(
                ComparisonRow(
                    output_file=str(release_file),
                    sep_index="",
                    slice_start=key,
                    event_timestamp="",
                    class_label="not_found",
                    matched_warning_time=None,
                    matched_warning_delta_minutes=None,
                    warning_count=0,
                    notes="No matching GSEP row found by slice_start key",
                )
            )
            continue

        event_ts = _parse_dt(gsep_row.get("timestamp", ""))
        warning_times = _load_times_from_release(release_file, match_time_field=match_time_field)

        if event_ts is None:
            label, matched, delta = "not_found", None, None
            notes = "Missing/invalid GSEP timestamp"
        else:
            label, matched, delta = _classify_event(event_ts=event_ts, warning_times=warning_times)
            notes = ""

        rows.append(
            ComparisonRow(
                output_file=str(release_file),
                sep_index=str(gsep_row.get("sep_index", "")),
                slice_start=str(gsep_row.get("slice_start", "")),
                event_timestamp=str(gsep_row.get("timestamp", "")),
                class_label=label,
                matched_warning_time=matched.strftime("%Y-%m-%d %H:%M:%S") if matched else None,
                matched_warning_delta_minutes=delta,
                warning_count=len(warning_times),
                notes=notes,
            )
        )

    return rows


def _write_rows_csv(rows: Sequence[ComparisonRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "output_file",
                "sep_index",
                "slice_start",
                "event_timestamp",
                "class_label",
                "matched_warning_time",
                "matched_warning_delta_minutes",
                "warning_count",
                "notes",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "output_file": r.output_file,
                    "sep_index": r.sep_index,
                    "slice_start": r.slice_start,
                    "event_timestamp": r.event_timestamp,
                    "class_label": r.class_label,
                    "matched_warning_time": r.matched_warning_time or "",
                    "matched_warning_delta_minutes": "" if r.matched_warning_delta_minutes is None else f"{r.matched_warning_delta_minutes:.2f}",
                    "warning_count": r.warning_count,
                    "notes": r.notes,
                }
            )


def _write_summary_json(rows: Sequence[ComparisonRow], out_json: Path) -> None:
    total = len(rows)
    close_n = sum(1 for r in rows if r.class_label == "close")
    not_found_n = total - close_n

    matched_deltas = [r.matched_warning_delta_minutes for r in rows if r.matched_warning_delta_minutes is not None]
    mean_delta = (sum(matched_deltas) / len(matched_deltas)) if matched_deltas else None

    summary = {
        "definitions": {
            "close": "At least one warning in [event_time - 30 min, event_time + 15 min]",
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
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare RELease warnings against GSEP event starts")
    p.add_argument(
        "--gsep-list-csv",
        type=str,
        default="/Users/ran/cs/oires/data/raw/gsep_ts/GSEP_List.csv",
        help="Path to GSEP_List.csv (uses timestamp as actual event start, slice_start for mapping)",
    )
    p.add_argument(
        "--release-output-dir",
        type=str,
        default="/Users/ran/cs/6850/project/oneoutput/release_batch_outputs",
        help="Directory containing RELease JSON outputs (release_*.json)",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="/Users/ran/cs/6850/project/oneoutput/release_hit_miss_comparison.csv",
        help="Per-file comparison output CSV",
    )
    p.add_argument(
        "--out-summary-json",
        type=str,
        default="/Users/ran/cs/6850/project/oneoutput/release_hit_miss_summary.json",
        help="Summary metrics JSON",
    )
    p.add_argument(
        "--match-time-field",
        type=str,
        choices=["warning_times_utc", "expected_onset_times_from_warnings_utc"],
        default="warning_times_utc",
        help="RELease JSON list field to compare against event timestamp",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    gsep_csv = Path(args.gsep_list_csv)
    release_dir = Path(args.release_output_dir)
    out_csv = Path(args.out_csv)
    out_summary = Path(args.out_summary_json)

    if not gsep_csv.is_file():
        raise FileNotFoundError(f"GSEP list CSV not found: {gsep_csv}")
    if not release_dir.is_dir():
        raise FileNotFoundError(f"RELease output directory not found: {release_dir}")

    gsep_map = _read_gsep_rows(gsep_csv)
    rows = _build_rows(
        gsep_map=gsep_map,
        release_dir=release_dir,
        match_time_field=args.match_time_field,
    )

    _write_rows_csv(rows, out_csv)
    _write_summary_json(rows, out_summary)

    total = len(rows)
    close_n = sum(1 for r in rows if r.class_label == "close")
    not_found_n = total - close_n
    print(f"Compared outputs: {total}")
    print(f"Matched field: {args.match_time_field}")
    print(f"close: {close_n}")
    print(f"not_found: {not_found_n}")
    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote summary JSON: {out_summary}")


if __name__ == "__main__":
    main()
