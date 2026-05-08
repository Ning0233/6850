from __future__ import annotations

import csv
from pathlib import Path


def parse_identifier(filename: str, label: int) -> str | None:
    stem = Path(filename).stem
    if "T" not in stem:
        return None
    date_part, time_part = stem.split("T", 1)
    time_parts = time_part.split("-")
    if len(time_parts) < 2:
        return None
    hh_mm = "-".join(time_parts[:2])
    prefix = "event" if label == 1 else "non-event"
    return f"{prefix}-{date_part}_{hh_mm}"


def build_index_rows(slice_root: Path, data_root: Path) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    for path in sorted(slice_root.rglob("*.csv")):
        if "non_event" in path.parts:
            label = 0
        else:
            label = 1
        identifier = parse_identifier(path.name, label)
        if identifier is None:
            continue
        rel_path = path.relative_to(data_root).as_posix()
        rows.append((identifier, rel_path, label))
    return rows


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_root = project_root / "data"
    slice_root = data_root / "2026-05-08" / "slice"
    output_path = data_root / "2026-05-08-train-index.csv"

    rows = build_index_rows(slice_root, data_root)

    with output_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["identifier", "file_root", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
