#!/usr/bin/env python3
"""Generate heatmaps from a trained delta-matrix JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _read_cells(path: Path) -> List[Dict[str, float]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list of matrix cells in {path}")
    if not data:
        raise ValueError(f"Matrix file is empty: {path}")
    return data


def _get_target_value(cell: Dict[str, float], target_field: str) -> float:
    if target_field != "auto":
        if target_field not in cell:
            raise KeyError(f"Missing target field '{target_field}' in matrix cell")
        return float(cell[target_field])

    if "target_peak_pfu" in cell:
        return float(cell["target_peak_pfu"])
    if "p30_peak_pfu" in cell:
        return float(cell["p30_peak_pfu"])
    raise KeyError("Matrix cell is missing both 'target_peak_pfu' and 'p30_peak_pfu'")


def _build_grids(
    cells: Sequence[Dict[str, float]],
    target_field: str,
) -> tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    i_vals = sorted({float(c["intensity_min"]) for c in cells})
    d_vals = sorted({float(c["delta_min"]) for c in cells})
    i_idx = {v: idx for idx, v in enumerate(i_vals)}
    d_idx = {v: idx for idx, v in enumerate(d_vals)}

    target_grid = np.full((len(d_vals), len(i_vals)), np.nan, dtype=float)
    conf_grid = np.full((len(d_vals), len(i_vals)), np.nan, dtype=float)

    for cell in cells:
        ii = i_idx[float(cell["intensity_min"])]
        dd = d_idx[float(cell["delta_min"])]
        target_grid[dd, ii] = _get_target_value(cell, target_field)
        conf_grid[dd, ii] = float(cell.get("confidence", 0.0))

    return target_grid, conf_grid, i_vals, d_vals


def _tick_labels(values: Sequence[float], max_ticks: int = 8) -> tuple[List[int], List[str]]:
    n = len(values)
    if n <= max_ticks:
        idx = list(range(n))
    else:
        idx = np.linspace(0, n - 1, max_ticks, dtype=int).tolist()
    labels = [f"{values[i]:.2g}" for i in idx]
    return idx, labels


def _plot_heatmap(
    grid: np.ndarray,
    out_path: Path,
    title: str,
    colorbar_label: str,
    intensity_values: Sequence[float],
    delta_values: Sequence[float],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(grid, origin="lower", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    x_idx, x_labels = _tick_labels(intensity_values)
    y_idx, y_labels = _tick_labels(delta_values)

    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.set_yticks(y_idx)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("Intensity bin lower edge")
    ax.set_ylabel("Delta bin lower edge")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate delta-matrix heatmaps")
    parser.add_argument(
        "--matrix-json",
        type=Path,
        required=True,
        help="Path to the trained delta matrix JSON.",
    )
    parser.add_argument(
        "--out-target-png",
        type=Path,
        default=Path("oneoutput/matrix_target_heatmap.png"),
        help="Output PNG for target proton prediction heatmap.",
    )
    parser.add_argument(
        "--out-confidence-png",
        type=Path,
        default=Path("oneoutput/matrix_confidence_heatmap.png"),
        help="Output PNG for confidence heatmap.",
    )
    parser.add_argument(
        "--target-field",
        type=str,
        default="auto",
        help="Target field in matrix cells. Use 'auto', 'target_peak_pfu', or 'p30_peak_pfu'.",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="Delta Matrix",
        help="Prefix used in plot titles.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    cells = _read_cells(args.matrix_json)
    target_grid, conf_grid, i_vals, d_vals = _build_grids(cells, target_field=args.target_field)

    _plot_heatmap(
        grid=target_grid,
        out_path=args.out_target_png,
        title=f"{args.title_prefix}: Target Proton Forecast",
        colorbar_label="Target Proton Flux (pfu)",
        intensity_values=i_vals,
        delta_values=d_vals,
    )
    _plot_heatmap(
        grid=conf_grid,
        out_path=args.out_confidence_png,
        title=f"{args.title_prefix}: Confidence",
        colorbar_label="Confidence",
        intensity_values=i_vals,
        delta_values=d_vals,
    )

    print(f"Wrote target heatmap: {args.out_target_png}")
    print(f"Wrote confidence heatmap: {args.out_confidence_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
