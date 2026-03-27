#!/usr/bin/env python3
"""Export Move 100 samples side-by-side for PowerAllFour2..11 datasets.

Uses move_energy_grid{n}.csv Move 100 start/end times to slice raw analyzer traces
from AllSpeedsAllAccelsRealRobot/PowerAllFour{n}.CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from extract_move_energy_grid import load_power_csv


def read_move_window(grid_csv: Path, move_number: int) -> Tuple[float, float, str, str]:
    with grid_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    idx = move_number - 1
    if idx < 0 or idx >= len(rows):
        raise ValueError(f"{grid_csv.name}: move {move_number} is out of range (1..{len(rows)})")

    row = rows[idx]
    start_s = float(row["Move Start (s)"])
    end_s = float(row["Move End (s)"])
    speed = row.get("Speed", "")
    accel = row.get("Accel", "")
    return start_s, end_s, speed, accel


def extract_segment_samples(raw_csv: Path, start_s: float, end_s: float) -> Tuple[np.ndarray, np.ndarray, float]:
    series = load_power_csv(raw_csv)

    # Include endpoints; small epsilon protects against floating-point edge mismatch.
    eps = 1e-9
    mask = (series.time_s >= (start_s - eps)) & (series.time_s <= (end_s + eps))
    seg_t = np.asarray(series.time_s[mask], dtype=float)
    seg_p = np.asarray(series.power_w[mask], dtype=float)

    if seg_t.size == 0:
        raise ValueError(f"No samples found in window {start_s}..{end_s} for {raw_csv.name}")

    seg_t_rel = seg_t - seg_t[0]
    return seg_t_rel, seg_p, float(seg_t[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Move 100 side-by-side values for datasets 2..11")
    parser.add_argument("--move-number", type=int, default=100, help="1-based move number (default: 100)")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("AllSpeedsAllAccelsRealRobot"),
        help="Directory containing PowerAllFour*.CSV",
    )
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=Path("."),
        help="Directory containing move_energy_grid*.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/move100_all10_side_by_side.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    dataset_ids = list(range(2, 12))

    traces: Dict[int, Tuple[np.ndarray, np.ndarray, float, float, str, str]] = {}
    # value tuple: (t_rel, p, abs_start, abs_end, speed, accel)
    for dataset_id in dataset_ids:
        grid_csv = args.grid_dir / f"move_energy_grid{dataset_id}.csv"
        raw_csv = args.input_dir / f"PowerAllFour{dataset_id}.CSV"

        if not grid_csv.exists():
            raise SystemExit(f"Missing grid CSV: {grid_csv}")
        if not raw_csv.exists():
            raise SystemExit(f"Missing raw CSV: {raw_csv}")

        start_s, end_s, speed, accel = read_move_window(grid_csv, args.move_number)
        t_rel, pwr, abs_start = extract_segment_samples(raw_csv, start_s, end_s)
        traces[dataset_id] = (t_rel, pwr, abs_start, end_s, speed, accel)

    max_len = max(len(traces[i][0]) for i in dataset_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)

        meta_header = ["dataset", "speed", "accel", "abs_start_s", "abs_end_s", "samples"]
        writer.writerow(meta_header)
        for dataset_id in dataset_ids:
            t_rel, pwr, abs_start, abs_end, speed, accel = traces[dataset_id]
            writer.writerow([
                f"PowerAllFour{dataset_id}",
                speed,
                accel,
                f"{abs_start:.6f}",
                f"{abs_end:.6f}",
                len(t_rel),
            ])

        writer.writerow([])

        header = ["sample_index"]
        for dataset_id in dataset_ids:
            header.append(f"PowerAllFour{dataset_id}_time_rel_s")
            header.append(f"PowerAllFour{dataset_id}_power_w")
        writer.writerow(header)

        for sample_idx in range(max_len):
            row: List[str] = [str(sample_idx)]
            for dataset_id in dataset_ids:
                t_rel, pwr, _, _, _, _ = traces[dataset_id]
                if sample_idx < len(t_rel):
                    row.append(f"{t_rel[sample_idx]:.6f}")
                    row.append(f"{pwr[sample_idx]:.6f}")
                else:
                    row.append("")
                    row.append("")
            writer.writerow(row)

    print(f"Wrote side-by-side CSV: {args.output}")


if __name__ == "__main__":
    main()
