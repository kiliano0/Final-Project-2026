import argparse
import csv
import math
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from EnergyModel4 import (
    SCARAEnergyModel,
    SIM_ACCEL_SCALE,
    SIM_BRAKE_EFFICIENCY,
    SIM_CONSTANT_POWER,
    SIM_GRAVITY,
    SIM_I1,
    SIM_I2,
    SIM_J1_HIGH_SPEED_REDUCTION,
    SIM_J1_LOW_SPEED_BOOST,
    SIM_J1_SHAPE_EXP,
    SIM_J1_SHAPE_REF_SPEED,
    SIM_J2_HIGH_SPEED_REDUCTION,
    SIM_J2_LOW_SPEED_BOOST,
    SIM_J2_SHAPE_EXP,
    SIM_J2_SHAPE_REF_SPEED,
    SIM_J3_HIGH_SPEED_BOOST,
    SIM_J3_LOW_SPEED_BOOST,
    SIM_J3_SHAPE_EXP,
    SIM_J3_SHAPE_REF_SPEED,
    SIM_J4_COULOMB,
    SIM_J4_HIGH_SPEED_BOOST,
    SIM_J4_INERTIA,
    SIM_J4_LOW_SPEED_BOOST,
    SIM_J4_SHAPE_EXP,
    SIM_J4_SHAPE_REF_SPEED,
    SIM_L1,
    SIM_L2,
    SIM_M1,
    SIM_M2,
    SIM_MOTOR_EFF,
    SIM_QUILL_DEADBAND,
    SIM_QUILL_GRAVITY_MOVING_ONLY,
    SIM_QUILL_MASS,
    SIM_REGEN_EFF,
    SIM_SCARA_HORIZONTAL,
    SIM_SPEED_EXP,
    SIM_SPEED_SQ,
    SIM_USE_POWER_TO_BRAKE,
    SIM_VISCOUS,
)

FILENAME_PATTERN = re.compile(
    r"Speed(?P<speed>[-+]?\d+(?:\.\d+)?)_Accel(?P<accel>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _parse_float(value: Optional[str]) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _extract_speed_accel(path: Path) -> Tuple[str, str]:
    match = FILENAME_PATTERN.search(path.stem)
    if not match:
        return "", ""
    return match.group("speed"), match.group("accel")


def load_trajectory_with_derivatives_from_csv(
    path: Path,
    time_col: str = "Time_Total",
    pos_cols: Optional[List[str]] = None,
    vel_cols: Optional[List[str]] = None,
    acc_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        if fields is None:
            raise ValueError(f"CSV has no header: {path}")

        if pos_cols is None:
            pos_cols = [f for f in fields if f.startswith("J") and "_Position" in f]
        if vel_cols is None:
            vel_cols = [f for f in fields if f.startswith("J") and "_Velocity" in f]
        if acc_cols is None:
            acc_cols = [f for f in fields if f.startswith("J") and "_Acceleration" in f]

        if not pos_cols or not vel_cols or not acc_cols:
            raise ValueError(f"Missing required joint columns in {path.name}")

        t_list: List[float] = []
        q_lists = [[] for _ in pos_cols]
        qd_lists = [[] for _ in vel_cols]
        qdd_lists = [[] for _ in acc_cols]

        for row in reader:
            t_list.append(_parse_float(row.get(time_col)))
            for i, col in enumerate(pos_cols):
                q_lists[i].append(_parse_float(row.get(col)))
            for i, col in enumerate(vel_cols):
                qd_lists[i].append(_parse_float(row.get(col)))
            for i, col in enumerate(acc_cols):
                qdd_lists[i].append(_parse_float(row.get(col)))

    if not t_list:
        raise ValueError(f"CSV has no data rows: {path.name}")

    t = np.array(t_list, dtype=float)
    q = np.vstack(q_lists).T
    qd = np.vstack(qd_lists).T
    qdd = np.vstack(qdd_lists).T

    for j, col in enumerate(pos_cols):
        base = col.split("_")[0]
        is_prismatic = base.upper().startswith("Z")
        if base.startswith("J"):
            try:
                if int(base[1:]) == 3:
                    is_prismatic = True
            except Exception:
                pass

        if is_prismatic:
            q[:, j] *= 1e-3
            qd[:, j] *= 1e-3
            qdd[:, j] *= 1e-3
        else:
            q[:, j] *= np.pi / 180.0
            qd[:, j] *= np.pi / 180.0
            qdd[:, j] *= np.pi / 180.0

    if len(t) > 1:
        dt = np.diff(t)
        valid_mask = np.concatenate([[True], dt > 1e-9])
        t = t[valid_mask]
        q = q[valid_mask]
        qd = qd[valid_mask]
        qdd = qdd[valid_mask]

    if len(t) < 2:
        raise ValueError(f"Not enough valid time samples after filtering: {path.name}")

    return t, q, qd, qdd


def build_model() -> SCARAEnergyModel:
    return SCARAEnergyModel(
        l1=SIM_L1,
        l2=SIM_L2,
        m1=SIM_M1,
        m2=SIM_M2,
        I1=SIM_I1,
        I2=SIM_I2,
        g=SIM_GRAVITY,
        motor_eff=SIM_MOTOR_EFF,
        regen_eff=SIM_REGEN_EFF,
        constant_power=SIM_CONSTANT_POWER,
        viscous_friction=SIM_VISCOUS,
        speed_squared_loss=SIM_SPEED_SQ,
        accel_scale=SIM_ACCEL_SCALE,
        speed_loss_exponent=SIM_SPEED_EXP,
        scara_horizontal=SIM_SCARA_HORIZONTAL,
        use_power_to_brake=SIM_USE_POWER_TO_BRAKE,
        brake_efficiency=SIM_BRAKE_EFFICIENCY,
        quill_mass=SIM_QUILL_MASS,
        quill_gravity_moving_only=SIM_QUILL_GRAVITY_MOVING_ONLY,
        quill_deadband=SIM_QUILL_DEADBAND,
        j1_low_speed_boost=SIM_J1_LOW_SPEED_BOOST,
        j1_high_speed_reduction=SIM_J1_HIGH_SPEED_REDUCTION,
        j1_shape_ref_speed=SIM_J1_SHAPE_REF_SPEED,
        j1_shape_exp=SIM_J1_SHAPE_EXP,
        j2_low_speed_boost=SIM_J2_LOW_SPEED_BOOST,
        j2_high_speed_reduction=SIM_J2_HIGH_SPEED_REDUCTION,
        j2_shape_ref_speed=SIM_J2_SHAPE_REF_SPEED,
        j2_shape_exp=SIM_J2_SHAPE_EXP,
        j3_low_speed_boost=SIM_J3_LOW_SPEED_BOOST,
        j3_high_speed_boost=SIM_J3_HIGH_SPEED_BOOST,
        j3_shape_ref_speed=SIM_J3_SHAPE_REF_SPEED,
        j3_shape_exp=SIM_J3_SHAPE_EXP,
        j4_low_speed_boost=SIM_J4_LOW_SPEED_BOOST,
        j4_high_speed_boost=SIM_J4_HIGH_SPEED_BOOST,
        j4_shape_ref_speed=SIM_J4_SHAPE_REF_SPEED,
        j4_shape_exp=SIM_J4_SHAPE_EXP,
        j4_inertia=SIM_J4_INERTIA,
        j4_coulomb=SIM_J4_COULOMB,
    )


def summarize_one(
    path: Path,
    model: SCARAEnergyModel,
    time_col: str,
    power_mode: str,
    collect_trace: bool = False,
) -> dict:
    t, q, qd, qdd = load_trajectory_with_derivatives_from_csv(path=path, time_col=time_col)

    elec_power_samples = np.empty(len(t), dtype=float)
    joint_power_rows: List[np.ndarray] = []
    for i in range(len(t)):
        _, _, elec_total, elec_per_joint = model.torques(q[i], qd[i], qdd[i])
        elec_power_samples[i] = elec_total
        if collect_trace:
            joint_power_rows.append(np.asarray(elec_per_joint, dtype=float))

    total_energy_j = float(np.trapezoid(elec_power_samples, t))
    duration_s = float(t[-1] - t[0])
    avg_power_w = total_energy_j / duration_s if duration_s > 0 else float(np.mean(elec_power_samples))

    speed, accel = _extract_speed_accel(path)
    power_value = total_energy_j if power_mode == "total_energy_j" else avg_power_w

    row = {
        "Speed": speed,
        "Accel": accel,
        "Time": f"{duration_s:.9g}",
        "Power": f"{power_value:.9g}",
        "_speed_sort": float(speed) if speed else float("inf"),
        "_accel_sort": float(accel) if accel else float("inf"),
        "_name": path.name,
    }
    if collect_trace:
        if joint_power_rows:
            joint_power = np.vstack(joint_power_rows)
        else:
            joint_power = np.zeros((len(t), 0), dtype=float)
        if joint_power.shape[1] < 4:
            padded = np.zeros((joint_power.shape[0], 4), dtype=float)
            padded[:, : joint_power.shape[1]] = joint_power
            joint_power = padded

        row["_t"] = t
        row["_p"] = elec_power_samples
        row["_joint_p"] = joint_power[:, :4]
        row["_base_p"] = np.full(len(t), float(model.constant_power), dtype=float)

    return row


def plot_power_pages(
    rows: List[dict],
    output_dir: Path,
    plots_per_page: int = 12,
    cols: int = 3,
) -> List[Path]:
    """Plot per-file power traces in paged grids (default 4x3 => 12 plots/page)."""
    if not rows:
        return []

    if plots_per_page <= 0:
        plots_per_page = 12
    if cols <= 0:
        cols = 3

    grid_rows = int(math.ceil(plots_per_page / cols))
    output_dir.mkdir(parents=True, exist_ok=True)
    page_paths: List[Path] = []
    page_count = int(math.ceil(len(rows) / plots_per_page))

    for page in range(page_count):
        start_idx = page * plots_per_page
        end_idx = min(len(rows), start_idx + plots_per_page)
        subset = rows[start_idx:end_idx]

        figure, axes = plt.subplots(
            grid_rows,
            cols,
            figsize=(5.4 * cols, 2.9 * grid_rows),
            constrained_layout=True,
        )
        if isinstance(axes, np.ndarray):
            axes_flat = axes.ravel().tolist()
        else:
            axes_flat = [axes]

        for panel_idx in range(grid_rows * cols):
            axis = axes_flat[panel_idx]
            if panel_idx >= len(subset):
                axis.axis("off")
                continue

            row = subset[panel_idx]
            t = row.get("_t")
            p_total = row.get("_p")
            joint_p = row.get("_joint_p")
            base_p = row.get("_base_p")
            if t is None or p_total is None or joint_p is None or base_p is None:
                axis.axis("off")
                continue

            move_idx = start_idx + panel_idx + 1
            speed = row.get("Speed", "")
            accel = row.get("Accel", "")
            stack_components = [
                np.asarray(base_p, dtype=float),
                np.asarray(joint_p[:, 0], dtype=float),
                np.asarray(joint_p[:, 1], dtype=float),
                np.asarray(joint_p[:, 2], dtype=float),
                np.asarray(joint_p[:, 3], dtype=float),
            ]
            stack_labels = ["Base", "J1", "J2", "J3", "J4"]
            stack_colors = ["#d9d9d9", "#4c78a8", "#f58518", "#54a24b", "#e45756"]

            axis.stackplot(t, stack_components, labels=stack_labels, colors=stack_colors, alpha=0.9)
            axis.plot(t, np.asarray(p_total, dtype=float), color="black", linewidth=1.0, label="Total")
            axis.set_title(f"Move {move_idx}: S{speed} A{accel}")
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("Power (W)")
            axis.set_ylim(bottom=80)
            axis.grid(True, alpha=0.25)
            if panel_idx == 0:
                axis.legend(loc="best", fontsize=8)

        page_path = output_dir / f"power_detail_page_{page + 1:02d}.png"
        figure.savefig(page_path, dpi=170)
        plt.close(figure)
        page_paths.append(page_path)

    return page_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-run SCARA energy model for all RoboDK export CSVs and output Speed/Accel/Time/Power summary."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../RoboDK_Exports"),
        help="Directory containing RoboDK export CSV files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("RoboDK_energy_summary.csv"),
        help="Path for the summary CSV output.",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="Time_Total",
        help="Time column to use from each input CSV.",
    )
    parser.add_argument(
        "--power-mode",
        choices=["total_energy_j", "avg_power_w"],
        default="total_energy_j",
        help="Power column meaning: total_energy_j (energy usage) or avg_power_w.",
    )
    parser.add_argument(
        "--plot-power-dir",
        type=Path,
        default=None,
        help="Optional output directory for paged per-move power plots (4x3 subplots per PNG).",
    )
    parser.add_argument(
        "--plot-power-per-page",
        type=int,
        default=12,
        help="Number of subplots per PNG page for power plots (default 12 = 4x3).",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    discovered = list(input_dir.glob("*.csv")) + list(input_dir.glob("*.CSV"))
    unique_by_name = {}
    for path in discovered:
        key = path.name.lower()
        if key not in unique_by_name:
            unique_by_name[key] = path
    files = sorted(unique_by_name.values(), key=lambda p: p.name.lower())
    if not files:
        raise SystemExit(f"No CSV files found in: {input_dir}")

    model = build_model()

    rows = []
    failures = []
    collect_trace = args.plot_power_dir is not None
    for path in files:
        try:
            rows.append(
                summarize_one(
                    path,
                    model,
                    args.time_col,
                    args.power_mode,
                    collect_trace=collect_trace,
                )
            )
        except Exception as exc:
            failures.append((path.name, str(exc)))

    rows.sort(key=lambda r: (r["_speed_sort"], r["_accel_sort"], r["_name"]))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Speed", "Accel", "Time", "Power"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "Speed": row["Speed"],
                "Accel": row["Accel"],
                "Time": row["Time"],
                "Power": row["Power"],
            })

    print(f"Processed {len(rows)} files from {input_dir}")
    print(f"Wrote summary CSV: {args.output_csv}")
    print(f"Power mode: {args.power_mode}")

    if args.plot_power_dir is not None:
        page_paths = plot_power_pages(
            rows=rows,
            output_dir=args.plot_power_dir,
            plots_per_page=int(args.plot_power_per_page),
            cols=3,
        )
        print(f"Power plot pages: {len(page_paths)} in {args.plot_power_dir}")

    if failures:
        print(f"Skipped {len(failures)} files due to errors:")
        for name, message in failures[:10]:
            print(f"  {name}: {message}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")


if __name__ == "__main__":
    main()
