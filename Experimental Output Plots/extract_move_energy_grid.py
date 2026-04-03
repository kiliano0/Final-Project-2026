#!/usr/bin/env python3
"""Extract per-move energy from power-analyzer CSV for speed/accel sweeps.

Expected motion pattern:
- Robot starts at idle for a while.
- Each move raises power above idle.
- Power drops back near idle, then waits ~1 second.

This script detects those move windows and outputs one row per move in the
format: Speed, Accel, Power Usage (J).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


HARD_IDLE_CUTOFF_W = 105.0
HARD_IDLE_CUTOFF_DURATION_S = 0.5


@dataclass
class PowerSeries:
    time_s: np.ndarray
    power_w: np.ndarray
    logging_interval_s: float


def _parse_hhmmss_to_seconds(value: str) -> float:
    """Parse HH:MM:SS.sss (or MM:SS.s) timestamp to seconds."""
    value = value.strip()
    parts = value.split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600.0 + minutes * 60.0 + seconds
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds
    raise ValueError(f"Invalid timestamp format: {value!r}")


def load_power_csv(path: Path) -> PowerSeries:
    """Load analyzer CSV and detect delimiter/header layout automatically."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    logging_interval_s = 0.1
    for line in lines:
        if line.startswith("#Logging Interval[s]"):
            if ";" in line:
                value = line.split(";", 1)[1].strip()
            elif "," in line:
                value = line.split(",", 1)[1].strip()
            else:
                value = ""
            try:
                logging_interval_s = float(value)
            except ValueError:
                pass
            break

    header_idx = None
    for index, line in enumerate(lines):
        if not line.startswith("#") and "P[W]" in line:
            header_idx = index
            break

    if header_idx is None:
        raise ValueError(f"Could not find data header row in {path}")

    delimiter = ";" if ";" in lines[header_idx] else ","
    header = [column.strip() for column in lines[header_idx].split(delimiter)]
    try:
        power_idx = header.index("P[W]")
    except ValueError as exc:
        raise ValueError("Could not find P[W] column") from exc

    timestamp_idx = header.index("Timestamp") if "Timestamp" in header else None
    time_idx = header.index("Time") if "Time" in header else None

    power_values: List[float] = []
    time_values: List[float] = []

    for line in lines[header_idx + 1 :]:
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split(delimiter)
        if len(parts) <= power_idx:
            continue

        power_str = parts[power_idx].strip().replace("E", "e")
        try:
            power = float(power_str)
        except ValueError:
            continue

        sample_time = None
        if timestamp_idx is not None and len(parts) > timestamp_idx:
            ts = parts[timestamp_idx].strip()
            if ts:
                try:
                    sample_time = _parse_hhmmss_to_seconds(ts)
                except ValueError:
                    sample_time = None

        # If timestamp is absent/unparseable, use explicit Time column when available.
        if sample_time is None and time_idx is not None and len(parts) > time_idx:
            time_str = parts[time_idx].strip()
            if time_str:
                try:
                    sample_time = float(time_str)
                except ValueError:
                    sample_time = None

        power_values.append(power)
        time_values.append(np.nan if sample_time is None else sample_time)

    if not power_values:
        raise ValueError(f"No valid power samples found in {path}")

    power_w = np.array(power_values, dtype=float)
    time_s = np.array(time_values, dtype=float)

    if np.isnan(time_s).all():
        time_s = np.arange(len(power_w), dtype=float) * logging_interval_s
    else:
        missing = np.isnan(time_s)
        if np.any(missing):
            estimated = np.arange(len(power_w), dtype=float) * logging_interval_s
            first_valid = np.flatnonzero(~missing)[0]
            estimated += time_s[first_valid] - estimated[first_valid]
            time_s[missing] = estimated[missing]
        time_s = time_s - time_s[0]

    return PowerSeries(time_s=time_s, power_w=power_w, logging_interval_s=logging_interval_s)


def _close_short_false_gaps(mask: np.ndarray, max_gap_samples: int) -> np.ndarray:
    """Fill small False gaps that are bracketed by True regions."""
    if max_gap_samples <= 0:
        return mask.copy()

    result = mask.copy()
    index = 0
    length = len(result)
    while index < length:
        if result[index]:
            index += 1
            continue
        gap_start = index
        while index < length and not result[index]:
            index += 1
        gap_end = index
        gap_size = gap_end - gap_start
        if (
            gap_size <= max_gap_samples
            and gap_start > 0
            and gap_end < length
            and result[gap_start - 1]
            and result[gap_end]
        ):
            result[gap_start:gap_end] = True
    return result


def detect_move_segments(
    power_w: np.ndarray,
    idle_power_w: float,
    start_margin_w: float,
    min_active_samples: int,
    start_trigger_samples: int,
    end_idle_band_w: float,
    settle_samples: int,
    move_threshold_w: float = 100.0,
    required_peaks: int = 2,
    peak_min_height_w: float | None = None,
    peak_min_separation_samples: int = 2,
    peak_rearm_threshold_w: float | None = None,
    start_backtrack_samples: int = 0,
    hard_idle_cutoff_w: float = HARD_IDLE_CUTOFF_W,
    hard_idle_cutoff_samples: int = 5,
) -> List[Tuple[int, int]]:
    """Return move segments as (start_idx, end_idx_exclusive).

    Segment start: power rises above move_threshold_w for start_trigger_samples.
    Segment end: after detecting required_peaks threshold excursions (peaks),
    end at the first sample that drops below idle power.

    Between peaks, power is allowed to dip near/below threshold; the move is
    still kept open until the required peak count is met.
    """
    segments: List[Tuple[int, int]] = []
    n = len(power_w)
    if n == 0:
        return segments

    start_threshold = float(move_threshold_w)
    end_threshold = float(move_threshold_w)
    idle_mask = np.abs(power_w - idle_power_w) <= end_idle_band_w
    start_confirm_needed = max(1, int(math.ceil(0.7 * max(start_trigger_samples, 1))))

    if peak_min_height_w is None:
        peak_min_height_w = start_threshold + 2.0
    peak_min_height_w = float(peak_min_height_w)
    if peak_rearm_threshold_w is None:
        # Use the start threshold itself for re-arm by default.
        # This avoids ending slightly too early on traces with noisy dips.
        peak_rearm_threshold_w = start_threshold
    peak_rearm_threshold_w = float(peak_rearm_threshold_w)
    peak_min_separation_samples = max(1, int(peak_min_separation_samples))
    start_backtrack_samples = max(0, int(start_backtrack_samples))
    required_peaks = max(1, int(required_peaks))
    hard_idle_cutoff_samples = max(1, int(hard_idle_cutoff_samples))

    index = 0
    while index < n:
        # Find move start: consecutive samples above start threshold.
        if power_w[index] <= start_threshold:
            index += 1
            continue

        trigger_end = index + start_trigger_samples
        if trigger_end > n:
            index += 1
            continue

        start_window = power_w[index:trigger_end]
        if int(np.sum(start_window > start_threshold)) < start_confirm_needed:
            index += 1
            continue

        start = index
        backtrack_floor = max(0, index - start_trigger_samples)
        while start > backtrack_floor and power_w[start - 1] > start_threshold:
            start -= 1
        start = max(0, start - start_backtrack_samples)
        search_idx = trigger_end

        # End rule priority:
        # 1) after required peaks, end on first sample <= idle power
        # 2) fallback: first <= threshold sample after required peaks
        # 3) fallback: settled idle-band region (safety only)
        end = n
        peak_count = 0
        in_excursion = True
        peak_max = float(power_w[start])
        first_threshold_return_after_required: int | None = None
        while search_idx < n:
            segment_len = search_idx - start + 1

            p = float(power_w[search_idx])

            # Hard stop rule: if power remains <= 105W for 0.5s, end move at
            # the start of that sustained low-power window, regardless of peaks.
            if p <= hard_idle_cutoff_w:
                cutoff_end = search_idx + hard_idle_cutoff_samples
                if cutoff_end <= n and np.all(power_w[search_idx:cutoff_end] <= hard_idle_cutoff_w):
                    end = search_idx
                    break

            if in_excursion:
                if p > peak_max:
                    peak_max = p
                if p <= peak_rearm_threshold_w:
                    if peak_max >= peak_min_height_w:
                        peak_count += 1
                    in_excursion = False
                    peak_max = -1e18
            else:
                if p > start_threshold:
                    in_excursion = True
                    peak_max = p

            if segment_len >= min_active_samples:
                if peak_count >= required_peaks:
                    # Move ends as soon as power drops to/below idle after required peaks.
                    # End index is exclusive; this keeps below-idle (regen) samples out of
                    # move energy and available for post-move regen integration.
                    if p <= idle_power_w:
                        end = search_idx
                        break
                    if first_threshold_return_after_required is None and p <= start_threshold:
                        first_threshold_return_after_required = search_idx

                if idle_mask[search_idx]:
                    settle_end = search_idx + settle_samples
                    if settle_end <= n and np.all(idle_mask[search_idx:settle_end]):
                        end = search_idx
                        break
            search_idx += 1

        if end == n and first_threshold_return_after_required is not None:
            end = first_threshold_return_after_required

        if end > start:
            segments.append((start, end))

        # Continue after the settle window if we found one, else stop at EOF.
        if end < n:
            index = min(n, end + settle_samples)
        else:
            break

    return segments


def choose_threshold_margin(
    power_w: np.ndarray,
    idle_power_w: float,
    min_active_samples: int,
    start_trigger_samples: int,
    end_idle_band_w: float,
    settle_samples: int,
    expected_moves: int,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Find near-idle start margin (W) that best matches expected move count."""
    candidates = np.arange(1.0, 12.5, 0.5)
    best_margin = 2.0
    best_segments: List[Tuple[int, int]] = []
    best_score = float("inf")

    for margin in candidates:
        segments = detect_move_segments(
            power_w=power_w,
            idle_power_w=idle_power_w,
            start_margin_w=float(margin),
            min_active_samples=min_active_samples,
            start_trigger_samples=start_trigger_samples,
            end_idle_band_w=end_idle_band_w,
            settle_samples=settle_samples,
        )
        move_count = len(segments)
        score = abs(move_count - expected_moves)
        if score < best_score:
            best_score = score
            best_margin = float(margin)
            best_segments = segments
        if score == 0:
            break

    return best_margin, best_segments


def auto_tune_detection(
    *,
    power_w: np.ndarray,
    idle_power_w: float,
    min_active_samples: int,
    start_trigger_samples: int,
    end_idle_band_w: float,
    settle_samples: int,
    move_threshold_w: float,
    required_peaks: int,
    peak_min_height_w: float | None,
    peak_min_separation_samples: int,
    peak_rearm_threshold_w: float | None,
    start_backtrack_samples: int,
    expected_moves: int,
    logging_interval_s: float,
    hard_idle_cutoff_w: float,
    hard_idle_cutoff_samples: int,
) -> Tuple[List[Tuple[int, int]], float, float, int]:
    """Search nearby parameters and return the best segmentation candidate."""
    base_score = abs(len(detect_move_segments(
        power_w=power_w,
        idle_power_w=idle_power_w,
        start_margin_w=move_threshold_w - idle_power_w,
        min_active_samples=min_active_samples,
        start_trigger_samples=start_trigger_samples,
        end_idle_band_w=end_idle_band_w,
        settle_samples=settle_samples,
        move_threshold_w=move_threshold_w,
        required_peaks=required_peaks,
        peak_min_height_w=peak_min_height_w,
        peak_min_separation_samples=peak_min_separation_samples,
        peak_rearm_threshold_w=peak_rearm_threshold_w,
        start_backtrack_samples=start_backtrack_samples,
        hard_idle_cutoff_w=hard_idle_cutoff_w,
        hard_idle_cutoff_samples=hard_idle_cutoff_samples,
    )) - expected_moves)

    threshold_candidates = sorted(
        {
            max(idle_power_w + 1.0, move_threshold_w - 2.0),
            max(idle_power_w + 1.0, move_threshold_w),
            max(idle_power_w + 1.0, move_threshold_w + 2.0),
            idle_power_w + 4.0,
            idle_power_w + 6.0,
            idle_power_w + 8.0,
            idle_power_w + 10.0,
        }
    )
    band_candidates = sorted({end_idle_band_w, max(end_idle_band_w, 7.0), max(end_idle_band_w, 9.0)})
    settle_s_base = settle_samples * logging_interval_s
    settle_candidates_s = sorted({settle_s_base, max(settle_s_base, 0.8), 1.0})

    best_segments: List[Tuple[int, int]] = []
    best_threshold = move_threshold_w
    best_band = end_idle_band_w
    best_settle_samples = settle_samples
    best_score = float(base_score)

    for threshold in threshold_candidates:
        for band in band_candidates:
            for settle_s in settle_candidates_s:
                candidate_settle_samples = max(1, int(round(settle_s / max(logging_interval_s, 1e-6))))
                candidate_segments = detect_move_segments(
                    power_w=power_w,
                    idle_power_w=idle_power_w,
                    start_margin_w=threshold - idle_power_w,
                    min_active_samples=min_active_samples,
                    start_trigger_samples=start_trigger_samples,
                    end_idle_band_w=band,
                    settle_samples=candidate_settle_samples,
                    move_threshold_w=threshold,
                    required_peaks=required_peaks,
                    peak_min_height_w=peak_min_height_w,
                    peak_min_separation_samples=peak_min_separation_samples,
                    peak_rearm_threshold_w=peak_rearm_threshold_w,
                    start_backtrack_samples=start_backtrack_samples,
                    hard_idle_cutoff_w=hard_idle_cutoff_w,
                    hard_idle_cutoff_samples=hard_idle_cutoff_samples,
                )
                score = abs(len(candidate_segments) - expected_moves)
                if score < best_score:
                    best_score = float(score)
                    best_segments = candidate_segments
                    best_threshold = float(threshold)
                    best_band = float(band)
                    best_settle_samples = int(candidate_settle_samples)
                if score == 0:
                    return candidate_segments, float(threshold), float(band), int(candidate_settle_samples)

    if best_segments:
        return best_segments, best_threshold, best_band, best_settle_samples

    # No better candidate found.
    return [], move_threshold_w, end_idle_band_w, settle_samples


def integrate_energy(time_s: np.ndarray, power_w: np.ndarray) -> float:
    if len(time_s) < 2:
        return 0.0
    return float(np.trapezoid(power_w, time_s))


def build_speed_accel_grid(
    speed_values: Sequence[int], accel_values: Sequence[int]
) -> List[Tuple[int, int]]:
    return [(speed, accel) for speed in speed_values for accel in accel_values]


def parse_range_triplet(start: int, stop: int, step: int) -> List[int]:
    if step <= 0:
        raise ValueError("Step must be > 0")
    if stop < start:
        raise ValueError("Stop must be >= start")
    return list(range(start, stop + 1, step))


def extend_segments(
    segments: Sequence[Tuple[int, int]],
    extension_samples: int,
    series_len: int,
) -> List[Tuple[int, int]]:
    """Extend each segment end by fixed samples without overlapping next segment."""
    if extension_samples <= 0 or not segments:
        return list(segments)

    extended: List[Tuple[int, int]] = []
    for idx, (start, end) in enumerate(segments):
        new_end = min(series_len, end + extension_samples)
        if idx + 1 < len(segments):
            next_start = segments[idx + 1][0]
            new_end = min(new_end, next_start)
        if new_end <= start:
            new_end = min(series_len, start + 1)
        extended.append((start, new_end))

    return extended


def plot_detected_regions(
    *,
    time_s: np.ndarray,
    power_w: np.ndarray,
    segments: Sequence[Tuple[int, int]],
    idle_power_w: float,
    start_margin_w: float,
    end_idle_band_w: float,
    output_path: Path,
) -> None:
    """Plot power trace with shaded regions used for per-move integration."""
    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)

    axis.plot(time_s, power_w, color="#1f77b4", linewidth=1.0, label="Measured power")
    axis.axhline(idle_power_w, color="#2ca02c", linestyle="--", linewidth=1.2, label="Estimated idle")
    axis.axhline(
        idle_power_w + start_margin_w,
        color="#d62728",
        linestyle=":",
        linewidth=1.2,
        label="Move start threshold",
    )
    axis.axhline(
        idle_power_w + end_idle_band_w,
        color="#9467bd",
        linestyle="--",
        linewidth=1.0,
        label="Idle settle band (+)",
    )
    axis.axhline(
        idle_power_w - end_idle_band_w,
        color="#9467bd",
        linestyle="--",
        linewidth=1.0,
        label="Idle settle band (-)",
    )

    for idx, (start, end) in enumerate(segments):
        if start >= len(time_s):
            continue
        end_idx = min(max(end - 1, start), len(time_s) - 1)
        x0 = float(time_s[start])
        x1 = float(time_s[end_idx])
        axis.axvspan(
            x0,
            x1,
            color="#ff7f0e",
            alpha=0.15,
            label="Integrated move region" if idx == 0 else None,
        )

    axis.set_title("Power Trace with Detected Move Regions")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Power (W)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_segment_detail_pages(
    *,
    time_s: np.ndarray,
    power_w: np.ndarray,
    segments: Sequence[Tuple[int, int]],
    idle_power_w: float,
    start_margin_w: float,
    end_idle_band_w: float,
    output_dir: Path,
    moves_per_page: int,
    pad_s: float,
) -> List[Path]:
    """Create paged zoom plots so move boundaries are easy to inspect."""
    output_dir.mkdir(parents=True, exist_ok=True)
    page_paths: List[Path] = []

    if moves_per_page <= 0:
        moves_per_page = 12

    page_count = int(math.ceil(len(segments) / moves_per_page)) if segments else 0
    cols = 3

    for page in range(page_count):
        start_move = page * moves_per_page
        end_move = min(len(segments), start_move + moves_per_page)
        count = end_move - start_move
        rows = int(math.ceil(count / cols))

        figure, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.1 * rows), constrained_layout=True)
        if isinstance(axes, np.ndarray):
            axes_flat = axes.ravel().tolist()
        else:
            axes_flat = [axes]

        for local_idx in range(rows * cols):
            axis = axes_flat[local_idx]
            if local_idx >= count:
                axis.axis("off")
                continue

            move_idx = start_move + local_idx
            seg_start, seg_end = segments[move_idx]
            seg_end_inclusive = min(max(seg_end - 1, seg_start), len(time_s) - 1)
            t0 = float(time_s[seg_start])
            t1 = float(time_s[seg_end_inclusive])

            window_start = max(float(time_s[0]), t0 - pad_s)
            window_end = min(float(time_s[-1]), t1 + pad_s)
            mask = (time_s >= window_start) & (time_s <= window_end)

            axis.plot(time_s[mask], power_w[mask], color="#1f77b4", linewidth=1.0)
            axis.axhline(idle_power_w, color="#2ca02c", linestyle="--", linewidth=0.9)
            axis.axhline(idle_power_w + start_margin_w, color="#d62728", linestyle=":", linewidth=0.9)
            axis.axhline(idle_power_w + end_idle_band_w, color="#9467bd", linestyle="--", linewidth=0.8)
            axis.axhline(idle_power_w - end_idle_band_w, color="#9467bd", linestyle="--", linewidth=0.8)

            axis.axvspan(t0, t1, color="#ff7f0e", alpha=0.18)
            axis.axvline(t0, color="#ff7f0e", linewidth=1.2)
            axis.axvline(t1, color="#ff7f0e", linewidth=1.2)

            axis.set_title(f"Move {move_idx + 1}: {t0:.2f}s to {t1:.2f}s")
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("Power (W)")
            axis.grid(True, alpha=0.25)

        page_path = output_dir / f"move_segment_detail_page_{page + 1:02d}.png"
        figure.savefig(page_path, dpi=170)
        plt.close(figure)
        page_paths.append(page_path)

    return page_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-move energy from power analyzer CSV")
    parser.add_argument("--input", required=True, help="Input analyzer CSV file")
    parser.add_argument("--output", default="move_energy_grid.csv", help="Output CSV path")

    parser.add_argument("--speed-start", type=int, default=10)
    parser.add_argument("--speed-stop", type=int, default=100)
    parser.add_argument("--speed-step", type=int, default=10)

    parser.add_argument("--accel-start", type=int, default=10)
    parser.add_argument("--accel-stop", type=int, default=100)
    parser.add_argument("--accel-step", type=int, default=10)

    parser.add_argument(
        "--idle-window-s",
        type=float,
        default=12.0,
        help="Initial time window used to estimate idle power",
    )
    parser.add_argument(
        "--start-margin-w",
        type=float,
        default=None,
        help="Move start threshold above idle (W). If omitted, auto-selected.",
    )
    parser.add_argument("--min-active-s", type=float, default=0.5)
    parser.add_argument(
        "--start-trigger-s",
        type=float,
        default=0.2,
        help="Time above start threshold required to trigger a move",
    )
    parser.add_argument(
        "--end-idle-band-w",
        type=float,
        default=5.0,
        help="Idle band half-width used to detect settled end-of-move",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.5,
        help="Required stable-in-idle-band duration to end a move",
    )
    parser.add_argument(
        "--power-mode",
        choices=["above_idle", "total"],
        default="total",
        help="Energy mode: total energy in each move window (includes baseline) or only above-idle energy",
    )
    parser.add_argument(
        "--move-threshold-w",
        type=float,
        default=100.0,
        help="Move start/end threshold power in watts.",
    )
    parser.add_argument(
        "--required-peaks",
        type=int,
        default=2,
        help="Number of peaks required before allowing move end at threshold.",
    )
    parser.add_argument(
        "--peak-min-height-w",
        type=float,
        default=None,
        help="Minimum peak height in watts for counting peaks (default: threshold+2W).",
    )
    parser.add_argument(
        "--peak-min-separation-s",
        type=float,
        default=0.20,
        help="Minimum time separation between counted peaks.",
    )
    parser.add_argument(
        "--peak-rearm-threshold-w",
        type=float,
        default=None,
        help="Power level that must be reached to re-arm peak counting between peaks (default: move-threshold+0.5W).",
    )
    parser.add_argument(
        "--start-backtrack-s",
        type=float,
        default=0.10,
        help="Shift detected move starts earlier by this many seconds.",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="Optional image path for overlay plot of detected move regions",
    )
    parser.add_argument(
        "--plot-detail-dir",
        default=None,
        help="Optional directory to save paged zoom plots of move start/end boundaries",
    )
    parser.add_argument(
        "--plot-detail-moves-per-page",
        type=int,
        default=12,
        help="Number of move zoom panels per detail page",
    )
    parser.add_argument(
        "--plot-detail-pad-s",
        type=float,
        default=0.6,
        help="Context time before/after each detected move in detail plots",
    )
    parser.add_argument(
        "--end-extension-s",
        type=float,
        default=0.1,
        help="Extend each detected move end by this many seconds.",
    )

    args = parser.parse_args()

    series = load_power_csv(Path(args.input))

    speed_values = parse_range_triplet(args.speed_start, args.speed_stop, args.speed_step)
    accel_values = parse_range_triplet(args.accel_start, args.accel_stop, args.accel_step)
    grid = build_speed_accel_grid(speed_values, accel_values)
    expected_moves = len(grid)

    min_idle_samples = max(5, int(round(args.idle_window_s / max(series.logging_interval_s, 1e-6))))
    min_idle_samples = min(min_idle_samples, len(series.power_w))
    # Hardcoded baseline requested to match PowerAllFour3 idle.
    idle_power_w = 101.9

    min_active_samples = max(2, int(round(args.min_active_s / max(series.logging_interval_s, 1e-6))))
    start_trigger_samples = max(1, int(round(args.start_trigger_s / max(series.logging_interval_s, 1e-6))))
    settle_samples = max(1, int(round(args.settle_s / max(series.logging_interval_s, 1e-6))))
    hard_idle_cutoff_samples = max(
        1,
        int(round(HARD_IDLE_CUTOFF_DURATION_S / max(series.logging_interval_s, 1e-6))),
    )

    requested_threshold_w = float(args.move_threshold_w)
    if args.start_margin_w is not None:
        requested_threshold_w = idle_power_w + float(args.start_margin_w)
    move_threshold_w = max(requested_threshold_w, idle_power_w + 1.0)
    start_margin_w = float(move_threshold_w - idle_power_w)

    end_idle_band_w = float(args.end_idle_band_w)
    peak_min_separation_samples = max(
        1,
        int(round(float(args.peak_min_separation_s) / max(series.logging_interval_s, 1e-6))),
    )
    start_backtrack_samples = max(
        0,
        int(round(float(args.start_backtrack_s) / max(series.logging_interval_s, 1e-6))),
    )
    segments = detect_move_segments(
        power_w=series.power_w,
        idle_power_w=idle_power_w,
        start_margin_w=start_margin_w,
        min_active_samples=min_active_samples,
        start_trigger_samples=start_trigger_samples,
        end_idle_band_w=end_idle_band_w,
        settle_samples=settle_samples,
        move_threshold_w=move_threshold_w,
        required_peaks=int(args.required_peaks),
        peak_min_height_w=args.peak_min_height_w,
        peak_min_separation_samples=peak_min_separation_samples,
        peak_rearm_threshold_w=args.peak_rearm_threshold_w,
        start_backtrack_samples=start_backtrack_samples,
        hard_idle_cutoff_w=HARD_IDLE_CUTOFF_W,
        hard_idle_cutoff_samples=hard_idle_cutoff_samples,
    )

    if len(segments) != expected_moves:
        tuned_segments, tuned_threshold_w, tuned_end_idle_band_w, tuned_settle_samples = auto_tune_detection(
            power_w=series.power_w,
            idle_power_w=idle_power_w,
            min_active_samples=min_active_samples,
            start_trigger_samples=start_trigger_samples,
            end_idle_band_w=end_idle_band_w,
            settle_samples=settle_samples,
            move_threshold_w=move_threshold_w,
            required_peaks=int(args.required_peaks),
            peak_min_height_w=args.peak_min_height_w,
            peak_min_separation_samples=peak_min_separation_samples,
            peak_rearm_threshold_w=args.peak_rearm_threshold_w,
            start_backtrack_samples=start_backtrack_samples,
            expected_moves=expected_moves,
            logging_interval_s=series.logging_interval_s,
            hard_idle_cutoff_w=HARD_IDLE_CUTOFF_W,
            hard_idle_cutoff_samples=hard_idle_cutoff_samples,
        )
        if abs(len(tuned_segments) - expected_moves) < abs(len(segments) - expected_moves):
            segments = tuned_segments
            move_threshold_w = tuned_threshold_w
            end_idle_band_w = tuned_end_idle_band_w
            settle_samples = tuned_settle_samples
            start_margin_w = float(move_threshold_w - idle_power_w)

    end_extension_samples = max(0, int(round(float(args.end_extension_s) / max(series.logging_interval_s, 1e-6))))
    if end_extension_samples > 0:
        segments = extend_segments(segments, end_extension_samples, len(series.time_s))

    row_count = min(len(grid), len(segments))
    # Keep CSV and plots in lockstep by using the same bounded segment set.
    segments_for_output = list(segments[:row_count])

    output_path = Path(args.output)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Speed",
                "Accel",
                "Power Usage (J)",
                "Regen Saved (J)",
                "Move Duration (s)",
                "Move Start (s)",
                "Move End (s)",
            ]
        )

        for index in range(row_count):
            speed, accel = grid[index]
            start, end = segments_for_output[index]
            segment_time = series.time_s[start:end]
            segment_power = series.power_w[start:end]

            next_start = (
                segments_for_output[index + 1][0]
                if index + 1 < len(segments_for_output)
                else len(series.time_s)
            )
            post_move_time = series.time_s[end:next_start]
            post_move_power = series.power_w[end:next_start]

            if args.power_mode == "above_idle":
                usable_power = np.maximum(segment_power - idle_power_w, 0.0)
            else:
                usable_power = segment_power

            energy_j = integrate_energy(segment_time, usable_power)
            regen_saved_j = integrate_energy(post_move_time, np.maximum(idle_power_w - post_move_power, 0.0))
            move_duration_s = float(segment_time[-1] - segment_time[0]) if len(segment_time) > 1 else 0.0
            move_start_s = float(segment_time[0]) if len(segment_time) else 0.0
            move_end_s = float(segment_time[-1]) if len(segment_time) else 0.0
            writer.writerow(
                [
                    speed,
                    accel,
                    f"{energy_j:.6f}",
                    f"{regen_saved_j:.6f}",
                    f"{move_duration_s:.3f}",
                    f"{move_start_s:.3f}",
                    f"{move_end_s:.3f}",
                ]
            )

    print(f"Input file: {args.input}")
    print(f"Samples: {len(series.power_w)}")
    print(f"Estimated idle power: {idle_power_w:.3f} W")
    print(f"Move threshold: {move_threshold_w:.3f} W")
    print(f"Start margin above idle: {start_margin_w:.3f} W")
    print(f"Required peaks: {int(args.required_peaks)}")
    print(f"End idle band: ±{end_idle_band_w:.3f} W")
    print(f"Settle duration: {settle_samples * series.logging_interval_s:.3f} s")
    print(
        f"Hard cutoff: <= {HARD_IDLE_CUTOFF_W:.1f} W for "
        f"{hard_idle_cutoff_samples * series.logging_interval_s:.3f} s"
    )
    print(f"End extension: {end_extension_samples * series.logging_interval_s:.3f} s")
    print(f"Detected moves: {len(segments)}")
    print(f"Expected moves from speed/accel grid: {expected_moves}")
    print(f"Rows written: {row_count}")
    print(f"Output: {output_path}")

    if args.plot_output:
        plot_path = Path(args.plot_output)
        plot_detected_regions(
            time_s=series.time_s,
            power_w=series.power_w,
            segments=segments_for_output,
            idle_power_w=idle_power_w,
            start_margin_w=start_margin_w,
            end_idle_band_w=end_idle_band_w,
            output_path=plot_path,
        )
        print(f"Plot: {plot_path}")

    if args.plot_detail_dir:
        detail_paths = plot_segment_detail_pages(
            time_s=series.time_s,
            power_w=series.power_w,
            segments=segments_for_output,
            idle_power_w=idle_power_w,
            start_margin_w=start_margin_w,
            end_idle_band_w=end_idle_band_w,
            output_dir=Path(args.plot_detail_dir),
            moves_per_page=int(args.plot_detail_moves_per_page),
            pad_s=float(args.plot_detail_pad_s),
        )
        print(f"Detail pages: {len(detail_paths)} in {args.plot_detail_dir}")

    if len(segments) != expected_moves:
        print(
            "Warning: detected move count does not match expected grid size. "
            "Adjust --start-margin-w, --start-trigger-s, --end-idle-band-w, or --settle-s if needed."
        )


if __name__ == "__main__":
    main()
