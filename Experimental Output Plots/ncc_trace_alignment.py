#!/usr/bin/env python3
"""Sub-sample alignment of repeated robot load traces via upsampled NCC.

Problem context
---------------
Power is sampled at dt = 0.1 s.  The robot true motion onset can fall
anywhere within that 0.1 s window, so integer-sample cross-correlation gives
lag = 0 for every run even when real sub-sample offsets exist.

Strategy
--------
1. Compute the *derivative* (gradient) of each run.
   - Flat idle regions have near-zero gradient -> they do not drive correlation.
   - Transitions and active-motion features have large gradients -> they dominate.

2. Upsample both the derivative signals to a fine grid (default fine_dt = 0.001 s)
   using cubic interpolation.

3. Run normalized cross-correlation on the fine-grid derivative signals,
   restricted to |lag| <= max_lag_s (default 0.15 s ~= 1.5 samples).

4. The peak of the NCC gives a fractional lag with fine_dt resolution.

5. Apply that fractional shift to the *original raw* signal via interpolation.

Public API
----------
find_motion_start(signal, dt, abs_threshold, deriv_threshold)
compute_alignment_signal(signal, method)
find_best_lag(reference, signal, dt, fine_dt, max_lag_s)
align_runs(runs, dt, reference_index, fine_dt, max_lag_s, align_method)
align_csv_by_move(csv_path, move, ...)
print_alignment_table(bundle)
plot_alignment_workflow(bundle, run_labels, title_prefix, show)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import correlate, correlation_lags

# ---------------------------------------------------------------------------
# Typing
# ---------------------------------------------------------------------------
ArrayLike1D = np.ndarray | pd.Series

_DEFAULT_RUN_COLS: List[str] = [f"Run{i}_Power_W" for i in range(2, 12)]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SubSampleAlignmentResult:
    """Per-run alignment output."""
    run_index: int
    aligned: np.ndarray
    aligned_fine: np.ndarray
    lag_s: float
    lag_fine_samples: int
    max_corr: float
    motion_start_idx: int


@dataclass
class AlignmentBundle:
    """All outputs from align_runs()."""
    t_ref: np.ndarray
    t_fine: np.ndarray
    reference: np.ndarray
    results: List[SubSampleAlignmentResult]
    aligned_on_ref_coarse: np.ndarray
    aligned_on_fine: np.ndarray
    ref_align_signal: np.ndarray
    run_align_signals: List[np.ndarray]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def find_motion_start(
    signal: np.ndarray,
    dt: float,
    abs_threshold: Optional[float] = None,
    deriv_threshold: Optional[float] = None,
) -> int:
    """Return the first coarse-sample index where motion begins.

    Detection order:
    1. Absolute deviation from initial baseline > abs_threshold
    2. Gradient magnitude > deriv_threshold
    Falls back to 0 if neither fires.
    """
    arr = np.asarray(signal, dtype=float).ravel()
    n_base = max(2, min(6, int(0.05 * len(arr))))
    baseline = float(np.nanmedian(arr[:n_base]))

    if abs_threshold is not None:
        cands = np.flatnonzero(np.abs(arr - baseline) > abs_threshold)
        if cands.size:
            return int(cands[0])

    if deriv_threshold is not None:
        cands = np.flatnonzero(np.abs(np.gradient(arr)) > deriv_threshold)
        if cands.size:
            return int(cands[0])

    return 0


def compute_alignment_signal(
    signal: np.ndarray,
    method: str = "derivative",
) -> np.ndarray:
    """Return the version of the signal used for cross-correlation.

    method = "derivative" : first gradient (highlights transitions, suppresses flat regions)
    method = "raw"        : signal as-is
    """
    arr = np.asarray(signal, dtype=float).ravel()
    if method == "derivative":
        return np.gradient(arr)
    if method == "raw":
        return arr.copy()
    raise ValueError(f"Unknown method {method!r}. Choose 'derivative' or 'raw'.")


def _zscore_safe(x: np.ndarray) -> np.ndarray:
    std = float(np.nanstd(x))
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - float(np.nanmean(x))) / std


def _upsample(signal: np.ndarray, dt: float, fine_dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Cubic-interpolate signal from dt to fine_dt. Returns (t_fine, signal_fine)."""
    arr = np.asarray(signal, dtype=float).ravel()
    t_coarse = np.arange(arr.size, dtype=float) * dt
    t_fine = np.arange(t_coarse[0], t_coarse[-1] + 0.5 * fine_dt, fine_dt)
    f = interp1d(t_coarse, arr, kind="cubic", bounds_error=False, fill_value=np.nan)
    fine = f(t_fine)
    nan_mask = np.isnan(fine)
    if nan_mask.any() and not nan_mask.all():
        ok = np.flatnonzero(~nan_mask)
        fine[: ok[0]] = fine[ok[0]]
        fine[ok[-1] + 1 :] = fine[ok[-1]]
    return t_fine, fine


def find_best_lag(
    reference: np.ndarray,
    signal: np.ndarray,
    dt: float,
    fine_dt: float = 0.001,
    max_lag_s: float = 0.15,
) -> Tuple[float, int, float]:
    """Estimate the fractional time lag that aligns signal to reference.

    Both inputs should be the alignment signal (e.g. derivative of raw run).

    Returns (lag_s, lag_fine_samples, max_corr).

    Convention: apply with  y_aligned(t) = y_raw(t - lag_s).
    Positive lag_s -> signal features occur EARLIER than reference (shift right).
    Negative lag_s -> signal features occur LATER  than reference (shift left).
    """
    _,  ref_fine = _upsample(reference, dt, fine_dt)
    _,  sig_fine = _upsample(signal,    dt, fine_dt)

    n = min(ref_fine.size, sig_fine.size)
    ref_fine = ref_fine[:n]
    sig_fine = sig_fine[:n]

    ref_z = _zscore_safe(ref_fine)
    sig_z = _zscore_safe(sig_fine)

    corr = correlate(ref_z, sig_z, mode="full", method="auto")
    lags = correlation_lags(ref_z.size, sig_z.size, mode="full")
    lag_s_all = lags.astype(float) * fine_dt

    valid = np.abs(lag_s_all) <= max_lag_s
    if not np.any(valid):
        return 0.0, 0, float("nan")

    overlap = correlate(np.ones(n), np.ones(n), mode="full")
    overlap = np.maximum(overlap, 1.0)
    ncc = corr / overlap

    valid_idx = np.flatnonzero(valid)
    best_local = int(np.nanargmax(ncc[valid]))
    best_i = int(valid_idx[best_local])

    return float(lag_s_all[best_i]), int(lags[best_i]), float(ncc[best_i])


def apply_fractional_shift_to_grid(
    signal: np.ndarray,
    dt: float,
    lag_s: float,
    t_out: np.ndarray,
) -> np.ndarray:
    """Evaluate raw signal shifted by lag_s seconds on any output grid.
    y_aligned(t) = y_signal(t - lag_s).  Out-of-range -> NaN.
    """
    arr = np.asarray(signal, dtype=float).ravel()
    t_sig = np.arange(arr.size, dtype=float) * dt
    f = interp1d(t_sig, arr, kind="linear", bounds_error=False, fill_value=np.nan)
    return f(t_out - lag_s)


def apply_fractional_shift(
    signal: np.ndarray,
    dt: float,
    lag_s: float,
    t_ref: np.ndarray,
) -> np.ndarray:
    """Backward-compatible wrapper for coarse-grid shifting."""
    return apply_fractional_shift_to_grid(signal=signal, dt=dt, lag_s=lag_s, t_out=t_ref)


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def align_runs(
    runs: Sequence[ArrayLike1D],
    dt: float = 0.1,
    reference_index: int = 0,
    fine_dt: float = 0.001,
    max_lag_s: float = 0.15,
    align_method: str = "derivative",
    abs_threshold: Optional[float] = None,
    deriv_threshold: Optional[float] = None,
) -> AlignmentBundle:
    """Align multiple runs to a reference using sub-sample upsampled NCC.

    Parameters
    ----------
    runs            : sequence of 1-D raw power/load signal arrays.
    dt              : coarse sample period in seconds (default 0.1 s).
    reference_index : which run to use as reference (default 0).
    fine_dt         : upsampled time step for NCC (default 0.001 s).
    max_lag_s       : maximum lag to search in seconds (default 0.15 s).
    align_method    : "derivative" (recommended) or "raw".
    abs_threshold   : abs deviation from baseline that marks motion onset.
    deriv_threshold : gradient magnitude that marks motion onset.
    """
    if len(runs) < 1:
        raise ValueError("Provide at least one run.")

    run_arrays = [np.asarray(r, dtype=float).ravel() for r in runs]
    ref_raw = run_arrays[reference_index]
    t_ref = np.arange(ref_raw.size, dtype=float) * dt
    t_fine = np.arange(t_ref[0], t_ref[-1] + 0.5 * fine_dt, fine_dt)

    ref_align = compute_alignment_signal(ref_raw, method=align_method)
    ref_motion = find_motion_start(ref_raw, dt, abs_threshold, deriv_threshold)

    run_align_signals: List[np.ndarray] = []
    results: List[SubSampleAlignmentResult] = []
    aligned_ref_list: List[np.ndarray] = []
    aligned_fine_list: List[np.ndarray] = []

    for idx, run_raw in enumerate(run_arrays):
        motion_start = find_motion_start(run_raw, dt, abs_threshold, deriv_threshold)
        run_align = compute_alignment_signal(run_raw, method=align_method)
        run_align_signals.append(run_align)

        if idx == reference_index:
            aligned = apply_fractional_shift_to_grid(run_raw, dt, 0.0, t_ref)
            aligned_fine = apply_fractional_shift_to_grid(run_raw, dt, 0.0, t_fine)
            aligned_ref_list.append(aligned)
            aligned_fine_list.append(aligned_fine)
            results.append(SubSampleAlignmentResult(
                run_index=idx,
                aligned=aligned,
                aligned_fine=aligned_fine,
                lag_s=0.0,
                lag_fine_samples=0,
                max_corr=1.0,
                motion_start_idx=ref_motion,
            ))
            continue

        lag_s, lag_fine, max_corr = find_best_lag(
            ref_align, run_align, dt=dt, fine_dt=fine_dt, max_lag_s=max_lag_s
        )
        aligned = apply_fractional_shift_to_grid(run_raw, dt, lag_s, t_ref)
        aligned_fine = apply_fractional_shift_to_grid(run_raw, dt, lag_s, t_fine)
        aligned_ref_list.append(aligned)
        aligned_fine_list.append(aligned_fine)
        results.append(SubSampleAlignmentResult(
            run_index=idx,
            aligned=aligned,
            aligned_fine=aligned_fine,
            lag_s=lag_s,
            lag_fine_samples=lag_fine,
            max_corr=max_corr,
            motion_start_idx=motion_start,
        ))

    return AlignmentBundle(
        t_ref=t_ref,
        t_fine=t_fine,
        reference=ref_raw,
        results=results,
        aligned_on_ref_coarse=np.vstack(aligned_ref_list),
        aligned_on_fine=np.vstack(aligned_fine_list),
        ref_align_signal=ref_align,
        run_align_signals=run_align_signals,
    )


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------

def align_csv_by_move(
    csv_path: str,
    move: int,
    dt: float = 0.1,
    fine_dt: float = 0.001,
    max_lag_s: float = 0.15,
    align_method: str = "derivative",
    run_cols: Optional[List[str]] = None,
    abs_threshold: Optional[float] = None,
    deriv_threshold: Optional[float] = None,
) -> Tuple[AlignmentBundle, List[str]]:
    """Load one move from the combined CSV and align all runs.

    Returns (AlignmentBundle, run_labels).
    """
    df = pd.read_csv(csv_path)
    move_df = df[df["Move"] == move].reset_index(drop=True)
    if move_df.empty:
        raise ValueError(f"Move {move} not found in {csv_path}")

    if run_cols is None:
        run_cols = [c for c in df.columns if c.endswith("_Power_W")]
    run_cols = [c for c in run_cols if c in move_df.columns]

    runs: List[np.ndarray] = []
    valid_labels: List[str] = []
    for c in run_cols:
        arr = move_df[c].values.astype(float)
        last_ok = np.where(~np.isnan(arr))[0]
        if last_ok.size == 0:
            continue
        arr = arr[: last_ok[-1] + 1]
        runs.append(arr)
        valid_labels.append(c.replace("_Power_W", ""))

    bundle = align_runs(
        runs,
        dt=dt,
        fine_dt=fine_dt,
        max_lag_s=max_lag_s,
        align_method=align_method,
        abs_threshold=abs_threshold,
        deriv_threshold=deriv_threshold,
    )
    return bundle, valid_labels


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_alignment_table(
    bundle: AlignmentBundle,
    run_labels: Optional[List[str]] = None,
) -> None:
    """Print lag and NCC for each run."""
    print(f"{'Run':<14}  {'motion_start':>13}  {'lag_s':>10}  "
          f"{'lag_fine':>10}  {'max_ncc':>9}")
    print("-" * 64)
    for r in bundle.results:
        label = run_labels[r.run_index] if run_labels else str(r.run_index)
        print(f"{label:<14}  {r.motion_start_idx:>13d}  {r.lag_s:>+10.4f}  "
              f"{r.lag_fine_samples:>10d}  {r.max_corr:>9.4f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_alignment_workflow(
    bundle: AlignmentBundle,
    run_labels: Optional[List[str]] = None,
    title_prefix: str = "Sub-Sample Alignment",
    show: bool = True,
    figsize: Tuple[float, float] = (13, 14),
) -> plt.Figure:
    """Four-panel diagnostic plot.

    Panel 1 - Original raw runs on coarse grid.
    Panel 2 - Alignment signals (derivative) on coarse grid.
    Panel 3 - Upsampled alignment signals on fine grid (cubic interpolation).
    Panel 4 - Final aligned raw runs + mean.
    """
    n_runs = len(bundle.results)
    cmap = plt.cm.get_cmap("tab10", n_runs)
    labels = run_labels or [str(r.run_index) for r in bundle.results]

    fig, axes = plt.subplots(4, 1, figsize=figsize, constrained_layout=True)

    dt = float(bundle.t_ref[1] - bundle.t_ref[0]) if bundle.t_ref.size > 1 else 0.1
    fine_dt = float(bundle.t_fine[1] - bundle.t_fine[0]) if bundle.t_fine.size > 1 else 0.001

    # --- Panel 1: Aligned raw runs on coarse reference grid (comparison view) ---
    ax = axes[0]
    for idx, r in enumerate(bundle.results):
        ax.plot(bundle.t_ref, r.aligned, color=cmap(idx), linewidth=1.2, alpha=0.85,
                label=f"{labels[idx]}  lag={r.lag_s:+.3f}s")
        ax.axvline(r.motion_start_idx * dt, color=cmap(idx),
                   linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_title(f"{title_prefix} — Aligned Runs on Coarse Reference Grid")
    ax.set_xlabel("Reference Time (s)")
    ax.set_ylabel("Power (W)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=7)

    # --- Panel 2: Derivative alignment signals on coarse grid ---
    ax = axes[1]
    for idx, r in enumerate(bundle.results):
        align_sig = bundle.run_align_signals[r.run_index]
        t_run = np.arange(align_sig.size, dtype=float) * dt
        ax.plot(t_run, align_sig, color=cmap(idx), linewidth=1.2, alpha=0.85,
                label=labels[idx])
        ax.axvline(r.motion_start_idx * dt, color=cmap(idx),
                   linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_title(f"{title_prefix} — Derivative Signals Used for NCC (coarse grid)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("d(Power)/d(sample)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=7)

    # --- Panel 3: Upsampled derivative signals on fine grid ---
    ax = axes[2]
    for idx, r in enumerate(bundle.results):
        align_sig = bundle.run_align_signals[r.run_index]
        t_fine_run, fine_sig = _upsample(align_sig, dt, fine_dt)
        # Shift the time axis by the estimated lag so they should now overlap
        ax.plot(t_fine_run - r.lag_s, fine_sig, color=cmap(idx), linewidth=0.8,
                alpha=0.7, label=f"{labels[idx]} (lag={r.lag_s:+.4f}s)")
    ax.set_title(f"{title_prefix} — Upsampled Derivative Signals After Lag Correction (fine grid)")
    ax.set_xlabel("Lag-corrected Time (s)")
    ax.set_ylabel("d(Power)/d(sample) upsampled")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=7)

    # --- Panel 4: Final aligned raw runs + mean on fine grid ---
    ax = axes[3]
    aligned_stack = bundle.aligned_on_fine
    mean_wave = np.nanmean(aligned_stack, axis=0)
    for idx in range(aligned_stack.shape[0]):
        ax.plot(bundle.t_fine, aligned_stack[idx], color=cmap(idx), linewidth=1.0,
                alpha=0.55, label=labels[idx])
    ax.plot(bundle.t_fine, mean_wave, color="black", linewidth=2.5,
            label="Mean", zorder=10)
    ax.set_title(f"{title_prefix} — Final Aligned Runs + Mean (Fine Grid)")
    ax.set_xlabel("Fine Time (s)")
    ax.set_ylabel("Power (W)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=7)

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Synthetic demo
# ---------------------------------------------------------------------------

def _make_synthetic_runs(
    n_runs: int = 8,
    n_samples: int = 120,
    dt: float = 0.1,
    max_frac_shift_s: float = 0.09,
    noise_std: float = 2.0,
    seed: int = 42,
) -> List[np.ndarray]:
    """Synthetic traces with sub-sample fractional shifts and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=float) * dt
    base = (
        95.0
        + 30.0 * np.exp(-0.5 * ((t - 2.0) / 0.4) ** 2)
        + 15.0 * np.exp(-0.5 * ((t - 5.5) / 0.3) ** 2)
    )
    runs: List[np.ndarray] = []
    for _ in range(n_runs):
        shift_s = rng.uniform(-max_frac_shift_s, max_frac_shift_s)
        f = interp1d(t, base, kind="cubic", bounds_error=False,
                     fill_value=(base[0], base[-1]))
        shifted = f(t - shift_s)
        noise = rng.normal(0.0, noise_std, size=n_samples)
        runs.append(shifted + noise)
    return runs


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def main_synthetic() -> None:
    """Demonstrate sub-sample alignment on synthetic data."""
    runs = _make_synthetic_runs(n_runs=8, n_samples=120, dt=0.1)
    bundle = align_runs(runs, dt=0.1, fine_dt=0.001, max_lag_s=0.15,
                        align_method="derivative")
    print_alignment_table(bundle)
    plot_alignment_workflow(bundle, title_prefix="Synthetic Sub-Sample Alignment")


def main_csv(
    csv_path: str = r"c:\Users\kytho\Documents\Energy_Model\combined_move_traces_runs2_to_11.csv",
    moves_to_plot: Sequence[int] = (1, 5, 10, 50),
    dt: float = 0.1,
    fine_dt: float = 0.001,
    max_lag_s: float = 0.15,
    align_method: str = "derivative",
) -> None:
    """Run sub-sample alignment on the combined move CSV for selected moves."""
    for mv in moves_to_plot:
        print(f"\n{'=' * 60}")
        print(f"Move {mv}")
        print(f"{'=' * 60}")
        try:
            bundle, labels = align_csv_by_move(
                csv_path, mv, dt=dt, fine_dt=fine_dt,
                max_lag_s=max_lag_s, align_method=align_method,
            )
        except ValueError as exc:
            print(f"  Skipped: {exc}")
            continue

        print_alignment_table(bundle, run_labels=labels)
        plot_alignment_workflow(
            bundle, run_labels=labels,
            title_prefix=f"Move {mv} | Sub-Sample Alignment",
            show=True,
        )


if __name__ == "__main__":
    # Swap to main_synthetic() to run the built-in demo without a CSV.
    main_csv()
