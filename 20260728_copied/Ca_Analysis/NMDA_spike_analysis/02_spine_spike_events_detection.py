"""
Spine-only NMDA spike analysis (before vs after comparison)
=============================================================

Workflow
--------
1. Select a `*_roi_traces.csv` file (produced by extract_spine_shaft_rois.py)
   via a file dialog.
2. A JSON file describing the recording's time structure must sit in the
   SAME folder as the csv. If exactly one *.json file is found there, it is
   used automatically; otherwise you'll be asked to pick one.

JSON format (frame units, sampling_rate in Hz)
------------------------------------------------
{
    "Time": {
        "Timer": false,
        "Continuous": [[-10000, 0], [0, 30000]],
        "sampling_rate": [30],
        "NMDA_spike_analysis": [[-9500, -500], [9000, 18000]]
    }
}

- "Continuous": list of [start, end] frame ranges, all relative to the
  before->after transition point (frame 0). Negative-side ranges = before
  segments, positive-side ranges = after segments. Ranges must be
  contiguous and given in chronological order once sorted by start.
  This supports recordings that are internally split into more than one
  physical acquisition segment (e.g. hardware frame-count limits) -- each
  [start,end] pair is treated as one physically continuous recording
  segment, and segment boundaries are checked for bleach/acquisition
  jumps independently.
- "sampling_rate": [Hz]
- "NMDA_spike_analysis": [[before_start, before_end], [after_start, after_end]]
  the actual comparison window, in the same relative-frame convention.

Analysis logic
---------------
1. Parse JSON, map relative frame ranges onto absolute csv frame indices.
2. For every internal boundary between consecutive Continuous segments,
   test for a systematic jump (mean F over a small window on each side,
   compared across all ROIs). Segments separated by a detected jump are
   kept strictly separate when computing the rolling F0 baseline (the
   rolling window is never allowed to cross that boundary); segments with
   no detected jump are merged into one continuous block for F0 purposes.
3. Explicit exponential-decay bleach correction is skipped -- the
   NMDA_spike_analysis windows are short and close to their respective
   segment start, so within-window bleach is assumed negligible. Slow
   drift is instead implicitly absorbed by the rolling-percentile F0.
4. F0 = rolling low-percentile baseline (per effective segment) -> dF/F0
   for the whole trace.
5. Within each NMDA_spike_analysis window: low-pass filter + light
   smoothing on dF/F0, per spine/shaft pair take
   diff = dff_spine - dff_shaft, estimate diff's own rolling baseline +
   noise-adaptive threshold, and flag contiguous excursions above
   threshold (min duration filter) as "spine-only" events. The event's
   peak dF/F0 (spine channel) is recorded.

Outputs (written next to the input csv)
-----------------------------------------
    <stem>_trace_overview.pdf     one page per spine/shaft pair, before &
                                   after side by side, smoothed dF/F0 +
                                   diff + adaptive threshold + shaded
                                   spine-only events
    <stem>_dff_distribution.pdf   before vs after histogram of spine-only
                                   peak dF/F0 (0-3.0 range, 0.1 bins, %)
    <stem>_spine_only_events.csv  pair, phase, peak_dff_spine,
                                   start_time_s, end_time_s (signed:
                                   before = negative, after = positive,
                                   seconds relative to the before->after
                                   transition)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    _TKINTER_AVAILABLE = True
except ImportError:
    _TKINTER_AVAILABLE = False

# ---------------------------------------------------------------
# Analysis parameters (defaults -- adjust here if needed)
# ---------------------------------------------------------------
LOWPASS_CUTOFF_HZ = 2.0
LOWPASS_ORDER = 2
SMOOTH_WIN_FRAMES = 3

F0_WINDOW_SEC = 60.0
F0_PERCENTILE = 8

DIFF_BASELINE_WINDOW_SEC = 60.0
K_THRESH = 2.5
MIN_EVENT_DURATION_FRAMES = 5

BOUNDARY_TEST_WINDOW_FRAMES = 500
BOUNDARY_JUMP_PCT_THRESH = 3.0     # median |jump%| above this -> flag
BOUNDARY_JUMP_CONSISTENCY = 0.7    # fraction of ROIs sharing the same sign

HIST_XMAX = 3.0
HIST_BIN_WIDTH = 0.1


# ---------------------------------------------------------------
# File selection
# ---------------------------------------------------------------
def select_csv_file():
    if not _TKINTER_AVAILABLE:
        raise RuntimeError("tkinter is not available in this environment; "
                            "pass the csv path as a command-line argument instead.")
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select *_roi_traces.csv to analyze",
        filetypes=[("CSV files", "*.csv")]
    )
    root.destroy()
    if not path:
        print("No file selected, exiting.")
        sys.exit(0)
    return path


def find_json_file(folder, root=None):
    candidates = [f for f in os.listdir(folder) if f.lower().endswith(".json")]
    if len(candidates) == 1:
        return os.path.join(folder, candidates[0])
    if len(candidates) == 0:
        raise FileNotFoundError(f"No .json time-structure file found in {folder}.")
    # multiple candidates -> ask
    if not _TKINTER_AVAILABLE:
        raise RuntimeError(f"Multiple JSON files found in {folder} and tkinter is "
                            f"not available to prompt for a choice: {candidates}")
    if root is None:
        root = tk.Tk()
        root.withdraw()
        owns_root = True
    else:
        owns_root = False
    path = filedialog.askopenfilename(
        title="Multiple JSON files found -- pick the time-structure file",
        initialdir=folder, filetypes=[("JSON files", "*.json")]
    )
    if owns_root:
        root.destroy()
    if not path:
        raise FileNotFoundError("No JSON file selected.")
    return path


# ---------------------------------------------------------------
# JSON parsing -> absolute frame ranges
# ---------------------------------------------------------------
def parse_time_json(json_path):
    with open(json_path) as f:
        spec = json.load(f)
    t = spec["Time"]
    fs = float(t["sampling_rate"][0])
    continuous = sorted(t["Continuous"], key=lambda x: x[0])
    offset = -continuous[0][0]   # anchor smallest relative start to absolute frame 0

    segments = []  # list of dicts: {abs_start, abs_end, label}
    for s, e in continuous:
        abs_s, abs_e = s + offset, e + offset
        label = "before" if e <= 0 else "after"
        segments.append({"abs_start": abs_s, "abs_end": abs_e, "label": label})

    nmda = t["NMDA_spike_analysis"]
    before_win = (nmda[0][0] + offset, nmda[0][1] + offset)
    after_win = (nmda[1][0] + offset, nmda[1][1] + offset)

    return {"fs": fs, "segments": segments, "offset": offset,
            "before_win": before_win, "after_win": after_win}


# ---------------------------------------------------------------
# Boundary jump test -> merge segments with no detected jump
# ---------------------------------------------------------------
def test_boundary_jump(piv, boundary_frame, window):
    rows = []
    for name in piv.columns:
        trace = piv[name].values
        if boundary_frame - window < 0 or boundary_frame + window > len(trace):
            continue
        before = trace[boundary_frame - window:boundary_frame]
        after = trace[boundary_frame:boundary_frame + window]
        mean_before, mean_after = before.mean(), after.mean()
        if mean_before == 0:
            continue
        jump_pct = (mean_after - mean_before) / mean_before * 100
        rows.append(jump_pct)
    rows = np.array(rows)
    if len(rows) == 0:
        return False, np.nan
    median_jump = np.median(rows)
    consistency = max((rows > 0).mean(), (rows < 0).mean())
    detected = (abs(median_jump) > BOUNDARY_JUMP_PCT_THRESH) and (consistency > BOUNDARY_JUMP_CONSISTENCY)
    return detected, median_jump


def merge_segments(piv, segments):
    """Return list of (start,end) 'effective' segments for F0 purposes:
    consecutive Continuous segments are merged unless a jump is detected
    at their shared boundary."""
    effective = [dict(segments[0])]
    for seg in segments[1:]:
        boundary = seg["abs_start"]
        jump_detected, median_jump = test_boundary_jump(piv, boundary, BOUNDARY_TEST_WINDOW_FRAMES)
        print(f"  boundary at frame {boundary}: median jump = {median_jump:.2f}%  "
              f"-> {'JUMP DETECTED, keeping segments separate' if jump_detected else 'no jump, merging'}")
        if jump_detected:
            effective.append(dict(seg))
        else:
            effective[-1]["abs_end"] = seg["abs_end"]
    return effective


# ---------------------------------------------------------------
# F0 / dF-F0
# ---------------------------------------------------------------
def rolling_pctl_f0(trace, win, pctl):
    s = pd.Series(trace)
    f0 = s.rolling(window=win, center=True, min_periods=max(10, win // 10)).quantile(pctl / 100.0)
    return f0.bfill().ffill().values


def compute_dff(piv, effective_segments, fs):
    win = int(round(F0_WINDOW_SEC * fs))
    dff = pd.DataFrame(index=piv.index, columns=piv.columns, dtype=np.float64)
    for name in piv.columns:
        trace_full = piv[name].values.astype(np.float64)
        for seg in effective_segments:
            s0, s1 = seg["abs_start"], seg["abs_end"]
            chunk = trace_full[s0:s1]
            f0 = rolling_pctl_f0(chunk, win, F0_PERCENTILE)
            dff.iloc[s0:s1, dff.columns.get_loc(name)] = (chunk - f0) / f0
    return dff.astype(np.float64)


# ---------------------------------------------------------------
# filter / smoothing / event detection
# ---------------------------------------------------------------
def make_lowpass(fs):
    return butter(LOWPASS_ORDER, LOWPASS_CUTOFF_HZ / (fs / 2), btype="low")


def process_trace(trace, b, a):
    lp = filtfilt(b, a, trace)
    return uniform_filter1d(lp, size=SMOOTH_WIN_FRAMES)


def rolling_median(x, win):
    s = pd.Series(x)
    m = s.rolling(window=win, center=True, min_periods=max(10, win // 10)).median()
    return m.bfill().ffill().values


def rolling_mad(residual, win):
    s = pd.Series(np.abs(residual))
    m = s.rolling(window=win, center=True, min_periods=max(10, win // 10)).median()
    return 1.4826 * m.bfill().ffill().values


def find_runs_above(diff_trace, threshold_trace, min_len):
    above = diff_trace > threshold_trace
    runs, start = [], None
    for i, v in enumerate(above):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(above) - start >= min_len:
        runs.append((start, len(above)))
    return runs


def get_pairs(roi_names, folder):
    spine_names = sorted([n for n in roi_names if n.startswith("spine")],
                          key=lambda s: int(s.replace("spine", "")))
    pairs = [(sp, f"shaft{sp.replace('spine', '')}") for sp in spine_names
             if f"shaft{sp.replace('spine', '')}" in roi_names]
    print(f"  same-number pairing ({len(pairs)} pairs)")
    return pairs


# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------
def run_pipeline(csv_path):
    folder = os.path.dirname(os.path.abspath(csv_path))
    stem = os.path.splitext(os.path.basename(csv_path))[0]

    json_path = find_json_file(folder)
    print(f"Using time-structure file: {json_path}")
    tspec = parse_time_json(json_path)
    fs = tspec["fs"]

    df = pd.read_csv(csv_path)
    piv = df.pivot(index="frame", columns="roi_name", values="raw_f").sort_index()
    roi_names = list(piv.columns)
    print(f"loaded {csv_path}: {piv.shape[0]} frames, {len(roi_names)} ROIs, fs={fs} Hz")

    pairs = get_pairs(roi_names, folder)

    print("checking segment boundaries...")
    effective_segments = merge_segments(piv, tspec["segments"])

    print("computing F0 / dF-F0 (per effective segment, no explicit bleach correction)...")
    dff = compute_dff(piv, effective_segments, fs)

    b, a = make_lowpass(fs)
    diff_win = int(round(DIFF_BASELINE_WINDOW_SEC * fs))

    before_win = tspec["before_win"]
    after_win = tspec["after_win"]

    all_events = {"before": [], "after": []}
    per_pair_processed = {}   # for the trace-overview figure

    for phase, win in [("before", before_win), ("after", after_win)]:
        w0, w1 = win
        for sp, sh in pairs:
            dff_sp = dff[sp].values[w0:w1].astype(np.float64)
            dff_sh = dff[sh].values[w0:w1].astype(np.float64)
            sp_p = process_trace(dff_sp, b, a)
            sh_p = process_trace(dff_sh, b, a)
            diff = sp_p - sh_p

            baseline = rolling_median(diff, diff_win)
            noise = rolling_mad(diff - baseline, diff_win)
            threshold = baseline + K_THRESH * noise

            runs = find_runs_above(diff, threshold, MIN_EVENT_DURATION_FRAMES)
            for (s, e) in runs:
                peak_val = sp_p[s:e].max()
                # convert local window index -> signed seconds relative to before/after transition
                rel_frame_start = (w0 + s) - tspec["offset"]
                rel_frame_end = (w0 + e) - tspec["offset"]
                all_events[phase].append({
                    "pair": f"{sp}-{sh}", "phase": phase,
                    "peak_dff_spine": peak_val,
                    "start_time_s": rel_frame_start / fs,
                    "end_time_s": rel_frame_end / fs,
                })

            per_pair_processed[(phase, f"{sp}-{sh}")] = {
                "sp": sp_p, "sh": sh_p, "diff": diff, "baseline": baseline,
                "threshold": threshold, "runs": runs
            }

    before_events = pd.DataFrame(all_events["before"])
    after_events = pd.DataFrame(all_events["after"])
    events_all = pd.concat([before_events, after_events], ignore_index=True)

    events_csv_path = os.path.join(folder, f"{stem}_spine_only_events.csv")
    events_all.to_csv(events_csv_path, index=False)
    print(f"saved {events_csv_path}  (before n={len(before_events)}, after n={len(after_events)})")

    # ---------------- distribution PDF ----------------
    bins = np.arange(0, HIST_XMAX + HIST_BIN_WIDTH, HIST_BIN_WIDTH)

    def to_pct_hist(vals):
        counts, edges = np.histogram(vals, bins=bins)
        pct = counts / counts.sum() * 100 if counts.sum() > 0 else counts.astype(float)
        return pct, edges

    pct_before, edges = to_pct_hist(before_events["peak_dff_spine"].values if len(before_events) else np.array([]))
    pct_after, _ = to_pct_hist(after_events["peak_dff_spine"].values if len(after_events) else np.array([]))
    ymax = np.ceil(max(pct_before.max(initial=0), pct_after.max(initial=0)) / 5) * 5
    ymax = max(ymax, 5)
    centers = (edges[:-1] + edges[1:]) / 2
    width = HIST_BIN_WIDTH * 0.9

    dist_pdf_path = os.path.join(folder, f"{stem}_dff_distribution.pdf")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    axes[0].bar(centers, pct_before, width=width, color="tab:blue", edgecolor="k", linewidth=0.3)
    axes[0].set_title(f"Before (n={len(before_events)} events)")
    axes[1].bar(centers, pct_after, width=width, color="tab:orange", edgecolor="k", linewidth=0.3)
    axes[1].set_title(f"After (n={len(after_events)} events)")
    for ax in axes:
        ax.set_xlim(0, HIST_XMAX)
        ax.set_xlabel("spine-only peak dF/F0")
        ax.set_ylim(0, ymax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("% of events")
    plt.tight_layout()
    fig.savefig(dist_pdf_path)
    plt.close(fig)
    print(f"saved {dist_pdf_path}")

    # ---------------- trace overview PDF ----------------
    trace_pdf_path = os.path.join(folder, f"{stem}_trace_overview.pdf")
    with PdfPages(trace_pdf_path) as pdf:
        for sp, sh in pairs:
            pb = per_pair_processed[("before", f"{sp}-{sh}")]
            pa = per_pair_processed[("after", f"{sp}-{sh}")]
            t_before = np.arange(len(pb["sp"])) / fs
            t_after = np.arange(len(pa["sp"])) / fs

            ymin = min(pb["sp"].min(), pb["sh"].min(), pa["sp"].min(), pa["sh"].min())
            ymax_ = max(pb["threshold"].max(), pa["threshold"].max(),
                        pb["sp"].max(), pa["sp"].max())
            pad = 0.05 * (ymax_ - ymin)

            fig, axes = plt.subplots(1, 2, figsize=(15, 3.8), sharey=True)
            for ax, phase_data, t_axis, title in [
                (axes[0], pb, t_before, f"{sp}-{sh}  BEFORE"),
                (axes[1], pa, t_after, f"{sp}-{sh}  AFTER"),
            ]:
                ax.plot(t_axis, phase_data["sh"], color="tab:cyan", lw=0.8, label="shaft dF/F0")
                ax.plot(t_axis, phase_data["sp"], color="tab:red", lw=0.8, label="spine dF/F0")
                ax.plot(t_axis, phase_data["diff"], color="gray", lw=0.6, alpha=0.6, label="diff")
                ax.plot(t_axis, phase_data["threshold"], color="black", lw=0.8, ls="--", label="threshold")
                for (s, e) in phase_data["runs"]:
                    ax.axvspan(t_axis[s], t_axis[min(e, len(t_axis) - 1)], color="gold", alpha=0.4, lw=0)
                ax.set_title(f"{title}  (n events = {len(phase_data['runs'])})", fontsize=9)
                ax.set_xlabel("time (s)")
                ax.set_ylim(ymin - pad, ymax_ + pad)
            axes[0].set_ylabel("dF/F0")
            axes[0].legend(loc="upper right", fontsize=6.5, framealpha=0.6, ncol=2)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"saved {trace_pdf_path}")

    return {"events_csv": events_csv_path, "dist_pdf": dist_pdf_path, "trace_pdf": trace_pdf_path,
            "before_events": before_events, "after_events": after_events}


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else select_csv_file()
    run_pipeline(csv_path)


if __name__ == "__main__":
    main()