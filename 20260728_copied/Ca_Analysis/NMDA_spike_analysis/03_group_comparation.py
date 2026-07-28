"""
Summary across mice: aggregate spine-only NMDA spike events
=============================================================

Workflow
--------
1. Select the TOP-LEVEL folder that contains one subfolder per mouse
   (each mouse's `*_spine_only_events.csv`, produced by
   spine_shaft_nmda_analysis.py, is searched for recursively underneath
   -- e.g. <root>/<mouse_id>/_GCaMP/suite2p/.../plane0_roi_traces_spine_only_events.csv).
2. All matching files are read and concatenated, tagged with a `mouse`
   column derived from the first path component under the selected root
   (i.e. the per-mouse folder name).
3. Outputs are written to <root>/_summary/ (created if it doesn't exist):
     summary_spine_only_events.csv   all events from all mice, with an
                                      added `mouse` column (first 15
                                      characters of the per-mouse folder
                                      name)
     summary_dff_distribution.pdf    before vs after histogram of
                                      spine-only peak dF/F0, pooled across
                                      all mice (same binning/axis
                                      convention as the per-mouse figures:
                                      0-1.5 range, 0.1 bins, %, shared y-axis)
"""

import os
import sys
import glob
import json as jsonlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import tkinter as tk
    from tkinter import filedialog
    _TKINTER_AVAILABLE = True
except ImportError:
    _TKINTER_AVAILABLE = False

HIST_XMAX = 1.5
HIST_BIN_WIDTH = 0.1

EVENTS_FILENAME_PATTERN = "*_spine_only_events.csv"


# ---------------------------------------------------------------
# Folder selection
# ---------------------------------------------------------------
def select_root_folder():
    if not _TKINTER_AVAILABLE:
        raise RuntimeError("tkinter is not available in this environment; "
                            "pass the root folder path as a command-line argument instead.")
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title="Select the top-level folder containing all mice"
    )
    root.destroy()
    if not folder:
        print("No folder selected, exiting.")
        sys.exit(0)
    return folder


def find_event_csvs(root_folder):
    pattern = os.path.join(root_folder, "**", EVENTS_FILENAME_PATTERN)
    all_matches = glob.glob(pattern, recursive=True)
    # exclude anything already inside a _summary folder (e.g. from a previous run)
    # so re-running the script doesn't fold last time's summary back in as a "mouse"
    filtered = [p for p in all_matches
                if "_summary" not in os.path.relpath(p, root_folder).split(os.sep)]
    return sorted(filtered)


def mouse_id_from_path(csv_path, root_folder):
    rel = os.path.relpath(csv_path, root_folder)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) > 1 else os.path.splitext(os.path.basename(csv_path))[0]


def find_json_in_folder(folder):
    """Return the path to the single *.json time-structure file in `folder`,
    or None if missing/ambiguous (caller should skip rate normalization for
    that mouse and warn)."""
    candidates = [f for f in os.listdir(folder) if f.lower().endswith(".json")]
    if len(candidates) == 1:
        return os.path.join(folder, candidates[0])
    return None


def parse_window_durations_sec(json_path):
    """Return {'before': seconds, 'after': seconds} -- the length of the
    NMDA_spike_analysis window used by spine_shaft_nmda_analysis.py, read
    from the same JSON it consumes."""
    with open(json_path) as f:
        spec = jsonlib.load(f)
    t = spec["Time"]
    fs = float(t["sampling_rate"][0])
    nmda = t["NMDA_spike_analysis"]
    return {
        "before": (nmda[0][1] - nmda[0][0]) / fs,
        "after": (nmda[1][1] - nmda[1][0]) / fs,
    }


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def run_summary(root_folder):
    csv_paths = find_event_csvs(root_folder)
    if not csv_paths:
        raise FileNotFoundError(
            f"No files matching '{EVENTS_FILENAME_PATTERN}' found under {root_folder}."
        )

    print(f"found {len(csv_paths)} events file(s):")
    dfs = []
    obs_minutes = {"before": 0.0, "after": 0.0}   # total pair-minutes observed, for Events/min
    for p in csv_paths:
        mouse = mouse_id_from_path(p, root_folder)[:15]
        d = pd.read_csv(p)
        d["mouse"] = mouse
        dfs.append(d)
        n_before = (d["phase"] == "before").sum()
        n_after = (d["phase"] == "after").sum()
        print(f"  {mouse:15s}  before={n_before:4d}  after={n_after:4d}   ({p})")

        json_path = find_json_in_folder(os.path.dirname(p))
        if json_path is None:
            print(f"    warning: no single json found in this folder -- "
                  f"{mouse} excluded from Events/min normalization")
            continue
        durations = parse_window_durations_sec(json_path)
        n_pairs = d["pair"].nunique()   # same pair list is analyzed in both phases
        obs_minutes["before"] += n_pairs * durations["before"] / 60.0
        obs_minutes["after"] += n_pairs * durations["after"] / 60.0

    print(f"\ntotal observation time for Events/min normalization: "
          f"before={obs_minutes['before']:.1f} pair-min  after={obs_minutes['after']:.1f} pair-min")

    all_events = pd.concat(dfs, ignore_index=True)
    n_mice = all_events["mouse"].nunique()

    summary_dir = os.path.join(root_folder, "_summary")
    os.makedirs(summary_dir, exist_ok=True)

    events_csv_path = os.path.join(summary_dir, "summary_spine_only_events.csv")
    all_events.to_csv(events_csv_path, index=False)
    print(f"\nsaved {events_csv_path}  (n mice = {n_mice}, n events total = {len(all_events)})")

    before = all_events[all_events["phase"] == "before"]
    after = all_events[all_events["phase"] == "after"]
    print(f"pooled: before n={len(before)}  after n={len(after)}")

    bins = np.arange(0, HIST_XMAX + HIST_BIN_WIDTH, HIST_BIN_WIDTH)

    def hist_counts(vals):
        counts, edges = np.histogram(vals, bins=bins)
        return counts, edges

    counts_before, edges = hist_counts(before["peak_dff_spine"].values)
    counts_after, _ = hist_counts(after["peak_dff_spine"].values)

    pct_before = counts_before / counts_before.sum() * 100 if counts_before.sum() > 0 else counts_before.astype(float)
    pct_after = counts_after / counts_after.sum() * 100 if counts_after.sum() > 0 else counts_after.astype(float)

    rate_before = counts_before / obs_minutes["before"] if obs_minutes["before"] > 0 else np.zeros_like(counts_before, dtype=float)
    rate_after = counts_after / obs_minutes["after"] if obs_minutes["after"] > 0 else np.zeros_like(counts_after, dtype=float)

    pct_ymax = np.ceil(max(pct_before.max(initial=0), pct_after.max(initial=0)) / 5) * 5
    pct_ymax = max(pct_ymax, 5)
    rate_ymax = max(rate_before.max(initial=0), rate_after.max(initial=0)) * 1.15
    rate_ymax = max(rate_ymax, 0.01)

    centers = (edges[:-1] + edges[1:]) / 2
    width = HIST_BIN_WIDTH * 0.9

    dist_pdf_path = os.path.join(summary_dir, "summary_dff_distribution.pdf")
    fig, axes = plt.subplots(3, 2, figsize=(11, 12))

    axes[0, 0].bar(centers, pct_before, width=width, color="tab:blue", edgecolor="k", linewidth=0.3)
    axes[0, 0].set_title(f"Before (n={len(before)} events, {n_mice} mice)")
    axes[0, 1].bar(centers, pct_after, width=width, color="tab:orange", edgecolor="k", linewidth=0.3)
    axes[0, 1].set_title(f"After (n={len(after)} events, {n_mice} mice)")
    for ax in axes[0, :]:
        ax.set_xlim(0, HIST_XMAX)
        ax.set_ylim(0, pct_ymax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel("% of events")
    axes[0, 1].set_ylabel("% of events")

    axes[1, 0].bar(centers, rate_before, width=width, color="tab:blue", edgecolor="k", linewidth=0.3, alpha=0.8)
    axes[1, 1].bar(centers, rate_after, width=width, color="tab:orange", edgecolor="k", linewidth=0.3, alpha=0.8)
    for ax in axes[1, :]:
        ax.set_ylim(0, rate_ymax)
        ax.set_xlim(0, HIST_XMAX)
        ax.set_xlabel("spine-only peak dF/F0")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1, 0].set_ylabel("Events / min")
    axes[1, 1].set_ylabel("Events / min")
    axes[1, 0].set_title(f"({obs_minutes['before']:.0f} pair-min observed)", fontsize=8)
    axes[1, 1].set_title(f"({obs_minutes['after']:.0f} pair-min observed)", fontsize=8)

    # ---- row 3: bin-free comparison (ECDF + KDE), before/after overlaid ----
    before_vals = before["peak_dff_spine"].values
    after_vals = after["peak_dff_spine"].values

    def ecdf(vals):
        x = np.sort(vals)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    ax_ecdf = axes[2, 0]
    if len(before_vals) > 0 and len(after_vals) > 0:
        xb, yb = ecdf(before_vals)
        xa, ya = ecdf(after_vals)
        ks_stat, ks_p = stats.ks_2samp(before_vals, after_vals)
        ax_ecdf.step(xb, yb, where="post", color="tab:blue", lw=1.5, label=f"before (n={len(before_vals)})")
        ax_ecdf.step(xa, ya, where="post", color="tab:orange", lw=1.5, label=f"after (n={len(after_vals)})")
        ax_ecdf.set_title(f"ECDF  (KS test: D={ks_stat:.3f}, p={ks_p:.3f})", fontsize=9)
        ax_ecdf.legend(loc="lower right", fontsize=8)
    ax_ecdf.set_xlim(0, HIST_XMAX)
    ax_ecdf.set_ylim(0, 1)
    ax_ecdf.set_xlabel("spine-only peak dF/F0")
    ax_ecdf.set_ylabel("cumulative fraction")
    ax_ecdf.spines["top"].set_visible(False)
    ax_ecdf.spines["right"].set_visible(False)

    ax_kde = axes[2, 1]
    if len(before_vals) > 1 and len(after_vals) > 1:
        xs = np.linspace(0, HIST_XMAX, 400)
        kde_b = stats.gaussian_kde(before_vals)   # bandwidth: Scott's rule (scipy default) = n^(-1/5) x sample std
        kde_a = stats.gaussian_kde(after_vals)
        # scale each KDE by (n_events / observation_minutes) so the curve's
        # area equals that phase's total Events/min -- otherwise gaussian_kde
        # normalizes each curve to integrate to 1 independently, which would
        # silently discard the same absolute-rate information %/KDE(density)
        # already can't show, duplicating row 1 instead of complementing row 2.
        rate_scale_b = len(before_vals) / obs_minutes["before"] if obs_minutes["before"] > 0 else 0
        rate_scale_a = len(after_vals) / obs_minutes["after"] if obs_minutes["after"] > 0 else 0
        kde_b_curve = kde_b(xs) * rate_scale_b
        kde_a_curve = kde_a(xs) * rate_scale_a
        ax_kde.plot(xs, kde_b_curve, color="tab:blue", lw=1.8, label=f"before (n={len(before_vals)})")
        ax_kde.fill_between(xs, kde_b_curve, color="tab:blue", alpha=0.15)
        ax_kde.plot(xs, kde_a_curve, color="tab:orange", lw=1.8, label=f"after (n={len(after_vals)})")
        ax_kde.fill_between(xs, kde_a_curve, color="tab:orange", alpha=0.15)
        ax_kde.legend(loc="upper right", fontsize=8)
    ax_kde.set_xlim(0, HIST_XMAX)
    ax_kde.set_xlabel("spine-only peak dF/F0")
    ax_kde.set_ylabel("Events / min  (KDE-smoothed)")
    ax_kde.set_title("Kernel density estimate, rate-scaled (Gaussian kernel, Scott's-rule bandwidth)", fontsize=9)
    ax_kde.spines["top"].set_visible(False)
    ax_kde.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(dist_pdf_path)
    plt.close(fig)
    print(f"saved {dist_pdf_path}")

    return {"events_csv": events_csv_path, "dist_pdf": dist_pdf_path, "all_events": all_events}


def main():
    root_folder = sys.argv[1] if len(sys.argv) > 1 else select_root_folder()
    run_summary(root_folder)


if __name__ == "__main__":
    main()