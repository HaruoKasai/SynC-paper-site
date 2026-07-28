import h5py
import numpy as np
import pandas as pd
import os
import re
import tifffile
import glob
import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

from EEG_Ca_treadmill_analysis import extract_params, select_folder
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import binary_dilation
from scipy.stats import ranksums
from itertools import combinations, product
import tkinter as tk
from tkinter import filedialog
import matplotlib as mpl
from matplotlib.cm import get_cmap

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42

# Exact paired permutation settings.  With n matched units there are 2**n
# possible independent condition-label swaps.  n=16 (65,536 permutations)
# is kept as a practical upper limit for exhaustive enumeration.
MAX_EXACT_PERMUTATION_N = 16


def exact_paired_permutation_test(x, y):
    """Two-sided exact paired label-swap permutation test of mean(y - x)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    diff = y[keep] - x[keep]
    n = len(diff)

    if n == 0:
        return np.nan, np.nan, n, "no paired data"
    if np.allclose(diff, 0.0):
        return 0.0, 1.0, n, "all paired differences are zero"
    if n > MAX_EXACT_PERMUTATION_N:
        return (
            float(np.mean(diff)), np.nan, n,
            f"exact enumeration requires 2^{n} permutations; limit is n={MAX_EXACT_PERMUTATION_N}"
        )

    observed = float(np.mean(diff))
    signs = np.asarray(list(product((-1.0, 1.0), repeat=n)), dtype=float)
    null_statistics = (signs @ diff) / n
    p_value = np.count_nonzero(
        np.abs(null_statistics) >= abs(observed) - 1e-12
    ) / len(null_statistics)
    return observed, float(p_value), n, "exact 2^n paired label-swap permutations"


def select_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    folder_path = filedialog.askdirectory(
        parent=root,
        title="Select the 'data' directory",
        initialdir=r"X:\Behavior\Ca_imaging"
    )

    root.destroy()
    return folder_path


def select_analysis_unit():
    root = tk.Tk()
    root.title("Select analysis unit")
    root.geometry("250x160")
    root.lift()
    root.attributes("-topmost", True)
    root.after_idle(root.attributes, "-topmost", False)

    choice = tk.StringVar(master=root, value="cell")

    tk.Label(root, text="Analysis unit").pack(padx=20, pady=10)

    tk.Radiobutton(root, text="Cell-level", variable=choice, value="cell").pack(anchor="w", padx=20)
    tk.Radiobutton(root, text="Mouse-level", variable=choice, value="mouse").pack(anchor="w", padx=20)

    selected = {"value": "cell"}

    def submit():
        selected["value"] = choice.get()
        root.destroy()

    tk.Button(root, text="OK", command=submit).pack(pady=10)

    root.mainloop()
    return selected["value"]


def short_mouse_label(name):
    """
    Convert folder/column names to the same mouse labels used in script 2.
    Example:
        20250724_z253-4 -> 250724_z253-4
        20250724_z253-4_cell12 -> 250724_z253-4
    """
    mouse_name = re.sub(r"_cell\d+$", "", os.path.basename(str(name)))
    return mouse_name[2:15]


def process_folder(data_folder, tw, analysis_unit="cell"):
    event_path = os.path.join(data_folder, "_Combined", "manual_event.csv")
    print("##### " + os.path.basename(data_folder) + " ######")

    if not os.path.exists(event_path):
        print("manual_event.csv was not found")
        return pd.DataFrame([])

    event_df = pd.read_csv(event_path)
    *_, contime = extract_params(data_folder)

    frame2p_df = pd.read_csv(os.path.join(data_folder, "_Combined", "2p_frame_time_combined.csv"))
    spks = np.load(os.path.join(data_folder, "_GCaMP", "_spks_cell.npy"))

    mask = np.zeros(len(event_df), dtype=bool)
    for t0, t1 in tw:
        mask |= (event_df["start_time"] >= t0 * 60) & (event_df["end_time"] <= t1 * 60)

    event_df_tw = event_df[mask]

    event_frames = []
    for _, row in event_df_tw.iterrows():
        frames = frame2p_df[
            (frame2p_df["time"] >= row["start_time"]) &
            (frame2p_df["time"] <= row["end_time"])
        ]["frame"].values
        event_frames.append(frames)

    cell_means_per_event = []
    for frames in event_frames:
        if len(frames) > 0:
            means = spks[:, frames].mean(axis=1)
        else:
            means = np.full(spks.shape[0], np.nan)
        cell_means_per_event.append(means)

    mean_df = pd.DataFrame(cell_means_per_event)
    mean_df["event_name"] = event_df_tw["event_name"].values

    tw_label = "_".join([f"{t0}-{t1}" for t0, t1 in tw])

    mean_df.to_csv(
        os.path.join(data_folder, "_GCaMP", f"_event_mean_fr_{tw_label}min.csv"),
        index=False
    )

    grouped_df = mean_df.groupby("event_name").mean()

    grouped_df.to_csv(
        os.path.join(data_folder, "_GCaMP", f"_event_type_mean_fr_{tw_label}min.csv")
    )

    if analysis_unit == "cell":
        return grouped_df

    elif analysis_unit == "mouse":
        mouse_mean = grouped_df.mean(axis=1)
        mouse_name = os.path.basename(data_folder)
        return pd.DataFrame({mouse_name: mouse_mean})

    else:
        raise ValueError("analysis_unit must be 'cell' or 'mouse'")


def process_group(path, analysis_unit="cell"):

    mouse_list = glob.glob(os.path.join(path, "202*"))
    os.makedirs(os.path.join(path, "_group_analysis"), exist_ok=True)

    tw_pre = [[-45, 0]]
    tw_post = [[0, 45]]
    tw_late = [[45, 90]]

    plot_index = [
        "Before_immobile",
        "After_immobile",
        "Late_immobile",
        "Before_mobile",
        "After_mobile",
        "Late_mobile",
        "After_StateC"
    ]

    plot_df = pd.DataFrame(index=plot_index)

    for mouse in mouse_list:
        mouse_name = os.path.basename(mouse)

        df_pre = process_folder(mouse, tw_pre, analysis_unit=analysis_unit)
        df_post = process_folder(mouse, tw_post, analysis_unit=analysis_unit)
        df_late = process_folder(mouse, tw_late, analysis_unit=analysis_unit)

        if analysis_unit == "mouse":
            mouse_series = pd.Series(index=plot_index, dtype=float, name=mouse_name)

            def get_mouse_value(source_df, keyword):
                matched_idx = [idx for idx in source_df.index if keyword in idx]

                if len(matched_idx) == 0:
                    return np.nan

                if len(matched_idx) > 1:
                    print(f"Warning: multiple matches for {keyword}: {matched_idx}")

                return source_df.loc[matched_idx[0]].iloc[0]

            mouse_series["Before_mobile"] = get_mouse_value(df_pre, "_mobile")
            mouse_series["Before_immobile"] = get_mouse_value(df_pre, "_immobile")

            mouse_series["After_mobile"] = get_mouse_value(df_post, "_mobile")
            mouse_series["After_immobile"] = get_mouse_value(df_post, "_immobile")
            mouse_series["After_StateC"] = get_mouse_value(df_post, "_StateC")

            mouse_series["Late_mobile"] = get_mouse_value(df_late, "_mobile")
            mouse_series["Late_immobile"] = get_mouse_value(df_late, "_immobile")

            plot_df = pd.concat([plot_df, mouse_series], axis=1)

        elif analysis_unit == "cell":
            cell_df = pd.DataFrame(index=plot_index)

            def assign_cell_values(new_name, source_df, keyword):
                matched_idx = [idx for idx in source_df.index if keyword in idx]

                if len(matched_idx) == 0:
                    return

                if len(matched_idx) > 1:
                    print(f"Warning: multiple matches for {keyword}: {matched_idx}")

                values = source_df.loc[matched_idx[0]]

                values.index = [
                    f"{mouse_name}_cell{cell_id}"
                    for cell_id in values.index
                ]

                cell_df.loc[new_name, values.index] = values

            assign_cell_values("Before_mobile", df_pre, "_mobile")
            assign_cell_values("Before_immobile", df_pre, "_immobile")

            assign_cell_values("After_mobile", df_post, "_mobile")
            assign_cell_values("After_immobile", df_post, "_immobile")
            assign_cell_values("After_StateC", df_post, "_StateC")

            assign_cell_values("Late_mobile", df_late, "_mobile")
            assign_cell_values("Late_immobile", df_late, "_immobile")

            plot_df = pd.concat([plot_df, cell_df], axis=1)

    tw_label = (
        f"pre_{tw_pre[0][0]}_{tw_pre[0][1]}"
        f"_post_{tw_post[0][0]}_{tw_post[0][1]}"
        f"_late_{tw_late[0][0]}_{tw_late[0][1]}"
    )

    # =========================
    # Statistics: exact paired permutation tests
    # Every possible within-unit swap of the two condition labels is
    # enumerated.  No asymptotic Wilcoxon/Friedman p values are reported.
    # =========================

    stat_rows = []

    pair_results = []

    for cond1, cond2 in combinations(plot_index, 2):
        pair_df = plot_df.loc[[cond1, cond2]].dropna(axis=1, how="any")
        statistic, p_val, n, note = exact_paired_permutation_test(
            pair_df.loc[cond1].values,
            pair_df.loc[cond2].values
        )

        pair_results.append({
            "test": "exact_paired_permutation",
            "comparison": f"{cond1} vs {cond2}",
            "n": n,
            "statistic": statistic,
            "p_value": p_val,
            "note": note
        })

    n_pairs = len(pair_results)

    for row in pair_results:
        if pd.notna(row["p_value"]):
            row["p_bonferroni"] = min(row["p_value"] * n_pairs, 1.0)
        else:
            row["p_bonferroni"] = np.nan

        stat_rows.append(row)

    stat_df = pd.DataFrame(stat_rows)

    stat_df.to_csv(
        os.path.join(
            path,
            "_group_analysis",
            f"_mean_fr_{analysis_unit}_{tw_label}_stats.csv"
        ),
        index=False
    )

    plot_df.to_csv(
        os.path.join(
            path,
            "_group_analysis",
            f"_mean_fr_{analysis_unit}_{tw_label}.csv"
        )
    )

    fig = plt.figure(figsize=(11.69, 8.27))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])

    # =========================
    # Plot individual trajectories
    # 個体ごとに色を変え、Legendに個体番号を表示
    # =========================

    cmap = get_cmap("tab20")

    if analysis_unit == "mouse":
        mouse_ids = [short_mouse_label(col) for col in plot_df.columns]
        mouse_to_color = {
            mouse_id: cmap(i % cmap.N)
            for i, mouse_id in enumerate(sorted(set(mouse_ids)))
        }

        for col in plot_df.columns:
            mouse_id = short_mouse_label(col)

            ax.plot(
                range(len(plot_df.index)),
                plot_df[col],
                color=mouse_to_color[mouse_id],
                linewidth=1.5,
                alpha=0.85,
                # marker="o",
                # markersize=3,
                label=mouse_id
            )

    else:
        # cell-levelでは、同じ個体由来のcellを同じ色にする
        mouse_ids = [short_mouse_label(col) for col in plot_df.columns]
        mouse_to_color = {
            mouse_id: cmap(i % cmap.N)
            for i, mouse_id in enumerate(sorted(set(mouse_ids)))
        }

        used_labels = set()

        for col in plot_df.columns:
            mouse_id = short_mouse_label(col)
            label = mouse_id if mouse_id not in used_labels else "_nolegend_"
            used_labels.add(mouse_id)

            ax.plot(
                range(len(plot_df.index)),
                plot_df[col],
                color=mouse_to_color[mouse_id],
                linewidth=0.6,
                alpha=0.35,
                marker="o",
                markersize=2,
                label=label
            )

    mean_values = plot_df.mean(axis=1)
    sem_values = plot_df.sem(axis=1)

    ax.bar(
        range(len(plot_df.index)),
        mean_values,
        color="none",
        edgecolor="black",
        alpha=0.8,
        width=0.4,
        yerr=sem_values,
        capsize=2,
        ecolor="black",
        error_kw=dict(alpha=0.7, lw=0.8),
        zorder=10
    )

    ax.set_xticks(range(len(plot_df.index)))
    ax.set_xticklabels(plot_index, rotation=30, ha="right")

    ax.set_ylim([0, 0.4])
    ax.set_ylabel("Mean firing rate")
    ax.set_title(f"Mean FR ({analysis_unit}-level)")

    ax.legend(
        title="Mouse",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=True
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])

    pdf_path = os.path.join(
        path,
        "_group_analysis",
        f"_mean_fr_{analysis_unit}_{tw_label}.pdf"
    )

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, dpi=300)

    plt.close(fig)


def main():
    path = select_folder()
    analysis_unit = select_analysis_unit()
    process_group(path, analysis_unit=analysis_unit)


if __name__ == "__main__":
    main()
