import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

from EEG_Ca_treadmill_analysis import extract_params
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from matplotlib.cm import get_cmap
import tkinter as tk
from tkinter import filedialog, simpledialog
from scipy.stats import wilcoxon, rankdata


def select_folder_and_window_params(default_window_len=150, default_n_windows=20):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    folder_path = filedialog.askdirectory(
        parent=root,
        title="Select the 'data' directory",
        initialdir=r"X:\Behavior\Ca_imaging"
    )

    if not folder_path:
        root.destroy()
        raise RuntimeError("Folder was not selected.")

    root.deiconify()
    root.title("Window parameters")
    root.geometry("320x170")
    root.lift()
    root.focus_force()
    root.attributes("-topmost", True)

    window_len_var = tk.StringVar(master=root, value=str(default_window_len))
    n_windows_var = tk.StringVar(master=root, value=str(default_n_windows))

    result = {
        "window_len": default_window_len,
        "n_windows": default_n_windows
    }

    tk.Label(root, text="window_len").pack(pady=(15, 0))
    tk.Entry(root, textvariable=window_len_var).pack()

    tk.Label(root, text="n_windows").pack(pady=(10, 0))
    tk.Entry(root, textvariable=n_windows_var).pack()

    def on_ok():
        try:
            result["window_len"] = int(window_len_var.get())
            result["n_windows"] = int(n_windows_var.get())
        except ValueError:
            result["window_len"] = default_window_len
            result["n_windows"] = default_n_windows

        root.quit()

    tk.Button(root, text="OK", command=on_ok).pack(pady=15)

    root.mainloop()
    root.destroy()

    return folder_path, result["window_len"], result["n_windows"]


def pcs_to_explain_variance(
    dff_event,
    thresholds=(0.5, 0.7),
    zscore_cells=True,
    remove_global_signal=True,
    min_frames=100
):
    X = np.nan_to_num(dff_event, nan=0.0).T
    n_frames, n_cells = X.shape

    if n_cells < 2 or n_frames < min_frames:
        return {thr: np.nan for thr in thresholds}

    if zscore_cells:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, ddof=1, keepdims=True)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd

    if remove_global_signal:
        X = X - X.mean(axis=1, keepdims=True)

    pca = PCA(svd_solver='full')
    pca.fit(X)

    csum = np.cumsum(pca.explained_variance_ratio_)

    results = {}
    for thr in thresholds:
        k = int(np.searchsorted(csum, thr) + 1)
        results[thr] = k

    return results


def _find_consecutive_runs(indices):
    if len(indices) == 0:
        return []

    runs = []
    start = 0

    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            runs.append((start, i - start))
            start = i

    runs.append((start, len(indices) - start))
    return runs


def sample_consecutive_windows(frame_indices, window_len=240, n_windows=7, seed=42, hop_len=None):
    rng = np.random.RandomState(seed)
    candidates = []
    runs = _find_consecutive_runs(frame_indices)

    if hop_len is None or hop_len < 1:
        for start_idx, run_len in runs:
            if run_len >= window_len:
                for off in range(0, run_len - window_len + 1):
                    candidates.append(start_idx + off)
    else:
        for start_idx, run_len in runs:
            if run_len >= window_len:
                for off in range(0, run_len - window_len + 1, hop_len):
                    candidates.append(start_idx + off)

    if len(candidates) == 0:
        return []

    pick = rng.choice(len(candidates), size=min(n_windows, len(candidates)), replace=False)
    starts = [candidates[i] for i in np.sort(pick)]

    windows = []
    for s in starts:
        win = np.array(frame_indices[s:s + window_len], dtype=int)
        windows.append(win)

    return windows


def participation_ratio(dff_batch: np.ndarray) -> float:
    cov = np.cov(dff_batch)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-12]

    if eigvals.size == 0:
        return np.nan

    pr = (eigvals.sum() ** 2) / (np.square(eigvals).sum())
    return pr



def population_coupling_by_cell(dff_binned: np.ndarray) -> np.ndarray:
    """
    Fast population coupling for each cell.

    For each cell, compute Spearman correlation between:
      1) that cell's activity
      2) mean activity of all other cells

    dff_binned shape: cells x binned_frames

    This version is vectorized. Avoid looping over cells with scipy.stats.spearmanr,
    which becomes very slow when n_cells is large.
    """
    X = np.asarray(dff_binned, dtype=float)
    n_cells, n_frames = X.shape

    if n_cells < 2 or n_frames < 2:
        return np.full(n_cells, np.nan)

    # Treat non-finite values conservatively. Your upstream data are usually finite,
    # but this avoids rankdata producing odd results when NaN/inf appears.
    X = np.where(np.isfinite(X), X, np.nan)

    # Leave-one-out population trace for every cell, computed in one shot.
    if np.any(~np.isfinite(X)):
        valid = np.isfinite(X).astype(float)
        X0 = np.nan_to_num(X, nan=0.0)
        sum_all = X0.sum(axis=0, keepdims=True)
        cnt_all = valid.sum(axis=0, keepdims=True)
        pop_sum = sum_all - X0
        pop_cnt = cnt_all - valid
        with np.errstate(invalid="ignore", divide="ignore"):
            Pop = pop_sum / pop_cnt
        Pop[pop_cnt <= 0] = np.nan
    else:
        Pop = (X.sum(axis=0, keepdims=True) - X) / (n_cells - 1)

    couplings = np.full(n_cells, np.nan, dtype=float)

    finite_rows = (
        np.all(np.isfinite(X), axis=1) &
        np.all(np.isfinite(Pop), axis=1) &
        (np.std(X, axis=1) > 0) &
        (np.std(Pop, axis=1) > 0)
    )

    if not np.any(finite_rows):
        return couplings

    Xv = X[finite_rows]
    Pv = Pop[finite_rows]

    # Spearman = Pearson correlation of rank-transformed traces.
    RX = rankdata(Xv, axis=1)
    RP = rankdata(Pv, axis=1)

    RX = RX - RX.mean(axis=1, keepdims=True)
    RP = RP - RP.mean(axis=1, keepdims=True)

    denom = np.sqrt(np.sum(RX * RX, axis=1) * np.sum(RP * RP, axis=1))
    ok = denom > 0

    vals = np.full(RX.shape[0], np.nan, dtype=float)
    vals[ok] = np.sum(RX[ok] * RP[ok], axis=1) / denom[ok]

    couplings[finite_rows] = vals
    return couplings


def process_folder(
    data_folder,
    analysis_conditions,
    data_pattern,
    records,
    pca_results,
    pr_results,
    corr_batch_results,
    mouse_corr_results,
    pop_coupling_results,
    seed,
    pca_batchsize,
    window_len,
    n_windows
):
    event_path = os.path.join(data_folder, "_Combined", "manual_event.csv")
    print("##### " + os.path.basename(data_folder) + " ######")

    if not os.path.exists(event_path):
        print("manual_event.csv was not found")
        return records, pca_results, pr_results, corr_batch_results, mouse_corr_results, pop_coupling_results

    event_df = pd.read_csv(event_path)
    *_, contime = extract_params(data_folder)

    frame2p_df = pd.read_csv(os.path.join(data_folder, "_Combined", "2p_frame_time_combined.csv"))
    spks = np.load(os.path.join(data_folder, "_GCaMP", "_spks_cell.npy"))
    Fc_all = np.load(os.path.join(data_folder, "_GCaMP", "suite2p_bleach_corrected", "F_corrected.npy"))
    iscell = np.load(os.path.join(data_folder, "_GCaMP", "suite2p", "plane0", "iscell.npy"))

    cell_indices = np.where(iscell[:, 0] == 1)[0]
    Fc = Fc_all[cell_indices]

    if data_pattern == "F":
        def compute_dff(Fc):
            dff = np.zeros_like(Fc)
            for i in range(Fc.shape[0]):
                baseline = np.percentile(Fc[i, :], 20)
                dff[i, :] = (Fc[i, :] - baseline) / (baseline + 1e-8)
            return dff

        dff = compute_dff(Fc)

    elif data_pattern == "spks":
        dff = spks

    else:
        raise ValueError("data_pattern must be 'F' or 'spks'")

    for tw_id, cond in enumerate(analysis_conditions):
        tw = cond["tw"]
        event_groups = cond["events"]

        print("time window:", tw)

        event_df_tw = event_df[
            (event_df["start_time"] >= tw[0] * 60) &
            (event_df["end_time"] <= tw[1] * 60)
        ].copy()

        for event_idx, (event_label, keyword) in enumerate(event_groups):
            group = event_df_tw[
                event_df_tw["event_name"].astype(str).str.contains(
                    keyword,
                    regex=False,
                    na=False
                )
            ].copy()

            if group.empty:
                print(f"skip: tw={tw}, event_label={event_label}, keyword={keyword}")
                continue

            event_name = event_label

            frame_indices = []
            for _, row in group.iterrows():
                frames = frame2p_df[
                    (frame2p_df["time"] >= row["start_time"]) &
                    (frame2p_df["time"] <= row["end_time"])
                ]["frame"].values

                frame_indices.extend(frames.tolist())

            frame_indices = sorted(set(frame_indices))

            if len(frame_indices) < 2:
                continue

            win_frames_list = sample_consecutive_windows(
                frame_indices,
                window_len=window_len,
                n_windows=n_windows,
                seed=seed,
                hop_len=window_len
            )

            if len(win_frames_list) < n_windows:
                print(event_name, "len(win_frames_list)", len(win_frames_list))
                continue

            binning_size = 3
            mouse_id = os.path.basename(data_folder)[2:15]

            dff_win0 = dff[:, win_frames_list[0]]
            n_cells0, n_frames0 = dff_win0.shape
            n_groups0 = n_frames0 // binning_size

            if n_groups0 < 2:
                continue

            dff_win0_b = (
                dff_win0[:, :n_groups0 * binning_size]
                .reshape(n_cells0, n_groups0, binning_size)
                .mean(axis=2)
            )

            n_cells_b = dff_win0_b.shape[0]
            triu_idx = np.triu_indices(n_cells_b, k=1)
            n_pairs = len(triu_idx[0])

            z_sum = np.zeros(n_pairs, dtype=float)
            valid_windows = 0

            thr_list = (0.3, 0.5, 0.8)

            n_cells_all = n_cells_b
            n_groups_cell = max(1, n_cells_all // pca_batchsize)
            batch_ids = (np.arange(n_cells_all) % n_groups_cell) + 1

            batch_sizes = {
                g: int(np.sum(batch_ids == g))
                for g in range(1, n_groups_cell + 1)
            }

            pca_agg = {
                g: {thr: [0.0, 0] for thr in thr_list}
                for g in range(1, n_groups_cell + 1)
            }

            pca_valid_windows = 0

            pr_agg_sum = {g: 0.0 for g in range(1, n_groups_cell + 1)}
            pr_agg_cnt = {g: 0 for g in range(1, n_groups_cell + 1)}

            corr_batch_zsum = {g: 0.0 for g in range(1, n_groups_cell + 1)}
            corr_batch_cnt = {g: 0 for g in range(1, n_groups_cell + 1)}

            mouse_corr_zsum = 0.0
            mouse_corr_cnt = 0

            pop_coupling_zsum = 0.0
            pop_coupling_cnt = 0
            pop_coupling_cell_zsum = np.zeros(n_cells_all, dtype=float)
            pop_coupling_cell_cnt = np.zeros(n_cells_all, dtype=int)

            all_windows_valid = True

            for win_frames in win_frames_list:
                dff_win = dff[:, win_frames]
                n_cells_w, n_frames_w = dff_win.shape
                n_groups_w = n_frames_w // binning_size

                if n_groups_w < 2:
                    all_windows_valid = False
                    break

                dff_win_b = (
                    dff_win[:, :n_groups_w * binning_size]
                    .reshape(n_cells_w, n_groups_w, binning_size)
                    .mean(axis=2)
                )

                corr_matrix, _ = spearmanr(dff_win_b, axis=1)

                if corr_matrix.ndim != 2:
                    all_windows_valid = False
                    break

                triu_idx = np.triu_indices_from(corr_matrix, k=1)
                r_vals = corr_matrix[triu_idx]
                r_vals = np.clip(r_vals, -0.999999, 0.999999)

                z_vals = np.arctanh(r_vals)

                m = np.isfinite(z_vals)
                z_sum[m] += z_vals[m]
                valid_windows += 1

                if np.sum(m) > 0:
                    mouse_corr_zsum += float(np.mean(z_vals[m]))
                    mouse_corr_cnt += 1

                pc_vals = population_coupling_by_cell(dff_win_b)
                pc_vals = np.clip(pc_vals, -0.999999, 0.999999)
                pc_z = np.arctanh(pc_vals)
                pc_m = np.isfinite(pc_z)

                if np.sum(pc_m) > 0:
                    pop_coupling_zsum += float(np.mean(pc_z[pc_m]))
                    pop_coupling_cnt += 1
                    pop_coupling_cell_zsum[pc_m] += pc_z[pc_m]
                    pop_coupling_cell_cnt[pc_m] += 1

                for g in range(1, n_groups_cell + 1):
                    if batch_sizes[g] < 2:
                        continue

                    cell_mask = (batch_ids == g)
                    dff_batch = dff_win_b[cell_mask, :]

                    corr_g, _ = spearmanr(dff_batch, axis=1)

                    if corr_g.ndim != 2 or corr_g.shape[0] < 2:
                        continue

                    tri_g = np.triu_indices_from(corr_g, k=1)
                    r_g = corr_g[tri_g]
                    z_g = np.arctanh(np.clip(r_g, -0.999999, 0.999999))
                    z_g = z_g[np.isfinite(z_g)]

                    if z_g.size > 0:
                        corr_batch_zsum[g] += float(z_g.mean())
                        corr_batch_cnt[g] += 1

                for g in range(1, n_groups_cell + 1):
                    cell_mask = (batch_ids == g)
                    dff_batch = dff_win_b[cell_mask, :]

                    k_dict = pcs_to_explain_variance(
                        dff_batch,
                        thresholds=thr_list,
                        zscore_cells=True,
                        remove_global_signal=False,
                        min_frames=1
                    )

                    for thr, kval in k_dict.items():
                        if np.isfinite(kval):
                            pca_agg[g][thr][0] += float(kval)
                            pca_agg[g][thr][1] += 1

                    pr_val = participation_ratio(dff_batch)

                    if np.isfinite(pr_val):
                        pr_agg_sum[g] += float(pr_val)
                        pr_agg_cnt[g] += 1

                pca_valid_windows += 1

            if (
                (not all_windows_valid) or
                (valid_windows < n_windows) or
                (pca_valid_windows < n_windows)
            ):
                continue

            z_mean = z_sum / n_windows
            r_mean = np.tanh(z_mean)

            for pair_id, r in enumerate(r_mean):
                records.append({
                    "mouse_id": mouse_id,
                    "event_name": event_name,
                    "event_keyword": keyword,
                    "tw_id": tw_id,
                    "tw_start_min": tw[0],
                    "tw_end_min": tw[1],
                    "pair_id": pair_id,
                    "r": float(r)
                })

            for g in range(1, n_groups_cell + 1):
                rec = {
                    "mouse_id": mouse_id,
                    "tw_id": tw_id,
                    "tw_start_min": tw[0],
                    "tw_end_min": tw[1],
                    "event_name": event_name,
                    "event_keyword": keyword,
                    "event_idx": event_idx,
                    "batch_id": g,
                    "n_cells_in_batch": batch_sizes[g],
                    "n_cells_total": int(n_cells_all),
                    "n_groups": int(n_groups_cell),
                }

                for thr in thr_list:
                    s, c = pca_agg[g][thr]
                    rec[f"k_thr_{int(thr * 100)}"] = (s / c) if c > 0 else np.nan

                pca_results.append(rec)

                cnt = pr_agg_cnt[g]
                pr_mean = (pr_agg_sum[g] / cnt) if cnt > 0 else np.nan
                n_in_batch = batch_sizes[g]

                pr_results.append({
                    "mouse_id": mouse_id,
                    "tw_id": tw_id,
                    "tw_start_min": tw[0],
                    "tw_end_min": tw[1],
                    "event_name": event_name,
                    "event_keyword": keyword,
                    "event_idx": event_idx,
                    "batch_id": g,
                    "n_cells_in_batch": n_in_batch,
                    "n_cells_total": int(n_cells_all),
                    "n_groups": int(n_groups_cell),
                    "pr": pr_mean,
                    "pr_norm": (
                        pr_mean / n_in_batch
                        if np.isfinite(pr_mean) and n_in_batch > 0
                        else np.nan
                    ),
                })

            for g in range(1, n_groups_cell + 1):
                c = corr_batch_cnt[g]

                if c > 0:
                    r_batch_mean = float(np.tanh(corr_batch_zsum[g] / c))
                else:
                    r_batch_mean = np.nan

                corr_batch_results.append({
                    "mouse_id": mouse_id,
                    "tw_id": tw_id,
                    "tw_start_min": tw[0],
                    "tw_end_min": tw[1],
                    "event_name": event_name,
                    "event_keyword": keyword,
                    "event_idx": event_idx,
                    "batch_id": g,
                    "n_cells_in_batch": batch_sizes[g],
                    "n_cells_total": int(n_cells_all),
                    "n_groups": int(n_groups_cell),
                    "r_batch_mean": r_batch_mean
                })

            if mouse_corr_cnt > 0:
                r_mouse_mean = float(np.tanh(mouse_corr_zsum / mouse_corr_cnt))
            else:
                r_mouse_mean = np.nan

            mouse_corr_results.append({
                "mouse_id": mouse_id,
                "tw_id": tw_id,
                "tw_start_min": tw[0],
                "tw_end_min": tw[1],
                "event_name": event_name,
                "event_keyword": keyword,
                "event_idx": event_idx,
                "n_cells_total": int(n_cells_all),
                "r_mouse_mean": r_mouse_mean
            })

            if pop_coupling_cnt > 0:
                pop_coupling_mouse_mean = float(np.tanh(pop_coupling_zsum / pop_coupling_cnt))
            else:
                pop_coupling_mouse_mean = np.nan

            valid_cell_mask = pop_coupling_cell_cnt > 0
            if np.sum(valid_cell_mask) > 0:
                pop_coupling_cell_mean = np.full(n_cells_all, np.nan, dtype=float)
                pop_coupling_cell_mean[valid_cell_mask] = np.tanh(
                    pop_coupling_cell_zsum[valid_cell_mask] /
                    pop_coupling_cell_cnt[valid_cell_mask]
                )
                pop_coupling_cell_mean_of_cells = float(np.nanmean(pop_coupling_cell_mean))
                n_cells_valid_pc = int(np.sum(valid_cell_mask))
            else:
                pop_coupling_cell_mean_of_cells = np.nan
                n_cells_valid_pc = 0

            pop_coupling_results.append({
                "mouse_id": mouse_id,
                "tw_id": tw_id,
                "tw_start_min": tw[0],
                "tw_end_min": tw[1],
                "event_name": event_name,
                "event_keyword": keyword,
                "event_idx": event_idx,
                "n_cells_total": int(n_cells_all),
                "n_cells_valid": n_cells_valid_pc,
                "population_coupling_mouse_mean": pop_coupling_mouse_mean,
                "population_coupling_cell_mean": pop_coupling_cell_mean_of_cells
            })

    return records, pca_results, pr_results, corr_batch_results, mouse_corr_results, pop_coupling_results


def plot_group_ecdfs(
    df: pd.DataFrame,
    group_cols=("tw_id", "event_name"),
    value_col="r",
    ax=None,
    event_order=("Before_mobile", "Before_immobile", "After_mobile", "After_immobile", "After_StateC"),
    lw=1.0,
    alpha=0.9,
    show_legend=True,
    legend_max=20
):
    if ax is None:
        ax = plt.gca()

    d = df[list(group_cols) + [value_col]].dropna().copy()

    if d.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(value_col)
        ax.set_ylabel("Cumulative prob.")
        return ax

    tws = np.sort(d[group_cols[0]].unique())
    evs = list(event_order)

    cmap = get_cmap("tab20")
    color_map = {}

    lines = []
    labels = []

    for ti, tw in enumerate(tws):
        for ei, ev in enumerate(evs):
            sub = d[
                (d[group_cols[0]] == tw) &
                (d[group_cols[1]] == ev)
            ]

            vals = np.sort(sub[value_col].to_numpy())
            n = len(vals)

            if n == 0:
                continue

            y = np.arange(1, n + 1) / n
            key = (tw, ev)

            if key not in color_map:
                color_map[key] = cmap((ti * len(evs) + ei) % cmap.N)

            line, = ax.step(
                vals,
                y,
                where="post",
                lw=lw,
                alpha=alpha,
                color=color_map[key]
            )

            lines.append(line)
            labels.append(f"tw={tw}, {ev} (n={n})")

    ax.set_xlabel("r")
    ax.set_ylabel("Cumulative prob.")
    ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.3, 0.3)

    if show_legend and len(lines) > 0:
        if len(lines) > legend_max:
            ax.legend(
                lines[:legend_max],
                labels[:legend_max],
                fontsize=8,
                loc="lower right",
                framealpha=0.9,
                ncol=1,
                title="ECDF (truncated)"
            )
        else:
            ax.legend(
                lines,
                labels,
                fontsize=8,
                loc="lower right",
                framealpha=0.9,
                ncol=1,
                title="ECDF"
            )

    return ax


def plot_bargraph(
    df,
    x,
    y,
    x2_order,
    ax,
    bar_width=0.7,
    block_gap=1.0,
    cmap_name="tab20",
    connect='id',
    xtick_every=1
):
    x1, x2 = x

    needed_cols = [x1, x2, y]
    optional_cols = ["mouse_id", "batch_id"]

    d = df[[c for c in needed_cols + optional_cols if c in df.columns]].dropna(subset=needed_cols).copy()

    if d.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return ax

    events = list(x2_order)
    tws = np.sort(d[x1].unique())

    ev_to_idx = {ev: i for i, ev in enumerate(events)}
    events_per_block = len(events)
    tw_to_block = {tw: bi for bi, tw in enumerate(tws)}

    d["_ev_idx"] = d[x2].map(ev_to_idx)
    d["_block"] = d[x1].map(tw_to_block)

    d = d[d["_ev_idx"].notna() & d["_block"].notna()].copy()
    d["_ev_idx"] = d["_ev_idx"].astype(int)

    # d["_x"] = (
    #     d["_block"].astype(float) * (events_per_block + float(block_gap))
    #     + d["_ev_idx"].astype(float)
    # )

    n_tw = len(tws)

    d["_x"] = (
            d["_ev_idx"].astype(float) * (n_tw + float(block_gap))
            + d["_block"].astype(float)
    )

    # ===== mean / sem =====
    means = []
    sems = []
    xs_bar = []
    xticklabels = []

    for ev in events:
        base_ev = events.index(ev) * (len(tws) + block_gap)

        for tw in tws:
            sub = d[(d[x1] == tw) & (d[x2] == ev)][y]

            means.append(sub.mean())
            sems.append(sub.sem())

            xs_bar.append(base_ev + tw_to_block[tw])
            xticklabels.append(f"{ev}\n{tw}")

    means = np.asarray(means)
    sems = np.nan_to_num(np.asarray(sems), nan=0.0)
    xs_bar = np.asarray(xs_bar)

    xs_bar = []
    xticklabels = []

    for i, ev in enumerate(events):
        base_ev = i * (len(tws) + block_gap)

        for tw in tws:
            xs_bar.append(base_ev + tw_to_block[tw])
            xticklabels.append(f"{ev}\n{tw}")

    xs_bar = np.array(xs_bar, dtype=float)

    ax.bar(
        xs_bar,
        means,
        width=bar_width,
        color="none",
        edgecolor="black",
        zorder=1,
        yerr=sems,
        ecolor="black",
        capsize=2,
        error_kw=dict(alpha=0.8, lw=0.8)
    )

    cmap = get_cmap(cmap_name)

    if connect == "id" and ("mouse_id" in d.columns) and ("batch_id" in d.columns):
        for k, ((mouse_id, batch_id), g) in enumerate(d.groupby(["mouse_id", "batch_id"])):
            g = g.sort_values("_x")

            gx = g["_x"].to_numpy(dtype=float)
            gy = g[y].to_numpy(dtype=float)

            mask = np.isfinite(gx) & np.isfinite(gy)

            if np.sum(mask) >= 2:
                ax.plot(
                    gx[mask],
                    gy[mask],
                    color=cmap(k % cmap.N),
                    lw=1.2,
                    alpha=0.7,
                    zorder=2
                )


    elif connect == "mouse" and ("mouse_id" in d.columns):

        for k, (mouse_id, g) in enumerate(d.groupby("mouse_id")):

            g = g.sort_values("_x")

            gx = g["_x"].to_numpy(dtype=float)

            gy = g[y].to_numpy(dtype=float)

            mask = np.isfinite(gx) & np.isfinite(gy)

            if np.sum(mask) >= 2:
                ax.plot(

                    gx[mask],

                    gy[mask],

                    color=cmap(k % cmap.N),

                    lw=1.5,

                    alpha=0.8,

                    zorder=2,

                    label=mouse_id  # ←追加

                )

        ax.legend(

            title="Mouse",

            fontsize=7,

            title_fontsize=8,

            loc="upper left",

            bbox_to_anchor=(1.02, 1),

            borderaxespad=0

        )

    ax.set_xticks(xs_bar[::xtick_every])
    ax.set_xticklabels(xticklabels[::xtick_every], rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_ylabel(y)

    return ax


def process_group(path, window_len, n_windows):
    event_groups_common = [
        ("mobile", "_mobile"),
        ("immobile", "_immobile"),
        ("StateC", "_StateC"),
    ]

    analysis_condition_sets = {
        "Be-45_0_Af0_45_Re45_120": [
            {
                "tw": [-45, 0],
                "events": event_groups_common
            },
            {
                "tw": [0, 45],
                "events": event_groups_common
            },
            {
                "tw": [45, 120],
                "events": event_groups_common
            },
    # "Be-40_0_Af5_45_Re50_90": [
    #     {
    #         "tw": [-40, 0],
    #         "events": event_groups_common
    #     },
    #     {
    #         "tw": [5, 45],
    #         "events": event_groups_common
    #     },
    #     {
    #         "tw": [50, 90],
    #         "events": event_groups_common
    #     },
    # ],
    #     "per10min": [
    #         {
    #             "tw": [-40, -30],
    #             "events": event_groups_common
    #         },
    #         {
    #             "tw": [-30, -20],
    #             "events": event_groups_common
    #         },
    #         {
    #             "tw": [-20, -10],
    #             "events": event_groups_common
    #         },
    #         {
    #             "tw": [-10, 0],
    #             "events": event_groups_common
    #         },
    #         {
    #             "tw": [0, 10],
    #             "events": event_groups_common
    #         },
    #         {
    #             "tw": [10, 20],
    #             "events": event_groups_common
    #         },
    #         {
    #             "tw": [20, 30],
    #             "events": event_groups_common
    #         },
    #         {
    #             "tw": [30, 40],
    #             "events": event_groups_common
    #         },
    #         {
    #             "tw": [40, 50],
    #             "events": event_groups_common
    #         },
    #         {
    #             "tw": [50, 60],
    #             "events": event_groups_common
    #         },
    #         {
    #             "tw": [60, 70],
    #             "events": event_groups_common
    #         },
    #         {
    #             "tw": [70, 80],
    #             "events": event_groups_common
    #         },
    #         {
    #             "tw": [80, 90],
    #             "events": event_groups_common
    #         },
    #         {
    #             "tw": [90, 100],
    #             "events": event_groups_common
    #         },
    #         {
    #             "tw": [100, 110],
    #             "events": event_groups_common
    #         },
    #         {
    #             "tw": [110, 120],
    #             "events": event_groups_common
    #         },
        ]

    }

    event_order = [
        "mobile",
        "immobile",
        "StateC"
    ]

    mouse_list = glob.glob(os.path.join(path, "202*"))

    os.makedirs(os.path.join(path, "_group_analysis"), exist_ok=True)

    for name, analysis_conditions in analysis_condition_sets.items():
        for seed in range(2):
            suffix = f"{name}_win{window_len}_nw{n_windows}_seed{seed}"

            fig = plt.figure(figsize=(len(analysis_conditions) * 3, 36))
            gs = gridspec.GridSpec(6, 1)

            plt.subplots_adjust(wspace=0.05, hspace=0.05)

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[2, 0])
            ax4 = fig.add_subplot(gs[3, 0])
            ax5 = fig.add_subplot(gs[4, 0])
            ax6 = fig.add_subplot(gs[5, 0])

            records_F = []
            pca_results_F = []
            pr_results_F = []
            corr_batch_results_F = []
            mouse_corr_results_F = []
            pop_coupling_results_F = []

            for mouse in mouse_list:
                (
                    records_F,
                    pca_results_F,
                    pr_results_F,
                    corr_batch_results_F,
                    mouse_corr_results_F,
                    pop_coupling_results_F
                ) = process_folder(
                    mouse,
                    analysis_conditions,
                    "F",
                    records_F,
                    pca_results_F,
                    pr_results_F,
                    corr_batch_results_F,
                    mouse_corr_results_F,
                    pop_coupling_results_F,
                    seed,
                    pca_batchsize=20,
                    window_len=window_len,
                    n_windows=n_windows
                )

            df_all_F = pd.DataFrame.from_records(records_F)
            df_pca_F = pd.DataFrame.from_records(pca_results_F)
            df_pr_F = pd.DataFrame.from_records(pr_results_F)
            df_corr_batch_F = pd.DataFrame.from_records(corr_batch_results_F)
            df_mouse_corr_F = pd.DataFrame.from_records(mouse_corr_results_F)
            df_pop_coupling_F = pd.DataFrame.from_records(pop_coupling_results_F)

            # ==========================================================
            # Wilcoxon signed-rank test (immobile tw0 vs tw1 vs tw2)
            # ==========================================================
            print("\n===== Wilcoxon signed-rank test: immobile =====")

            imm = df_corr_batch_F[df_corr_batch_F["event_name"] == "immobile"]

            pivot = imm.pivot_table(
                index=["mouse_id", "batch_id"],
                columns="tw_id",
                values="r_batch_mean"
            ).dropna(subset=[0, 1, 2])

            for a, b in [(0, 1), (1, 2), (0, 2)]:
                stat, p = wilcoxon(pivot[a], pivot[b])
                print(
                    f"tw{a} vs tw{b}: "
                    f"n={len(pivot)}, "
                    f"W={stat:.1f}, "
                    f"p={p:.6f}"
                )

            df_all_F.to_csv(
                os.path.join(path, "_group_analysis", f"Corr_{suffix}.csv"),
                index=False
            )

            df_pca_F.to_csv(
                os.path.join(path, "_group_analysis", f"PCA_{suffix}.csv"),
                index=False
            )

            df_pr_F.to_csv(
                os.path.join(path, "_group_analysis", f"PartitionRatio_{suffix}.csv"),
                index=False
            )

            df_corr_batch_F.to_csv(
                os.path.join(path, "_group_analysis", f"CorrBatch_{suffix}.csv"),
                index=False
            )

            df_mouse_corr_F.to_csv(
                os.path.join(path, "_group_analysis", f"CorrMouseAllCells_{suffix}.csv"),
                index=False
            )

            df_pop_coupling_F.to_csv(
                os.path.join(path, "_group_analysis", f"PopulationCouplingMouse_{suffix}.csv"),
                index=False
            )

            plot_bargraph(
                df_corr_batch_F,
                x=["tw_id", "event_name"],
                y="r_batch_mean",
                x2_order=event_order,
                ax=ax1,
                connect="id"
            )
            ax1.set_title("Batch-wise pairwise correlation")

            plot_bargraph(
                df_pr_F,
                x=["tw_id", "event_name"],
                y="pr",
                x2_order=event_order,
                ax=ax2,
                connect="id"
            )
            ax2.set_title("Participation ratio")

            plot_bargraph(
                df_pca_F,
                x=["tw_id", "event_name"],
                y="k_thr_50",
                x2_order=event_order,
                ax=ax3,
                connect="id"
            )
            ax3.set_title("PCA: PCs explaining 50% variance")

            plot_group_ecdfs(
                df_all_F,
                group_cols=("tw_id", "event_name"),
                value_col="r",
                ax=ax4,
                event_order=event_order,
                lw=0.4,
                alpha=0.9,
                show_legend=True,
                legend_max=20
            )
            ax4.set_title("ECDF of pairwise r")

            plot_bargraph(
                df_mouse_corr_F,
                x=["tw_id", "event_name"],
                y="r_mouse_mean",
                x2_order=event_order,
                ax=ax5,
                connect="mouse"
            )
            ax5.set_title("Mouse-wise all-cell pairwise correlation")

            plot_bargraph(
                df_pop_coupling_F,
                x=["tw_id", "event_name"],
                y="population_coupling_mouse_mean",
                x2_order=event_order,
                ax=ax6,
                connect="mouse"
            )
            ax6.set_title(
                "Mouse-wise population coupling "
                "(cell vs population activity; Okun et al. 2015)"
            )

            plt.tight_layout()

            pdf_path = os.path.join(
                path,
                "_group_analysis",
                f"Corr_PCA_{suffix}.pdf"
            )

            with PdfPages(pdf_path) as pdf:
                pdf.savefig(fig, dpi=300)

            plt.close(fig)


def main():
    path, window_len, n_windows = select_folder_and_window_params()
    process_group(path, window_len, n_windows)


if __name__ == "__main__":
    main()