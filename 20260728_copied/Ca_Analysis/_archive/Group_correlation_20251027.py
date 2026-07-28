
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12   })
from EEG_Ca_treadmill_analysis import extract_params
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
import tkinter as tk
from tkinter import filedialog

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select the 'data' directory", initialdir=r"X:\Behavior\Ca_imaging")
    root.destroy()
    return folder_path


def get_rho_at_percentile(rho_sorted, cdf, target=0.5):
    idx = np.argmin(np.abs(cdf - target))
    return rho_sorted[idx]

def pcs_to_explain_variance(
    dff_event, thresholds=(0.5, 0.7),
    zscore_cells=True,
    remove_global_signal=True,
    min_frames=100
):
    # dff_event: (n_cells, n_frames)
    X = np.nan_to_num(dff_event, nan=0.0).T  # (frames, cells)
    n_frames, n_cells = X.shape
    if n_cells < 2 or n_frames < min_frames:
        return np.nan

    # 細胞ごと標準化（列ごと）
    if zscore_cells:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, ddof=1, keepdims=True)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd

    # グローバル信号回帰（行平均を引く）
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

def plot_box(
    fig, gs,
    df_all: pd.DataFrame,
    tw_id: int,
    event_order=("Before_mobile","Before_immobile","After_mobile","After_immobile","After_StateC"),
    box_unit="pair",          # "pair"（全ペアのr） or "mouse"（各マウス平均のr）
    whis=1.5, #(5, 95),             # ひげ：分位（外れ値でつぶれないように5–95%）
    show_n=True               # 箱の上に n 表示
):
    """ tw_id の箱ひげ図を描く（スロットは gs[0, tw_id]）。 """

    dftw = df_all[df_all["tw_id"] == tw_id].copy()
    if dftw.empty:
        print(f"[tw_id={tw_id}] データが空です")
        return

    # 箱に入れる単位を選択
    if box_unit == "pair":
        df_plot = dftw[["event_name", "r"]].dropna()
    elif box_unit == "mouse":
        # 各マウス×イベントで平均（中央値にしたいなら .median() に変更）
        df_plot = (dftw.groupby(["mouse_id", "event_name"])["r"]
                        .mean().reset_index()[["event_name", "r"]])
    else:
        raise ValueError("box_unit must be 'pair' or 'mouse'")

    ax = fig.add_subplot(gs[0, tw_id])
    # イベント順を固定しつつ、データのある位置だけ箱を描く
    x = np.arange(len(event_order))
    data, positions, ns = [], [], []
    for i, ev in enumerate(event_order):
        vals = df_plot.loc[df_plot["event_name"] == ev, "r"].to_numpy()
        vals = vals[~np.isnan(vals)]
        ns.append(len(vals))
        if len(vals) > 0:
            data.append(vals)
            positions.append(i)

    if len(data) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    else:
        bp = ax.boxplot(
            data, positions=positions, widths=0.6, whis=whis,
            showmeans=False, meanline=False, patch_artist=True, flierprops=dict(marker='.', markersize=0.2, alpha=0.4)
        )
        # 見やすさ少しだけ調整（色はデフォルトのまま）
        for b in bp["boxes"]:
            b.set_alpha(0.6)
        for k in ("whiskers", "caps", "medians", "means"):
            for obj in bp[k]:
                obj.set_alpha(0.8)

    # 軸や目盛り
    ax.set_xticks(x)
    ax.set_xticklabels(event_order, rotation=15)
    ax.set_xlim(-0.5, len(event_order) - 0.5)
    ax.axhline(0, color="k", lw=0.7, alpha=0.4)
    ax.grid(alpha=0.25, axis="y")
    ax.set_ylabel("Spearman r")
    ax.set_ylim(-0.075, 0.15)

    # y範囲はロバストに（全イベント合算の1–99%で余白を追加）
    # if len(df_plot) > 0:
    #     q1, q99 = np.nanpercentile(df_plot["r"], [1, 99])
    #     pad = max((q99 - q1) * 0.1, 0.02)
    #     ax.set_ylim(q1 - pad, q99 + pad)

    # n を表示（データが無いイベントはスキップ）
    if show_n:
        y_top = ax.get_ylim()[1]
        for i, n in enumerate(ns):
            if n > 0:
                ax.text(i, y_top, f"n={n}", ha="center", va="bottom", fontsize=7)


def _find_consecutive_runs(indices):
    """昇順ユニークなframe_indicesから連続ラン(開始idx, 長さ)のリストを返す"""
    if len(indices) == 0:
        return []
    runs = []
    start = 0
    for i in range(1, len(indices)):
        if indices[i] != indices[i-1] + 1:
            runs.append((start, i - start))
            start = i
    runs.append((start, len(indices) - start))
    return runs

def sample_consecutive_windows(frame_indices, window_len=322, n_windows=20, seed=42, hop_len=None):
    """
     昇順ユニーク frame_indices から、長さ window_len の連続フレーム列を抽出。
     hop_len が None のときは従来通り 1 フレーム刻み（最大に重複）、
     hop_len を指定すると、そのフレーム数ぶん開始位置を飛ばして候補を作る（重複抑制）。
    """
    rng = np.random.RandomState(seed)
    candidates = []
    runs = _find_consecutive_runs(frame_indices)

    if hop_len is None or hop_len < 1:
        # 1フレーム刻み（従来仕様）
        for start_idx, run_len in runs:
            if run_len >= window_len:
                for off in range(0, run_len - window_len + 1):
                    candidates.append(start_idx + off)
    else:
        # hop_len フレーム刻み（重複抑制）
        for start_idx, run_len in runs:
            if run_len >= window_len:
                for off in range(0, run_len - window_len + 1, hop_len):
                    candidates.append(start_idx + off)

    if len(candidates) == 0:
        return []

    # 候補数が少なければそのまま、十分あればランダムに n_windows 個
    pick = rng.choice(len(candidates), size=min(n_windows, len(candidates)), replace=False)
    starts = [candidates[i] for i in np.sort(pick)]

    windows = []
    for s in starts:
        win = np.array(frame_indices[s:s + window_len], dtype=int)
        windows.append(win)
    return windows

def participation_ratio_from_corr(corr_mat: np.ndarray) -> float:
    """Spearman/pearsonの相関行列からParticipation Ratioを計算"""
    # 数値対策：対称化 + 微小ダイアゴナル
    C = 0.5*(corr_mat + corr_mat.T)
    np.fill_diagonal(C, 1.0)
    C = C + 1e-9*np.eye(C.shape[0])
    vals = np.linalg.eigvalsh(C)
    vals = np.clip(vals, 0, None)  # 数値誤差の負を0に
    s1 = vals.sum()           # 相関行列なら ≈ Ncells
    s2 = np.sum(vals**2)
    return (s1*s1)/s2 if s2 > 0 else np.nan

def participation_ratio(dff_batch: np.ndarray) -> float:
    """
    dff_batch: shape = (n_cells, n_frames)
    共分散行列に基づく Participation Ratio を計算
    """
    # 共分散行列を計算（行:細胞, 列:時間）
    cov = np.cov(dff_batch)  # shape (n_cells, n_cells)

    # 固有値を計算
    eigvals = np.linalg.eigvalsh(cov)  # 対称行列なので安定なeigvalshを推奨
    eigvals = eigvals[eigvals > 1e-12]  # 数値誤差のゼロ/負値を除去

    # Participation Ratio
    pr = (eigvals.sum() ** 2) / (np.square(eigvals).sum())
    return pr

def process_folder(data_folder, analysis_time_window, data_pattern, records, pca_results,pr_results,corr_batch_results, seed, pca_batchsize):
    event_path = os.path.join(data_folder, "_Combined", "manual_event.csv")
    # event_path = os.path.join(data_folder, "_Combined", "manual_event.csv")
    print("##### " + os.path.basename(data_folder) + " ######")

    if not os.path.exists(event_path):
        print("event_combined.csv was not found")
    else:
        event_df = pd.read_csv(event_path)
        *_, contime = extract_params(data_folder)
        frame2p_df = pd.read_csv(os.path.join(data_folder, "_Combined", "2p_frame_time_combined.csv"))
        spks = np.load(os.path.join(data_folder, "_GCaMP", "_spks_cell.npy"))  # F_correctedをもとに生成されたもののはず
        Fc_all = np.load(os.path.join(data_folder, "_GCaMP", "suite2p_bleach_corrected", "F_corrected.npy"))
        iscell = np.load(os.path.join(data_folder, "_GCaMP", "suite2p", "plane0", "iscell.npy"))
        cell_indices = np.where(iscell[:, 0] == 1)[0]
        Fc = Fc_all[cell_indices]

        event_names = sorted(event_df['event_name'].unique())
        event_name_to_idx = {name: i for i, name in enumerate(event_names)}
        event_num = event_df.groupby('event_name').ngroups



        if data_pattern == "F":
            def compute_dff(Fc, win=100):
                dff = np.zeros_like(Fc)
                for i in range(Fc.shape[0]):
                    baseline = np.percentile(Fc[i, :], 20)
                    dff[i, :] = (Fc[i, :] - baseline) / (baseline + 1e-8)
                return dff

            dff = compute_dff(Fc)
            v = 0.1
        if data_pattern == "spks":
            dff = spks
            v = 0.02

        for tw_id, tw in enumerate(analysis_time_window):
            print(tw)
            event_df_tw = event_df[
                (event_df["start_time"] >= tw[0] * 60) & (event_df["end_time"] <= tw[1] * 60)
                ]

            for event_name, group in event_df_tw.groupby('event_name'):
                event_idx = event_name_to_idx[event_name]
                # 各eventごとのフレームを収集
                frame_indices = []
                for _, row in group.iterrows():
                    frames = frame2p_df[
                        (frame2p_df['time'] >= row['start_time']) &
                        (frame2p_df['time'] <= row['end_time'])
                        ]['frame'].values
                    frame_indices.extend(frames.tolist())

                frame_indices = sorted(set(frame_indices))  # 重複除去＆昇順

                if len(frame_indices) < 2:
                    continue  # 相関が計算できない場合はスキップ

                # --- windowサンプリング設定 ---
                window_len = 240
                n_windows = 7

                # 連続windowをサンプリング
                win_frames_list = sample_consecutive_windows(
                    frame_indices, window_len=window_len, n_windows=n_windows, seed=seed, hop_len=window_len
                )
                # print(win_frames_list)
                # 20本取れなければ、この(mouse,event,tw)は完全スキップ（recordもpca_resultsも作らない）
                if len(win_frames_list) < n_windows:
                    print(event_name, "len(win_frames_list)", len(win_frames_list))
                    continue

                # 各windowで相関とPCAを求める。どれか1本でもbinning後に短すぎる（n_groups<2）ならスキップ
                binning_size = 3  # 従来どおりのbinning（60–100ms相当）
                mouse_id = os.path.basename(data_folder)[4:15]

                # まず shape/ペアの固定用に最初のwindowをbinning
                dff_win0 = dff[:, win_frames_list[0]]
                n_cells0, n_frames0 = dff_win0.shape
                n_groups0 = n_frames0 // binning_size
                if n_groups0 < 2:
                    continue
                dff_win0_b = dff_win0[:, :n_groups0 * binning_size].reshape(n_cells0, n_groups0, binning_size).mean(axis=2)

                # ペアの上三角インデックスを固定
                n_cells_b = dff_win0_b.shape[0]
                triu_idx = np.triu_indices(n_cells_b, k=1)
                n_pairs = len(triu_idx[0])

                # Fisher z の蓄積
                z_sum = np.zeros(n_pairs, dtype=float)
                valid_windows = 0

                # PCA集計の準備（バッチ分割は従来ロジック）
                thr_list = (0.3, 0.5, 0.8)
                n_cells_all = n_cells_b  # binning後のセル数（=元のセル数）
                n_groups_cell = max(1, n_cells_all // pca_batchsize)
                batch_ids = (np.arange(n_cells_all) % n_groups_cell) + 1 #TODO これだと、ぴったり20細胞になるとは限らない。　n_used_cells = (n_cells_all // pca_batchsize) * pca_batchsizeとかをいれたらよい
                batch_sizes = {g: int(np.sum(batch_ids == g)) for g in range(1, n_groups_cell + 1)}
                # 閾値ごとの合計・カウントをバッチ毎に保持
                pca_agg = {g: {thr: [0.0, 0] for thr in thr_list} for g in range(1, n_groups_cell + 1)}
                pca_valid_windows = 0
                pr_agg_sum = {g: 0.0 for g in range(1, n_groups_cell + 1)}
                pr_agg_cnt = {g: 0 for g in range(1, n_groups_cell + 1)}

                corr_batch_zsum = {g: 0.0 for g in range(1, n_groups_cell + 1)}
                corr_batch_cnt = {g: 0 for g in range(1, n_groups_cell + 1)}

                # ---- 各windowで相関とPCAを計算 ----
                all_windows_valid = True
                for win_frames in win_frames_list:
                    dff_win = dff[:, win_frames]
                    n_cells_w, n_frames_w = dff_win.shape
                    n_groups_w = n_frames_w // binning_size
                    if n_groups_w < 2:
                        all_windows_valid = False
                        break

                    # binning
                    dff_win_b = dff_win[:, :n_groups_w * binning_size].reshape(n_cells_w, n_groups_w, binning_size).mean(
                        axis=2)

                    # --- 相関（Pearson） ---
                    # セル×セルの相関行列（セルを行に）
                    # corr_matrix = np.corrcoef(dff_win_b)
                    # r_vals = corr_matrix[triu_idx]
                    # r_vals = np.clip(r_vals, -0.999999, 0.999999)  # 数値安定化

                    #Spearman
                    corr_matrix, _ = spearmanr(dff_win_b, axis=1)
                    triu_idx = np.triu_indices_from(corr_matrix, k=1)
                    r_vals = corr_matrix[triu_idx]

                    z_vals = np.arctanh(r_vals)
                    # ここでは20本揃っている前提だが、念のため有限値のみ加算
                    m = np.isfinite(z_vals)
                    z_sum[m] += z_vals[m]
                    valid_windows += 1

                    for g in range(1, n_groups_cell + 1):
                        if batch_sizes[g] < 2:
                            continue
                        cell_mask = (batch_ids == g)
                        dff_batch = dff_win_b[cell_mask, :]
                        # バッチ内相関
                        corr_g, _ = spearmanr(dff_batch, axis=1)
                        if corr_g.ndim != 2 or corr_g.shape[0] < 2:
                            continue
                        tri_g = np.triu_indices_from(corr_g, k=1)
                        r_g = corr_g[tri_g]
                        z_g = np.arctanh(np.clip(r_g, -0.999999, 0.999999))
                        z_g = z_g[np.isfinite(z_g)]
                        if z_g.size > 0:
                            # この window における「バッチ内ペアの Fisher z 平均」
                            corr_batch_zsum[g] += float(z_g.mean())
                            corr_batch_cnt[g] += 1



                    # --- PCA（しきい値を説明するPC数） ---
                    for g in range(1, n_groups_cell + 1):
                        cell_mask = (batch_ids == g)
                        dff_batch = dff_win_b[cell_mask, :]

                        k_dict = pcs_to_explain_variance(
                            dff_batch, thresholds=thr_list,
                            zscore_cells=True, remove_global_signal=False, min_frames=1
                        )
                        for thr, kval in k_dict.items():
                            if np.isfinite(kval):
                                pca_agg[g][thr][0] += float(kval)
                                pca_agg[g][thr][1] += 1

                        # idx = np.where(cell_mask)[0]
                        # if idx.size < 2:
                        #     continue
                        # sub_corr = corr_matrix[np.ix_(idx, idx)]
                        # pr_val = participation_ratio_from_corr(sub_corr)
                        pr_val = participation_ratio(dff_batch)# 1〜N_batch
                        if np.isfinite(pr_val):
                            pr_agg_sum[g] += float(pr_val)
                            pr_agg_cnt[g] += 1
                    pca_valid_windows += 1

                # どこかのwindowで不適合（binning後短すぎ）だったらスキップ
                if (not all_windows_valid) or (valid_windows < n_windows) or (pca_valid_windows < n_windows):
                    # 20本すべてが有効でなければ、レコード非作成
                    continue

                # ---- 相関のFisher z平均 -> rに戻す、recordsへ格納 ----
                z_mean = z_sum / n_windows
                r_mean = np.tanh(z_mean)
                for pair_id, r in enumerate(r_mean):
                    records.append({
                        "mouse_id": mouse_id,
                        "event_name": event_name,
                        "tw_id": tw_id,
                        "pair_id": pair_id,
                        "r": float(r)
                    })

                # ---- PCA: 20 window平均でpca_resultsへ格納（1レコード/バッチ）----
                for g in range(1, n_groups_cell + 1):
                    rec = {
                        "mouse_id": mouse_id,
                        "tw_id": tw_id,
                        "tw_start_min": tw[0],
                        "tw_end_min": tw[1],
                        "event_name": event_name,
                        "event_idx": event_idx,
                        "batch_id": g,
                        "n_cells_in_batch": batch_sizes[g],
                        "n_cells_total": int(n_cells_all),
                        "n_groups": int(n_groups_cell),
                    }
                    for thr in thr_list:
                        s, c = pca_agg[g][thr]
                        # 20 windowすべてで有効なら c は n_windows のはずだが、安全に c>0 を確認
                        rec[f"k_thr_{int(thr * 100)}"] = (s / c) if c > 0 else np.nan
                    pca_results.append(rec)

                    cnt = pr_agg_cnt[g]
                    pr_mean = (pr_agg_sum[g] / cnt) if cnt > 0 else np.nan
                    n_in_batch = batch_sizes[g]
                    rec_pr = {
                        "mouse_id": mouse_id,
                        "tw_id": tw_id,
                        "tw_start_min": tw[0],
                        "tw_end_min": tw[1],
                        "event_name": event_name,
                        "event_idx": event_idx,
                        "batch_id": g,
                        "n_cells_in_batch": n_in_batch,
                        "n_cells_total": int(n_cells_all),
                        "n_groups": int(n_groups_cell),
                        # 生PR（1〜n_cells_in_batch）と、比較しやすい正規化PR（0〜1）
                        "pr": pr_mean,
                        "pr_norm": (pr_mean / n_in_batch) if np.isfinite(
                            pr_mean) and n_in_batch > 0 else np.nan,
                    }
                    pr_results.append(rec_pr)
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
                        "event_idx": event_idx,
                        "batch_id": g,
                        "n_cells_in_batch": batch_sizes[g],
                        "n_cells_total": int(n_cells_all),
                        "n_groups": int(n_groups_cell),
                        "r_batch_mean": r_batch_mean
                    })

    return records, pca_results, pr_results,corr_batch_results


def plot_violin(df, x, y, x2_order, ax,
                block_gap=1.0,
                alpha_points=0.8,
                jitter_width=0.2,
                cmap_name="tab20",
                xtick_every=1,
                max_points_per_group=None):
    """
    (x[0], x[1]) ごとの violin plot。
    - 空グループはスキップ
    - 1点だけのグループは点のみ描画
    - 2点以上は violin を描画
    - 散布点はまとめて一括描画で高速化
    """
    x1, x2 = x
    d = df[[x1, x2, y]].dropna().copy()

    events = list(x2_order)
    # tws = pd.unique(d[x1])  # 出現順
    tws = np.sort(d[x1].unique())


    ev_to_idx = {ev: i for i, ev in enumerate(events)}
    tw_to_block = {tw: bi for bi, tw in enumerate(tws)}
    events_per_block = len(events)

    # 位置 & データ収集
    all_positions, all_labels = [], []
    groups_vals = []  # 各グループの numpy 配列（空も含む）

    for tw in tws:
        base = tw_to_block[tw] * (events_per_block + block_gap)
        for ev in events:
            pos = base + ev_to_idx[ev]
            vals = d[(d[x1] == tw) & (d[x2] == ev)][y].to_numpy()
            if max_points_per_group is not None and len(vals) > max_points_per_group:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(vals), size=max_points_per_group, replace=False)
                vals = vals[idx]
            all_positions.append(pos)
            all_labels.append(f"{tw}\n{ev}")
            groups_vals.append(vals)

    # グループを「violin用(>=2)」「singleton(=1)」「empty(=0)」に分ける
    pos_violin, data_violin = [], []
    pos_singleton, data_singleton = [], []

    for pos, vals in zip(all_positions, groups_vals):
        n = len(vals)
        if n >= 2:
            pos_violin.append(pos)
            data_violin.append(vals)
        elif n == 1:
            pos_singleton.append(pos)
            data_singleton.append(vals)  # 1要素配列

    # --- violin（データ>=2のみ）---
    if len(data_violin) > 0:
        parts = ax.violinplot(
            dataset=data_violin,
            positions=pos_violin,
            widths=0.8,
            showmeans=False,
            showmedians=True,
            showextrema=False
        )
        for pc in parts['bodies']:
            pc.set_facecolor("lightgray")
            pc.set_edgecolor("black")
            pc.set_alpha(0.6)
        if 'cmedians' in parts:
            parts['cmedians'].set_color("black")
            parts['cmedians'].set_linewidth(1.0)

    # --- 散布点を一括描画（violin と singleton の両方）---
    cmap = get_cmap(cmap_name)
    n_colors = cmap.N
    rng = np.random.default_rng(0)
    half = jitter_width / 2.0

    # まず長さを数えて配列を確保
    total_pts = int(sum(len(v) for v in groups_vals))
    if total_pts > 0:
        x_points = np.empty(total_pts, dtype=float)
        y_points = np.empty(total_pts, dtype=float)
        c_points = np.empty((total_pts, 4), dtype=float)
        off = 0

        for pos, vals in zip(all_positions, groups_vals):
            n = len(vals)
            if n == 0:
                continue
            jit = rng.uniform(-half, half, size=n)
            x_points[off:off+n] = pos + jit
            y_points[off:off+n] = vals
            idxs = np.arange(n) % n_colors
            c_points[off:off+n] = cmap(idxs)
            off += n

        ax.scatter(x_points, y_points,
                   s=8, alpha=alpha_points,
                   c=c_points, edgecolors='none',
                   zorder=2, rasterized=True)

    # xticks（必要なら間引く）
    ax.set_xticks(all_positions[::xtick_every])
    ax.set_xticklabels(all_labels[::xtick_every], rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    return ax


# ========= 高速化版 BARGRAPH =========
def plot_bargraph(df, x, y, x2_order, ax,
                  bar_width=0.7,
                  jitter_width=0.4,
                  block_gap=1.0,
                  alpha_points=0.9,
                  cmap_name="tab20",
                  connect='index',         # 'none' | 'index'
                  xtick_every=1,           # xtick を間引く
                  max_points_per_group=None # 各グループで点を上限サンプル
                  ):
    """
    (x[0], x[1]) ごとの平均バー＋個別点（バー内で点ごとに色）。
    - 散布は1回にまとめて高速化
    - 線接続(connect='index')は LineCollection で一括描画
    - xtick 間引き／点間引き対応
    """
    x1, x2 = x
    d = df[[x1, x2, y]].dropna().copy()

    events = list(x2_order)
    tws = pd.unique(d[x1])  # 出現順

    ev_to_idx = {ev: i for i, ev in enumerate(events)}
    events_per_block = len(events)
    tw_to_block = {tw: bi for bi, tw in enumerate(tws)}

    # 各行にバー中心 x を付与（ベクトル）
    d["_ev_idx"] = d[x2].map(ev_to_idx)
    d["_block"]  = d[x1].map(tw_to_block)
    d = d[d["_ev_idx"].notna()].copy()
    d["_ev_idx"] = d["_ev_idx"].astype(int)

    base = (d["_block"].to_numpy(dtype=float) *
            (events_per_block + float(block_gap)))
    x_center = base + d["_ev_idx"].to_numpy(dtype=float)

    # バー平均（高速）
    stats = (
        d.groupby([x1, x2], sort=False)[y]
        .agg(['mean', 'sem'])
        .reindex(pd.MultiIndex.from_product([tws, events], names=[x1, x2]))
    )
    means = stats['mean'].to_numpy()
    sems = stats['sem'].to_numpy()
    sems = np.nan_to_num(sems, nan=0.0)  # サンプル1個などでNaNになるのを回避


    # means = (
    #     d.groupby([x1, x2], sort=False)[y].mean()
    #       .reindex(pd.MultiIndex.from_product([tws, events], names=[x1, x2]))
    # )

    # 全バーの中心 x 座標
    xs_bar = []
    for tw in tws:
        base_tw = tw_to_block[tw] * (events_per_block + block_gap)
        xs_bar.extend([base_tw + i for i in range(events_per_block)])
    xs_bar = np.array(xs_bar, dtype=float)

    # 散布色：各バー内の点番号（cumcount）で決める
    # 先にグループごとにサンプル間引き（必要な時のみ）
    if max_points_per_group is not None:
        d["_tmp_idx"] = d.groupby([x1, x2]).cumcount()
        # 乱数で各グループから同数上限をサンプル
        rng = np.random.default_rng(0)
        keep = []
        for (tw, ev), g in d.groupby([x1, x2]):
            if len(g) <= max_points_per_group:
                keep.append(g.index)
            else:
                keep.append(rng.choice(g.index, size=max_points_per_group, replace=False))
        keep_idx = np.concatenate(keep)
        d = d.loc[np.sort(keep_idx)].copy()
        d.drop(columns=["_tmp_idx"], inplace=True)

    d["_in_bar_idx"] = d.groupby([x1, x2]).cumcount()
    cmap = get_cmap(cmap_name)
    n_colors = cmap.N
    colors = cmap((d["_in_bar_idx"].to_numpy() % n_colors))

    # ジッター（ベクトル一括）
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-jitter_width / 2.0, jitter_width / 2.0, size=len(d))
    x_points = x_center[:len(d)] + jitter
    y_points = d[y].to_numpy()

    # === 描画 ===
    # バー
    ax.bar(
        xs_bar, means, width=bar_width,color="none",
        edgecolor="black", zorder=1,
        yerr=sems, ecolor='black', capsize=2,
        error_kw=dict(alpha=0.8, lw=0.8)
    )

    # 散布（1回）
    # ax.scatter(x_points, y_points, s=22, alpha=alpha_points,
    #            c=colors, edgecolors='none', zorder=2, rasterized=True)

    # 線接続（大量の ax.plot を避け、LineCollection で一括）
    if connect == 'index':
        lists = (d.groupby([x1, x2])[y]
                 .apply(list)
                 .reindex(pd.MultiIndex.from_product([tws, events], names=[x1, x2])))

        for tw in tws:
            base_tw = tw_to_block[tw] * (events_per_block + block_gap)
            xs_tw = np.array([base_tw + i for i in range(events_per_block)], dtype=float)

            vals_seq = [lists.loc[(tw, ev)] if isinstance(lists.loc[(tw, ev)], list) else [] for ev in events]
            max_k = max((len(v) for v in vals_seq), default=0)

            # 各バッチ（同じk）を全イベントでつなぐ
            for k in range(max_k):
                ys_k = []
                for vlist in vals_seq:
                    ys_k.append(vlist[k] if k < len(vlist) else np.nan)
                ys_k = np.array(ys_k, dtype=float)

                # nanを除外して連続線を描く
                mask = np.isfinite(ys_k)
                if np.sum(mask) >= 2:
                    ax.plot(xs_tw[mask], ys_k[mask],
                            color=cmap(k % cmap.N),
                            lw=1.2, alpha=0.7, zorder=2)

    # 軸など
    xticklabels = []
    for tw in tws:
        xticklabels.extend([f"{tw}\n{ev}" for ev in events])
    ax.set_xticks(xs_bar[::xtick_every])
    ax.set_xticklabels(xticklabels[::xtick_every], rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    return ax

# def connect_across_tw(ax, df, xcol="tw_id", evcol="event_name", ycol="r_batch_mean",
#                       idcol="mouse_id", ev_from="Before_immobile", tw_from=0,
#                       ev_to="After_mobile", tw_to=1, events_order=None, block_gap=1.0):
#     if events_order is None:
#         events_order = ["Before_mobile","Before_immobile","After_mobile","After_immobile","After_StateC"]
#
#     # tw_id, event -> x座標位置を再現
#     ev_to_idx = {ev: i for i, ev in enumerate(events_order)}
#     events_per_block = len(events_order)
#
#     # 各 tw_id のベース位置（blockごとのオフセット）
#     tws = sorted(df[xcol].unique())
#     tw_to_block = {tw: bi for bi, tw in enumerate(tws)}
#
#     x_from = tw_to_block[tw_from] * (events_per_block + block_gap) + ev_to_idx[ev_from]
#     x_to   = tw_to_block[tw_to]   * (events_per_block + block_gap) + ev_to_idx[ev_to]
#
#     # 各マウス（またはID）ごとに接続
#     for mid, g in df.groupby(idcol):
#         y1 = g.loc[(g[xcol]==tw_from) & (g[evcol]==ev_from), ycol]
#         y2 = g.loc[(g[xcol]==tw_to)   & (g[evcol]==ev_to),   ycol]
#         if len(y1)==1 and len(y2)==1:
#             ax.plot([x_from, x_to], [y1.values[0], y2.values[0]],
#                     color="black", alpha=0.5, lw=1.2, zorder=3)

def plot_group_ecdfs(
    df: pd.DataFrame,
    group_cols=("tw_id", "event_name"),
    value_col="r",
    ax=None,
    event_order=("Before_mobile","Before_immobile","After_mobile","After_immobile","After_StateC"),
    lw=1.0,
    alpha=0.9,
    show_legend=True,
    legend_max=20
):
    """
    df を group_cols（("tw_id","event_name")）で分け、value_col（"r"）の ECDF を ax に重ね描き。
    """
    if ax is None:
        ax = plt.gca()

    d = df[list(group_cols) + [value_col]].dropna().copy()
    if d.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(value_col)
        ax.set_ylabel("Cumulative prob.")
        return ax

    # tw_id を昇順、event_name は所望の順序で並べて色割り当て
    tws = np.sort(d[group_cols[0]].unique())
    evs = list(event_order)
    cmap = get_cmap("tab20")
    color_map = {}

    lines = []
    labels = []

    for ti, tw in enumerate(tws):
        for ei, ev in enumerate(evs):
            sub = d[(d[group_cols[0]] == tw) & (d[group_cols[1]] == ev)]
            vals = np.sort(sub[value_col].to_numpy())
            n = len(vals)
            if n == 0:
                continue

            # ECDF（階段関数）
            y = np.arange(1, n + 1) / n
            # カラー決定：tw と event の組合せで安定に
            key = (tw, ev)
            if key not in color_map:
                color_map[key] = cmap((ti * len(evs) + ei) % cmap.N)

            line, = ax.step(vals, y, where="post",
                            lw=lw, alpha=alpha, color=color_map[key])
            lines.append(line)
            labels.append(f"tw={tw}, {ev} (n={n})")

    ax.set_xlabel("r")
    ax.set_ylabel("Cumulative prob.")
    ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.6)

    # 範囲が既知なら固定（任意）
    # ax.set_xlim(-0.1, 0.15)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.3, 0.3)

    if show_legend and len(lines) > 0:
        # 伝説が増えすぎると邪魔なので制限
        if len(lines) > legend_max:
            # 先頭だけ表示し、残りは省略
            ax.legend(lines[:legend_max], labels[:legend_max], fontsize=8, loc="lower right", framealpha=0.9, ncol=1, title="ECDF (truncated)")
        else:
            ax.legend(lines, labels, fontsize=8, loc="lower right", framealpha=0.9, ncol=1, title="ECDF")
    return ax

def process_group (path):
    # analysis_time_window = [[-30,0], [0,30], [30,60]]
    time_window_sets = {
        # "-60-60min": [[-60, 60]],
        # "-60-30min": [[-60, 30]],
        # "-20-40min": [[-20,40]],
        # "-30-0_0-40min": [[-30, 0], [0, 40]],
        # # "-20-0_0-30_40-80min": [[-20, 0], [0, 30], [40,80]],
        # "per20min": [[-40, -20],[-20, 0], [0, 20], [20, 40], [40,60]],
        # "per25min": [[-50, -25], [-25, 0], [0, 25], [25, 50], [50, 75]],
        # "per40min": [[-40, 0], [0, 40], [40, 80]],
        # "per50min": [[-50, 0], [0, 50], [50, 100]],
        "per40min":[[-40, 0], [5, 45], [50, 90]],
        # "per40min-2": [[-40, 0], [5, 45], [45, 85]],
        # "per35min": [[-35, 0], [10, 45], [45, 80]],
        # "per30min": [[-40, -10], [10, 40], [40, 70]]
        # "per5min": [[i, i + 5] for i in range(-30, 60, 5)]
    }

    event_order = [ "Before_mobile", "Before_immobile", "After_mobile", "After_immobile", "After_StateC"]

    mouse_list = glob.glob(os.path.join(path, "202*"))

    for name, analysis_time_window in time_window_sets.items():
        for seed in range(2):

            fig = plt.figure(figsize=(len(analysis_time_window) * 5, 15))
            gs = gridspec.GridSpec(5, 1)
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            ax1, ax2, ax3, ax4, ax5= fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0]),fig.add_subplot(gs[3, 0]),fig.add_subplot(gs[4, 0])
            records_F = []
            records_spks = []
            pca_results_F = []
            pr_results_F = []
            corr_batch_results_F = []

            for mouse in mouse_list:
                records_F, pca_results_F, pr_results_F,corr_batch_results_F= process_folder (mouse, analysis_time_window, "F", records_F, pca_results_F,pr_results_F, corr_batch_results_F,seed, pca_batchsize=20)
                # records_spks = process_folder(mouse, analysis_time_window, "spks", records_spks)

            df_all_F = pd.DataFrame.from_records(records_F)
            df_all_F.to_csv (os.path.join(path, "_group_analysis", "Corr_"+name+"_seed"+str(seed)+".csv"))
            df_pca_F = pd.DataFrame.from_records(pca_results_F)
            df_pca_F.to_csv(os.path.join(path, "_group_analysis", "PCA_" + name + "_seed"+str(seed)+".csv"))
            df_pr_F = pd.DataFrame.from_records(pr_results_F)
            df_pr_F.to_csv(os.path.join(path, "_group_analysis", "PartitionRatio_" + name +"_seed"+str(seed)+ ".csv"))
            df_corr_batch_F = pd.DataFrame.from_records(corr_batch_results_F)
            df_corr_batch_F.to_csv(os.path.join(path, "_group_analysis", f"CorrBatch_{name}_seed{seed}.csv"),
                                   index=False)
            # df_summary_F = (df_all_F
            #       .groupby(["mouse_id","event_name","tw_id"], as_index=False)
            #       .agg(r_median=("r","median"),
            #            r_mean=("r","mean"),
            #            n_pairs=("r","size")))
            # df_summary_F.to_csv (os.path.join(path, "_group_analysis", "Corr_"+name+".csv"))

            # plot_violin(df_all_F, x=["tw_id", "event_name"], y="r", x2_order=event_order, ax=ax1)
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
            ax4.set_title("ECDF of r by [tw_id, event_name]")

            plot_bargraph(df_corr_batch_F, x=["tw_id", "event_name"], y="r_batch_mean",
                          x2_order=event_order, ax=ax1)
            # connect_across_tw(ax1, df_corr_batch_F,
            #                   ev_from="Before_immobile", tw_from=0,
            #                   ev_to="After_mobile", tw_to=1,
            #                   events_order=event_order)


            plot_bargraph(df_pr_F,x=["tw_id", "event_name"], y="pr", x2_order=event_order, ax=ax2)
            plot_bargraph(df_pca_F, x=["tw_id", "event_name"], y="k_thr_50", x2_order=event_order, ax=ax3)


            plt.tight_layout()
            # plt.legend(fontsize=1, labelspacing=0.1)
            pdf_path = os.path.join(path, "_group_analysis", "Corr_PCA_"+name+"_seed"+str(seed)+".pdf")
            with PdfPages(pdf_path) as pdf:
                pdf.savefig(fig, dpi=300)
            plt.close(fig)

def main():
    # path= r"X:\Behavior\Ca_imaging"
    path = select_folder()
    process_group(path)


if __name__ == "__main__":
    main()