
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


# # def plot_spaghetti_bars(
# #     fig, gs,
# #     df_all: pd.DataFrame,
# #     tw_id: int,
# #     event_order=("Before_mobile","Before_immobile","After_mobile","After_immobile","After_StateC"),
# #     bar_mode="pair_mean",     # "pair_mean" (=全ペア平均) か "mouse_mean" (=各マウス平均の平均)
# #     max_lines=None,            # 描画する折れ線をサンプリング（重いときに整数指定）
# #     line_alpha=0.2, line_lw=0.7,
# #     bar_alpha=0.9
# ):
#
#     dftw = df_all[df_all["tw_id"] == tw_id].copy()
#     if dftw.empty:
#         print(f"[tw_id={tw_id}] データが空です")
#         return
#
#     # 折れ線用に pivot（行= mouse_id×pair_id、列= event_name）
#     line_df = dftw.pivot_table(index=["mouse_id","pair_id"],
#                                columns="event_name",
#                                values="r",
#                                aggfunc="mean")  # 念のため mean
#     # 列の順序を揃える（無いイベントは落ちるので reindex で補完）
#     line_df = line_df.reindex(columns=list(event_order))
#
#     # ---- 棒グラフの平均 ----
#     if bar_mode == "pair_mean":
#         bar_vals = dftw.groupby("event_name")["r"].mean().reindex(event_order)
#     elif bar_mode == "mouse_mean":
#         # マウスごとの平均 → その平均（マウス数で同じ重み）
#         tmp = dftw.groupby(["mouse_id","event_name"])["r"].mean().reset_index()
#         bar_vals = tmp.groupby("event_name")["r"].mean().reindex(event_order)
#     else:
#         raise ValueError("bar_mode must be 'pair_mean' or 'mouse_mean'")
#
#     # ---- 描画 ----
#     ax = fig.add_subplot(gs[0, tw_id])
#     # 折れ線（スパゲッティ）
#     x = np.arange(len(event_order))
#     pairs_index = line_df.index.to_list()
#
#     if max_lines is not None and max_lines < len(pairs_index):
#         rng = np.random.default_rng(0)
#         pairs_index = rng.choice(pairs_index, size=max_lines, replace=False)
#
#     # for idx in pairs_index:
#     #     y = line_df.loc[idx].values.astype(float)
#     #     m = ~np.isnan(y)
#     #     if m.sum() >= 2:
#     #         ax.plot(x[m], y[m], alpha=line_alpha, lw=line_lw, color="gray", zorder=1, rasterized=True)
#
#     # 棒グラフ（平均）
#     ax.bar(x, bar_vals.values.astype(float), alpha=bar_alpha, zorder=3)
#
#     ax.set_xticks(x)
#     ax.set_xticklabels(event_order, rotation=15)
#     ax.set_xlim(-0.5, len(event_order)-0.5)
#     ax.set_ylim(-0.075,0.15)
#     # ax.grid(alpha=0.2, axis="y")

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

def process_folder(data_folder, analysis_time_window, data_pattern, records, pca_results, pca_batchsize):
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

                dff_event = dff[:, frame_indices]
                #binning average
                group_size = 3  # 60ms相当 100ms相当
                n_cells, n_frames = dff_event.shape
                n_groups = n_frames // group_size  # 余りは切り捨て
                reshaped = dff_event[:, :n_groups * group_size].reshape(n_cells, n_groups, group_size)
                dff_event = reshaped.mean(axis=2)

                # 相関計算
                corr_matrix, _ = spearmanr(dff_event, axis=1)
                triu_idx = np.triu_indices_from(corr_matrix, k=1)
                r_values = corr_matrix[triu_idx]
                r_values = r_values[~np.isnan(r_values)]

                mouse_id = os.path.basename(data_folder)[4:15]
                for pair_id, r in enumerate(r_values):
                    records.append({
                        "mouse_id": mouse_id,
                        "event_name": event_name,
                        "tw_id": tw_id,
                        "pair_id": pair_id,  # 同一(mouse,event,tw)内でのローカルなペア番号
                        "r": float(r)
                    })

                #PCA
                n_cells = dff_event.shape[0]
                n_groups = max(1, n_cells // pca_batchsize)  # 例: 140細胞 & 30 → 4グループ
                # 1..n_groups を循環付番（cell 0 -> 1, cell 1 -> 2, ...）
                batch_ids = (np.arange(n_cells) % n_groups) + 1

                # 各バッチで PCA 要素数（しきい値別）を算出
                for g in range(1, n_groups + 1):
                    cell_mask = (batch_ids == g)
                    dff_batch = dff_event[cell_mask, :]

                    k_dict = pcs_to_explain_variance(
                        dff_batch, thresholds=(0.3, 0.5, 0.8),
                        zscore_cells=True, remove_global_signal=False, min_frames=100
                    )

                    rec = {
                        "mouse_id": mouse_id, #if 'mouse_id' in globals() else None,
                        "tw_id": tw_id,
                        "tw_start_min": tw[0],
                        "tw_end_min": tw[1],
                        "event_name": event_name,
                        "event_idx": event_idx,
                        "batch_id": g,
                        "n_cells_in_batch": int(cell_mask.sum()),
                        "n_cells_total": int(n_cells),
                        "n_groups": int(n_groups),
                    }
                    # しきい値別の k を列展開（例: k_thr_0p2, k_thr_0p5, …）
                    for thr, kval in k_dict.items():
                        rec[f"k_thr_{str(int(thr*100))}"] = kval

                    pca_results.append(rec)

    return records, pca_results


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
    tws = pd.unique(d[x1])  # 出現順

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
    means = (
        d.groupby([x1, x2], sort=False)[y].mean()
          .reindex(pd.MultiIndex.from_product([tws, events], names=[x1, x2]))
    )

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
    ax.bar(xs_bar, means.to_numpy(), width=bar_width, edgecolor="black", zorder=1)

    # 散布（1回）
    ax.scatter(x_points, y_points, s=22, alpha=alpha_points,
               c=colors, edgecolors='none', zorder=2, rasterized=True)

    # 線接続（大量の ax.plot を避け、LineCollection で一括）
    if connect == 'index':
        # 各 (tw, ev) ごとの配列を作り、k番目同士を繋ぐセグメント集合を構築
        lists = (d.groupby([x1, x2])[y]
                   .apply(list)
                   .reindex(pd.MultiIndex.from_product([tws, events], names=[x1, x2])))

        segments = []   # [[(x1,y1),(x2,y2),...], ...] ではなく、線分ごとに [(x1,y1),(x2,y2)]
        colors_lc = []  # 線の色は点の色と同じ規則（k番目で色循環）

        for tw in tws:
            base_tw = tw_to_block[tw] * (events_per_block + block_gap)
            xs_tw = np.array([base_tw + i for i in range(events_per_block)], dtype=float)
            # その tw の各イベントの値リスト
            vals_seq = [lists.loc[(tw, ev)] if isinstance(lists.loc[(tw, ev)], list) else [] for ev in events]
            max_k = max((len(v) for v in vals_seq), default=0)
            for k in range(max_k):
                xs_k, ys_k = [], []
                for i, vlist in enumerate(vals_seq):
                    if k < len(vlist):
                        xs_k.append(xs_tw[i])
                        ys_k.append(vlist[k])
                if len(xs_k) >= 2:
                    # 隣り合う点同士で線分を作る（折れ線を線分群に分解）
                    for i in range(len(xs_k)-1):
                        segments.append([(xs_k[i], ys_k[i]), (xs_k[i+1], ys_k[i+1])])
                        colors_lc.append(cmap(k % n_colors))

        if segments:
            lc = LineCollection(segments, colors=colors_lc, linewidths=1.0, alpha=0.6, zorder=1.4)
            ax.add_collection(lc)

    # xtick（間引き）
    xticklabels = []
    for tw in tws:
        xticklabels.extend([f"{tw}\n{ev}" for ev in events])

    ax.set_xticks(xs_bar[::xtick_every])
    ax.set_xticklabels(xticklabels[::xtick_every], rotation=45, ha="right")

    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    return ax


def process_group (path):
    # analysis_time_window = [[-30,0], [0,30], [30,60]]
    time_window_sets = {
        "-60-60min": [[-60, 60]],
        "-60-30min": [[-60, 30]],
        "-20-40min": [[-20,40]],
        "per20min": [[-20, 0], [0, 20], [20, 40]],
        "per5min": [[i, i + 5] for i in range(-30, 60, 5)]
    }

    event_order = [ "Before_mobile", "Before_immobile", "After_mobile", "After_immobile", "After_StateC"]

    mouse_list = glob.glob(os.path.join(path, "202*"))

    for name, analysis_time_window in time_window_sets.items():
        fig = plt.figure(figsize=(len(analysis_time_window)*4, 12))
        gs = gridspec.GridSpec(3, 1)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        ax1,ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]),fig.add_subplot(gs[2, 0])
        records_F = []
        records_spks = []
        pca_results_F = []
        for mouse in mouse_list:
            records_F, pca_results_F = process_folder (mouse, analysis_time_window, "F", records_F, pca_results_F, pca_batchsize=30)
            # records_spks = process_folder(mouse, analysis_time_window, "spks", records_spks)

        df_all_F = pd.DataFrame.from_records(records_F)
        df_all_F.to_csv (os.path.join(path, "_group_analysis", "Corr_"+name+".csv"))
        df_pca_F = pd.DataFrame.from_records(pca_results_F)
        df_pca_F.to_csv(os.path.join(path, "_group_analysis", "PCA_" + name + ".csv"))
        # df_summary_F = (df_all_F
        #       .groupby(["mouse_id","event_name","tw_id"], as_index=False)
        #       .agg(r_median=("r","median"),
        #            r_mean=("r","mean"),
        #            n_pairs=("r","size")))
        # df_summary_F.to_csv (os.path.join(path, "_group_analysis", "Corr_"+name+".csv"))

        plot_violin(df_all_F, x=["tw_id", "event_name"], y="r", x2_order=event_order, ax=ax1)

        plot_bargraph(df_pca_F,x=["tw_id", "event_name"], y="k_thr_30", x2_order=event_order, ax=ax2)
        plot_bargraph(df_pca_F, x=["tw_id", "event_name"], y="k_thr_50", x2_order=event_order, ax=ax3)


        plt.tight_layout()
        # plt.legend(fontsize=1, labelspacing=0.1)
        pdf_path = os.path.join(path, "_group_analysis", "Corr_PCA_"+name+".pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, dpi=300)
        plt.close(fig)

def main():
    path= r"X:\Behavior\Ca_imaging"
    process_group(path)


if __name__ == "__main__":
    main()