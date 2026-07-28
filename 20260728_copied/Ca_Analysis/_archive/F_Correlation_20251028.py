import h5py
import numpy as np
import pandas as pd
import os
import tifffile
import glob
import matplotlib.pyplot as plt
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12
})
from EEG_Ca_treadmill_analysis import extract_params, select_folder, plot_timeseries
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap
from Group_correlation import participation_ratio, sample_consecutive_windows


# ==========================
# Utilities
# ==========================
def get_rho_at_percentile(rho_sorted, cdf, target=0.5):
    idx = np.argmin(np.abs(cdf - target))
    return rho_sorted[idx]


def pcs_to_explain_variance(
    dff_event, thresholds=(0.5, 0.7),
    zscore_cells=True,
    remove_global_signal=True,
    min_frames=100
):
    """
    dff_event: (n_cells, n_frames)
    return: {thr: k}
    """
    X = np.nan_to_num(dff_event, nan=0.0).T  # (frames, cells)
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


# ==========================
# Main processing
# ==========================
def process(data_folder):
    # event_path = os.path.join(data_folder, "_Combined", "event_combined.csv")
    event_path = os.path.join(data_folder, "_Combined", "manual_event.csv")
    print("##### " + os.path.basename(data_folder) + " ######")

    if not os.path.exists(event_path):
        print("manual_event.csv was not found")
        return

    event_df = pd.read_csv(event_path)
    *_, contime = extract_params(data_folder)
    frame2p_df = pd.read_csv(os.path.join(data_folder, "_Combined", "2p_frame_time_combined.csv"))

    # Data load
    spks = np.load(os.path.join(data_folder, "_GCaMP", "_spks_cell.npy"))
    Fc_all = np.load(os.path.join(data_folder, "_GCaMP", "suite2p_bleach_corrected", "F_corrected.npy"))
    iscell = np.load(os.path.join(data_folder, "_GCaMP", "suite2p", "plane0", "iscell.npy"))
    cell_indices = np.where(iscell[:, 0] == 1)[0]
    Fc = Fc_all[cell_indices]

    # ---- Pre-clustering (running 無関係で、t<0 の期間で Spearman 相関) ----
    before_frames = frame2p_df[frame2p_df['time'] < 0]['frame'].values.tolist()
    before_Fc = Fc[:, before_frames]
    corr_matrix_before, _ = spearmanr(before_Fc, axis=1)  # (n_cells, n_cells)
    corr_matrix_before = np.nan_to_num(corr_matrix_before, nan=0.0)

    kmeans = KMeans(n_clusters=4, random_state=0)
    cluster_labels = kmeans.fit_predict(corr_matrix_before)
    centers = kmeans.cluster_centers_
    mean_vals = centers.mean(axis=1)
    new_order = np.argsort(mean_vals)
    label_map = {old: new for new, old in enumerate(new_order)}
    cluster_labels = np.array([label_map[label] for label in cluster_labels])

    df = pd.DataFrame({
        "original_cell_index": np.arange(len(cluster_labels)),
        "cluster_label": cluster_labels
    })
    df_sorted = df.sort_values(by=["cluster_label", "original_cell_index"]).reset_index(drop=True)
    kmeans_sorted_indices = df_sorted["original_cell_index"].to_numpy()
    df_sorted["sort_order"] = np.arange(len(df_sorted))
    df_sorted.to_csv(os.path.join(data_folder, "_GCaMP", "kmeans_sorting.csv"), index=False)

    # 並び替え適用
    spks = spks[kmeans_sorted_indices]
    Fc = Fc[kmeans_sorted_indices]

    # ---- Analysis settings ----
    time_window_sets = {
        # "-30-0_0-40min": [[-30, 0], [0, 40]],
        # "per20min": [[-20, 0], [0, 20], [20, 40]],
        # "per5min": [[i, i + 5] for i in range(-30, 60, 5)]
        "per40min": [[-40, 0], [5, 45], [50, 90]]
    }
    window_len = 240
    n_windows = 7
    seed = 0
    pca_batchsize = 20
    group_size = 3  # binning frames per group (60-100 ms 相当)

    # n_groups_cell はセル数から固定（データセット内で不変）
    n_cells_all = Fc.shape[0]
    n_groups_cell = max(1, n_cells_all // pca_batchsize)

    # ---- Main loop over data types ----
    for data_pattern in ["F"]: #, "spks"
        if data_pattern == "F":
            def compute_dff(Fc_in, win=100):
                dff_out = np.zeros_like(Fc_in)
                for i in range(Fc_in.shape[0]):
                    baseline = np.percentile(Fc_in[i, :], 20)
                    dff_out[i, :] = (Fc_in[i, :] - baseline) / (baseline + 1e-8)
                return dff_out
            dff_all = compute_dff(Fc)
            v = 0.2  # heatmap range
        else:
            dff_all = spks
            v = 0.02  # heatmap range

        for name, analysis_time_window in time_window_sets.items():
            time_array = np.array([(start + end) / 2 for start, end in analysis_time_window])
            event_names = sorted(event_df['event_name'].unique())
            event_name_to_idx = {nm: i for i, nm in enumerate(event_names)}
            event_num = len(event_names)

            # 相関パーセンタイルの格納配列
            rho_50_array = np.full((event_num, len(analysis_time_window)), np.nan)
            rho_25_array = np.full((event_num, len(analysis_time_window)), np.nan)
            rho_75_array = np.full((event_num, len(analysis_time_window)), np.nan)

            # ★ PR の時系列（g ごと）: dict[g] -> (event_num, n_tw)
            pr_array = {g: np.full((event_num, len(analysis_time_window)), np.nan)
                        for g in range(1, n_groups_cell + 1)}

            # Figure 準備
            if len(analysis_time_window) < 12:
                fig = plt.figure(figsize=(3*(len(analysis_time_window)+2), 8.27))
                gs = gridspec.GridSpec(event_num, len(analysis_time_window) + 2,
                                       width_ratios=[1]*len(analysis_time_window) + [2, 2])
            else:
                fig = plt.figure(figsize=(3*2, 8.27))
                gs = gridspec.GridSpec(event_num, 2)

            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            ax1_dict = {}  # eventごとの CDF プロット領域

            # ---- iterate time windows ----
            for tw_id, tw in enumerate(analysis_time_window):
                # この時間窓に完全に含まれる event だけ抽出
                event_df_tw = event_df[
                    (event_df["start_time"] >= tw[0]*60) & (event_df["end_time"] <= tw[1]*60)
                ]

                for event_name, group in event_df_tw.groupby('event_name'):
                    event_idx = event_name_to_idx[event_name]

                    # 該当 event のフレーム一覧
                    frame_indices = []
                    for _, row in group.iterrows():
                        frames = frame2p_df[
                            (frame2p_df['time'] >= row['start_time']) &
                            (frame2p_df['time'] <= row['end_time'])
                        ]['frame'].values
                        frame_indices.extend(frames.tolist())
                    frame_indices = sorted(set(frame_indices))  # 重複除去＆昇順

                    if len(frame_indices) < window_len:
                        # そもそも連続 window が取れない
                        continue

                    # 連続 window サンプリング
                    win_frames_list = sample_consecutive_windows(
                        frame_indices, window_len=window_len, n_windows=n_windows, seed=seed, hop_len=window_len
                    )
                    if len(win_frames_list) < n_windows:
                        # 20本そろわなければこの (event, tw) はスキップ
                        continue

                    # Fisher z 平均のための初期化（ペア shape 固定）
                    dff_win0 = dff_all[:, win_frames_list[0]]
                    n_cells0, n_frames0 = dff_win0.shape
                    n_groups0 = n_frames0 // group_size
                    if n_groups0 < 2:
                        continue
                    dff_win0_b = dff_win0[:, :n_groups0*group_size].reshape(n_cells0, n_groups0, group_size).mean(axis=2)

                    # 上三角のテンプレ
                    triu_idx_template = np.triu_indices(dff_win0_b.shape[0], k=1)
                    n_pairs = len(triu_idx_template[0])
                    z_sum_vec = np.zeros(n_pairs, dtype=float)  # CDF/percentile用
                    valid_windows = 0

                    # PR 集計（gごと）
                    batch_ids = (np.arange(dff_win0_b.shape[0]) % n_groups_cell) + 1
                    pr_agg_sum = {g: 0.0 for g in range(1, n_groups_cell + 1)}
                    pr_agg_cnt = {g: 0 for g in range(1, n_groups_cell + 1)}

                    # ---- window loop ----
                    all_windows_valid = True
                    for win_frames in win_frames_list:
                        dff_win = dff_all[:, win_frames]
                        n_cells_w, n_frames_w = dff_win.shape
                        n_groups_w = n_frames_w // group_size
                        if n_groups_w < 2:
                            all_windows_valid = False
                            break

                        # binning
                        dff_win_b = dff_win[:, :n_groups_w*group_size].reshape(
                            n_cells_w, n_groups_w, group_size
                        ).mean(axis=2)

                        # --- Spearman 相関 ---
                        corr_matrix, _ = spearmanr(dff_win_b, axis=1)
                        # 上三角を z にして合算（平均相関のため）
                        r_vals = corr_matrix[triu_idx_template]
                        z_vals = np.arctanh(np.clip(r_vals, -0.999999, 0.999999))
                        m = np.isfinite(z_vals)
                        if m.any():
                            if z_vals.shape[0] == z_sum_vec.shape[0]:
                                z_sum_vec[m] += z_vals[m]
                                valid_windows += 1
                            else:
                                all_windows_valid = False
                                break
                        else:
                            all_windows_valid = False
                            break

                        # --- gごとに PR を集計 ---
                        for g in range(1, n_groups_cell + 1):
                            cell_mask = (batch_ids == g)
                            dff_batch = dff_win_b[cell_mask, :]
                            pr_val = participation_ratio(dff_batch)  # 1〜N_batch の範囲
                            if np.isfinite(pr_val):
                                pr_agg_sum[g] += float(pr_val)
                                pr_agg_cnt[g] += 1

                    if (not all_windows_valid) or (valid_windows < n_windows):
                        continue

                    # ---- （1）平均相関ベクトル → パーセンタイル用に分布化 ----
                    z_mean_vec = z_sum_vec / n_windows
                    r_mean_vec = np.tanh(z_mean_vec)
                    r_sorted = np.sort(r_mean_vec)
                    cum_dist = np.linspace(0, 1, len(r_sorted))
                    rho_50_array[event_idx, tw_id] = get_rho_at_percentile(r_sorted, cum_dist, 0.5)
                    rho_25_array[event_idx, tw_id] = get_rho_at_percentile(r_sorted, cum_dist, 0.25)
                    rho_75_array[event_idx, tw_id] = get_rho_at_percentile(r_sorted, cum_dist, 0.75)

                    # ---- （2）平均相関“行列”を復元して Heatmap に使用 ----
                    nC = dff_win0_b.shape[0]
                    r_mean_mat = np.eye(nC, dtype=float)
                    # 上三角に代入
                    r_mean_mat[triu_idx_template] = r_mean_vec
                    # 下三角に反映（対称化）
                    r_mean_mat[(triu_idx_template[1], triu_idx_template[0])] = r_mean_vec

                    # ---- Heatmap / CDF ----
                    if len(analysis_time_window) < 12:
                        ax0 = fig.add_subplot(gs[event_idx, tw_id])
                        cbar = True if tw_id == len(analysis_time_window)-1 else False
                        # colors = [(0.0, 'blue'), (0.5, 'black'), (1.0, 'red')]
                        colors = [(0.0, '#0064ff'), (0.5, 'black'), (1.0, '#ff6400')]
                        cmap = LinearSegmentedColormap.from_list("custom_black_center", colors)
                        sns.heatmap(r_mean_mat, cmap=cmap, vmin=-v, vmax=v, square=True,
                                    cbar_kws={'label': 'Spearman ρ (mean over 20×10s)'},
                                    ax=ax0, xticklabels=False, yticklabels=False, cbar=cbar)
                        ax0.set_title(event_name)


                        #個別にpng保存
                        heatmap_png_path = os.path.join(data_folder, "_GCaMP", f"_{data_pattern}_correlation_{name}_{str(v)}_{str(event_idx)}_{str(tw_id)}.png")
                        fig_heatmap = plt.figure(figsize=(8, 5))
                        ax_heat = fig_heatmap.add_axes([0, 0, 1, 1])
                        sns.heatmap(r_mean_mat, cmap=cmap, vmin=-v, vmax=v, square=True,
                                    # cbar_kws={'label': 'Spearman ρ (mean over 20×10s)'},
                                    ax=ax_heat, xticklabels=False, yticklabels=False, cbar=False,)
                        # plot_heatmap(
                        #     ax_heat,
                        #     avg_data["t_stft"],
                        #     avg_data["f_stft"],
                        #     10 * np.log10(avg_data["power_spectrum"] + 1e-10),
                        #     "",  # f"{group} {state} {type}",
                        #     "",
                        #     80,
                        #     "rainbow",
                        #     time,
                        #     False,
                        #     1,
                        #     38
                        # )
                        ax_heat.set_axis_off()  # これで軸・目盛り・枠まとめて非表示
                        # 念のため（不要だが安全策）
                        ax_heat.set_xticks([]);
                        ax_heat.set_yticks([])
                        for spine in ax_heat.spines.values():
                            spine.set_visible(False)
                        fig_heatmap.savefig(heatmap_png_path, dpi=300, bbox_inches='tight', pad_inches=0)
                        plt.close(fig_heatmap)


                    if event_idx not in ax1_dict:
                        ax1 = fig.add_subplot(gs[event_idx, -2])
                        ax1_dict[event_idx] = ax1
                    else:
                        ax1 = ax1_dict[event_idx]

                    ax1.plot(r_sorted, cum_dist, color=plt.get_cmap("tab20")(tw_id),
                             lw=0.5, label=f"{tw[0]}-{tw[1]}min_{event_name}")
                    ax1.grid(True)

                    # ---- PR の gごとの平均を保存（時系列化）----
                    for g in range(1, n_groups_cell + 1):
                        if pr_agg_cnt[g] > 0:
                            pr_array[g][event_idx, tw_id] = pr_agg_sum[g] / pr_agg_cnt[g]
                        else:
                            pr_array[g][event_idx, tw_id] = np.nan

            # CDF 軸の体裁
            for ax1 in ax1_dict.values():
                handles, labels = ax1.get_legend_handles_labels()
                if len(handles) > 0:
                    ax1.legend(fontsize=6, labelspacing=0.3)
                ax1.set_xlabel('Correlation coefficient (ρ)')
                ax1.set_ylabel('Cumulative distribution')
                ax1.set_xlim([-0.2, 0.4])

            # ---- 右端の時系列図：相関のパーセンタイル + PR(g毎) ----
            for event_idx in range(event_num):
                ax2 = fig.add_subplot(gs[event_idx, -1])
                plot_timeseries(time_array, rho_50_array[event_idx, :], 1, ax2, "blue", 1,
                                None, None, None, label="ρ 50%", alpha=1)
                plot_timeseries(time_array, rho_25_array[event_idx, :], 1, ax2, "lightblue", 1,
                                None, None, None, label="ρ 25%", alpha=1)
                plot_timeseries(time_array, rho_75_array[event_idx, :], 1, ax2, "k", 1,
                                None, None, (0, 0.2), label="ρ 75%", alpha=1)

                ax2.set_xlabel("Time (min)")
                ax2.set_ylabel("ρ percentile")

                ax2_t = ax2.twinx()
                # gごとの PR を重ね描き
                for g in range(1, n_groups_cell + 1):
                    vals = pr_array[g][event_idx, :]
                    ax2_t.plot(time_array, vals, marker='o', linewidth=1, alpha=0.9,
                               label=f"PR g{g}")
                ax2_t.set_ylabel('Participation ratio')
                ax2_t.set_ylim([0,18])

                # 凡例（重複多いので右上にまとめる）
                handles2, labels2 = ax2_t.get_legend_handles_labels()
                if len(handles2) > 0:
                    ax2_t.legend(fontsize=6, loc="upper right")

            plt.tight_layout()
            pdf_path = os.path.join(data_folder, "_GCaMP", f"_{data_pattern}_correlation_{name}_{str(v)}.pdf")
            with PdfPages(pdf_path) as pdf:
                pdf.savefig(fig, dpi=300)
            plt.close(fig)


def main():
    data_folder = select_folder()
    # data_folder = r"X:\Behavior\Ca_imaging\20250724_z253-4_IRES-2x_GCaMP-3e12_soma_imaging_EEG"  # for development
    process(data_folder)


if __name__ == "__main__":
    main()
