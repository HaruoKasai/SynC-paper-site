import h5py
import numpy as np
import pandas as pd
import os
import tifffile
import glob
import matplotlib.pyplot as plt
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12   })
from EEG_Ca_treadmill_analysis import extract_params, select_folder, plot_timeseries
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap

def get_rho_at_percentile(rho_sorted, cdf, target=0.5):
    idx = np.argmin(np.abs(cdf - target))
    return rho_sorted[idx]

def pcs_to_explain_variance(
    dff_event, threshold=0.5,
    zscore_cells=True,
    remove_global_signal=False,
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
    k = int(np.searchsorted(csum, threshold) + 1)
    return k

def process_folder(data_folder):
    # event_path = os.path.join(data_folder, "_Combined", "event_combined.csv")
    event_path = os.path.join(data_folder, "_Combined", "manual_event.csv")
    print("##### " + os.path.basename(data_folder) + " ######")

    if not os.path.exists(event_path):
        print("event_combined.csv was not found")
    else:
        event_df = pd.read_csv(event_path)
        *_,contime = extract_params(data_folder)
        frame2p_df = pd.read_csv(os.path.join(data_folder, "_Combined", "2p_frame_time_combined.csv"))
        spks = np.load(os.path.join(data_folder, "_GCaMP", "_spks_cell.npy")) #F_correctedをもとに生成されたもののはず
        # F = np.load(os.path.join(data_folder, "_GCaMP", "suite2p", "plane0","F.npy"))
        # Fneu = np.load(os.path.join(data_folder, "_GCaMP", "suite2p", "plane0","Fneu.npy"))
        # neucoeff = 0.7
        # Fc = F - neucoeff * Fneu
        Fc_all = np.load(os.path.join(data_folder, "_GCaMP", "suite2p_bleach_corrected","F_corrected.npy"))
        iscell = np.load(os.path.join(data_folder, "_GCaMP", "suite2p", "plane0", "iscell.npy"))
        cell_indices = np.where(iscell[:, 0] == 1)[0]
        Fc = Fc_all[cell_indices]

        # matches = glob.glob(os.path.join(data_folder, "_GCaMP", "*-0min*eventwise*.csv"))
        # if matches:
        #     cellsort_csv = matches[0]
        #     print("Cell Sorted")
        #     sort_df = pd.read_csv(cellsort_csv)
        #     sorted_indices = sort_df.sort_values("correlation", ascending=False)["cell"].to_numpy(dtype=int)
        #     print("spks",spks.shape)
        #     spks = spks[sorted_indices]
        #     Fc = Fc[sorted_indices]

        # clustering beforeの走ってるデータ (runningに関係なく)
        before_frames = frame2p_df[frame2p_df['time'] <0]['frame'].values.tolist()
        before_Fc = Fc[:, before_frames]
        corr_matrix_before, _ = spearmanr(before_Fc, axis=1)  # (n_cells, n_cells)
        corr_matrix_before = np.nan_to_num(corr_matrix_before, nan=0.0)
        kmeans = KMeans(n_clusters=4, random_state=0)
        cluster_labels = kmeans.fit_predict(corr_matrix_before)
        centers = kmeans.cluster_centers_
        mean_vals = centers.mean(axis=1)
        new_order = np.argsort(mean_vals)

        # 古いラベル → 新しいラベルへのマッピング
        label_map = {old: new for new, old in enumerate(new_order)}
        cluster_labels  = np.array([label_map[label] for label in cluster_labels])

        # --- DataFrameを作ってクラスタ + 元インデックス順に並べる ---
        df = pd.DataFrame({
            "original_cell_index": np.arange(len(cluster_labels)),
            "cluster_label": cluster_labels
        })

        # cluster_label → original_cell_index 順に並べる
        df_sorted = df.sort_values(by=["cluster_label", "original_cell_index"]).reset_index(drop=True)

        # 並び替え順のインデックスを取得
        kmeans_sorted_indices = df_sorted["original_cell_index"].to_numpy()

        # 並び順を追加して保存
        df_sorted["sort_order"] = np.arange(len(df_sorted))
        df_sorted.to_csv(os.path.join(data_folder, "_GCaMP", "kmeans_sorting.csv"), index=False)

        # --- 並び替えを適用 ---
        spks = spks[kmeans_sorted_indices]
        Fc = Fc[kmeans_sorted_indices]

        for data_pattern in ["F", "spks"]: #["spks", "F"]
            if data_pattern=="F":
                def compute_dff(Fc, win=100):
                    dff = np.zeros_like(Fc)
                    for i in range(Fc.shape[0]):
                        baseline = np.percentile(Fc[i, :], 20)
                        dff[i, :] = (Fc[i, :] - baseline) / (baseline + 1e-8)
                    return dff

                dff = compute_dff(Fc)
                v = 0.1
            if data_pattern=="spks":
                dff = spks
                v = 0.02

            # analysis_time_window = [[-30,-20], [-20,-10], [-10,0], [0,10], [10,20],[20,30],[30,40],[40,50],[50,60]]#[[-30,0], [0,30], [30,60]]
            # analysis_time_window = [[-20, 0], [0, 20], [20, 40]]
            # analysis_time_window = [[i, i + 5] for i in range(-30, 60, 5)]
            # analysis_time_window = [[i, i + 1] for i in range(-30, 60)]
            time_window_sets = {
                "per20min": [[-20, 0], [0, 20], [20, 40]],
                "per5min": [[i, i + 5] for i in range(-30, 60, 5)]
            }
            for name, analysis_time_window in time_window_sets.items():
                time_array = np.array([(start + end) / 2 for start, end in analysis_time_window])
                event_names = sorted(event_df['event_name'].unique())
                event_name_to_idx = {name: i for i, name in enumerate(event_names)}
                event_num = event_df.groupby('event_name').ngroups

                rho_50_array =np.full((event_num, len(analysis_time_window)), np.nan)
                rho_25_array = np.full((event_num, len(analysis_time_window)), np.nan)
                rho_75_array = np.full((event_num, len(analysis_time_window)), np.nan)

                k50_array = np.full((event_num, len(analysis_time_window)), np.nan)
                k80_array = np.full((event_num, len(analysis_time_window)), np.nan)
                k90_array = np.full((event_num, len(analysis_time_window)), np.nan)



                if len(analysis_time_window)<12:
                    fig = plt.figure(figsize=(3*(len(analysis_time_window)+2),8.27))
                    gs = gridspec.GridSpec(event_num, len(analysis_time_window) + 2,width_ratios=[1] * len(analysis_time_window) + [2]*2)
                else:
                    fig = plt.figure(figsize=(3 * 2, 8.27))
                    gs = gridspec.GridSpec(event_num, 2)

                plt.subplots_adjust(wspace=0.05, hspace=0.05)
                ax1_dict = {}

                for tw_id, tw in enumerate(analysis_time_window):
                    event_df_tw =  event_df[
                                        (event_df["start_time"] >= tw[0]*60) & (event_df["end_time"] <= tw[1]*60)
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
                        # group_size = 3  # 100ms相当
                        # n_cells, n_frames = dff_event.shape
                        # n_groups = n_frames // group_size  # 余りは切り捨て
                        #
                        # # reshape → 平均 → 新しい行列
                        # reshaped = dff_event[:, :n_groups * group_size].reshape(n_cells, n_groups, group_size)
                        # dff_event = reshaped.mean(axis=2)
                        # print("dff_event", dff_event.shape)

                        # 相関計算
                        corr_matrix, _ = spearmanr(dff_event, axis=1)
                        triu_idx = np.triu_indices_from(corr_matrix, k=1)
                        r_values = corr_matrix[triu_idx]
                        r_values = r_values[~np.isnan(r_values)]
                        r_sorted = np.sort(r_values)
                        cum_dist = np.linspace(0, 1, len(r_sorted))

                        rho_50_array[event_idx, tw_id] = get_rho_at_percentile(r_sorted, cum_dist, target=0.5)
                        rho_75_array[event_idx,tw_id] = get_rho_at_percentile(r_sorted, cum_dist, target=0.75)
                        rho_25_array[event_idx,tw_id] = get_rho_at_percentile(r_sorted, cum_dist, target=0.25)
                        k50_array[event_idx, tw_id] = pcs_to_explain_variance(dff_event, threshold=0.5)
                        k80_array[event_idx, tw_id] = pcs_to_explain_variance(dff_event, threshold=0.8)
                        k90_array[event_idx, tw_id] = pcs_to_explain_variance(dff_event, threshold=0.9)

                        if len(analysis_time_window) < 12:
                            ax0 = fig.add_subplot(gs[event_idx, tw_id])
                            cbar = True if tw_id == len(analysis_time_window)-1 else False
                            colors = [
                                (0.0, 'blue'),  # 最小値の色
                                (0.5, 'black'),  # 中央の色
                                (1.0, 'red')  # 最大値の色
                            ]
                            cmap = LinearSegmentedColormap.from_list("custom_black_center", colors)
                            sns.heatmap(corr_matrix, cmap=cmap, vmin =-v, vmax=v, square=True,
                                        cbar_kws={'label': 'Spearman ρ'}, ax=ax0, xticklabels=False, yticklabels=False, cbar=cbar)
                            # ax0.set_title(f'Corr Matrix: {event_name}')
                            ax0.set_title(event_name)
                            # ax0.set_xlabel('Cells')
                            # ax0.set_ylabel('Cells')

                        if event_idx not in ax1_dict:
                            ax1 = fig.add_subplot(gs[event_idx, -2])
                            ax1_dict[event_idx] = ax1
                        else:
                            ax1 = ax1_dict[event_idx]

                        ax1.plot(r_sorted, cum_dist, color=plt.get_cmap("tab20")(tw_id), lw=0.5, label =str(tw[0])+"-"+str(tw[1])+"min_"+event_name)
                        # ax1.set_xlabel('Correlation coefficient (ρ)')
                        # ax1.set_ylabel('Cumulative distribution')
                        # ax1.set_title(f'CDF: {event_name}')
                        ax1.grid(True)
                for ax1 in ax1_dict.values():
                    ax1.legend(fontsize=6, labelspacing=0.3)
                    ax1.set_xlabel('Correlation coefficient (ρ)')
                    ax1.set_ylabel('Cumulative distribution')
                    ax1.set_xlim([-0.2, 0.4])


                for event_idx in range(event_num):
                    ax2 = fig.add_subplot(gs[event_idx, -1])
                    plot_timeseries(time_array, rho_50_array[event_idx,:], 1, ax2, "blue", 1, None, None, None, label=None, alpha=1)
                    plot_timeseries(time_array, rho_25_array[event_idx, :], 1, ax2, "lightblue", 1, None, None, None, label=None,alpha=1)
                    plot_timeseries(time_array, rho_75_array[event_idx, :], 1, ax2, "k", 1, None, None, (0,0.2), label=None,alpha=1)

                    ax2_t = ax2.twinx()
                    ax2_t.plot(time_array, k50_array[event_idx, :], marker='o', linewidth=1, alpha=0.9)
                    ax2_t.plot(time_array, k80_array[event_idx, :], marker='o', linewidth=1, alpha=0.9)
                    ax2_t.plot(time_array, k90_array[event_idx, :], marker='o', linewidth=1, alpha=0.9)
                    ax2_t.set_ylabel('PCs for XX% variance')

                plt.tight_layout()
                plt.legend(fontsize=1, labelspacing=0.1)
                pdf_path = os.path.join(data_folder, "_GCaMP", "_"+data_pattern+"_correlation_"+name+".pdf")
                with PdfPages(pdf_path) as pdf:
                    pdf.savefig(fig, dpi=300)
                plt.close(fig)


def main():
    data_folder = select_folder()
    # data_folder = r"X:\Behavior\Ca_imaging\20250724_z253-4_IRES-2x_GCaMP-3e12_soma_imaging_EEG"  # for development
    process_folder(data_folder)

if __name__ == "__main__":
    main()