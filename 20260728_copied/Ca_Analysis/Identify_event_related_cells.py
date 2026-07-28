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
from EEG_Ca_treadmill_analysis import extract_params, select_folder
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import binary_dilation
from scipy.stats import ranksums, spearmanr
from itertools import combinations


def process_folder(data_folder):
    event_path = os.path.join(data_folder, "_Combined", "event_combined.csv")
    print("##### " + os.path.basename(data_folder) + " ######")

    if not os.path.exists(event_path):
        print("event_combined.csv was not found")
    else:
        event_df = pd.read_csv(event_path)
        *_,contime = extract_params(data_folder)
        # 各　何frameかを抽出
        # frame_counts = []
        # tiff_list = sorted(glob.glob(os.path.join(data_folder, "_GCaMP", "*.tif")))
        # for tiff_path in tiff_list:
        #     with tifffile.TiffFile(tiff_path) as tif:
        #         frame_counts.append(len(tif.pages))
        # frame_cumsum = np.cumsum([0] + frame_counts[:-1])
        #
        # frame2p_csv_list = []
        # for exp in glob.glob(os.path.join(data_folder, "[!_]*")):
        #     frame2p_csv_list.append(os.path.join(exp, "results", "2p_frame_time.csv"))
        # frame_df_list = [pd.read_csv(csv_path) for csv_path in frame2p_csv_list]
        #
        # adjusted_frame_df_list = []
        # for i, df in enumerate(frame_df_list):
        #     adjusted_df = df.copy()
        #     adjusted_df["frame"] += frame_cumsum[i]  # 累積オフセットを加算
        #     adjusted_frame_df_list.append(adjusted_df)
        # frame_df = pd.concat(adjusted_frame_df_list, ignore_index=True)
        # frame_df.to_csv(os.path.join(data_folder, "_Combined", "2p_frame_time_combined.csv"))
        frame2p_df = pd.read_csv(os.path.join(data_folder, "_Combined", "2p_frame_time_combined.csv"))
        # ops = np.load(os.path.join(data_folder, "_GCaMP", "suite2p", "plane0", "ops.npy"), allow_pickle=True).item()
        # print(ops['frames_per_file'])
        # spks_all = np.load(os.path.join(data_folder, "_GCaMP", "suite2p", "plane0", "spks.npy"))
        # iscell = np.load(os.path.join(data_folder, "_GCaMP", "suite2p", "plane0", "iscell.npy"))
        # cell_indices = np.where(iscell[:, 0] == 1)[0]
        # spks = spks_all[cell_indices]
        spks = np.load(os.path.join(data_folder, "_GCaMP", "_spks_cell.npy"))
        spks_z = np.load(os.path.join(data_folder, "_GCaMP", "_spks_cell_zscore.npy"))
        # spks_df = pd.DataFrame(spks)
        # spks_df.to_csv (os.path.join(data_folder, "_GCaMP", "_spks.csv"))

        analysis_time_window = [[-30,0], [0,30], [30,60]]
        for tw in analysis_time_window:
            event_df_tw =  event_df[
                                (event_df["start_time"] >= tw[0]*60) & (event_df["end_time"] <= tw[1]*60)
                            ]

            event_frames = []
            for _, row in event_df_tw.iterrows():
                frames = frame2p_df[(frame2p_df['time'] >= row['start_time']) & (frame2p_df['time'] <= row['end_time'])][
                    'frame'].values
                event_frames.append(frames)

            # 各イベントに対して、各細胞の平均発火頻度を計算
            cell_means_per_event = []
            cell_meansz_per_event = []
            for frames in event_frames:
                if len(frames) > 0:
                    means = spks[:, frames].mean(axis=1)
                    means_z = spks_z[:, frames].mean(axis=1)
                else:
                    means = np.full(spks.shape[0], np.nan)
                    means_z = np.full(spks_z.shape[0], np.nan)
                # print(means.shape)
                cell_means_per_event.append(means)
                cell_meansz_per_event.append(means_z)

            # データフレームに変換
            mean_df = pd.DataFrame(cell_means_per_event)
            mean_df['event_name'] = event_df_tw['event_name'].values
            mean_df.to_csv(os.path.join(data_folder, "_GCaMP", "_mean_fr.csv"))
            mean_z_df = pd.DataFrame(cell_meansz_per_event)
            mean_z_df['event_name'] = event_df_tw['event_name'].values
            mean_z_df.to_csv(os.path.join(data_folder, "_GCaMP", "_mean_zscore.csv"))


            unique_events = mean_df["event_name"].unique()
            event_pairs = list(combinations(unique_events, 2))

            # 各細胞ごとにイベントペア間でrank sum testを実行
            results = []
            for event_a, event_b in event_pairs:
                group_a = mean_z_df[mean_z_df["event_name"] == event_a].drop(columns=["event_name"])
                group_b = mean_z_df[mean_z_df["event_name"] == event_b].drop(columns=["event_name"])

                for cell_idx in range(spks.shape[0]):
                    vals_a = group_a[cell_idx].dropna()
                    vals_b = group_b[cell_idx].dropna()

                    if len(vals_a) > 0 and len(vals_b) > 0:
                        stat, pval = ranksums(vals_a, vals_b)
                        diff = vals_b.mean() - vals_a.mean()
                        regulation = (
                            "upregulated" if pval < 0.05 and diff > 0
                            else "downregulated" if pval < 0.05 and diff < 0
                            else "no_change"
                        )

                        # correlation between binary condition (0: event_a, 1: event_b) and activity
                        combined_vals = pd.concat([vals_a, vals_b], ignore_index=True)
                        condition_vector = np.array([0] * len(vals_a) + [1] * len(vals_b))
                        if len(combined_vals) > 1:
                            corr, pval_corr = spearmanr(condition_vector, combined_vals)
                        else:
                            corr, pval_corr = np.nan, np.nan

                    else:
                        pval = np.nan
                        regulation = "insufficient_data"
                        corr, pval_corr = np.nan, np.nan

                    results.append({
                        "cell": cell_idx,
                        "event_a": event_a,
                        "event_b": event_b,
                        "mean_a": vals_a.mean() if len(vals_a) > 0 else np.nan,
                        "mean_b": vals_b.mean() if len(vals_b) > 0 else np.nan,
                        "pval": pval,
                        "regulation": regulation,
                        "correlation": corr,
                        "pval_corr": pval_corr
                    })
            # 結果を保存
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(data_folder, "_GCaMP",str(tw[0])+"-"+str(tw[1])+"min_eventwise_ranksum.csv"), index=False)


def main():
    data_folder = select_folder()
    # data_folder = r"X:\Behavior\Ca_imaging\20250707_z251-2_SynC-GCaMP"  # for development
    process_folder(data_folder)

if __name__ == "__main__":
    main()