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
from scipy.stats import ranksums
from itertools import combinations


def process_folder(data_folder):
    event_path = os.path.join(data_folder, "_Combined", "event_combined.csv")
    man_event_path = os.path.join(data_folder, "_Combined", "manual_event.csv")
    output_names = ["","_manual"]
    # event_path = os.path.join(data_folder, "_Combined", "manual_event.csv")
    print("##### " + os.path.basename(data_folder) + " ######")


    for idx, event_path in enumerate([event_path, man_event_path]):
        if not os.path.exists(event_path):
            print("event_combined.csv was not found")
        else:
            event_df = pd.read_csv(event_path)
            *_,contime = extract_params(data_folder)
            frame2p_df = pd.read_csv(os.path.join(data_folder, "_Combined", "2p_frame_time_combined.csv"))
            spks = np.load(os.path.join(data_folder, "_GCaMP", "_spks_cell.npy"))
            spks_z = np.load(os.path.join(data_folder, "_GCaMP", "_spks_cell_zscore.npy"))

            # analysis_time_window = [[-30,0], [0,30], [30,60]]
            analysis_time_window = [[-60,60]]


            fig = plt.figure(figsize=(11.69,8.27))
            gs = gridspec.GridSpec(1, len(analysis_time_window))
            plt.subplots_adjust(wspace=0.05, hspace=0.05)

            for tw_id, tw in enumerate(analysis_time_window):
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
                mean_df.to_csv(os.path.join(data_folder, "_GCaMP", "_event_mean_fr_"+str(tw[0])+"-"+str(tw[1])+"min.csv"))
                grouped_df = mean_df.groupby('event_name').mean()
                grouped_df.to_csv(os.path.join(data_folder, "_GCaMP", "_event_type_mean_fr_" + str(tw[0]) + "-" + str(tw[1]) + "min.csv"))

                #plot
                ax = fig.add_subplot(gs[0, tw_id])
                for col in grouped_df.columns:
                    ax.plot(grouped_df.index, grouped_df[col], color='gray', linewidth=1, alpha=0.6)
                mean_values = grouped_df.mean(axis=1)
                ax.bar(grouped_df.index, mean_values, color='orange', alpha=0.8, width=0.4, label=str(tw[0])+"-"+str(tw[1])+"min")
                ax.set_ylim([0,0.7])


            plt.tight_layout()
            plt.legend(fontsize=1, labelspacing=0.1)
            pdf_path = os.path.join(data_folder, "_GCaMP", "_mean_fr"+output_names[idx]+".pdf")
            with PdfPages(pdf_path) as pdf:
                pdf.savefig(fig, dpi=300)
            plt.close(fig)


                # mean_z_df = pd.DataFrame(cell_meansz_per_event)
                # mean_z_df['event_name'] = event_df_tw['event_name'].values
                # mean_z_df.to_csv(os.path.join(data_folder, "_GCaMP", "event_mean_zscore_"+str(tw[0])+"-"+str(tw[1])+"min.csv"))


                # unique_events = mean_df["event_name"].unique()
                # event_pairs = list(combinations(unique_events, 2))
                #
                # # 各細胞ごとにイベントペア間でrank sum testを実行
                # results = []
                # for event_a, event_b in event_pairs:
                #     group_a = mean_z_df[mean_z_df["event_name"] == event_a].drop(columns=["event_name"])
                #     group_b = mean_z_df[mean_z_df["event_name"] == event_b].drop(columns=["event_name"])
                #
                #     for cell_idx in range(spks.shape[0]):
                #         vals_a = group_a[cell_idx].dropna()
                #         vals_b = group_b[cell_idx].dropna()
                #
                #         if len(vals_a) > 0 and len(vals_b) > 0:
                #             stat, pval = ranksums(vals_a, vals_b)
                #             diff = vals_b.mean() - vals_a.mean()
                #             if pval < 0.05:
                #                 regulation = "upregulated" if diff > 0 else "downregulated"
                #             else:
                #                 regulation = "no_change"
                #         else:
                #             pval = np.nan
                #             regulation = "insufficient_data"
                #
                #         results.append({
                #             "cell": cell_idx,
                #             "event_a": event_a,
                #             "event_b": event_b,
                #             "mean_a": vals_a.mean() if len(vals_a) > 0 else np.nan,
                #             "mean_b": vals_b.mean() if len(vals_b) > 0 else np.nan,
                #             "pval": pval,
                #             "regulation": regulation
                #         })
                # # 結果を保存
                # results_df = pd.DataFrame(results)
                # results_df.to_csv(os.path.join(data_folder, "_GCaMP",str(tw[0])+"-"+str(tw[1])+"min_eventwise_ranksum.csv"), index=False)


def main():
    data_folder = select_folder()
    # data_folder = r"X:\Behavior\Ca_imaging\20250718_z254-1_IRES-2x_GCaMP-5e12_soma_imaging_EEG"  # for development
    process_folder(data_folder)

if __name__ == "__main__":
    main()