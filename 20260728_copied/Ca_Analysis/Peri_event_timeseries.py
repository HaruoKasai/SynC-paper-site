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
import matplotlib.pyplot as plt
plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42  # TrueTypeフォントで保存
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


def extract_transition_timing(event_df, event_a, event_b):
    # df = event_df.sort_values('start_time').reset_index(drop=True)
    df = event_df
    # a と b 以外を除外
    df = df[df['event_name'].isin([event_a, event_b])].reset_index(drop=True)

    current_event = df['event_name']
    next_event = df['event_name'].shift(-1)
    next_start = df['start_time'].shift(-1)

    # 条件をベクトルで評価
    a_to_b_mask = (current_event == event_a) & (next_event == event_b)
    b_to_a_mask = (current_event == event_b) & (next_event == event_a)

    # 結果の抽出
    a_to_b_times = next_start[a_to_b_mask].dropna().tolist()
    b_to_a_times = next_start[b_to_a_mask].dropna().tolist()

    return a_to_b_times, b_to_a_times


def process_folder(data_folder):
    spks_z = np.load(os.path.join(data_folder, "_GCaMP", "_spks_cell_zscore.npy"))
    frame2p_df = pd.read_csv(os.path.join(data_folder, "_Combined", "2p_frame_time_combined.csv"))
    event_df = pd.read_csv(os.path.join(data_folder, "_Combined", "event_combined.csv"))


    #before活動をもとに、細胞を分類
    before_csv = glob.glob(os.path.join(data_folder, "_GCaMP", "-*0min*.csv"))[0]
    df_before = pd.read_csv(before_csv)

    event_pairs = df_before[['event_a', 'event_b']].drop_duplicates()
    reg_list = sorted (df_before['regulation'].drop_duplicates().to_list() )
    print(reg_list)

    analysis_time_window = [[-30, 0], [0, 30], [30, 60]]
    for _, row in event_pairs.iterrows():
        a, b = row['event_a'], row['event_b']

        fig = plt.figure(figsize=(8.27,11.69))
        gs = gridspec.GridSpec(len(reg_list)*2, len(analysis_time_window)) # height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        for r, regulation in enumerate(reg_list):
            print(a, b, regulation)
            # 条件に一致する行を抽出
            sub_df = df_before[
                (df_before['event_a'] == a) &
                (df_before['event_b'] == b) &
                (df_before['regulation'] == regulation)
                ]

            # セル番号をリストで取得して保存
            # cell_list = {}
            # cell_list[(a, b, regulation)] = sub_df['cell'].tolist()
            cell_list = sub_df['cell'].tolist()
            a_to_b_times_all, b_to_a_times_all = extract_transition_timing(event_df, a, b)
            # print("a to b", a_to_b_times_all)
            # print("b to a", b_to_a_times_all)
            for idx, time_list_all in enumerate([a_to_b_times_all, b_to_a_times_all]):
                print("idx", idx)
                # print(time_list_all)
                for tw_id, tw in enumerate(analysis_time_window):
                    # print(time_list_all)
                    time_list = [x for x in time_list_all if tw[0] * 60 <= x <= tw[1] * 60]
                    # print(time_list)
                    if len(time_list) > 0:
                        t_pre, t_post = -10, 10
                        bin_width = 1
                        n_bins = int((t_post - t_pre) / bin_width)

                        # Binの中心を0にするための順序付きbin index
                        # 例: [9, 8, ..., 0, 1, 2, ..., 19]  ← 中央から左右へ
                        bin_centers = np.linspace(t_pre + bin_width / 2, t_post - bin_width / 2,
                                                  n_bins)  # e.g. [-9.5, -8.5, ..., 9.5]
                        bin_order = np.arange(n_bins)

                        data = np.zeros([len(cell_list), len(time_list), n_bins])
                        for t,time in enumerate(time_list):
                            frames = frame2p_df[(frame2p_df['time'] >= time+t_pre) & (frame2p_df['time'] <= time+t_post)]['frame'].values
                            n_frames = len(frames)
                            print(n_frames)

                            frame_rate = n_frames / (t_post - t_pre)  # ≒ 33 Hz
                            bin_size_frames = int(bin_width * frame_rate)
                            start_indices = np.arange(n_bins) * bin_size_frames
                            print(start_indices)

                            binned_chunks = np.stack([
                                spks_z[np.ix_(cell_list, frames[start:start + bin_size_frames])]
                                for start in start_indices
                            ], axis=0)
                            print(binned_chunks.shape)
                            binned_mean = np.mean(binned_chunks, axis=2)
                            data[:, t, :] = binned_mean.T

                        # print("data")
                        # print(data)
                        average = np.mean(data, axis=1)
                        print(average)
                        ax = fig.add_subplot(gs[r*2+idx, tw_id])
                        cmap = plt.get_cmap("nipy_spectral", 100)  # or "hsv", "turbo", etc.
                        colors = [cmap(i) for i in range(100)]
                        for cell in range(len(average)):
                            color = colors[cell % 100]
                            plot_timeseries(bin_centers, average[cell], 1, ax, color, 0.2, None, None, (-0.2,1), label=None, alpha=0.3)
                            ax.margins(x=0)
                        inter_cell_average = np.mean(average, axis=0)
                        plot_timeseries(bin_centers, inter_cell_average, 1, ax, "k", 1, None, None, (-0.2, 1), label=None,alpha=1)
                        # print("average shape", average.shape)
                        # print(average)
                    if len(time_list) == 0:
                        print(f"  [skipped] {a}->{b} {regulation} tw_id={tw_id} (no events)")
                        continue

        plt.tight_layout()
        plt.legend(fontsize=1, labelspacing=0.1)
        pdf_path = os.path.join(data_folder,"_GCaMP", a+"-vs-"+b+".pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, dpi=300)
        plt.close(fig)


def main():
    data_folder = select_folder()
    # data_folder = r"X:\Behavior\Ca_imaging\20250718_z254-1_IRES-2x_GCaMP-5e12_soma_imaging_EEG"  # for development
    process_folder(data_folder)

if __name__ == "__main__":
    main()