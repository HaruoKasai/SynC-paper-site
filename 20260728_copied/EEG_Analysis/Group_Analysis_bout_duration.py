import pandas as pd
import os
import glob
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import json
import h5py

def extract_group_analysis_params(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        group_dict = data["Group"]
        state_dict = data["state"]
    return group_dict, state_dict

def select_json_path():
    root = tk.Tk()
    root.withdraw()  # ウィンドウを表示しない
    file_path = filedialog.askopenfilename(title="Select a group_analysis_param json file", initialdir=r"X:\Behavior")
    root.destroy()
    return file_path

def load_dataset(name, file):
    if name not in file:
        return None  # データセットが存在しない場合
    data = file[name][:]  # データ取得
    return data if not np.isnan(data).all() else None

def open_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        OF_tp = load_dataset("all_OF_tp", f)
        velocity = load_dataset("all_v", f)
    try:
        event_df = pd.read_hdf(h5_path, key="event_df")
    except (KeyError, FileNotFoundError):
        event_df = None

    return OF_tp, velocity, event_df


def binning(timepoint, data, bin_size, min_time, max_time):
    # bins = np.arange(min_time, max_time + bin_size, bin_size)
    # bin_means, _, _ = binned_statistic(timepoint, data, statistic='mean', bins=bins)
    # bin_centers = (bins[:-1] + bins[1:]) / 2

    valid = ~np.isnan(data)
    timepoint = timepoint[valid]
    data = data[valid]

    # ビンの定義
    bins = np.arange(min_time, max_time + bin_size, bin_size)
    bin_indices = np.digitize(timepoint, bins) - 1  # 0-based index

    # 無効なインデックスを除去（ビン外）
    in_range = (bin_indices >= 0) & (bin_indices < len(bins) - 1)
    bin_indices = bin_indices[in_range]
    data = data[in_range]

    # 合計とカウントを計算 → 平均
    sums = np.bincount(bin_indices, weights=data, minlength=len(bins) - 1)
    counts = np.bincount(bin_indices, minlength=len(bins) - 1)
    means = np.full(len(bins) - 1, np.nan)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero]

    bin_centers = (bins[:-1] + bins[1:]) / 2


    return bin_centers, means

def compute_bin_event_ratios(df, bin_size=60, min_time=-3600, max_time=10800):
    # イベントの開始・終了・ラベルを抽出
    starts = df['start_time'].values
    ends = df['end_time'].values
    labels = df['event_name'].values

    # ビン定義
    bin_edges = np.arange(min_time, max_time + bin_size, bin_size)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_edges) - 1

    # 各カテゴリの時間を記録
    statec_time = np.zeros(n_bins)
    NREM_time = np.zeros(n_bins)

    # ベクトル化処理：各イベントをすべてのビンと比較して重なり計算
    for start, end, label in zip(starts, ends, labels):
        overlap_start = np.maximum(start, bin_edges[:-1])
        overlap_end = np.minimum(end, bin_edges[1:])
        durations = np.clip(overlap_end - overlap_start, 0, bin_size)

        if label == 'StateC':
            statec_time += durations
        elif label == 'NREM':
            NREM_time += durations

    total_time = bin_size
    known_time = statec_time + NREM_time
    active_time = np.clip(total_time - known_time, 0, bin_size)

    # パーセンテージに変換
    result_df = pd.DataFrame({
        'bin_start': bin_edges[:-1],
        'bin_end': bin_edges[1:],
        'bin_center': bin_centers,
        'StateC_%': statec_time / total_time * 100,
        'NREM_%': NREM_time / total_time * 100,
        'Active_%': active_time / total_time * 100,
    })

    print(result_df)

    return result_df

def event_mask(tp_array, data_array, event_df, extra_sec):
    for _, row in event_df.iterrows():
        start = row['start_time'] - extra_sec
        end = row['end_time'] + extra_sec
        mask = (tp_array >= start) & (tp_array <= end)
        data_array[mask] = np.nan

    return data_array

def process_group(json_path, group_dict, state_dict):
    dir = os.path.dirname(os.path.dirname(json_path))
    json_dir = os.path.dirname(json_path)
    g_num = len(group_dict)
    print(g_num)
    fig = plt.figure(figsize=(10, g_num*3))
    gs = gridspec.GridSpec(g_num, 3) #StateC/SWS/motive #height_ratios=[1, 1, 1, 1, 1, 1]
    plt.subplots_adjust(wspace=0.05, hspace=0.05)


    for g, (group, exp_list) in enumerate(group_dict.items()):
        axes = [fig.add_subplot(gs[g, i]) for i in range(3)]
        ax0, ax1, ax2 = axes

        event_files = []
        SWS_durations = []
        motive_durations = []
        StateC_durations = []
        for folder in exp_list:
            csv_pattern = os.path.join(dir, folder, "_Combined","manual_event.csv")
            event_files.extend(glob.glob(csv_pattern))
        for e, event_file in enumerate (event_files):
            df = pd.read_csv(event_file, low_memory=False)
            df = df[df['start_time'] > 0] # OK
            if len(df) >1:
                if group=="Control":
                    df_s = df
                else:
                    df_s = df[df['event_name'] == 'StateC']
                    #interimC = stateCとstateCの間
                intervals = df_s['start_time'].iloc[1:].values - df_s['end_time'].iloc[:-1].values
                motive_durations.extend(intervals)

            s_df = df[df['event_name'] == 'NREM'].sort_values(by='start_time').reset_index(drop=True)
            c_df = df[df['event_name'] == 'StateC'].sort_values(by='start_time').reset_index(drop=True)
            s_durations = s_df['end_time'] - s_df['start_time']
            SWS_durations.extend(s_durations)
            c_durations = c_df['end_time'] - c_df['start_time']
            StateC_durations.extend(c_durations)

        # bins = np.arange(0, 1000 + 100, 50)
        bins = np.arange(0, 1000 + 100, 75)

        state_list = ["StateC","SWS", "Motive"]


        c_counts, bin_edges = np.histogram(StateC_durations, bins=bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = np.diff(bin_edges)
        c_total = c_counts.sum()
        c_probabilities = c_counts / c_total
        ax0.bar(bin_edges[:-1], c_counts, width=bin_width, align='edge', edgecolor='black', color="lightblue", bottom =5e-1) #,bottom=1e-3
        # ax0.bar(bin_edges[:-1], c_probabilities, width=bin_width, align='edge', edgecolor='black', color="lightblue", bottom=1e-3)
        # # ax0.plot(bin_centers, c_probabilities, marker='.', color='lightblue', linewidth=2)
        ax0.set_yscale('log')
        ax0.set_xlim(0, 600)
        ax0.set_ylim(5e-1,1e3)
        # ax0.set_ylim(1e-3, 1)

        s_counts, bin_edges = np.histogram(SWS_durations, bins=bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = np.diff(bin_edges)
        s_total = s_counts.sum()
        s_probabilities = s_counts / s_total
        ax1.bar(bin_edges[:-1], s_counts, width=bin_width, align='edge', edgecolor='black', color="blue",bottom =5e-1)
        # ax1.bar(bin_edges[:-1], s_probabilities, width=bin_width, align='edge', edgecolor='black', color="blue", bottom=1e-3 )
        # ax1.plot(bin_centers, s_probabilities, marker='.', color='blue', linewidth=2)
        ax1.set_yscale('log')
        ax1.set_xlim(0, 600)
        ax1.set_ylim(5e-1,1e3)
        # ax1.set_ylim(1e-3, 1)

        bins = np.arange(0, 1000 + 80, 80)

        m_counts, bin_edges = np.histogram(motive_durations, bins=bins, density=False)
        bin_width = np.diff(bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        m_total = m_counts.sum()
        m_probabilities = m_counts / m_total
        ax2.bar(bin_edges[:-1], m_counts, width=bin_width, align='edge', edgecolor='black', color="orange", bottom =5e-1)
        # ax2.bar(bin_edges[:-1], m_probabilities, width=bin_width, align='edge', edgecolor='black', color="orange", bottom=1e-3)
        # ax2.plot(bin_centers, m_probabilities, marker='.', color='orange', linewidth=2)
        ax2.set_yscale('log')
        ax2.set_xlim(0, 1000)
        ax2.set_ylim(5e-1, 1e3)
        # ax2.set_ylim(1e-3, 1)

        # ax0.hist(StateC_durations, bins=bins, color='skyblue', edgecolor='black',density=True)
        #
        # ax1.hist(SWS_durations, bins=bins, color='blue', edgecolor='black', density=True)
        # ax1.set_xlim(0, 600)
        # ax2.hist(motive_durations, bins=bins, color='white', edgecolor='black', density=True)
        # ax2.set_xlim(0, 4000)
        for a, ax in enumerate([ax0,ax1, ax2]):
            ax.set_xlabel('Duration (s)')
            ax.set_ylabel('Probability')
            ax.set_title(state_list[a])
            ax.margins(x=0)


    plt.tight_layout()
    pdf_path = os.path.join(json_dir,
                            "_bout_duration_histogram.pdf")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)





def main():
    # json_path = select_json_path()
    json_path = r"X:\Behavior\Openfield_EEG\_Group_bout_duration\_group_analysis_param_bout_duration.json"
    group_dict, state_dict = extract_group_analysis_params(json_path)
    process_group(json_path, group_dict, state_dict)


if __name__ == "__main__":
    main()