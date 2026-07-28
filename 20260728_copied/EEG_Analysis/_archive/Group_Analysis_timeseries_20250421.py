import pandas as pd
from _archive.EEG_Analysis import plot_timeseries
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

def process_group(json_path, group_dict,state_dict):
    dir = os.path.dirname(os.path.dirname(json_path))
    for group in group_dict:

        fig = plt.figure(figsize=(8, 25))
        gs = gridspec.GridSpec(6, 1, height_ratios=[1, 1, 1, 1, 1, 1])
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        axes = [fig.add_subplot(gs[i, 0]) for i in range(6)]
        ax0, ax1, ax2, ax3, ax4, ax5 = axes

        h5_files = []
        event_files = []
        group_velocity = []
        group_active_velocity = []
        group_active = []
        group_NREM = []
        group_stateC = []

        subgroup_list = group_dict[group]
        for subgroup in subgroup_list:
            h5_pattern = os.path.join(dir, subgroup, "[!_]*", "_Combined","data.h5")
            csv_pattern = os.path.join(dir, subgroup, "[!_]*", "_Combined","manual_event.csv")
            h5_files.extend(glob.glob(h5_pattern))
            event_files.extend(glob.glob(csv_pattern))

        for h, h5_file in enumerate(h5_files):
            print(h5_file)
            OF_tp, velocity, event_df= open_h5(h5_file)
            bin_tp, bin_v = binning(OF_tp, velocity, 1200, -3600, 10800)
            if event_df is None:
                active_velocity=velocity
            else:
                active_velocity = event_mask(OF_tp, velocity, event_df, 5)
            _, bin_v_active = binning(OF_tp, active_velocity, 1200, -3600, 10800)
            bin_tp = bin_tp/60 #sec to min
            # plot_timeseries(bin_tp, bin_v, 1, ax0, "gray", 0.25, "", None,(0, 250), None)
            plot_timeseries(bin_tp, bin_v_active, 1, ax0, "gray", 0.25, "", None, (0, 150), None)
            print(bin_v_active)
            group_velocity.append(bin_v)
            group_active_velocity.append(bin_v_active)

        for e, event_file in enumerate (event_files):
            df = pd.read_csv(event_file, low_memory=False)
            ratio_df = compute_bin_event_ratios(df, bin_size=300, min_time=-3600, max_time=10800)
            plot_timeseries(ratio_df["bin_center"].values/60, ratio_df["Active_%"].values, 1, ax1, "gray", 0.25, None, None, (0,100),None)
            group_active.append(ratio_df["Active_%"])
            group_NREM.append(ratio_df["NREM_%"])
            group_stateC.append(ratio_df["StateC_%"])


        mean_v = np.nanmean(group_velocity, axis=0)
        mean_v_active = np.nanmean(group_active_velocity, axis=0)
        plot_timeseries(bin_tp, mean_v, 1, ax0, "blue", 1, "Velocity", "mm/s", (0, 150), None)
        plot_timeseries(bin_tp, mean_v_active, 1, ax0, "red", 1, "Velocity", "mm/s", (0, 150), None)
        mean_active = np.nanmean(group_active, axis=0)
        mean_NREM = np.nanmean(group_NREM, axis=0)
        mean_stateC = np.nanmean(group_stateC, axis=0)
        plot_timeseries(ratio_df["bin_center"].values/60, mean_active, 1, ax1, "blue", 1, "State", "%", (0, 100), "Active")
        plot_timeseries(ratio_df["bin_center"].values / 60, mean_NREM, 1, ax1, "green", 1, "State", "%",
                        (0, 100), "NREM")
        plot_timeseries(ratio_df["bin_center"].values / 60, mean_stateC, 1, ax1, "red", 1, "State", "%",
                        (0, 100), "StateC")

        for ax in axes:
            ax.set_xlabel("Time (min)")
            ax.margins(x=0)
        plt.tight_layout()
        pdf_path = os.path.join(dir, "_Group_Analysis_timeseries",
                                group + "_timeseries.pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, dpi=300)
        plt.close(fig)





def main():
    # json_path = select_json_path()
    json_path = r"X:\Behavior\Openfield_EEG\_Group_Analysis_timeseries\_group_analysis_param_timeseries_0422.json"
    group_dict, state_dict = extract_group_analysis_params(json_path)
    process_group(json_path, group_dict, state_dict)


if __name__ == "__main__":
    main()