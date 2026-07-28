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

import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42  # TrueTypeフォントで保存
# plt.rcParams['pdf.fonttype'] = 42
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import json
import seaborn as sns
from _archive import EEG_Analysis as EA
import lib.DLCAnalysis as DA
from Group_Analysis_timeseries import event_mask


def extract_group_analysis_params(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        group_dict = data["Group"]
        electrode_dict = data["Electrode"]
        state_dict = data["state"]
    return group_dict, electrode_dict, state_dict

def select_json_path():
    root = tk.Tk()
    root.withdraw()  # ウィンドウを表示しない
    file_path = filedialog.askopenfilename(title="Select a group_analysis_param json file", initialdir=r"X:\Behavior")
    root.destroy()
    return file_path

def calc_center(arena, ratio):
    arena = np.array(arena)
    center_x = (arena[0, 0] + arena[1, 0]) / 2
    center_y = (arena[0, 1] + arena[1, 1]) / 2

    # 元の幅と高さ
    width = arena[1, 0] - arena[0, 0]
    height = arena[1, 1] - arena[0, 1]

    # 新しい幅と高さ（40% に縮小）
    new_width = width * ratio
    new_height = height * ratio

    # 新しい四角形の頂点を計算
    center = np.array([
        [center_x - new_width / 2, center_y - new_height / 2],  # 左上
        [center_x + new_width / 2, center_y + new_height / 2]   # 右下
        ])

    return center

def calculate_center_time(dlc_dir,contime, data_list, event_df, time_blocks):
    in_center_array_list = []
    time_array_list = []
    for c in range(len(contime)):
        dlc_exp_dir = glob.glob(os.path.join(dlc_dir, "day*"))[c]
        dlc_h5_path = os.path.join(dlc_exp_dir, "dlc_raw.h5")
        param_ind = os.path.join(dlc_exp_dir, "param_individual.json")
        df = pd.read_hdf(dlc_h5_path, key='dlc_data')

        # `df["time"]` を 秒単位 の float に変換
        start_time = contime[c][0] * 60  # 分 → 秒に変換
        df["time"] = start_time + (df["time"] - df["time"].iloc[0]).dt.total_seconds()

        # `end_time` も秒単位の `float`
        end_time = contime[c][1] * 60
        df_filtered = df[df["time"] <= end_time]

        # `arena` と `center` を計算
        arena = DA.get_roi_coordinate("arena_box", param_ind=param_ind)
        center = calc_center(arena, 0.5)

        # `centroid_x`, `centroid_y` を取得
        centroid_x = df_filtered["centroid"]["x"].values
        centroid_y = df_filtered["centroid"]["y"].values

        # `center` の範囲内か判定し、0/1 の配列として保存
        in_center = ((center[0, 0] <= centroid_x) & (centroid_x <= center[1, 0]) &
                     (center[0, 1] <= centroid_y) & (centroid_y <= center[1, 1])).astype(int)

        # `list.extend()` でリストに追加
        in_center_array_list.extend(in_center.tolist())
        time_array_list.extend(df_filtered["time"].tolist())

    # `time_array_list` を NumPy 配列 (`float` の秒単位) に変換
    time_array = np.array(time_array_list, dtype=np.float64)

    # `in_center_array_list` を NumPy 配列 (`int32`) に変換
    in_center_array = np.array(in_center_array_list, dtype=np.float64)

    if event_df is not None:
        in_center_array = event_mask(time_array, in_center_array, event_df, 0)

    # `time_blocks` に基づき `center_time_percent` を計算
    for t,timeblock in enumerate(time_blocks):
        # `time_blocks` も秒単位の float に変換
        timeblock = np.array([timeblock[0] * 60, timeblock[1] * 60], dtype=np.float64)

        # `time_array` の範囲内のインデックスを取得
        valid_indices = np.where((timeblock[0] <= time_array) & (time_array <= timeblock[1]))[0]
        filtered_in_center = in_center_array[valid_indices]

        # `center_time_percent` を計算
        if len(filtered_in_center) > 0:
            # center_time_percent = np.sum(filtered_in_center) / len(filtered_in_center) * 100
            center_time_percent = np.nansum(filtered_in_center) / np.sum(~np.isnan(filtered_in_center)) * 100
        else:
            center_time_percent = 0
        print(center_time_percent)
        data_list[t].append(center_time_percent)

    return data_list


def process_group(json_path, group_dict, electrode_dict, state_dict):
    dir = os.path.dirname(os.path.dirname(json_path))
    for g, (group, exp_list) in enumerate(group_dict.items()):
        data_list = [[] for _ in range(2)] # before vs after
        for exp_name in exp_list:
            exp = os.path.join(dir, exp_name)
            print("$$$$$$$$$$$$$$")
            print(exp)
            if os.path.exists(exp):
                _, dlc_dir, *_, contime = EA.extract_params(exp)
                event_df = pd.read_csv(os.path.join(exp, "_Combined", "manual_event.csv"))
                if dlc_dir is not None:
                    time_blocks = [[-40, 0], [30, 90]]
                    data_list = calculate_center_time(dlc_dir, contime, data_list, event_df, time_blocks)
                else:
                    print("No dlc data")
            else:
                print("Something's wrong in Exp name")
        df = pd.DataFrame({"Before": data_list[0], "After": data_list[1]})
        #jsonのマウス順と、csvのマウス順は対応していないので注意！！
        df.to_csv(os.path.join(os.path.dirname(json_path), "Center_time_"+group + ".csv"))
        df_melted = df.melt(var_name="Group", value_name="Time in center (%)")

        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        plt.subplots_adjust(wspace=0.05, hspace=0.05)


        # sns.violinplot(x="Group", y="Value", data=df_melted, inner="quartile", alpha=0.5)
        # sns.boxplot(data=df_melted, x="Group", y="Value", showfliers=False)
        sns.barplot(x="Group", y="Time in center (%)", data=df_melted, estimator=np.mean, errorbar=('ci', 68), edgecolor='black',alpha=1, facecolor='none')
        # sns.stripplot(x="Group", y="Value", data=df_melted, jitter=True, color="black", alpha=0.7)
        for i in range(len(data_list[0])):
            plt.plot(["Before", "After"], [data_list[0][i], data_list[1][i]], color="gray", #linestyle="--",
                     alpha=0.5)
        plt.ylim(0,60)

        plt.tight_layout()
        pdf_path = os.path.join(dir, "_Group_Analysis_Behavior", "Center_time_"+
                                group + ".pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, dpi=300)
        plt.close(fig)



def main():
    # json_path = select_json_path()
    json_path = r"X:\Behavior\Openfield_EEG\_Group_Analysis_Behavior\_group_analysis_velocity_centertime.json"
    group_dict, electrode_dict, state_dict= extract_group_analysis_params(json_path)
    process_group(json_path, group_dict, electrode_dict, state_dict)


if __name__ == "__main__":
    main()