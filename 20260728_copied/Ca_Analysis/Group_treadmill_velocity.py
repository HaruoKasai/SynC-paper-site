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
import tkinter as tk
from tkinter import filedialog
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42  # TrueTypeフォントで保存

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select the 'data' directory", initialdir=r"X:\Behavior\Ca_imaging")
    root.destroy()
    return folder_path

def load_dataset(name, file):
    if name not in file:
        return None  # データセットが存在しない場合
    data = file[name][:]  # データ取得
    return data if not np.isnan(data).all() else None

def open_h5 (h5_path):
    with h5py.File(h5_path, "r") as f:
        table_v = load_dataset("all_table_v", f)
        table_tp = load_dataset("all_table_tp", f)

    return table_tp, table_v

def process_folder(data_folder, tw):

    # event_path = os.path.join(data_folder, "_Combined", "event_combined.csv")
    event_path = os.path.join(data_folder, "_Combined", "manual_event.csv")
    print("##### " + os.path.basename(data_folder) + " ######")
    tp, v = open_h5(os.path.join(data_folder, "_Combined", "data.h5"))
    if not os.path.exists(event_path):
        print("event_combined.csv was not found")
        return pd.DataFrame([])
    else:
        event_df = pd.read_csv(event_path)
        *_,contime = extract_params(data_folder)
        # frame2p_df = pd.read_csv(os.path.join(data_folder, "_Combined", "2p_frame_time_combined.csv"))
        # spks = np.load(os.path.join(data_folder, "_GCaMP", "_spks_cell.npy"))
        # spks_z = np.load(os.path.join(data_folder, "_GCaMP", "_spks_cell_zscore.npy"))

        event_df_tw =  event_df[
                            (event_df["start_time"] >= tw[0]*60) & (event_df["end_time"] <= tw[1]*60)
                        ]

        starts = event_df_tw["start_time"].values
        ends = event_df_tw["end_time"].values
        mean_vs = np.array([np.mean(v[(tp >= s) & (tp <= e)]) for s, e in zip(starts, ends)])

        event_df_tw["velocity"] = mean_vs

        ##StateC以外の時間平均を出す
        statec_rows = event_df_tw[event_df_tw["event_name"].str.contains("StateC", case=False, na=False)]
        exclude_intervals = list(zip(statec_rows["start_time"], statec_rows["end_time"]))
        mask = np.ones_like(tp, dtype=bool)
        for start, end in exclude_intervals:
            mask &= ~((tp >= start) & (tp <= end))
        mask &= (tp >= tw[0]*60) & (tp <= tw[1]*60)
        mean_v = np.mean(v[mask])
        new_row = {
            "start_time": tw[0]*60,
            "end_time": tw[1]*60,
            "event_name": "Awake_all",
            "velocity": mean_v
        }
        event_df_tw = pd.concat([event_df_tw, pd.DataFrame([new_row])], ignore_index=True)

        event_df_tw["tw"] = str(tw[0])+"-"+str(tw[1])+"min"
        event_df_tw["mouse"] = os.path.basename(data_folder)[:15]
        #
        #
        # # データフレームに変換
        # mean_df = pd.DataFrame(cell_means_per_event)
        # mean_df['event_name'] = event_df_tw['event_name'].values
        # mean_df.to_csv(os.path.join(data_folder, "_GCaMP",
        #                             "_event_mean_fr_" + str(tw[0]) + "-" + str(tw[1]) + "min.csv"))
        # grouped_df = mean_df.groupby('event_name').mean()
        # grouped_df.to_csv(
        #     os.path.join(data_folder, "_GCaMP", "_event_type_mean_fr_" + str(tw[0]) + "-" + str(tw[1]) + "min.csv"))
        print(event_df_tw)
        return event_df_tw


def process_group (path):
    # analysis_time_window = [[-40,-10],[10,40],[40,70]]
    # analysis_time_window = [[-30, 0], [0, 30]]
    analysis_time_window = [[-20, 0], [0, 20]]
    mouse_list = glob.glob(os.path.join(path, "202*"))

    for tw_id, tw in enumerate(analysis_time_window):
        fig = plt.figure(figsize=(11.69, 8.27))
        gs = gridspec.GridSpec(1, 1)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        event_names = [ "Before_mobile", "Before_immobile", "After_mobile", "After_immobile", "After_StateC", "Awake_all"]
        dfs = []
        for mouse in mouse_list:
            df_mouse = process_folder (mouse, tw)
            dfs.append(df_mouse)
            # df = pd.concat([df, df_mouse], axis=1)
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv (os.path.join(path, "_group_analysis", "_velocity_"+str(tw[0])+"-"+str(tw[1])+ "min.csv"))

        means = []
        sems = []
        for name in event_names:
            # 該当するイベント名を含む行を抽出
            subset = df[df["event_name"].str.contains(name, case=False, na=False)]
            v = subset["velocity"].values
            means.append(np.mean(v))
            sems.append(np.std(v, ddof=1) / np.sqrt(len(v)))  # SEM = SD / sqrt(n)

        # plot
        ax = fig.add_subplot(gs[0, 0])
        x = np.arange(len(event_names))
        ax.bar(x, means, color='none',edgecolor="black", alpha=0.8, width=0.4,
                yerr = sems,  # ここでエラーバーを追加
                capsize = 2, ecolor = 'black'#, error_kw = dict(alpha=0.7, lw=0.8)
                )

        for i, name in enumerate(event_names):
            subset = df[df["event_name"].str.contains(name, case=False, na=False)]
            v = subset["velocity"].values
            # 少し横にずらして重ならないように
            ax.scatter(
                np.full_like(v, x[i]) + np.random.uniform(-0.1, 0.1, size=len(v)),
                v,
                color="black", s=8, alpha=0.7
            )

        ax.set_xticks(x)
        ax.set_xticklabels(event_names, rotation=30, ha="right")
        # ax.set_ylim([0, 1.2])

        plt.tight_layout()
        plt.legend(fontsize=1, labelspacing=0.1)
        pdf_path = os.path.join(path, "_group_analysis", "_velocity_"+str(tw[0])+"-"+str(tw[1])+ "min.pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, dpi=300)
        plt.close(fig)


def main():
    path = select_folder()
    process_group(path)

if __name__ == "__main__":
    main()