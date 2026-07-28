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

def process_folder(data_folder, tw):
    # event_path = os.path.join(data_folder, "_Combined", "event_combined.csv")
    event_path = os.path.join(data_folder, "_Combined", "manual_event.csv")
    print("##### " + os.path.basename(data_folder) + " ######")

    if not os.path.exists(event_path):
        print("event_combined.csv was not found")
        return pd.DataFrame([])
    else:
        event_df = pd.read_csv(event_path)
        *_,contime = extract_params(data_folder)
        frame2p_df = pd.read_csv(os.path.join(data_folder, "_Combined", "2p_frame_time_combined.csv"))
        spks = np.load(os.path.join(data_folder, "_GCaMP", "_spks_cell.npy"))
        # spks_z = np.load(os.path.join(data_folder, "_GCaMP", "_spks_cell_zscore.npy"))

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
        for frames in event_frames:
            if len(frames) > 0:
                means = spks[:, frames].mean(axis=1)
            else:
                means = np.full(spks.shape[0], np.nan)
            # print(means.shape)
            cell_means_per_event.append(means)

        # データフレームに変換
        mean_df = pd.DataFrame(cell_means_per_event)
        mean_df['event_name'] = event_df_tw['event_name'].values
        mean_df.to_csv(os.path.join(data_folder, "_GCaMP",
                                    "_event_mean_fr_" + str(tw[0]) + "-" + str(tw[1]) + "min.csv"))
        grouped_df = mean_df.groupby('event_name').mean()
        grouped_df.to_csv(
            os.path.join(data_folder, "_GCaMP", "_event_type_mean_fr_" + str(tw[0]) + "-" + str(tw[1]) + "min.csv"))

        return grouped_df




def process_group (path):
    # analysis_time_window = [[-30,0], [0,30], [30,60]]
    # analysis_time_window = [[-20, 40], [-30,30], [-40, 40], [-60,60], [-30,60],[-60,30],[-20,0], [0,20], [20,40]]
    # analysis_time_window = [[-40,-10],[-40,-5],[5,40],[5,45],[10,45], [10, 40]]
    # analysis_time_window = [[-40, -10], [5, 35], [10, 40], [40, 70], [35, 65]
    #                         ,[-40,-5],[-35,-5],[5,40],[10,45],[45,80],[40,75]]
    analysis_time_window = [[-40,-10],[10,40],[40,70]]
    # analysis_time_window = [[-60, 55], [-30, 30], [-40, 40], [-60, 60], [-30, 60], [-60, 30], [-20, 0], [0, 20],
    #                         [20, 40]]
    mouse_list = glob.glob(os.path.join(path, "202*"))

    for tw_id, tw in enumerate(analysis_time_window):
        fig = plt.figure(figsize=(11.69, 8.27))
        gs = gridspec.GridSpec(1, 1)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        event_names = [ "Before_mobile", "Before_immobile", "After_mobile", "After_immobile", "After_StateC"]
        df= pd.DataFrame(index=event_names)
        for mouse in mouse_list:
            df_mouse = process_folder (mouse, tw)
            df = pd.concat([df, df_mouse], axis=1)

        df.to_csv (os.path.join(path, "_group_analysis", "_mean_fr_"+str(tw[0])+"-"+str(tw[1])+ "min.csv"))
        # plot
        ax = fig.add_subplot(gs[0, 0])
        for col in df.columns:
            ax.plot(df.index, df[col], color='gray', linewidth=0.1, alpha=0.6)
        mean_values = df.mean(axis=1)
        sem_values = df.sem(axis=1)
        ax.bar(df.index, mean_values, color='none',edgecolor="black", alpha=0.8, width=0.4,
               label=str(tw[0]) + "-" + str(tw[1]) + "min",
                yerr = sem_values,  # ここでエラーバーを追加
                capsize = 2, ecolor = 'black', error_kw = dict(alpha=0.7, lw=0.8)
                )
        ax.set_ylim([0, 1.2])

        plt.tight_layout()
        plt.legend(fontsize=1, labelspacing=0.1)
        pdf_path = os.path.join(path, "_group_analysis", "_mean_fr_"+str(tw[0])+"-"+str(tw[1])+ "min.pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, dpi=300)
        plt.close(fig)


def main():
    path = select_folder()
    process_group(path)

if __name__ == "__main__":
    main()