# from pyqtgraph.examples.GLMeshItem import theta

from _archive.EEG_Analysis import plot_heatmap, plot_timeseries
import matplotlib.pyplot as plt
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12   })

import os
import glob
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
# from pyqtgraph.examples.DateAxisItem_QtDesigner import window
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12   })
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import json
# from functools import partial
import h5py


"""
#TODO
PETH.pyでtimeblock一つしかh5に保存できないようになってしまっている。
これを直して、groupanalysisも直す必要あり
"""

def extract_group_analysis_params(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        group_dict = data["Group"]
        electrode_dict = data["Electrode"]
        state_dict = data["state"]
        PETH_time = data["PETH_time"]
    return group_dict, electrode_dict, state_dict, PETH_time

def select_json_path():
    root = tk.Tk()
    root.withdraw()  # ウィンドウを表示しない
    file_path = filedialog.askopenfilename(title="Select a group_analysis_param json file", initialdir=r"X:\Behavior")
    root.destroy()
    return file_path

def open_PETH_h5(h5_path):
    data = {}
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            data[key] = f[key][()]
    return data

def process_group(json_path, group_dict, electrode_dict, state_dict, PETH_time):
    dir = os.path.dirname(os.path.dirname(json_path))
    param_name = os.path.basename(json_path)[22:-5]
    for state in state_dict:
        fig = plt.figure(figsize=(8, 21))
        gs = gridspec.GridSpec(8, 2, height_ratios=[1, 1, 1, 1,1,1,1,1])
        # tab10 = plt.get_cmap("tab10")
        plt.subplots_adjust(hspace=0.5)
        epoch_types = ["start", "end"]

        for group in group_dict:
            subgroup_list = group_dict[group]
            for e, type in enumerate(epoch_types):
                axes = [fig.add_subplot(gs[i, e]) for i in range(8)]
                ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axes

                all_data = {}
                h5_files = []
                for subgroup in subgroup_list:
                    file_pattern = os.path.join(dir, subgroup, "*", "_Combined", state+"_"+type+"_"+str(PETH_time[type][0])+"s_"+str(PETH_time[type][1])+"s_PETH_average*.h5") #TODO 本当はelectrode位置ごとに解析わける
                    # TODO 現状、同一個体に複数のelectrodeセットが解析されていると、すべて拾う（emg等については重ねて平均される）ようになってしまっている
                    h5_files.extend(glob.glob(file_pattern))
                print(h5_files)
                for h5_file in h5_files:
                    data = open_PETH_h5(h5_file)
                    for key, values in data.items():
                        if key not in all_data:
                            all_data[key] = []
                        all_data[key].append(values)

                    plot_timeseries(data["OF_tp"], data["velocity"], 4, ax0, "gray",0.3, None, None, (0,50), None)
                    plot_timeseries(data["emg_tp"], data["emg_rms"], 4, ax1, "gray", 0.3, None, None, (0,200),None)
                    plot_timeseries(data["power_time_array"], 10 * np.log10(data["delta"] + 1e-10), 1, ax3, "#1f77b4", 0.3, None, None, (55,85),None)
                    plot_timeseries(data["power_time_array"], 10 * np.log10(data["theta"] + 1e-10), 1, ax4, "#1f77b4", 0.5, None, None,(55, 85), None)
                    plot_timeseries(data["power_time_array"], 10 * np.log10(data["alpha"] + 1e-10), 1, ax5, "#1f77b4", 0.5, None, None,(55, 85), None)
                    plot_timeseries(data["power_time_array"], 10 * np.log10(data["beta"] + 1e-10), 1, ax6, "#1f77b4", 0.5, None, None,(55, 85), None)
                    plot_timeseries(data["power_time_array"], 10 * np.log10(data["gamma"] + 1e-10), 1, ax7, "#1f77b4", 0.5, None, None,(55, 85), None)



                if all_data:
                    avg_data = {}
                    for key, values_list in all_data.items():
                        stacked_values = np.stack(values_list)
                        avg_data[key] = np.nanmean(stacked_values, axis=0)

                    plot_timeseries(avg_data["OF_tp"], avg_data["velocity"], 4, ax0, "gray",2.5, "Velocity", "mm/s", (0,50), None)
                    plot_timeseries(avg_data["emg_tp"], avg_data["emg_rms"], 4, ax1, "gray", 2.5, "EMG-RMS", None, (0,200),None)
                    plot_timeseries(avg_data["power_time_array"], 10 * np.log10(avg_data["delta"] + 1e-10), 1, ax3, "#1f77b4", 2.5, "delta power", "(dB)", (55,85),None)
                    plot_timeseries(avg_data["power_time_array"], 10 * np.log10(avg_data["theta"] + 1e-10), 1, ax4, "#1f77b4", 2.5, "theta power", "(dB)",(55, 85), None)
                    plot_timeseries(avg_data["power_time_array"], 10 * np.log10(avg_data["alpha"] + 1e-10), 1, ax5, "#1f77b4", 2.5, "alpha power", "(dB)",(55, 85), None)
                    plot_timeseries(avg_data["power_time_array"], 10 * np.log10(avg_data["beta"] + 1e-10), 1, ax6, "#1f77b4", 2.5, "beta power", "(dB)",(55, 85), None)
                    plot_timeseries(avg_data["power_time_array"], 10 * np.log10(avg_data["gamma"] + 1e-10), 1, ax7, "#1f77b4", 2.5, "gamma power", "(dB)",(55, 85), None)
                    plot_heatmap(ax2, avg_data["t_stft"], avg_data["f_stft"], 10 * np.log10(avg_data["power_spectrum"] + 1e-10), "STFT dB Power", "Frequency (Hz)", 100, "rainbow", [-10, 33])
            plt.tight_layout()

            pdf_path = os.path.join(dir, "_Group_Analysis", group+"_"+state+"_"+param_name+"_PETH.pdf")


            with PdfPages(pdf_path) as pdf:
                pdf.savefig(fig, dpi=300)
            plt.close(fig)



def main():
    json_path = select_json_path()
    group_dict, electrode_dict, state_dict, PETH_time = extract_group_analysis_params(json_path)
    process_group(json_path, group_dict, electrode_dict, state_dict, PETH_time)


if __name__ == "__main__":
    main()