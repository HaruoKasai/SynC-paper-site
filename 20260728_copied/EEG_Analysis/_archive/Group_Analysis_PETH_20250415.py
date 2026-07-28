from _archive.EEG_Analysis import plot_heatmap, plot_timeseries
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
        DLC_type = data["DLC_type"]
    return group_dict, electrode_dict, state_dict, PETH_time, DLC_type

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

def process_group(json_path, group_dict, electrode_dict, state_dict, PETH_time_list, dlc_type):
    dir = os.path.dirname(os.path.dirname(json_path))
    # param_name = os.path.basename(json_path)[22:-5]
    band_list = ["delta", "theta", "alpha", "beta", "gamma"]

    # pupillometryの場合は、OF_tp, velocityにpupil_tp, pupil_sizeを代入して計算
    if dlc_type == "Pupillometry":  # if OF_tp is None and pupil_tp is not None:
        ax0_title = "Pupil size"
        ax0_ylable = "Δ(%)"
        ax0_ylim = (-50, 150)
    else:
        ax0_title = "Velocity"
        ax0_ylable = "mm/s"
        ax0_ylim = (0, 50)


    for state in state_dict:
        epoch_types = ["start", "end"]
        for group in group_dict:
            subgroup_list = group_dict[group]
            for time in PETH_time_list:
                fig = plt.figure(figsize=(8, 25))
                gs = gridspec.GridSpec(10, 2, height_ratios=[1, 1, 1, 1, 1, 1,1,0.5, 0.5, 0.5])
                plt.subplots_adjust(wspace=0.05, hspace=0.05)
                plot=False
                velocity_window = int((time[1]-time[0])/4)
                breath_window = 1
                table_window = 1
                rms_window = 2 #int((time[1] - time[0]) / 20)
                power_window = 1 #if time[1] - time[0]<300 else 2
                baseline_list = [[] for _ in range(5)]
                heat_base_list =[]
                for e, type in enumerate(epoch_types):
                    axes = [fig.add_subplot(gs[i, e]) for i in range(10)]
                    ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9= axes
                    all_data = {}
                    h5_files = []
                    for subgroup in subgroup_list:
                        #PETH.pyでstrict/roughを使ってたときは、それぞれの秒数でデータが異なったので個別に保存してとっていた。→allに変更。すべて短いPETHもすべて同じデータから取り出す方針に変更
                        # file_pattern = os.path.join(dir, subgroup, "*", "_Combined", state+"_"+type+"_"+str(time[0])+"s_"+str(time[1])+"s_PETH_average_all*.h5") #TODO 本当はelectrode位置ごとに解析わける
                        # # TODO 現状、同一個体に複数のelectrodeセットが解析されていると、すべて拾う（emg等については重ねて平均される）ようになってしまっている

                        file_pattern = os.path.join(dir, subgroup, "*", "_Combined", state+"_"+type+"_-180s_180s_PETH_average_all*.h5")
                        h5_files.extend(glob.glob(file_pattern))
                    # print(h5_files)
                    for h, h5_file in enumerate(h5_files):
                        data = open_PETH_h5(h5_file)

                        Omin = min(len(data["OF_tp"]), len(data["velocity"]))
                        OF_tp_indices = (time[0] <= data["OF_tp"][:Omin]) & (data["OF_tp"][:Omin] <= time[1])
                        data["OF_tp"] = data["OF_tp"][:Omin][OF_tp_indices]
                        print(len(data["OF_tp"]))
                        print(data["OF_tp"])
                        data["velocity"] = data["velocity"][:Omin][OF_tp_indices]

                        emg_tp_indices = (time[0] <= data["emg_tp"]) & (data["emg_tp"] <= time[1])
                        data["emg_tp"] = data["emg_tp"][emg_tp_indices]
                        data["emg_rms"] = data["emg_rms"][emg_tp_indices]

                        t_stft_indices = (time[0] <= data["t_stft"]) & (data["t_stft"] <= time[1])
                        data["t_stft"] = data["t_stft"][t_stft_indices]
                        data["power_spectrum"] = data["power_spectrum"][:,t_stft_indices]

                        if "breath_tp" in data:
                            breath_tp_indices = (time[0] <= data["breath_tp"]) & (data["breath_tp"] <= time[1])
                            data["breath_tp"] = data["breath_tp"][breath_tp_indices]
                            data["breathing_rate"] = data["breathing_rate"][breath_tp_indices]
                        if "table_tp" in data:
                            table_tp_indices = (time[0] <= data["table_tp"]) & (data["table_tp"] <= time[1])
                            data["table_tp"] = data["table_tp"][table_tp_indices]
                            data["table_velocity"] = data["table_velocity"][table_tp_indices]


                        power_array_indices =  (time[0] <= data["power_time_array"]) & (data["power_time_array"] <= time[1])
                        data["power_time_array"] = data["power_time_array"][power_array_indices]
                        for band in band_list:
                            data[band] = data[band][power_array_indices]

                        #normalize
                        negative_indices = data["power_time_array"] < 0
                        for b, band in enumerate(band_list):
                            if e==0: #start
                                base = np.mean(data[band][negative_indices])
                                baseline_list[b].append(base)
                            else: #end
                                base = baseline_list[b][h]
                            data[band] = (data[band] / base - 1) * 100

                        if e==0: #start
                            # plot_heatmap(ax2, avg_data["t_stft"], avg_data["f_stft"], 10 * np.log10(avg_data["power_spectrum"] + 1e-10), "STFT dB Power", "Frequency (Hz)", 100, "rainbow", [-10, 33])
                            negative_indices = data["t_stft"]<0
                            heat_base = np.mean(data["power_spectrum"][:, negative_indices]) #sumにすると秒数かわるたびにcolor code変わってしまう
                            heat_base_list.append(heat_base)
                        else: #end
                            heat_base = np.array(heat_base_list[h])
                        data["power_spectrum"] = data["power_spectrum"] / heat_base

                        for key, values in data.items():
                            if key not in all_data:
                                all_data[key] = []
                            all_data[key].append(values)
                        print(h5_file)
                        if not any(np.isnan(values).any() for values in data.values()):
                            plot_timeseries(data["OF_tp"], data["velocity"], velocity_window, ax0, "gray",0.3, None, None, ax0_ylim, None)
                            plot_timeseries(data["emg_tp"], data["emg_rms"], rms_window, ax1, "gray", 0.3, None, None, (0,250),None)

                            #Decibel plot
                            # plot_timeseries(data["power_time_array"], 10 * np.log10(data["delta"] + 1e-10), power_window, ax3, "#1f77b4", 0.3, None, None, (55,85),None)
                            # plot_timeseries(data["power_time_array"], 10 * np.log10(data["theta"] + 1e-10), power_window, ax4, "#1f77b4", 0.5, None, None,(55, 85), None)
                            # plot_timeseries(data["power_time_array"], 10 * np.log10(data["alpha"] + 1e-10), power_window, ax5, "#1f77b4", 0.5, None, None,(55, 85), None)
                            # plot_timeseries(data["power_time_array"], 10 * np.log10(data["beta"] + 1e-10), power_window, ax6, "#1f77b4", 0.5, None, None,(55, 85), None)
                            # plot_timeseries(data["power_time_array"], 10 * np.log10(data["gamma"] + 1e-10), power_window, ax7, "#1f77b4", 0.5, None, None,(55, 85), None)
                            plot_timeseries(data["power_time_array"], data["gamma"], power_window, ax3, "#1f77b4", 0.3, None, None, (-80,80),None)
                            plot_timeseries(data["power_time_array"], data["delta"], power_window, ax4, "#1f77b4", 0.3, None, None, (-80,320),None)
                            plot_timeseries(data["power_time_array"], data["theta"], power_window, ax7, "#1f77b4", 0.3, None, None, (-100,200),None)
                            plot_timeseries(data["power_time_array"], data["alpha"], power_window, ax8, "#1f77b4", 0.3, None, None, (-100,200),None)
                            plot_timeseries(data["power_time_array"], data["beta"], power_window, ax9, "#1f77b4", 0.3, None, None, (-100,200),None)
                            # plot_timeseries(data["power_time_array"], data["gamma"], power_window, ax3, "#1f77b4", 0.3, None, None, (0,4e6),None)
                            # plot_timeseries(data["power_time_array"], data["delta"], power_window, ax4, "#1f77b4", 0.3, None, None, (0,3e7),None)
                            # plot_timeseries(data["power_time_array"], data["theta"], power_window, ax5, "#1f77b4", 0.3, None, None, (1e6,4e6),None)
                            # plot_timeseries(data["power_time_array"], data["alpha"], power_window, ax6, "#1f77b4", 0.3, None, None, (1e6,4e6),None)
                            # plot_timeseries(data["power_time_array"], data["beta"], power_window, ax7, "#1f77b4", 0.3, None, None, (1e6,4e6),None)

                            if "breath_tp" in data:
                                plot_timeseries(data["breath_tp"], data["breathing_rate"], breath_window, ax5, "gray",0.3, None, None, (50,450), None)
                            if "table_tp" in data:
                                plot_timeseries(data["table_tp"], data["table_velocity"], breath_window, ax6, "gray",0.3, None, None, (-50,250), None)


                        else:
                            print("the file includes NaN")
                    if all_data:
                        plot=True
                        avg_data = {}
                        for key, values_list in all_data.items():
                            min_len = min(arr.shape[0] for arr in values_list)
                            trimmed_values = [arr[:min_len] for arr in values_list]
                            stacked_values = np.stack(trimmed_values)
                            avg_data[key] = np.nanmean(stacked_values, axis=0)

                        plot_timeseries(avg_data["OF_tp"], avg_data["velocity"], velocity_window, ax0, "gray",2.5, ax0_title, ax0_ylable, ax0_ylim, None)
                        plot_timeseries(avg_data["emg_tp"], avg_data["emg_rms"], rms_window, ax1, "gray", 2.5, "EMG-RMS", None, (0,250),None)
                        # plot_timeseries(avg_data["power_time_array"], 10 * np.log10(avg_data["delta"] + 1e-10), power_window, ax3, "#1f77b4", 2.5, "delta power", "(dB)", (55,85),None)
                        plot_timeseries(avg_data["power_time_array"], avg_data["gamma"], power_window, ax3, "#1f77b4", 2.5, "gamma power", "Δ(%)",(-80,80), None)
                        plot_timeseries(avg_data["power_time_array"], avg_data["delta"], power_window, ax4, "#1f77b4", 2.5, "delta power", "Δ(%)", (-80,320),None)
                        plot_timeseries(avg_data["power_time_array"], avg_data["theta"], power_window, ax7, "#1f77b4", 2.5, "theta power", "Δ(%)",(-100,200), None)
                        plot_timeseries(avg_data["power_time_array"], avg_data["alpha"], power_window, ax8, "#1f77b4", 2.5, "alpha power", "Δ(%)",(-100,200), None)
                        plot_timeseries(avg_data["power_time_array"], avg_data["beta"], power_window, ax9, "#1f77b4", 2.5, "beta power", "Δ(%)",(-100,200), None)

                        if "breath_tp" in avg_data:
                            plot_timeseries(avg_data["breath_tp"], avg_data["breathing_rate"], breath_window, ax5, "gray",2.5, "Breathing rate", "BPM", (50,450), None)
                        if "table_tp" in avg_data:
                            plot_timeseries(avg_data["table_tp"], avg_data["table_velocity"], breath_window, ax6, "gray",2.5, "Velocity on table", "mm/s", (-50,250), None)

                        # plot_heatmap(ax2, avg_data["t_stft"], avg_data["f_stft"], 10 * np.log10(avg_data["power_spectrum"] + 1e-10), "STFT dB Power", "Frequency (Hz)", 100, "rainbow", [-10, 33])
                        plot_heatmap(ax2, avg_data["t_stft"], avg_data["f_stft"], 10 * np.log10(avg_data["power_spectrum"] + 1e-10),
                                     "STFT dB Power", "Frequency (Hz)", 100, "rainbow", [-15, 18]) #[-5, 8]
                        temp=10 * np.log10(avg_data["power_spectrum"] + 1e-10)

                        for ax in axes:
                            # ax.set_yticks([])  # y軸の目盛りを消す
                            # ax.yaxis.set_visible(False)  # y軸のラベルも消す
                            # ax.set_xticks([])  # y軸の目盛りを消す
                            # ax.xaxis.set_visible(False)  # y軸のラベルも消す
                            # ymin, ymax = ax.get_ylim()
                            # print(ymax)
                            # yticks = list(range(int(ymin), int(ymax+1), int((ymax-ymin)/5)))
                            # ax.set_yticks(yticks)
                            # ax.set_xlim([time[0],time[1]])
                            # xtick_bin = min(min(abs(time[0]),time[1]), int(max(abs(time[0]),time[1])/4))
                            # xticks = list(range(time[0], time[1] + 1, xtick_bin))
                            # if abs(time[0])<time[1]:
                            #     xtick_label=[time[0]]+list(range(0,time[1]+1, int(time[1]/5)))
                            # else:
                            #     xtick_label=list(range(time[0],1, int(abs(time[0]/5))))
                            # ax.set_xticks(xticks)
                            # ax.set_xticklabels(['' if x not in xtick_label else str(x) for x in xticks])  # 指定の位置以外は空文字
                            ax.set_xlabel("Time (sec)")
                            ax.margins(x=0)
                if plot:
                    plt.tight_layout()
                    pdf_path = os.path.join(dir, "_Group_Analysis_PETH", group+"_"+state+"_"+str(time[0])+"s_"+str(time[1])+"s_PETH.pdf")
                    with PdfPages(pdf_path) as pdf:
                        pdf.savefig(fig, dpi=300)
                    plt.close(fig)

def main():
    json_path = select_json_path()
    # json_path =r"X:\Behavior\Openfield_EEG\_Group_Analysis\_group_analysis_param.json"
    group_dict, electrode_dict, state_dict, PETH_time, DLC_type = extract_group_analysis_params(json_path)
    process_group(json_path, group_dict, electrode_dict, state_dict, PETH_time, DLC_type)


if __name__ == "__main__":
    main()