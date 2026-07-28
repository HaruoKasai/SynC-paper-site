import pandas as pd
from _archive.EEG_Analysis import plot_heatmap, plot_timeseries, extract_params
import os
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

"""
def detect_gamma_drop_time (gamma_array, threshold_percent, duration):
    
    gamma_arrayは-180~180secの2秒ごとのデータ　len=180
    ベースからthreshold_percent以上低下し、それが、duration秒以上つづいたとき、threshold_percent以上低下した最初の時間を返す

    sampling_interval = 2  # データは2秒ごと
    num_points = len(gamma_array)

    # 時間軸の生成
    time_array = np.arange(-179, 180, sampling_interval)

    # ベースライン: たとえば-180～-60秒の平均
    baseline_mask = time_array < -60
    baseline_mean = np.mean(gamma_array[baseline_mask])

    # しきい値: ベースラインからthreshold_percent低下
    threshold_value = baseline_mean * (1 - threshold_percent / 100)

    # durationに対応するデータ点数
    duration_points = int(duration / sampling_interval)

    # 判定
    for i in range(num_points - duration_points + 1):
        window = gamma_array[i:i + duration_points]
        if np.all(window <= threshold_value):
            return time_array[i]

    return None
"""


def detect_gamma_drop_time(gamma_array, sd_multiplier, duration, epoch_type, window_size=3):
    """
    gamma_array: numpy array, -179~179secの2秒ごとのデータ (len=180)
    sd_multiplier: SDの何倍を閾値にするか
    duration: 継続時間 (秒)
    epoch_type: "start" or "end"（drop開始 or 回復終了）
    window_size: スムージングの窓サイズ（デフォルト5点）

    返り値: 閾値を満たす最初の時刻 (秒, relative to 0)、ただし-60秒以降。なければ None
    """
    sampling_interval = 2  # 2秒ごと
    time_array = np.arange(-179, 181, sampling_interval)[:len(gamma_array)]

    # ベースライン (-100～-60秒)
    base_start = -40
    base_end = -20
    baseline_mask = (time_array >= base_start) & (time_array <= base_end)
    baseline_mean = np.mean(gamma_array[baseline_mask])
    baseline_sd = np.std(gamma_array[baseline_mask])

    # スムージング（移動平均）
    smoothed_gamma = pd.Series(gamma_array).rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()

    # 閾値設定
    if epoch_type == "start":
        threshold = baseline_mean - sd_multiplier * baseline_sd
    elif epoch_type == "end":
        threshold = baseline_mean + sd_multiplier * baseline_sd
    else:
        raise ValueError("epoch_type must be 'start' or 'end'")

    duration_points = int(duration / sampling_interval)

    for i in range(len(smoothed_gamma) - duration_points + 1):
        current_time = time_array[i]
        if current_time < base_end:
            continue  # base_end以前はスキップ

        window = smoothed_gamma[i:i + duration_points]
        if epoch_type == "start" and np.all(window <= threshold):
            return current_time
        elif epoch_type == "end" and np.all(window >= threshold):
            return current_time

    return None

def plot_shade(tp, avg, sem, window_size,ax,color, alpha=0.3):
    valid_len = (len(sem) // window_size) * window_size

    ave_tp = tp[:valid_len].reshape(-1, window_size).mean(axis=1)
    ave_data = avg[:valid_len].reshape(-1, window_size).mean(axis=1)
    sem_binned_array = sem[:valid_len].reshape(-1, window_size)
    sem_binned =np.sqrt(np.sum(sem_binned_array**2, axis=1) / (window_size**2))

    ax.fill_between(ave_tp, ave_data - sem_binned, ave_data + sem_binned, color=color, alpha=alpha)

def process_group(json_path, group_dict, electrode_dict, state_dict, PETH_time_list, dlc_type):
    dir = os.path.dirname(os.path.dirname(json_path))
    group_analysis_dir = os.path.dirname(json_path)
    # param_name = os.path.basename(json_path)[22:-5]
    band_list = ["delta", "theta", "alpha", "beta", "gamma", "high_gamma", "low_gamma"]

    # pupillometryの場合は、OF_tp, velocityにpupil_tp, pupil_sizeを代入して計算
    ax0_title = None
    if dlc_type == "Pupillometry":  # if OF_tp is None and pupil_tp is not None:
        # ax0_title = "Pupil size"
        ax0_ylable = "ΔPupil size(%)"
        ax0_ylim = (-50, 150)
    else:
        # ax0_title = "Velocity"
        ax0_ylable = "Velocity (mm/s)"
        ax0_ylim = (0, 30)


    for state in state_dict:
        epoch_types = ["start", "end"]

        for elec_num, (electrode_name, electrode_list) in enumerate(electrode_dict.items()):
            for g, (group, exp_list) in enumerate(group_dict.items()):
                for time in PETH_time_list:
                    # alignment_types = ["", "Gamma_Aligned"] #時間をalignするのを試そうとしたが、不要そうなのでいったん中止 20250502
                    alignment_types = [""]
                    for align_type in alignment_types:
                        fig = plt.figure(figsize=(6, 25))
                        gs = gridspec.GridSpec(10, 2, height_ratios=[1, 1, 1, 1, 1, 1,1,1, 1, 1])
                        plt.subplots_adjust(wspace=0.05, hspace=0.05)
                        plot=False
                        velocity_window = int((time[1]-time[0])/4) if dlc_type=="Openfield" else int((time[1]-time[0])/2)
                        breath_window = 5
                        table_window = 1
                        hr_window = 20
                        rms_window = 2 #int((time[1] - time[0]) / 20)
                        power_window = 1 #if time[1] - time[0]<300 else 2
                        baseline_list = [[] for _ in range(5)]
                        heat_base_list =[]
                        for e, type in enumerate(epoch_types):
                            axes = [fig.add_subplot(gs[i, e]) for i in range(10)]
                            ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9= axes
                            all_data = {}
                            h5_files = []
                            for exp_name in exp_list:
                                # combined_dir = os.path.join(dir, exp_name, "_Combined")
                                combined_dir = os.path.normpath(os.path.join(dir, exp_name, "_Combined")) #相対pathも扱えるように
                                for elec in electrode_list:
                                    target_file = f"{state}_{type}_-180s_180s_PETH_average_all_{elec}.h5"
                                    full_path = os.path.join(combined_dir, target_file)
                                    if os.path.isfile(full_path):
                                        h5_files.append(full_path)
                                        break  # 最初に見つかったファイルだけ追加する
                            print(time)
                            # print(h5_files)
                            for h, h5_file in enumerate(h5_files):
                                print(h5_file)
                                data = open_PETH_h5(h5_file)
                                # exp_ab = "_".join(os.path.basename(os.path.dirname(os.path.dirname(h5_file))).split("_")[:2])
                                exp_ab = os.path.basename(os.path.dirname(os.path.dirname(h5_file)))[:15]
                                #TODO "Aligned"については、ここで、gammaが落ちるタイミングを検出して各時間をずらす。
                                if align_type=="Gamma_Aligned":
                                    time_zero = detect_gamma_drop_time (data["gamma"], 0.5, 20, type)
                                    # print(str(h)+"__time_zero="+str(time_zero))


                                Omin = min(len(data["OF_tp"]), len(data["velocity"]))
                                OF_tp_indices = (time[0] <= data["OF_tp"][:Omin]) & (data["OF_tp"][:Omin] <= time[1])
                                data["OF_tp"] = data["OF_tp"][:Omin][OF_tp_indices]
                                data["velocity"] = data["velocity"][:Omin][OF_tp_indices]

                                """
                                Pupillometryの解析に、一部Openfieldのheartrateの結果をいれる
                                このとき、これらのマウスのVelovityがpupil sizeとしてplotされてしまう（Pupil dataは便宜上"velocity"として保存してあるので）のを防ぐ
                                """

                                mouse_dlc_type, *_ = extract_params(os.path.dirname(os.path.dirname(h5_file)))
                                if dlc_type=="Pupillometry" and mouse_dlc_type=="Openfield":
                                    print("dlc type: Pupillometry, Mouse dlc type: Openfield")
                                    data["velocity"] = np.full_like(data["velocity"], np.nan)

                                emg_tp_indices = (time[0] <= data["emg_tp"]) & (data["emg_tp"] <= time[1])
                                data["emg_tp"] = data["emg_tp"][emg_tp_indices]
                                data["emg_rms"] = data["emg_rms"][emg_tp_indices]

                                t_stft_indices = (time[0] <= data["t_stft"]) & (data["t_stft"] <= time[1])
                                data["t_stft"] = data["t_stft"][t_stft_indices]
                                data["power_spectrum"] = data["power_spectrum"][:,t_stft_indices]
                                # print(data)
                                # if "breath_tp" in data:


                                # if np.any(data["breathing_rate"] != 0):
                                breath_tp_indices = (time[0] <= data["breath_tp"]) & (data["breath_tp"] <= time[1])
                                data["breath_tp"] = data["breath_tp"][breath_tp_indices]
                                data["breathing_rate"] = data["breathing_rate"][breath_tp_indices]
                                # if "table_tp" in data:
                                # if np.any(data["table_velocity"] != 0):
                                table_tp_indices = (time[0] <= data["table_tp"]) & (data["table_tp"] <= time[1])
                                data["table_tp"] = data["table_tp"][table_tp_indices]
                                data["table_velocity"] = data["table_velocity"][table_tp_indices]

                                # if np.any(data["heartrate"] != 0):
                                if "hr_tp" in data:
                                    hr_tp_indices = (time[0] <= data["hr_tp"]) & (data["hr_tp"] <= time[1])
                                    data["hr_tp"] = data["hr_tp"][hr_tp_indices]
                                    data["heartrate"] = data["heartrate"][hr_tp_indices]

                                power_array_indices =  (time[0] <= data["power_time_array"]) & (data["power_time_array"] <= time[1])
                                data["power_time_array"] = data["power_time_array"][power_array_indices]
                                for band in band_list:
                                    data[band] = data[band][power_array_indices]

                                #normalize
                                """
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
                                """

                                #binning



                                for key, values in data.items():
                                    if isinstance(values, np.ndarray) and np.all(values == 0):
                                        continue
                                    if key not in all_data:
                                        all_data[key] = []
                                    all_data[key].append(values)

                                # print(all_data["hr_tp"])
                                # if not any(np.isnan(values).any() for values in data.values()):
                                    # print(data)
                                    # print("#################################")
                                    # print(data["hr_tp"])
                                    # print("$$$$$$$")
                                    # print(data["breath_tp"])
                                plot_timeseries(data["OF_tp"], data["velocity"], velocity_window, ax0, plt.get_cmap("tab20")(h),0.3, None, None, ax0_ylim, None)
                                plot_timeseries(data["emg_tp"], data["emg_rms"], rms_window, ax1, plt.get_cmap("tab20")(h), 0.3, None, None, (0,200),None)

                                #Decibel plot
                                # plot_timeseries(data["power_time_array"], 10 * np.log10(data["delta"] + 1e-10), power_window, ax3, "#1f77b4", 0.3, None, None, (55,85),None)
                                # plot_timeseries(data["power_time_array"], 10 * np.log10(data["theta"] + 1e-10), power_window, ax4, "#1f77b4", 0.5, None, None,(55, 85), None)
                                # plot_timeseries(data["power_time_array"], 10 * np.log10(data["alpha"] + 1e-10), power_window, ax5, "#1f77b4", 0.5, None, None,(55, 85), None)
                                # plot_timeseries(data["power_time_array"], 10 * np.log10(data["beta"] + 1e-10), power_window, ax6, "#1f77b4", 0.5, None, None,(55, 85), None)
                                # plot_timeseries(data["power_time_array"], 10 * np.log10(data["gamma"] + 1e-10), power_window, ax7, "#1f77b4", 0.5, None, None,(55, 85), None)

                                plot_timeseries(data["power_time_array"], data["gamma"], power_window, ax3, plt.get_cmap("tab20")(h), 0.3, None, None, (0,3e6),None)#(-80,80) delta percent表示のときの値
                                plot_timeseries(data["power_time_array"], data["high_gamma"], power_window, ax5, plt.get_cmap("tab20")(h), 0.3, None, None, (0, 3e6), None)
                                plot_timeseries(data["power_time_array"], data["low_gamma"], power_window, ax6, plt.get_cmap("tab20")(h), 0.3, None, None, (0, 3e6), exp_ab)
                                plot_timeseries(data["power_time_array"], data["delta"], power_window, ax4, plt.get_cmap("tab20")(h), 0.3, None, None, (0,3e7),None)#(-80,320)
                                # plot_timeseries(data["power_time_array"], data["theta"], power_window, ax7, plt.get_cmap("tab10")(h), 0.3, None, None, (-100,200),exp_ab)#(-100,200)
                                # plot_timeseries(data["power_time_array"], data["alpha"], power_window, ax8, plt.get_cmap("tab10")(h), 0.3, None, None, (-100,200),None)#(-100,200)
                                # plot_timeseries(data["power_time_array"], data["beta"], power_window, ax9, plt.get_cmap("tab10")(h), 0.3, None, None, (-100,200),None)#(-100,200)


                                if np.any(data["breathing_rate"] != 0):
                                    plot_timeseries(data["breath_tp"], data["breathing_rate"], breath_window, ax7, plt.get_cmap("tab20")(h),0.3, None, None, (50,450), None)
                                if np.any(data["table_velocity"] != 0):
                                    plot_timeseries(data["table_tp"], data["table_velocity"], table_window, ax8, plt.get_cmap("tab20")(h),0.3, None, None, (-50,250), None)
                                if "hr_tp" in data and "heartrate" in data:
                                    # print(data["heartrate"])
                                    if np.any(data["heartrate"] != 0):
                                        plot_timeseries(data["hr_tp"], data["heartrate"], hr_window, ax9, plt.get_cmap("tab20")(h),0.3, None, None, (200,1000), None)
                                    else:
                                        print("%%%%%%%%%")
                                else:
                                    print("&&&&&&&&&&&&&")

                                # else:
                                #     print(data)
                                #     print("the file includes NaN")
                            if all_data:
                                plot=True
                                avg_data = {}
                                sem_data = {}


                                #TODO temp
                                if group=="SynC-pupil":
                                    print(time)
                                    print(all_data["velocity"])
                                    velocity_list = all_data["velocity"][:6]
                                    # 各arrayの長さをチェック（おそらく同じ長さ）
                                    lengths = [len(v) for v in velocity_list]
                                    print(lengths)
                                    assert len(set(lengths)) == 1, "長さが不揃いです"

                                    # スタックして2D配列化（shape: [n_animals, n_timepoints]）
                                    velocity_array = np.stack(velocity_list, axis=0)

                                    # DataFrame化して保存（index=Falseで行番号なし）
                                    df = pd.DataFrame(velocity_array)
                                    df.to_csv(r"X:\Behavior\Turntable_EEG\_Group_Analysis_PETH\temp_velocity.csv",index=False )

                                for key, values_list in all_data.items():
                                    min_len = min(arr.shape[0] for arr in values_list)
                                    trimmed_values = [arr[:min_len] for arr in values_list]
                                    stacked_values = np.stack(trimmed_values)
                                    avg_data[key] = np.nanmean(stacked_values, axis=0)
                                    sem_data[key] = np.nanstd(stacked_values, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(stacked_values), axis=0))
                                    # print("")
                                    # print("#####")
                                    # print(key)
                                    # print("#NUmber")
                                    # print(np.sum(~np.isnan(stacked_values), axis=0))


                                plot_timeseries(avg_data["OF_tp"], avg_data["velocity"], velocity_window, ax0, "gray",1, ax0_title, ax0_ylable, ax0_ylim, None)
                                plot_timeseries(avg_data["emg_tp"], avg_data["emg_rms"], rms_window, ax1, "gray", 2.5, None, "EMG-RMS", (0,200),None)
                                # plot_timeseries(avg_data["power_time_array"], 10 * np.log10(avg_data["delta"] + 1e-10), power_window, ax3, "#1f77b4", 2.5, "delta power", "(dB)", (55,85),None)
                                plot_timeseries(avg_data["power_time_array"], avg_data["gamma"], power_window, ax3, "#1f77b4", 2.5, None, "gamma power",(0,3e6), None)#Δ(%)
                                plot_timeseries(avg_data["power_time_array"], avg_data["high_gamma"], power_window, ax5, "#1f77b4", 2.5, None, "High gamma",(0,1.5e6), None)#Δ(%)
                                plot_timeseries(avg_data["power_time_array"], avg_data["low_gamma"], power_window, ax6, "#1f77b4", 2.5, None, "Low gamma",(0,1.5e6), None)#Δ(%)
                                plot_timeseries(avg_data["power_time_array"], avg_data["delta"], power_window, ax4, "#1f77b4", 2.5, None, "delta power", (0,3e7),None)#Δ(%)
                                # plot_timeseries(avg_data["power_time_array"], avg_data["theta"], power_window, ax7, "#1f77b4", 2.5, "theta power", "Δ(%)",(-100,200), None)
                                # plot_timeseries(avg_data["power_time_array"], avg_data["alpha"], power_window, ax8, "#1f77b4", 2.5, "alpha power", "Δ(%)",(-100,200), None)
                                # plot_timeseries(avg_data["power_time_array"], avg_data["beta"], power_window, ax9, "#1f77b4", 2.5, "beta power", "Δ(%)",(-100,200), None)

                                print(sem_data["velocity"])
                                plot_shade(avg_data["OF_tp"],avg_data["velocity"],sem_data["velocity"],velocity_window, ax0,"gray")
                                plot_shade(avg_data["emg_tp"],avg_data["emg_rms"],sem_data["emg_rms"],rms_window, ax1,"gray")
                                plot_shade(avg_data["power_time_array"],avg_data["gamma"],sem_data["gamma"],power_window, ax3,"#1f77b4")
                                plot_shade(avg_data["power_time_array"],avg_data["high_gamma"],sem_data["high_gamma"],power_window, ax5,"#1f77b4")
                                plot_shade(avg_data["power_time_array"],avg_data["low_gamma"],sem_data["low_gamma"],power_window, ax6,"#1f77b4")
                                plot_shade(avg_data["power_time_array"], avg_data["delta"], sem_data["delta"],power_window, ax4, "#1f77b4")



                                # if "breath_tp" in avg_data:
                                if "breathing_rate" in avg_data and np.any(avg_data["breathing_rate"] != 0):
                                    plot_timeseries(avg_data["breath_tp"], avg_data["breathing_rate"], breath_window, ax7, "gray",2.5, None, "Breathing rate (BPM)", (50,450), None)
                                    plot_shade(avg_data["breath_tp"], avg_data["breathing_rate"], sem_data["breathing_rate"],breath_window, ax7, "#1f77b4")
                                if "table_velocity" in avg_data and np.any(avg_data["table_velocity"] != 0):
                                    plot_timeseries(avg_data["table_tp"], avg_data["table_velocity"], breath_window, ax8, "gray",2.5, None, "Velocity (mm/s)", (-50,250), None)
                                if "hr_tp" in avg_data and "heartrate" in avg_data:
                                    if np.any(avg_data["heartrate"] != 0):
                                        plot_timeseries(avg_data["hr_tp"], avg_data["heartrate"], hr_window, ax9, "gray",2.5, None, "Heartrate (BPM)", (200,1000), None)
                                        plot_shade(avg_data["hr_tp"], avg_data["heartrate"],sem_data["heartrate"], hr_window, ax9, "#1f77b4")
                                # plot_heatmap(ax2, avg_data["t_stft"], avg_data["f_stft"], 10 * np.log10(avg_data["power_spectrum"] + 1e-10), "STFT dB Power", "Frequency (Hz)", 100, "rainbow", [-10, 33])
                                plot_heatmap(ax2, avg_data["t_stft"], avg_data["f_stft"], 10 * np.log10(avg_data["power_spectrum"] + 1e-10),
                                             "STFT dB Power", "Frequency (Hz)", 100, "rainbow", [-8, 45]) #[-5, 8]

                                for ax in axes:

                                    ax.set_xlabel("Time (sec)")
                                    ax.margins(x=0)
                        if plot:
                            plt.tight_layout()
                            plt.legend(fontsize=1, labelspacing=0.1)
                            pdf_path = os.path.join(group_analysis_dir, align_type+"_"+group+"_"+state+"_"+str(time[0])+"s_"+str(time[1])+"s_PETH_"+electrode_name+".pdf")
                            with PdfPages(pdf_path) as pdf:
                                pdf.savefig(fig, dpi=300)
                            plt.close(fig)

def main():
    # json_path = select_json_path()
    json_path =r"X:\Behavior\Turntable_EEG\_Group_Analysis_PETH\__group_analysis_param_Turntable_PETH.json"

    # json_path = r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH\__group_analysis_param_PETH_0605.json"


    group_dict, electrode_dict, state_dict, PETH_time, DLC_type = extract_group_analysis_params(json_path)
    process_group(json_path, group_dict, electrode_dict, state_dict, PETH_time, DLC_type)


if __name__ == "__main__":
    main()