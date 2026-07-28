import pandas as pd
from EEG_Analysis import plot_timeseries, extract_params, binning
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

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42  # TrueTypeフォントで保存
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import json
import h5py
from scipy.stats import sem



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
        Bargraph_time = data["Bargraph_time"]
    return group_dict, electrode_dict, state_dict, PETH_time, DLC_type, Bargraph_time

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

def plot_heatmap(ax, t, f, power_db, title, ylabel, fmax, cmap, t_range, color_bar = True, vmin=None, vmax=None):
    """
    ax       : matplotlib Axes
    t        : 時間軸 (1D)
    f        : 周波数軸 (1D)
    power_db : 10*log10(power) 済みの2D配列 (f×t)
    """
    # プロット
    im = ax.pcolormesh(t, f, power_db, shading="auto", cmap=cmap,
                       vmin=vmin if vmin else np.nanmin(power_db),
                       vmax=vmax if vmax else np.nanmax(power_db))
    ax.set_ylim(0, fmax)
    ax.set_xlim(t_range)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if color_bar:
        # カラーバーを log化前の値で表示
        cbar = plt.colorbar(im, ax=ax)
        # 元の値（線形パワー）のスケールに戻す
        # log10^-1 して ticklabel を上書き
        ticks = cbar.get_ticks()
        tick_labels = [f"{10**(t/10):.1f}" for t in ticks]  # 単位は元の µV²
        cbar.set_ticklabels(tick_labels)
        cbar.set_label("Power (µV²)")   # 元スケールのラベル

    return im

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
    """
    window_sizeでbinningしたときのsemの近似計算（binningしてからsem計算ではなくて、sem配列をbinningして平均的なのをとる）は、結構実際の値とずれるので使わない
    """

    valid_len = (len(sem) // window_size) * window_size

    ave_tp = tp[:valid_len].reshape(-1, window_size).mean(axis=1)
    ave_data = avg[:valid_len].reshape(-1, window_size).mean(axis=1)
    sem_binned_array = sem[:valid_len].reshape(-1, window_size)
    sem_binned =np.sqrt(np.sum(sem_binned_array**2, axis=1) / (window_size**2))

    ax.fill_between(ave_tp, ave_data - sem_binned, ave_data + sem_binned, color=color, alpha=alpha, linewidth=0)


def Bargraph (tp_list, data_list, time_list, dir, group,state_name, epoch_type, variable):
    per_subject_means = []
    for a, (hr_tp, heartrate) in enumerate(zip(tp_list, data_list)):
        subject_means = []
        for t, (start, end) in enumerate(time_list):
            mask = (hr_tp >= start) & (hr_tp < end)
            values = heartrate[mask]
            if np.any(~np.isnan(values)):
                subject_means.append(np.nanmean(values))
            else:
                subject_means.append(np.nan)
        per_subject_means.append(subject_means)

    per_subject_means = np.array(per_subject_means)  # shape: (n_subjects, n_time_windows)
    df_ = pd.DataFrame(per_subject_means)
    df_.to_csv(os.path.join(dir, "bar_"+variable+"_"+group+"_"+state_name+"_"+epoch_type+".csv"))
    # 折れ線グラフ：各個体
    plt.figure(figsize=(8, 6))
    for subj_vals in per_subject_means:
        plt.plot(range(len(time_list)), subj_vals, marker='', color='gray', alpha=0.6)

    # 棒グラフ：個体間平均とSEM
    mean_vals = np.nanmean(per_subject_means, axis=0)
    sem_vals = sem(per_subject_means, axis=0, nan_policy='omit')

    bar_positions = np.arange(len(time_list))
    plt.bar(bar_positions, mean_vals, yerr=sem_vals, alpha=0.6, color='blue', width=0.4)

    # ラベル調整
    labels = [f"{start}–{end}" for start, end in time_list]
    plt.xticks(bar_positions, labels)
    # plt.grid(True)
    plt.savefig(os.path.join(dir, "bar_"+variable+"_"+group+"_"+state_name+"_"+epoch_type+".pdf"), dpi=300)
    plt.close()

def save_all_data(all_data, outdir, group, state, epoch_type, time, electrode_name, subject_labels=None):
    os.makedirs(outdir, exist_ok=True)
    if subject_labels is None:
        n_subj_guess = len(next(iter(all_data.values())))
        subject_labels = [f"subj{i+1}" for i in range(n_subj_guess)]

    time_key_map = {
        "velocity": "OF_tp",
        "emg_rms": "emg_tp",
        "gamma": "power_time_array",
        "delta": "power_time_array",
        "breathing_rate": "breath_tp",
        "heartrate": "hr_tp",
        "table_velocity": "table_tp",
    }

    def _common_min_len(arr_list):
        return min(len(a) for a in arr_list)

    for key, values_list in all_data.items():
        if not values_list:
            continue

        # --- power_spectrum は保存しない ---
        if key == "power_spectrum":
            continue

        first = values_list[0]

        # --- 1D 系列 ---
        if first.ndim == 1:
            min_len = _common_min_len(values_list)
            trimmed = [arr[:min_len] for arr in values_list]
            stacked = np.stack(trimmed, axis=1)  # (T, Nsubj)

            cols = subject_labels if len(subject_labels) == stacked.shape[1] \
                   else [f"subj{i+1}" for i in range(stacked.shape[1])]
            df = pd.DataFrame(stacked, columns=cols)

            # 時間軸を index に設定
            time_key = None
            if key.endswith("_tp"):
                # tpそのものは個別保存しても良いがスキップ可
                continue
            else:
                time_key = time_key_map.get(key, None)

            if time_key and time_key in all_data and len(all_data[time_key]) == len(values_list):
                t_min_len = _common_min_len(all_data[time_key])
                final_len = min(min_len, t_min_len)
                time_arr = all_data[time_key][0][:final_len]
                df = df.iloc[:final_len, :]
                df.index = time_arr
                df.index.name = "time(sec)"

            fname = f"{group}_{state}_{epoch_type}_{int(time[0])}-{int(time[1])}s_{electrode_name}_{key}.csv"
            df.to_csv(os.path.join(outdir, fname))


def process_group(json_path, group_dict, electrode_dict, state_dict, PETH_time_list, dlc_type, bargraph_time):

    dir = os.path.dirname(os.path.dirname(json_path))
    group_analysis_dir = os.path.dirname(json_path)
    # param_name = os.path.basename(json_path)[22:-5]
    band_list = ["delta", "theta", "alpha", "beta", "gamma", "high_gamma", "low_gamma"]

    # pupillometryの場合は、OF_tp, velocityにpupil_tp, pupil_sizeを代入して計算
    ax0_title = None
    if dlc_type == "Pupillometry":  # if OF_tp is None and pupil_tp is not None:
        # ax0_title = "Pupil size"
        ax0_ylable = "ΔPupil size(%)"
        ax0_ylim = (-50, 50)
    else:
        # ax0_title = "Velocity"
        ax0_ylable = "Velocity (mm/s)"
        ax0_ylim = (0, 30)


    for state in state_dict:
        epoch_types = ["start", "end"]

        for elec_num, (electrode_name, electrode_list) in enumerate(electrode_dict.items()):
            for g, (group, exp_list) in enumerate(group_dict.items()):
                if group=="Excluded":
                    continue

                for time in PETH_time_list:
                    # alignment_types = ["", "Gamma_Aligned"] #時間をalignするのを試そうとしたが、不要そうなのでいったん中止 20250502
                    alignment_types = [""]
                    for align_type in alignment_types:
                        fig = plt.figure(figsize=(9, 25))
                        gs = gridspec.GridSpec(10, 2, height_ratios=[1, 1, 1, 1, 1, 1,1,1, 1, 1])
                        plt.subplots_adjust(wspace=0.05, hspace=0.05)
                        plot=False
                        # velocity_window = int((time[1]ya-time[0])/4) if dlc_type=="Openfield" else int((time[1]-time[0])/1.5)
                        # velocity_window = 2
                        if time[1]<30:
                            velocity_window = 20 #-20~20 sec
                            rms_window = 2  # int((time[1] - time[0]) / 20)
                            power_window = 1  # if time[1] - time[0]<300 else 2
                        else:
                            velocity_window = 60
                            rms_window = 4  # int((time[1] - time[0]) / 20)
                            power_window = 3  # if time[1] - time[0]<300 else 2
                        breath_window = 5
                        table_window = 1
                        hr_window = 20
                        rms_window = 2 #int((time[1] - time[0]) / 20)
                        power_window = 1 #if time[1] - time[0]<300 else 2

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
                                exp_ab = os.path.basename(os.path.dirname(os.path.dirname(h5_file)))[:15]
                                #TODO "Aligned"については、ここで、gammaが落ちるタイミングを検出して各時間をずらす。
                                if align_type=="Gamma_Aligned":
                                    time_zero = detect_gamma_drop_time (data["gamma"], 0.5, 20, type)


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
                                if dlc_type=="Pupillometry":
                                    #あらためて、delta化のベースを調整
                                    data["velocity"] += 100
                                    basemask = data["OF_tp"] < 0
                                    base = np.nanmean(data["velocity"][basemask])
                                    if np.isnan(base):  # 有効なデータが1つもなかった場合
                                        data["velocity"] = np.full_like(data["velocity"], np.nan)
                                    else:
                                        data["velocity"] = data["velocity"] / base * 100 - 100

                                emg_tp_indices = (time[0] <= data["emg_tp"]) & (data["emg_tp"] <= time[1])
                                data["emg_tp"] = data["emg_tp"][emg_tp_indices]
                                data["emg_rms"] = data["emg_rms"][emg_tp_indices]

                                t_stft_indices = (time[0] <= data["t_stft"]) & (data["t_stft"] <= time[1])
                                data["t_stft"] = data["t_stft"][t_stft_indices]
                                data["power_spectrum"] = data["power_spectrum"][:,t_stft_indices]




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

                                if np.all((data["heartrate"] == 0) | np.isnan(data["heartrate"])):
                                    data["heartrate"] = np.full_like(data["heartrate"], np.nan)
                                if np.all((data["breathing_rate"] == 0) | np.isnan(data["breathing_rate"])):
                                    data["breathing_rate"] = np.full_like(data["breathing_rate"], np.nan)


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
                                data["OF_tp"] = binning(data["OF_tp"], velocity_window)
                                data["velocity"] = binning(data["velocity"], velocity_window)
                                data["emg_tp"] = binning(data["emg_tp"], rms_window)
                                data["emg_rms"] = binning(data["emg_rms"], rms_window)
                                data["power_time_array"] = binning(data["power_time_array"], power_window)
                                data["gamma"]= binning(data["gamma"], power_window)
                                data["high_gamma"] = binning(data["high_gamma"], power_window)
                                data["low_gamma"] = binning(data["low_gamma"], power_window)
                                data["delta"] = binning(data["delta"], power_window)
                                data["breath_tp"] = binning(data["breath_tp"], breath_window)
                                data["breathing_rate"]= binning(data["breathing_rate"], breath_window)
                                data["table_tp"] = binning(data["table_tp"], table_window)
                                data["table_velocity"] = binning(data["table_velocity"], table_window)
                                data["hr_tp"] = binning(data["hr_tp"], hr_window)
                                data["heartrate"] = binning(data["heartrate"], hr_window)


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
                                plot_timeseries(data["OF_tp"], data["velocity"], 1, ax0, plt.get_cmap("tab20")(h),0.3, None, None, ax0_ylim, None)
                                plot_timeseries(data["emg_tp"], data["emg_rms"], 1, ax1, plt.get_cmap("tab20")(h), 0.3, None, None, (0,200),None)

                                #Decibel plot
                                # plot_timeseries(data["power_time_array"], 10 * np.log10(data["delta"] + 1e-10), power_window, ax3, "#1f77b4", 0.3, None, None, (55,85),None)
                                # plot_timeseries(data["power_time_array"], 10 * np.log10(data["theta"] + 1e-10), power_window, ax4, "#1f77b4", 0.5, None, None,(55, 85), None)
                                # plot_timeseries(data["power_time_array"], 10 * np.log10(data["alpha"] + 1e-10), power_window, ax5, "#1f77b4", 0.5, None, None,(55, 85), None)
                                # plot_timeseries(data["power_time_array"], 10 * np.log10(data["beta"] + 1e-10), power_window, ax6, "#1f77b4", 0.5, None, None,(55, 85), None)
                                # plot_timeseries(data["power_time_array"], 10 * np.log10(data["gamma"] + 1e-10), power_window, ax7, "#1f77b4", 0.5, None, None,(55, 85), None)

                                plot_timeseries(data["power_time_array"], data["gamma"], 1, ax3, plt.get_cmap("tab20")(h), 0.3, None, None, (0,3e6),None)#(-80,80) delta percent表示のときの値
                                plot_timeseries(data["power_time_array"], data["high_gamma"], 1, ax5, plt.get_cmap("tab20")(h), 0.3, None, None, (0, 3e6), None)
                                plot_timeseries(data["power_time_array"], data["low_gamma"], 1, ax6, plt.get_cmap("tab20")(h), 0.3, None, None, (0, 3e6), exp_ab)
                                plot_timeseries(data["power_time_array"], data["delta"], 1, ax4, plt.get_cmap("tab20")(h), 0.3, None, None, (0,3e7),None)#(-80,320)
                                # plot_timeseries(data["power_time_array"], data["theta"], power_window, ax7, plt.get_cmap("tab10")(h), 0.3, None, None, (-100,200),exp_ab)#(-100,200)
                                # plot_timeseries(data["power_time_array"], data["alpha"], power_window, ax8, plt.get_cmap("tab10")(h), 0.3, None, None, (-100,200),None)#(-100,200)
                                # plot_timeseries(data["power_time_array"], data["beta"], power_window, ax9, plt.get_cmap("tab10")(h), 0.3, None, None, (-100,200),None)#(-100,200)


                                if np.any(data["breathing_rate"] != 0):
                                    plot_timeseries(data["breath_tp"], data["breathing_rate"], 1, ax7, plt.get_cmap("tab20")(h),0.3, None, None, (50,450), None)
                                if np.any(data["table_velocity"] != 0):
                                    plot_timeseries(data["table_tp"], data["table_velocity"], 1, ax8, plt.get_cmap("tab20")(h),0.3, None, None, (-50,250), None)
                                if "hr_tp" in data and "heartrate" in data:
                                    # print(data["heartrate"])
                                    if np.any(data["heartrate"] != 0):
                                        plot_timeseries(data["hr_tp"], data["heartrate"], 1, ax9, plt.get_cmap("tab20")(h),0.3, None, None, (-100,1000), None)
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
                                save_all_data(all_data, group_analysis_dir, group, state, type, time, electrode_name)
                                #Bargraph quantification
                                print("bargraph_time", bargraph_time)
                                print("time", time)
                                if dlc_type=="Pupillometry" and bargraph_time[0][0]==time[0] and type=="start":
                                    Bargraph(all_data["OF_tp"], all_data["velocity"], bargraph_time, group_analysis_dir, group, state, type, "Pupil")
                                    Bargraph(all_data["hr_tp"], all_data["heartrate"], bargraph_time,
                                             group_analysis_dir, group, state, type, "HR")
                                    Bargraph(all_data["breath_tp"], all_data["breathing_rate"], bargraph_time,
                                             group_analysis_dir, group, state, type, "Breathing")


                                # #TODO temp
                                # if group=="SynC-pupil":
                                #     print(time)
                                #     print(all_data["velocity"])
                                #     velocity_list = all_data["velocity"][:6]
                                #     # 各arrayの長さをチェック（おそらく同じ長さ）
                                #     lengths = [len(v) for v in velocity_list]
                                #     print(lengths)
                                #     assert len(set(lengths)) == 1, "長さが不揃いです"
                                #
                                #     # スタックして2D配列化（shape: [n_animals, n_timepoints]）
                                #     velocity_array = np.stack(velocity_list, axis=0)
                                #
                                #     # DataFrame化して保存（index=Falseで行番号なし）
                                #     df = pd.DataFrame(velocity_array)
                                #     df.to_csv(r"X:\Behavior\Turntable_EEG\_Group_Analysis_PETH\temp_velocity.csv",index=False )
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


                                plot_timeseries(avg_data["OF_tp"], avg_data["velocity"], 1, ax0, "gray",1, ax0_title, ax0_ylable, ax0_ylim, None)
                                plot_timeseries(avg_data["emg_tp"], avg_data["emg_rms"], 1, ax1, "gray", 2.5, None, "EMG-RMS", (0,200),None)
                                # plot_timeseries(avg_data["power_time_array"], 10 * np.log10(avg_data["delta"] + 1e-10), power_window, ax3, "#1f77b4", 2.5, "delta power", "(dB)", (55,85),None)
                                plot_timeseries(avg_data["power_time_array"], avg_data["gamma"], 1, ax3, "#1f77b4", 2.5, None, "gamma power",(0,3e6), None)#Δ(%)
                                plot_timeseries(avg_data["power_time_array"], avg_data["high_gamma"], 1, ax5, "#1f77b4", 2.5, None, "High gamma",(0,1.5e6), None)#Δ(%)
                                plot_timeseries(avg_data["power_time_array"], avg_data["low_gamma"], 1, ax6, "#1f77b4", 2.5, None, "Low gamma",(0,1.5e6), None)#Δ(%)
                                plot_timeseries(avg_data["power_time_array"], avg_data["delta"], 1, ax4, "#1f77b4", 2.5, None, "delta power", (0,3e7),None)#Δ(%)
                                # plot_timeseries(avg_data["power_time_array"], avg_data["theta"], power_window, ax7, "#1f77b4", 2.5, "theta power", "Δ(%)",(-100,200), None)
                                # plot_timeseries(avg_data["power_time_array"], avg_data["alpha"], power_window, ax8, "#1f77b4", 2.5, "alpha power", "Δ(%)",(-100,200), None)
                                # plot_timeseries(avg_data["power_time_array"], avg_data["beta"], power_window, ax9, "#1f77b4", 2.5, "beta power", "Δ(%)",(-100,200), None)

                                # print(sem_data["velocity"])
                                plot_shade(avg_data["OF_tp"],avg_data["velocity"],sem_data["velocity"],1, ax0,"gray")
                                plot_shade(avg_data["emg_tp"],avg_data["emg_rms"],sem_data["emg_rms"],1, ax1,"gray")
                                plot_shade(avg_data["power_time_array"],avg_data["gamma"],sem_data["gamma"],1, ax3,"#1f77b4")
                                plot_shade(avg_data["power_time_array"],avg_data["high_gamma"],sem_data["high_gamma"],1, ax5,"#1f77b4")
                                plot_shade(avg_data["power_time_array"],avg_data["low_gamma"],sem_data["low_gamma"],1, ax6,"#1f77b4")
                                plot_shade(avg_data["power_time_array"], avg_data["delta"], sem_data["delta"],1, ax4, "#1f77b4")



                                # if "breath_tp" in avg_data:
                                if "breathing_rate" in avg_data and np.any(avg_data["breathing_rate"] != 0):
                                    plot_timeseries(avg_data["breath_tp"], avg_data["breathing_rate"], 1, ax7, "gray",1.5, None, "Breathing rate (BPM)", (100,400), None)
                                    plot_shade(avg_data["breath_tp"], avg_data["breathing_rate"], sem_data["breathing_rate"],1, ax7, "#1f77b4")
                                if "table_velocity" in avg_data and np.any(avg_data["table_velocity"] != 0):
                                    plot_timeseries(avg_data["table_tp"], avg_data["table_velocity"], 1, ax8, "gray",2.5, None, "Velocity (mm/s)", (-50,250), None)
                                if "hr_tp" in avg_data and "heartrate" in avg_data:
                                    if np.any(avg_data["heartrate"] != 0):
                                        plot_timeseries(avg_data["hr_tp"], avg_data["heartrate"], 1, ax9, "gray",1.5, None, "Heartrate (BPM)", (400,800), None)
                                        plot_shade(avg_data["hr_tp"], avg_data["heartrate"],sem_data["heartrate"], 1, ax9, "#1f77b4")
                                # plot_heatmap(ax2, avg_data["t_stft"], avg_data["f_stft"], 10 * np.log10(avg_data["power_spectrum"] + 1e-10), "STFT dB Power", "Frequency (Hz)", 100, "rainbow", [-10, 33])
                                # plot_heatmap(ax2, avg_data["t_stft"], avg_data["f_stft"], 10 * np.log10(avg_data["power_spectrum"] + 1e-10),
                                #              "STFT dB Power", "Frequency (Hz)", 80, "rainbow", [0, 35]) #[-5, 8]
                                im = plot_heatmap(ax2,avg_data["t_stft"],avg_data["f_stft"], 10 * np.log10(avg_data["power_spectrum"] + 1e-10),
                                    "STFT dB Power","Frequency (Hz)",80,"rainbow",time, True, 1, 38
                                )
                                heatmap_png_path = os.path.join(
                                    group_analysis_dir,
                                    f"{align_type}_{group}_{state}_{int(time[0])}s_{int(time[1])}s_PETH_{electrode_name}_{type}_heatmap.png"
                                )
                                fig_heatmap= plt.figure(figsize=(8, 5))
                                ax_heat = fig_heatmap.add_axes([0, 0, 1, 1])
                                plot_heatmap(
                                    ax_heat,
                                    avg_data["t_stft"],
                                    avg_data["f_stft"],
                                    10 * np.log10(avg_data["power_spectrum"] + 1e-10),
                                    "", #f"{group} {state} {type}",
                                    "",
                                    80,
                                    "rainbow",
                                    time,
                                    False,
                                    1,
                                    38
                                )
                                # ax_heat.set_xlabel("Time (sec)")
                                # ax_heat.set_ylabel("Frequency (Hz)")

                                ax_heat.set_axis_off()  # これで軸・目盛り・枠まとめて非表示
                                # 念のため（不要だが安全策）
                                ax_heat.set_xticks([]);
                                ax_heat.set_yticks([])
                                for spine in ax_heat.spines.values():
                                    spine.set_visible(False)

                                fig_heatmap.savefig(heatmap_png_path, dpi=300, bbox_inches='tight', pad_inches=0)
                                plt.close(fig_heatmap)
                                for ax in axes:
                                    # ax.set_xlabel("Time (sec)")
                                    ax.margins(x=0)
                        if plot:
                            plt.tight_layout()
                            plt.legend(fontsize=1, labelspacing=0.1, handlelength=0.9, handletextpad=0.2, borderpad=0.2,ncol=2, loc="upper left")
                            pdf_path = os.path.join(group_analysis_dir, align_type+"_"+group+"_"+state+"_"+str(time[0])+"s_"+str(time[1])+"s_PETH_"+electrode_name+".pdf")
                            with PdfPages(pdf_path) as pdf:
                                pdf.savefig(fig, dpi=300)
                            plt.close(fig)

def main():
    # json_path = select_json_path()
    # json_path =r"X:\Behavior\Turntable_EEG\_Group_Analysis_PETH\__group_analysis_param_Turntable_PETH.json"

    # json_path = r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH\__group_analysis_param_PETH_MVConly_0820.json"
    # json_path = r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH\__group_analysis_param_PETH_MVConly_0924.json"
    json_path = r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH\__group_analysis_param_PETH_0926.json"
    group_dict, electrode_dict, state_dict, PETH_time, DLC_type, Bargraph_time = extract_group_analysis_params(json_path)
    process_group(json_path, group_dict, electrode_dict, state_dict, PETH_time, DLC_type, Bargraph_time)


if __name__ == "__main__":
    main()