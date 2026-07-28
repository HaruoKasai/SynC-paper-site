import pandas as pd
from EEG_Analysis import extract_params
import os
os.system('cls')
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from PETH import open_h5
import json


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


def emg_rms_aligned(emg: np.ndarray, analog_tp: np.ndarray, t_stft: np.ndarray) -> np.ndarray:
    """
    EMG を t_stft の各 bin に合わせて RMS を計算して返す（len = len(t_stft)）
    - emg, analog_tp を 1D に正規化
    - t_stft が等間隔でなくても OK（隣接中心の中点を境界にする）
    """
    # 1) 形状を揃える
    emg = np.asarray(emg).squeeze()
    analog_tp = np.asarray(analog_tp).squeeze()
    t_stft = np.asarray(t_stft).squeeze()

    if emg.ndim != 1:
        raise ValueError(f"emg must be 1D after squeeze; got shape {emg.shape}")
    if analog_tp.ndim != 1:
        raise ValueError(f"analog_tp must be 1D after squeeze; got shape {analog_tp.shape}")

    # emg 長さと analog_tp 長さを合わせる（チャンネル×時間だった場合の救済）
    if emg.shape[0] != analog_tp.shape[0]:
        # もし emg が (C, N) で残っているなら N を合わせる
        if emg.ndim == 1:
            pass
        else:
            # ここには基本来ないが、保険として例外
            raise ValueError(f"Length mismatch: emg({emg.shape[0]}) vs analog_tp({analog_tp.shape[0]})")

    # 2) t_stft の bin 境界を作る（可変幅対応）
    #    中点を境界に： edges[i] = (t_i + t_{i-1})/2
    edges = np.empty(len(t_stft) + 1, dtype=float)
    if len(t_stft) == 1:
        # 幅が不明な単一点の場合は適当な幅（例えば1秒）を仮定
        dt = 1.0
        edges[0] = t_stft[0] - dt / 2
        edges[1] = t_stft[0] + dt / 2
    else:
        centers = t_stft
        mid = (centers[:-1] + centers[1:]) / 2.0
        edges[1:-1] = mid
        # 端の外挿幅は隣の差分を使用
        left_dt = centers[1] - centers[0]
        right_dt = centers[-1] - centers[-2]
        edges[0] = centers[0] - left_dt / 2.0
        edges[-1] = centers[-1] + right_dt / 2.0

    # 3) searchsorted で各 bin の範囲インデックスを得る
    # analog_tp は単調増加を想定
    idx_st = np.searchsorted(analog_tp, edges[:-1], side='left')
    idx_en = np.searchsorted(analog_tp, edges[1:], side='left')

    # 4) 各 bin の RMS を計算
    rms = np.full(len(t_stft), np.nan, dtype=float)
    for i, (s, e) in enumerate(zip(idx_st, idx_en)):
        if e > s:
            seg = emg[s:e]  # ← ここでもう形状不一致は起きない
            rms[i] = np.sqrt(np.mean(seg ** 2))

    return rms

def calculate_band_power_v2(freqs, power_spectrum, lower_bound, upper_bound, normalize=False, to_db=False):
    power_spectrum = np.nanmean(power_spectrum, axis=1)
    band_mask = (freqs >= lower_bound) & (freqs < upper_bound)
    band_power = np.sum(power_spectrum[band_mask])
    # print(band_power)
    if normalize:
        band_power = band_power / np.sum(power_spectrum) * 100
    if to_db:
        band_power = 10 * np.log10(band_power + 1e-10)
    return band_power


def process_group(json_path, group_dict):
    dir = os.path.dirname(os.path.dirname(json_path))
    group_analysis_dir = os.path.dirname(json_path)
    # param_name = os.path.basename(json_path)[22:-5]
    band_list = ["delta", "theta", "alpha", "beta", "gamma", "high_gamma", "low_gamma"]

    # for elec_num, (electrode_name, electrode_list) in enumerate(electrode_dict.items()):

    target_keys_list  = [["M1-Ce", "M1-V1"]] #,["V1-Ce"]
    for target_keys in target_keys_list:
        group_num = len(group_dict.items())
        fig = plt.figure(figsize=(25, 25))
        gs = gridspec.GridSpec(3, group_num)

        fig_s = plt.figure(figsize=(25, 25))
        gs_s = gridspec.GridSpec(6, group_num)

        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        for g, (group, exp_list) in enumerate(group_dict.items()):
            h5_files = []
            axes = [fig.add_subplot(gs[i, g]) for i in range(3)]
            axes_s = [fig_s.add_subplot(gs_s[i, g]) for i in range(6)]

            for exp_name in exp_list:
                combined_dir = os.path.join(dir, exp_name, "_Combined")
                path = os.path.join(combined_dir, "data.h5")
                if os.path.isfile(path):
                    h5_files.append(path)
                    # break

            time_bin = 10
            states = ['awake', 'interimC', 'stateC','nrem','awake_before','awake_after',]
    
            # states = ['awake', 'nrem','rem', 'interimC', 'stateC']
            bands = {'gamma': (30, 80), 'delta': (0.5, 4)}
            results = {state: {b: [] for b in bands} for state in states}
            results_emg ={state: [] for state in states}
            spectra_by_state = {state: [] for state in states}

            for h5_file in h5_files:
                print(f"Processing: {h5_file}")
                data = open_h5(h5_file)
                analog_tp = data[1]
                t_stft = data[3]
                f_stft = data[4]
                print(f_stft)
                emg = data[7][0]
                print(emg.shape)
                emg_rms = emg_rms_aligned(emg, analog_tp, t_stft)

                _,_,EEG_ch_dict, *_  = extract_params(os.path.dirname(os.path.dirname(h5_file)))
                eeg_keys = list(EEG_ch_dict.keys())
                EEG_ch = next((i for i, k in enumerate(eeg_keys) if k in target_keys), None)
                linear_power = data[5][EEG_ch]

                freq_mask = f_stft <= 80
                f_stft = f_stft[freq_mask]
                linear_power = linear_power[freq_mask, :]

                low_freq_mask = (f_stft >= 0) & (f_stft < 2)
                lf_power = np.sum(linear_power[low_freq_mask], axis=0)
                threshold = np.mean(lf_power) + 3 * np.std(lf_power)
                above_threshold = lf_power > threshold
                x = 5  #3sdをこえたところの前後x秒もnanにする
                artifact_mask = binary_dilation(above_threshold, structure=np.ones(2 * x + 1))
                # linear_power[:, artifact_mask] = np.nan

                df = pd.read_csv(os.path.join(os.path.dirname(h5_file), "manual_event.csv"))
                exclude = os.path.join(os.path.dirname(h5_file), "exclude.csv")
                if os.path.exists(exclude):
                    print("exclude path", exclude)
                    exclude_df = pd.read_csv(exclude)
                    exclude_mask = np.zeros_like(t_stft, dtype=bool)
                    for _, row in exclude_df.iterrows():
                        start, end = row['start_time'], row['end_time']
                        print(start)
                        if row['event_name'] == 'exclude':
                            exclude_mask |= (t_stft >= start) & (t_stft < end)
                    linear_power[:, exclude_mask] = np.nan



                #TODO stftは1秒ごと　binの切り方を確認。場合によって、最初の1秒切る
                nrem_mask = np.zeros_like(t_stft, dtype=bool)
                statec_mask = np.zeros_like(t_stft, dtype=bool)
                rem_mask = np.zeros_like(t_stft, dtype=bool)

                for _, row in df.iterrows():
                    start, end = row['start_time'], row['end_time']
                    if row['event_name'] == 'NREM':
                        nrem_mask |= (t_stft >= start) & (t_stft < end)
                    elif row['event_name'] == 'StateC':
                        statec_mask |= (t_stft >= start) & (t_stft < end)
                    elif row['event_name'] == 'REM':
                        rem_mask = (t_stft >= start) & (t_stft < end)
                motive_mask = ~(nrem_mask | statec_mask | rem_mask )

                motive_artifact_mask = artifact_mask & motive_mask
                linear_power[:, motive_artifact_mask] = np.nan


                awake_before_mask = motive_mask & (t_stft < 0)

                interimc_mask = np.zeros_like(t_stft, dtype=bool)
                awake_after_mask = np.zeros_like(t_stft, dtype=bool)

                statec_times = df[df['event_name'] == 'StateC'][['start_time', 'end_time']].values
                for (end1, start2) in zip(statec_times[:-1, 1], statec_times[1:, 0]):
                    interimc_mask |= motive_mask & (t_stft >= end1) & (t_stft < start2) & (t_stft >= 0)

                # nrem_times = df[df['event_name'] == 'NREM'][['start_time', 'end_time']].values
                # for (end1, start2) in zip(nrem_times[:-1, 1], nrem_times[1:, 0]):
                #     awake_after_mask |= motive_mask & (t_stft >= end1) & (t_stft < start2) & (t_stft >= 0)
                awake_after_mask = motive_mask & (t_stft >=0)
                awake_mask = awake_before_mask | awake_after_mask

                mask_dict = {
                    'awake': awake_mask,
                    'interimC': interimc_mask,
                    'stateC': statec_mask,
                    'nrem': nrem_mask,
                    'awake_before': awake_before_mask,
                    'awake_after': awake_after_mask,
                }
                # mask_dict = {
                #     'awake': awake_mask,
                #     'nrem': nrem_mask,
                #     'rem': rem_mask,
                #     'interimC': interimc_mask,
                #     'stateC': statec_mask
                # }


                for state, mask in mask_dict.items():
                    power = linear_power[:, mask]
                    mean_power = np.nanmean(power, axis=-1)  # shape = (freq,)
                    # if np.all(np.isnan(mean_power)):
                    #     continue
                    spectra_by_state[state].append(mean_power)

                    emg_rms_mean = np.nanmean(emg_rms[mask])
                    results_emg[state].append(emg_rms_mean)
                    for band, (fmin, fmax) in bands.items():
                        band_val = calculate_band_power_v2(f_stft, power, fmin, fmax)
                        results[state][band].append(band_val)



            print(results)
            df_rows = []
            for state in results:
                for band in results[state]:
                    for subj_idx, value in enumerate(results[state][band]):
                        df_rows.append([group, subj_idx, band, state, value])
                for subj_idx, value in enumerate(results_emg[state]):
                    df_rows.append([group, subj_idx, "emg", state, value])
            df = pd.DataFrame(df_rows, columns=["Group", "SubjectIndex", "Band", "State", "Power"])
            output_path = os.path.join(r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH", f"{group}_power_summary_{target_keys[0]}.csv")
            df.to_csv(output_path, index=False)

            for s, state in enumerate(spectra_by_state):
                ax_s = axes_s[s]
                for value in spectra_by_state[state]:
                    ax_s.plot(f_stft, value, alpha=0.3, linewidth=0.5)
                arr = np.vstack(spectra_by_state[state])
                mean_value = np.nanmean(arr, axis=0)
                sem_value = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
                ax_s.plot(f_stft, mean_value, linewidth=1.5, color="k")
                ax_s.fill_between(f_stft, mean_value - sem_value, mean_value + sem_value, alpha=0.3, color = "k")

                ax_s.set_xlabel("Frequency (Hz)")
                ax_s.set_ylabel("Power (μV²)")
                # ax_s.set_ylim(0,1200)
                ax_s.set_ylim(1e0, 1e4)  # log用に下限>0にするのが必須
                ax_s.set_yscale("log")
                ax_s.set_xlim(0, 80)
                ax_s.set_title(state)
                ax_s.legend()

            # プロット
            n_subj = len(h5_files)
            for i, band in enumerate(['gamma', 'delta', 'emg']):
                ax = axes[i]
                if band =='emg':
                    means = [np.nanmean(results_emg[state]) for state in states]
                    sems = [np.nanstd(results_emg[state]) / np.sqrt(np.sum(~np.isnan(results_emg[state]))) for state in states]
                else:
                    means = [np.nanmean(results[state][band]) for state in states]
                    sems = [np.nanstd(results[state][band]) / np.sqrt(np.sum(~np.isnan(results[state][band]))) for state in states]
                x = np.arange(len(states))
                ax.bar(x, means, yerr=sems, capsize=5,facecolor = "none", edgecolor = "black")

                for subj in range(n_subj):
                    if band == 'emg':
                        subj_vals = [results_emg[state][subj] if subj < len(results_emg[state]) else np.nan for state in states]
                    else:
                        subj_vals = [results[state][band][subj] if subj < len(results[state][band]) else np.nan for state in states]
                    label = os.path.basename(os.path.dirname(os.path.dirname(h5_files[subj])))[:15]
                    ax.plot(x, subj_vals,  marker ="none", linewidth = 0.5, color = plt.get_cmap("tab20")(subj), label=label) #marker=".", alpha=0.6,

                ax.legend(loc='best')

                if band =="gamma":
                    ax.set_ylim(0,2000)
                elif band =="delta":
                    ax.set_ylim(0,20000)
                elif band =="emg":
                    ax.set_ylim(0, 500)
                ax.set_xticks(x)
                ax.set_xticklabels(states, rotation=45)
                ax.set_ylabel('Power')
                ax.set_title(f'{band.capitalize()} Power (mean ± SEM)')

        # plt.tight_layout()
        # plt.legend(loc='best')

        pdf_path = os.path.join(r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH", "_power_summary_"+target_keys[0]+".pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, dpi=300)
        plt.close(fig)



        pdf_s_path = os.path.join(r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH","_spectrum_" + target_keys[0] + ".pdf")
        with PdfPages(pdf_s_path) as pdf:
            pdf.savefig(fig_s, dpi=300)
        plt.close(fig)

    

                

def main():
    # json_path = r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH\__group_analysis_param_PETH_0605.json"
    # json_path = r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH\__group_analysis_param_PETH_MVConly_0820.json"
    # json_path = r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH\__group_analysis_param_PETH_MVConly_0924.json"
    json_path = r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH\__group_analysis_param_PETH_0926.json"
    group_dict, electrode_dict, state_dict, PETH_time, DLC_type = extract_group_analysis_params(json_path)
    
    process_group(json_path, group_dict)


if __name__ == "__main__":
    main()