import pandas as pd
from _archive.EEG_Analysis import extract_params
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

    group_num = len(group_dict.items())
    fig = plt.figure(figsize=(25, 25))
    gs = gridspec.GridSpec(2, group_num)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for g, (group, exp_list) in enumerate(group_dict.items()):
        h5_files = []
        axes = [fig.add_subplot(gs[i, g]) for i in range(2)]

        for exp_name in exp_list:
            combined_dir = os.path.join(dir, exp_name, "_Combined")
            path = os.path.join(combined_dir, "data.h5")
            if os.path.isfile(path):
                h5_files.append(path)
                # break
        
        time_bin = 10 
        states = ['awake', 'awake_before','awake_after','nrem', 'interimC', 'stateC']
        bands = {'gamma': (30, 100), 'delta': (0, 4)}
        results = {state: {b: [] for b in bands} for state in states}

        for h5_file in h5_files:
            print(f"Processing: {h5_file}")
            data = open_h5(h5_file)
            analog_tp = data[1]
            t_stft = data[3]
            f_stft = data[4]

            _,_,EEG_ch_dict, *_  = extract_params(os.path.dirname(os.path.dirname(h5_file)))
            target_keys = ["M1-Ce", "M1-V1"]
            eeg_keys = list(EEG_ch_dict.keys())
            EEG_ch = next((i for i, k in enumerate(eeg_keys) if k in target_keys), None)
            linear_power = data[5][EEG_ch]

            low_freq_mask = (f_stft >= 0) & (f_stft < 2)
            lf_power = np.sum(linear_power[low_freq_mask], axis=0)
            threshold = np.mean(lf_power) + 3 * np.std(lf_power)
            above_threshold = lf_power > threshold
            x = 5  #3sdをこえたところの前後x秒もnanにする
            artifact_mask = binary_dilation(above_threshold, structure=np.ones(2 * x + 1))
            # linear_power[:, artifact_mask] = np.nan

            df = pd.read_csv(os.path.join(os.path.dirname(h5_file), "manual_event.csv"))




            #TODO stftは1秒ごと　binの切り方を確認。場合によって、最初の1秒切る
            nrem_mask = np.zeros_like(t_stft, dtype=bool)
            statec_mask = np.zeros_like(t_stft, dtype=bool)

            for _, row in df.iterrows():
                start, end = row['start_time'], row['end_time']
                if row['event_name'] == 'NREM':
                    nrem_mask |= (t_stft >= start) & (t_stft < end)
                elif row['event_name'] == 'StateC':
                    statec_mask |= (t_stft >= start) & (t_stft < end)
            motive_mask = ~(nrem_mask | statec_mask)

            motive_artifact_mask = artifact_mask & motive_mask
            linear_power[:, motive_artifact_mask] = np.nan


            awake_before_mask = motive_mask & (t_stft < 0)

            interimc_mask = np.zeros_like(t_stft, dtype=bool) 
            awake_after_mask = np.zeros_like(t_stft, dtype=bool) 

            statec_times = df[df['event_name'] == 'StateC'][['start_time', 'end_time']].values
            for (end1, start2) in zip(statec_times[:-1, 1], statec_times[1:, 0]):
                interimc_mask |= motive_mask & (t_stft >= end1) & (t_stft < start2) & (t_stft >= 0)

            nrem_times = df[df['event_name'] == 'NREM'][['start_time', 'end_time']].values
            for (end1, start2) in zip(nrem_times[:-1, 1], nrem_times[1:, 0]):
                awake_after_mask |= motive_mask & (t_stft >= end1) & (t_stft < start2) & (t_stft >= 0)

            awake_mask = awake_before_mask | awake_after_mask

            mask_dict = {
                'awake': awake_mask,
                'awake_before': awake_before_mask,
                'awake_after': awake_after_mask,
                'nrem': nrem_mask,
                'interimC': interimc_mask,
                'stateC': statec_mask
            }
            for state, mask in mask_dict.items():
                power = linear_power[:, mask]
                for band, (fmin, fmax) in bands.items():    
                    band_val = calculate_band_power_v2(f_stft, power, fmin, fmax)
                    results[state][band].append(band_val)

        print(results)
        df_rows = []
        for state in results:
            for band in results[state]:
                for subj_idx, value in enumerate(results[state][band]):
                    df_rows.append([group, subj_idx, band, state, value])
        df = pd.DataFrame(df_rows, columns=["Group", "SubjectIndex", "Band", "State", "Power"])
        output_path = os.path.join(r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH", f"{group}_power_summary.csv")
        df.to_csv(output_path, index=False)



        # プロット
        n_subj = len(h5_files)
        for i, band in enumerate(['gamma', 'delta']):
            ax = axes[i]
            means = [np.nanmean(results[state][band]) for state in states]
            sems = [np.nanstd(results[state][band]) / np.sqrt(np.sum(~np.isnan(results[state][band]))) for state in states]
            x = np.arange(len(states))

            ax.bar(x, means, yerr=sems, capsize=5,facecolor = None, edgecolor = "black")
            for subj in range(n_subj):
                subj_vals = [results[state][band][subj] if subj < len(results[state][band]) else np.nan for state in states]
                label = os.path.basename(os.path.dirname(os.path.dirname(h5_files[subj])))[:15]
                ax.plot(x, subj_vals, marker=".", alpha=0.6, linewidth = 0.5, color = plt.get_cmap("tab10")(subj), label=label)
            
            ax.legend(loc='best')

            if band =="gamma":
                ax.set_ylim(0,2000)
            elif band =="delta":
                ax.set_ylim(0,20000)
            ax.set_xticks(x)
            ax.set_xticklabels(states, rotation=45)
            ax.set_ylabel('Power')
            ax.set_title(f'{band.capitalize()} Power (mean ± SEM)')

    # plt.tight_layout()
    # plt.legend(loc='best')

    pdf_path = r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH\_power_summary.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)
    

                

def main():
    json_path = r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH\__group_analysis_param_PETH_0605.json"
    group_dict, electrode_dict, state_dict, PETH_time, DLC_type = extract_group_analysis_params(json_path)
    
    process_group(json_path, group_dict)


if __name__ == "__main__":
    main()