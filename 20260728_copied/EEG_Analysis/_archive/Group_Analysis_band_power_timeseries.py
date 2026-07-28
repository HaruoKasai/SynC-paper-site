import pandas as pd
from _archive.EEG_Analysis import plot_timeseries, calculate_spectrum, calculate_band_power
import os
os.system('cls')
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from Group_Analysis_timeseries import event_mask
import json
import h5py

def open_h5_selective(h5_path, needed_keys=['all_analog_tp', 'all_eeg', 'sampling_rate']):
    """H5ファイルから必要なデータだけを選択的に読み込む"""
    data = {}
    with h5py.File(h5_path, "r") as f:
        if 'all_analog_tp' in needed_keys:
            data['analog_tp'] = f["all_analog_tp"][:]
        if 'all_eeg' in needed_keys:
            data['eeg'] = f["all_eeg"][:]
        if 'sampling_rate' in needed_keys:
            data['sampling_rate'] = f["sampling_rate"][()]
    
    # event_dfの読み込み
    try:
        data['event_df'] = pd.read_hdf(h5_path, key="event_df")
    except (KeyError, FileNotFoundError):
        data['event_df'] = None
        
    return data



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

def plot_timeseries_power_v2(eeg, analog_tp, sampling_rate, time_bin, ax_list, lw, legend =True, dB=True):
    columns = ['Delta (0.5-4 Hz)', 'Theta (4-8 Hz)', 'Alpha (8-12 Hz)', 'Beta (12-30 Hz)', 'Gamma (30-100 Hz)', 'High gamma (60-100Hz)', 'Low gamma (30-60Hz)']
    # powers = {col: [] for col in columns}
    time_bins = int(len(analog_tp) / (sampling_rate * time_bin))
    powers = {col: np.zeros(time_bins) for col in columns}   

    for t in range(time_bins):
        start = analog_tp[0]+ t*time_bin
        end = start + time_bin
        mask = (analog_tp >= start) & (analog_tp < end)
        segment = eeg[mask]
        freqs, power_spectrum = calculate_spectrum(segment, sampling_rate)

        powers['Delta (0.5-4 Hz)'][t] = calculate_band_power(freqs, power_spectrum, 0.5, 4, to_db=False)
        powers['Theta (4-8 Hz)'][t] = calculate_band_power(freqs, power_spectrum, 4, 8, to_db=False)
        powers['Alpha (8-12 Hz)'][t] = calculate_band_power(freqs, power_spectrum, 8, 12, to_db=False)
        powers['Beta (12-30 Hz)'][t] = calculate_band_power(freqs, power_spectrum, 12, 30, to_db=False)
        powers['Gamma (30-100 Hz)'][t] = calculate_band_power(freqs, power_spectrum, 30, 100, to_db=False)
        powers['High gamma (60-100Hz)'][t] = calculate_band_power(freqs, power_spectrum, 60, 100, to_db=False)
        powers['Low gamma (30-60Hz)'][t] = calculate_band_power(freqs, power_spectrum, 30, 60, to_db=False)

    df_linear= pd.DataFrame(powers)

    # df.to_csv(os.path.join(output_dir, "power_time_series.csv"), index=False)
    if dB==True:
        df = 10 * np.log10(df_linear + 1e-10)
    else:
        df = df_linear
    for c,col in enumerate(columns[:5]): #high/low gammaはplotしない
        ax = ax_list[c]
        if ax is not None:
            ax.plot(analog_tp[0]+time_bin/2 + np.arange(len(df)) * time_bin, df[col], label=col, lw=lw)
            ax.set_title("Band Power Time Series")
            ax.set_ylabel("Power (dB)")
            ax.set_ylim(55, 85)
            if legend:
                ax.legend(loc="upper right")

    return powers #後のaverage計算用に、DB化していないものを出力

def process_single_h5(args):
    """単一のH5ファイルを処理する関数（並列処理用）"""
    h5_file, time_bin, max_tp = args

    try:
        # 必要なデータだけを選択的に読み込む
        h5_data = open_h5_selective(h5_file)
        analog_tp = h5_data['analog_tp']
        eeg = h5_data['eeg']
        sampling_rate = h5_data['sampling_rate']
        event_df = h5_data['event_df']
        
        # EEGを一つに絞る
        eeg = eeg[0]
        eeg = event_mask(analog_tp, eeg, event_df, extra_sec=0)
        
        # パワー計算（プロットなし）
        powers = plot_timeseries_power_v2(eeg, analog_tp, sampling_rate, time_bin=time_bin, 
                                        ax_list=[None,None,None,None,None], lw=0.4, legend=False, dB=False)
        
        tp = np.arange(analog_tp[0]+time_bin/2, analog_tp[-1], time_bin)
        gamma = np.array(powers['Gamma (30-100 Hz)'])
        
        # max_tpの範囲に合わせてgamma_arrayを作成
        gamma_result = np.full(len(max_tp), np.nan)
        inds = np.searchsorted(max_tp, tp)
        valid = (inds >= 0) & (inds < len(max_tp))
        gamma_result[inds[valid]] = gamma[valid]
        
        return gamma_result
        
    except Exception as e:
        print(f"Error processing {h5_file}: {e}")
        return np.full(len(max_tp), np.nan)




def process_group(json_path, group_dict, electrode_dict, state_dict, PETH_time_list, dlc_type):
    dir = os.path.dirname(os.path.dirname(json_path))
    group_analysis_dir = os.path.dirname(json_path)
    # param_name = os.path.basename(json_path)[22:-5]
    band_list = ["delta", "theta", "alpha", "beta", "gamma", "high_gamma", "low_gamma"]

    # for elec_num, (electrode_name, electrode_list) in enumerate(electrode_dict.items()):

    group_num = len(group_dict.items())
    fig = plt.figure(figsize=(25, 25))
    gs = gridspec.GridSpec(2, group_num)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # time_binを先に定義
    time_bin = 10
    max_tp = np.arange(-7200+time_bin/2, 36000, time_bin)
    
    for g, (group, exp_list) in enumerate(group_dict.items()):
        h5_files = []
        axes = [fig.add_subplot(gs[i, g]) for i in range(2)]
        ax0, ax1= axes

        for exp_name in exp_list:
            combined_dir = os.path.join(dir, exp_name, "_Combined")
            path = os.path.join(combined_dir, "data.h5")
            if os.path.isfile(path):
                h5_files.append(path)
                # break
        
        # 並列処理でH5ファイルを処理
        print(f"Processing {len(h5_files)} files in parallel...")
        
        # 引数リストを作成
        args_list = [(h5_file, time_bin, max_tp) for h5_file in h5_files]
        
        # 並列処理を実行
        with Pool(processes=min(cpu_count(), len(h5_files))) as pool:
            gamma_results = pool.map(process_single_h5, args_list)
        
        # 結果を配列に変換
        gamma_array = np.array(gamma_results)

        mean_gamma = np.nanmean(gamma_array, axis=0)
        for i in range(len(gamma_array)):
            plot_timeseries(max_tp, gamma_array[i], window_size=1, ax=ax0, color=plt.get_cmap("tab10")(i), lw=0.1, title=None, ylabel=None, ylim=(0, 3e7))
        plot_timeseries(max_tp, mean_gamma, window_size=1, ax=ax0, color='#1f77b4', lw=0.1, title=None, ylabel="Gamma power", ylim=(0, 3e7))

            
    pdf_path = r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH\_timeseries.pdf"        
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)    
    

        

def main():
    json_path = r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH\__group_analysis_param_PETH_0415.json"
    # json_path = r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH\__group_analysis_param_PETH_0415_temp.json"
    group_dict, electrode_dict, state_dict, PETH_time, DLC_type = extract_group_analysis_params(json_path)
    process_group(json_path, group_dict, electrode_dict, state_dict, PETH_time, DLC_type)


if __name__ == "__main__":
    main()