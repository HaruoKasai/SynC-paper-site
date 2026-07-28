from joblib import Parallel, delayed
import os
import glob
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from pyqtgraph.examples.DateAxisItem_QtDesigner import window
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12   })
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter, stft, find_peaks
import scipy.signal as signal
from matplotlib.backends.backend_pdf import PdfPages
from neo.io import BlackrockIO
from skimage.measure import EllipseModel
import json
import multiprocessing
# from functools import partial
import lib.DLCAnalysis as DA
import h5py


def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select the 'data' directory", initialdir=r"X:\Behavior")
    root.destroy()
    return folder_path

def rotary_analysis(A, B, Z, exp_duration, time_bin=1, radius=50, resolution=100):
    sampling_rate = 2000  # 2000 Hz
    num_rows = int(exp_duration * sampling_rate)
    signal = np.zeros((num_rows, 2), dtype=int)

    A_indices = (np.array(A) * sampling_rate).astype(int)
    B_indices = (np.array(B) * sampling_rate).astype(int)

    signal[A_indices, 0] = 1
    signal[B_indices, 1] = 1

    def toggle_pattern(array, toggle_points):
        toggle_state = np.zeros_like(array)
        toggle_state[toggle_points] = 1
        cumulative_toggles = np.cumsum(toggle_state)
        return cumulative_toggles % 2

    signal[:, 0] = toggle_pattern(signal[:, 0], A_indices)
    signal[:, 1] = toggle_pattern(signal[:, 1], B_indices)

    A_changes = (np.diff(signal[:, 0], prepend=0) == 1).astype(int)
    B_values = signal[:, 1]
    bin_size = int(time_bin * sampling_rate)
    A_transition_indices = np.where(A_changes == 1)[0]
    B_values_at_transitions = B_values[A_transition_indices]

    bin_edges = np.arange(0, num_rows + 1, bin_size)
    bin_indices = np.digitize(A_transition_indices, bin_edges) - 1

    bin_counts = np.bincount(bin_indices, weights=(1 - 2 * B_values_at_transitions), minlength=len(bin_edges) - 1)
    time_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 / sampling_rate

    circumference = 2 * np.pi * radius  # mm
    mm_per_resolution = circumference / resolution

    velocities = bin_counts * mm_per_resolution / time_bin  # velocity in mm/s
    if np.sum(velocities) <= 0:
        velocities = -1 * velocities  # 顺向方向修正
    return velocities, time_centers

def plot_binned_timeseries(data, time_centers, ylabel, ax):
    ax.grid(False)
    ax.plot(time_centers, data, lw=1)
    xticks = np.arange(0, time_centers[-1], 30)
    xtick_labels = np.arange(0, time_centers[-1], 30)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-400, 400)
    bin = time_centers[1] - time_centers[0]
    ax.set_xlim(time_centers[0] - bin / 2, time_centers[-1] + bin / 2)
    ax.margins(x=0)
    return ax


def extract_raw_data(file_path, EEG_ch_dict, EMG_ch_dict, analog_dict, cont_time):
    reader = BlackrockIO(filename=file_path)
    block = reader.read_block()
    raw_signals = [seg.analogsignals[0] for seg in block.segments]
    Ch0_signal = raw_signals[0][:, 0]
    sr = int(Ch0_signal.sampling_rate.magnitude)
    exp_dur = (cont_time[1] - cont_time[0]) * 60
    EEG_ch_list = [[v - 1 if v is not None else None for v in values] for values in EEG_ch_dict.values()]
    EMG_ch_list = [[v - 1 if v is not None else None for v in values] for values in EMG_ch_dict.values()]

    # EEG
    Rec_signals = np.array([raw_signals[0][:, ch[0]].magnitude.flatten() for ch in EEG_ch_list])
    Ref_signals = np.array([raw_signals[0][:, ch[1]].magnitude.flatten() if ch[1] is not None else np.zeros_like(
        raw_signals[0][:, ch[0]].magnitude.flatten()) for ch in EEG_ch_list])
    eeg_signals = Rec_signals - Ref_signals
    eeg_signals = eeg_signals[:, :exp_dur * sr]

    # EMG
    Rec_signals = np.array([raw_signals[0][:, ch[0]].magnitude.flatten() for ch in EMG_ch_list])
    Ref_signals = np.array([raw_signals[0][:, ch[1]].magnitude.flatten() if ch[1] is not None else np.zeros_like(
        raw_signals[0][:, ch[0]].magnitude.flatten()) for ch in EMG_ch_list])
    emg_signals = Rec_signals - Ref_signals
    emg_signals = emg_signals[:, :exp_dur * sr]

    start_time = raw_signals[0].t_start.magnitude
    digital_input_data = None
    channel_times = {}
    breathe = None
    tem = None

    try:
        # analog
        analog_input_signals = [seg.analogsignals[1] for seg in block.segments]
        breathe = analog_input_signals[0][:, analog_dict["Breathing"]-1].magnitude.flatten()
        tem = analog_input_signals[0][:, analog_dict["Temperature"]-1].magnitude.flatten()
        breathe = breathe[:exp_dur * sr]
        tem = tem[:exp_dur * sr]

    except:
        pass

    for segment in block.segments:
        for event_array in segment.events:
            if event_array.name == 'digital_input_port':
                digital_input_data = event_array
    try:
        times = digital_input_data.times.rescale('s').magnitude
        labels = digital_input_data.labels.astype(int)
        num_channels = 16
        bitwise_states = np.array([[(label >> i) & 1 for i in range(num_channels)] for label in labels])
        channel_times = {f'channel_{i}': [] for i in range(num_channels)}
        for i in range(num_channels):
            changes = np.where(np.diff(bitwise_states[:, i]) != 0)[0] + 1
            channel_times[f'channel_{i}'] = times[changes] - start_time
        channel_times = {key: values[values <= exp_dur] for key, values in channel_times.items()}
    except:
        pass
    return emg_signals, eeg_signals, sr, breathe, tem, channel_times, digital_input_data

def calculate_band_power(freqs, power_spectrum, lower_bound, upper_bound, normalize=False, to_db=False):
    # TODO 計算正しく直す。cf. Memorandoms 2025.09.22 /  EEG_analysis_20250922.py
    band_mask = (freqs >= lower_bound) & (freqs < upper_bound)
    band_power = np.sum(power_spectrum[band_mask])

    if normalize:
        band_power = band_power / np.sum(power_spectrum) * 100
    if to_db:
        band_power = 10 * np.log10(band_power + 1e-10)
    return band_power

def plot_timeseries_power(eeg, analog_tp, sampling_rate, time_bin, ax_list, lw, legend =True, dB=True):
    columns = ['Delta (0.5-4 Hz)', 'Theta (4-8 Hz)', 'Alpha (8-12 Hz)', 'Beta (12-30 Hz)', 'Gamma (30-80 Hz)', 'High gamma (60-100Hz)', 'Low gamma (30-60Hz)']
    powers = {col: [] for col in columns}
    time_bins = int(len(eeg) / (sampling_rate * time_bin))

    for t in range(time_bins):
        start = t * sampling_rate * time_bin
        end = start + sampling_rate * time_bin
        segment = eeg[start:end]
        freqs, power_spectrum = calculate_spectrum(segment, sampling_rate)

        powers['Delta (0.5-4 Hz)'].append(calculate_band_power(freqs, power_spectrum, 0.5, 4, to_db=False))
        powers['Theta (4-8 Hz)'].append(calculate_band_power(freqs, power_spectrum, 4, 8, to_db=False))
        powers['Alpha (8-12 Hz)'].append(calculate_band_power(freqs, power_spectrum, 8, 12, to_db=False))
        powers['Beta (12-30 Hz)'].append(calculate_band_power(freqs, power_spectrum, 12, 30, to_db=False))
        powers['Gamma (30-80 Hz)'].append(calculate_band_power(freqs, power_spectrum, 30, 100, to_db=False))
        powers['High gamma (60-100Hz)'].append(calculate_band_power(freqs, power_spectrum, 60, 100, to_db=False))
        powers['Low gamma (30-60Hz)'].append(calculate_band_power(freqs, power_spectrum, 30, 60, to_db=False))

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

# def calculate_band_power_v2 (linear_power, f_stft):
#     bands = {
#         'Delta (0.5-4 Hz)': (0.5, 4),
#         'Theta (4-8 Hz)': (4, 8),
#         'Alpha (8-12 Hz)': (8, 12),
#         'Beta (12-30 Hz)': (12, 30),
#         'Gamma (30-100 Hz)': (30, 100),
#     }
#     band_powers = {}
#
#     for band, (f_low, f_high) in bands.items():
#         idx = np.where((f_stft >= f_low) & (f_stft < f_high))[0]  # 該当周波数のインデックス取得
#         band_powers[band] = 10 * np.log10(np.sum(linear_power[idx, :], axis=0) + 1e-10)
#
#     return band_powers

def calculate_spectrum(eeg, sampling_rate):
    n = len(eeg)
    eeg -= np.mean(eeg)
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)
    fft_vals = np.fft.fft(eeg)
    power_spectrum = np.abs(fft_vals) ** 2 / n
    return freqs[:n // 2], power_spectrum[:n // 2]


def plot_heatmap(ax, t, f, power, title, ylabel, freq_limit, cmap, power_range):
    pcm = ax.pcolormesh(
        t, f, power, shading='gouraud', rasterized=True, cmap=cmap,
        vmin=power_range[0], vmax=power_range[1]
    )
    ax.set_title(title)
    # ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, freq_limit)
    # cbar = plt.colorbar(pcm, ax=ax)
    # cbar.set_label('Power' + (' (µV²)' if 'Linear' in title else ' (dB)'))

def adaptive_threshold(signal, window_size, sigma_factor):
    if len(signal) < window_size:
        return np.full_like(signal, np.mean(signal))
    rolling_std = pd.Series(signal).rolling(window=window_size, min_periods=1).std().to_numpy()
    return np.mean(signal) + sigma_factor * rolling_std

def compute_breathing_frequency(breathe, sampling_rate, window_size=2):
    breathe = breathe - np.mean(breathe)

    nyquist = 0.5 * sampling_rate
    b, a = signal.butter(2, [1 / nyquist, 10 / nyquist], btype='band')
    filtered_signal = signal.filtfilt(b, a, breathe)

    threshold_values = adaptive_threshold(filtered_signal, int(sampling_rate * 2), sigma_factor=0.25)

    min_peak_distance = int(sampling_rate * 0.1)  # 最小間隔0.1秒（最大10Hz）
    peaks, _ = signal.find_peaks(filtered_signal, distance=min_peak_distance)
    # 適応閾値を適用
    valid_peaks = peaks[filtered_signal[peaks] > threshold_values[peaks]]
    if len(valid_peaks) < 2:
        return np.array([]), np.array([])

    peak_intervals = np.diff(valid_peaks) / sampling_rate  # ピーク間の時間間隔（秒）
    breathing_rates = 60 / peak_intervals  # BPM (breaths per minute)
    timestamps = valid_peaks[1:] / sampling_rate  # 呼吸数が計算される時間（秒）

    # 移動平均の適用
    # if len(breathing_rates) >= 3:
    #     breathing_rates = uniform_filter1d(breathing_rates, size=3, mode='nearest')

    df = pd.DataFrame({'timestamp': timestamps, 'breathing_rate': breathing_rates})
    df['window'] = (df['timestamp'] // window_size).astype(int)
    result = df.groupby('window')['breathing_rate'].mean().reset_index()
    all_windows = pd.DataFrame({'window': range(df['window'].min(), df['window'].max() + 1)})
    result = all_windows.merge(result, on='window', how='left').fillna(0)
    result['timestamp'] = result['window'] * window_size

    return result['timestamp'].to_numpy(), result['breathing_rate'].to_numpy()

def calculate_heartrate(ecg_data, sampling_rate, bin_size):
    min_distance = int(0.05 * sampling_rate)
    sd =np.std(ecg_data)
    r_peaks, _ = find_peaks(ecg_data, distance=min_distance, prominence=sd*3)

    r_peak_times = r_peaks / sampling_rate
    rr_intervals = np.diff(r_peak_times)
    rr_times = r_peak_times[:-1] + rr_intervals / 2
    hr_values = 60.0 / rr_intervals

    total_duration = len(ecg_data) / sampling_rate
    bin_edges = np.arange(0, total_duration + bin_size, bin_size)
    bin_indices = np.digitize(rr_times, bin_edges) - 1

    num_bins = len(bin_edges) - 1
    sum_hr = np.bincount(bin_indices, weights=hr_values, minlength=num_bins)
    count_hr = np.bincount(bin_indices, minlength=num_bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        hr_timeseries = sum_hr / count_hr
        hr_timeseries[count_hr == 0] = np.nan  # データがないところをNaNに

    return hr_timeseries, bin_edges[:-1]+bin_size/2

def extract_params(json_dir):
    dlc_json = os.path.join(json_dir, "_analysis_param.json")
    with open(dlc_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
        dlc_type = data["DLC"].get("type", None)
        dlc_dir = data["DLC"].get("dir", None)
        EEG_ch_dict = data["EEG"]
        EMG_ch_dict = data["EMG"]
        analog_dict = data["Analog"]
        timer = data["Time"]["Timer"]
        contime = data["Time"]["Continuous"]
    return dlc_type, dlc_dir, EEG_ch_dict, EMG_ch_dict, analog_dict, contime

def pupillometry(dir):
    csv = glob.glob(os.path.join(dir, "*filtered.csv"))[0]
    df = pd.read_csv(csv, low_memory=False)

    num_points = 8
    x_indices = [1 + i * 3 for i in range(num_points)]
    y_indices = [x + 1 for x in x_indices]
    l_indices = [y + 1 for y in y_indices]

    time_steps = len(df) - 2  # Excluding the first two header rows
    coords_x = df.iloc[2:, x_indices].astype(float).values
    coords_y = df.iloc[2:, y_indices].astype(float).values
    likelihood = np.mean(df.iloc[2:, l_indices].astype(float).values, axis=1)

    def fit_ellipse(t):
        ellipse = EllipseModel()
        if ellipse.estimate(np.column_stack([coords_x[t], coords_y[t]])):
            _, _, a, b, _ = ellipse.params  # Major axis, minor axis
            return np.pi * (a / 2) * (b / 2)
        return np.nan

    # 並列処理 (num_jobs=-1 で CPU コア数を最大限活用)
    max_jobs = min(multiprocessing.cpu_count(), 63)
    areas = Parallel(n_jobs=max_jobs, backend="threading")(delayed(fit_ellipse)(t) for t in range(time_steps))
    areas = np.array(areas)

    #Extract real frame time in video
    df = pd.read_csv(os.path.join(dir, "_timestamp.csv"))
    df["time"] = pd.to_datetime(df["time"])
    real_frame_time = (df["time"].iloc[-1] - df["time"].iloc[0]) / (len(df) - 1)  # real frame time
    real_frame_time = round(pd.to_timedelta(real_frame_time).total_seconds(), 5)

    return areas, real_frame_time, likelihood

def Openfield(index, exp_dir, dlc_dir, event_df, start_time, velocity_boundary):
    try:
        arena_mm_per_pix = 0.6
        dlc_exp_dir = glob.glob(os.path.join(dlc_dir, "day*"))[index]
        dlc_h5_path = os.path.join(dlc_exp_dir, "dlc_raw.h5")
        param_ind = os.path.join(dlc_exp_dir, "param_individual.json")
        df = pd.read_hdf(dlc_h5_path, key='dlc_data')
        dlc_output_dir = os.path.join(exp_dir, "_DLC_analysis")
        if not os.path.exists(dlc_output_dir):
            os.makedirs(dlc_output_dir)
        df.to_csv(os.path.join(dlc_output_dir, "dlc_data.csv"))

        real_frame_time = (df["time"].iloc[-1] - df["time"].iloc[0]) / (len(df) - 1)  # real frame time
        real_frame_time = round(pd.to_timedelta(real_frame_time).total_seconds(), 5)
        velocity, cumulative_distance, frames_extracted_by_velocity, likelihood = DA.time_series_velocity(df, real_frame_time,arena_mm_per_pix,"centroid", velocity_boundary)

        for v in range(len(frames_extracted_by_velocity)):
            event_name = "Centroid~" + str(velocity_boundary[v]) + "mm_per_s"
            event_df = DA.frame_to_sec(frames_extracted_by_velocity[v], real_frame_time, event_df, event_name,start_time, tolerable_frame_drop=2, min_duration=10) #min_duration:sec

        arena_coordinate = DA.get_roi_coordinate("arena_box", param_ind=param_ind)
        distance_to_center, frame_approaching, frame_leaving = DA.time_series_distance_to_object(df, arena_coordinate,real_frame_time,arena_mm_per_pix,body_part="centroid",distance_to_boundary_mm=100)

        # event_df = DLCAnal.frame_to_sec_v2(frame_approaching, real_frame_time, event_df,
        #                                    event_name="Approaching (-5~0 sec)", before_sec=-5, after_sec=0,
        #                                    exp_duration=exp_duration)

        # event_df = DLCAnal.frame_to_sec_v2(frame_leaving, real_frame_time, event_df, event_name="Leaving (0-5 sec)",
        #                                    before_sec=0, after_sec=5, exp_duration=exp_duration)

        # DLCAnal.time_series_angle_to_object_direction(df, object_coordinate=coordinate, real_frame_time=real_frame_time,
        #                                               exp_duration=exp_duration, fig=fig, ax=(10, slice(0, 3)), gs=gs)
        event_df.to_csv(os.path.join(dlc_output_dir, "event.csv"))

        return real_frame_time, velocity, cumulative_distance, distance_to_center, event_df, likelihood
    except Exception as e:
        print("Something's wrong in processing openfield" )
        return None, None, None, None, None, None

def plot_timeseries(tp, data, window_size, ax, color, lw, title, ylabel, ylim, label=None, alpha=1):
    """
    tp, data: numpy
    window_size: int
    """
    valid_len = (len(data) // window_size) * window_size
    ave_data = data[:valid_len].reshape(-1, window_size).mean(axis=1)
    ave_tp = tp[:valid_len].reshape(-1, window_size).mean(axis=1)
    ax.set_ylim(ylim)
    ax.plot(ave_tp, ave_data,lw=lw, color=color, label = label, alpha = alpha)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if label:
        ax.legend(loc="upper right")

def binning (data, window_size):
    valid_len = (len(data) // window_size) * window_size
    ave_data = data[:valid_len].reshape(-1, window_size).mean(axis=1)
    return ave_data

def save_eeg_analysis_results(dlc_type, analog_tp,
                              eeg,t_stft, f_stft, linear_power,dB_power,
                              emg, ecg, sampling_rate, heartrate, hr_tp, breathe, tem, breathing_rate, pupil_size, pupil_tp,
                              table_velocities, table_tp,
                              OF_tp, velocity, cumulative_distance,distance,velocity_boundary, likelihood, event_df,manual_event,
                              output_dir, pdf_name, figsize):

    pdf_path = os.path.join(output_dir, pdf_name)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(10, 1, height_ratios=[0.4, 1, 1, 1, 1, 0.4, 1, 1, 1, 1])
    plt.subplots_adjust(hspace=0.5)

    total_time = len(eeg) / sampling_rate

    axes = [fig.add_subplot(gs[i, 0]) for i in range(10)]
    ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9= axes

    # 1. Raw EEG Signal
    plot_timeseries(analog_tp, eeg, window_size=1, ax=ax0, color='#555555', lw=0.1, title=pdf_name, ylabel="Amplitude (µV)", ylim=(-500, 500))

    # 2. STFT Linear Power
    plot_heatmap(ax1, t_stft, f_stft, linear_power, "STFT Linear Power", "Frequency (Hz)", 100, "rainbow", [-10, 100])

    # 3. STFT dB Power
    plot_heatmap(ax2, t_stft, f_stft, dB_power, "STFT dB Power", "Frequency (Hz)", 80, "rainbow", [-10, 33])

    # 4. STFT Low-Frequency dB Power)
    plot_heatmap(ax3, t_stft, f_stft, dB_power, "STFT dB Power (0-20 Hz)", "Frequency (Hz)", 20, "rainbow", [-10, 33])

    # 5. Band Power Time Series
    powers = plot_timeseries_power(eeg, analog_tp, sampling_rate, time_bin=2, ax_list=[ax4,ax4,ax4,ax4,ax4], lw=0.4)
    #上のstftの計算結果を使おうとしたが、計算法が違い、これまで(2025.2)のグラフとずれてしまうので、計算は一部かぶっていて時間はかかるが、ひとまずこのまま使う。

    # 6. Raw EMG Signal
    plot_timeseries(analog_tp, emg, window_size=1, ax=ax5, color='#555555', lw=0.1, title="Raw EMG Signal",ylabel="Amplitude (µV)", ylim=(-1000, 1000))

    if dlc_type == "Pupillometry":
        # 7. Breathe
        plot_timeseries(analog_tp, breathe, window_size=4, ax=ax7, color = 'k',lw=0.1, title="Breathe", ylabel="Amplitude (µV)", ylim=(1800, 2400))

        ax7_2 = ax7.twinx()
        window_size = total_time / len(breathing_rate)
        time = np.arange(window_size / 2, total_time, window_size) + analog_tp[0]
        ax7_2.plot(time, breathing_rate, label="Breathing_rate (BPM)", color="red", linewidth=0.5)
        ax7_2.set_ylabel("Breathing_rate (BPM)", color="red")
        ax7_2.tick_params(axis='y', labelcolor="red")
        ax7_2.set_ylim([40, 420])

        # 8 Pupillometry
        plot_timeseries(pupil_tp, pupil_size, window_size=4, ax=ax8, color='green', lw=0.2, title="Pupil size",ylabel="(a.u.)", ylim=(200, 3500))
        plot_timeseries(pupil_tp, likelihood, window_size=4, ax=ax8.twinx(), color='gray', lw=0.25,title="Pupil size", ylabel="likelihood", ylim=(0, 1))

        # 9. Velocity
        if table_velocities is not None and table_tp is not None:
            plot_binned_timeseries(table_velocities, table_tp, ylabel="Velocity (mm/s)", ax=ax6)
        else:
            ax6.set_title("No Rotary Encoder Data Available")
            ax6.set_ylabel("Velocity (mm/s)")

        # # 10. Temperature
        # plot_timeseries(analog_tp, tem, window_size=1, ax=ax9, color='red', lw=0.1, title="Rectal Temperature",ylabel="Amplitude (µV)", ylim=(3300, 3800))

    elif dlc_type =="Openfield":
        plot_timeseries(OF_tp, velocity , window_size=4, ax=ax6, color='blue',lw=0.25, title="Velocity (Centroid)", ylabel="Velocity (mm/s)",ylim=(0, 200)) # 約20fpsなので、約5Hzでplot?? #TODO check
        plot_timeseries(OF_tp, velocity, window_size=20*60, ax=ax6, color='olive', lw=0.5, title="Velocity (Centroid)",ylabel="Velocity (mm/s)", ylim=(0, 200))
        plot_timeseries(OF_tp, likelihood, window_size=20, ax=ax6.twinx(), color='gray', lw=0.25, title="Velocity (Centroid)",ylabel="likelihood", ylim=(0,1))

        plot_timeseries(OF_tp, distance, window_size=1, ax=ax7, color='green', lw=0.5, title="Distance to Center (Centroid)", ylabel="(mm)", ylim=(0, 300))

        for y in velocity_boundary:
            ax6.axhline(y=y, color='gray', linewidth = 0.2)

    if ecg is not None:
        # plot_timeseries(analog_tp, ecg, window_size=1, ax=ax9.twinx(), color='gray', lw=0.01, title="", ylabel="Amplitude (µV)", ylim=(-500, 1000))
        # print("######")
        # print(heartrate)
        plot_timeseries(hr_tp, heartrate, window_size=4, ax=ax9, color='#1f77b4', lw=0.5, title="ECG", ylabel="BPM", ylim=(200, 1000))

    added_labels = set()
    event_df = event_df if manual_event is None else manual_event
    # for i, event_df in enumerate([event_df, manual_event]):
    # if event_df is not None:
    event_list = event_df["event_name"].unique().tolist()
    for e, event in enumerate(event_list):
        df = event_df[event_df['event_name'] == event]
        df = df.reset_index(drop=True)
        for ep in range(len(df)):
            label = event if event not in added_labels else ""
            ax6.axvspan(df.loc[ep, "start_time"],df.loc[ep, "end_time"], color=plt.get_cmap("tab10")(e), alpha=0.3, linewidth =0, label=label)
            added_labels.add(event)
    ax6.legend(fontsize=8)


    for ax in axes:
        xticks = list(range(int(analog_tp[0]), int(analog_tp[-1]) + 1, 1 * 60))
        ax.set_xticks(xticks)
        xtick_labels = [str(x)+"("+str(int(x/60))+")" if i % 5 == 0 else "" for i, x in enumerate(xticks)]
        ax.set_xticklabels(xtick_labels)
        ax.set_xlabel("Time (sec)")
        ax.set_xlim(min(xticks), max(xticks))
        ax.margins(x=0)

    plt.tight_layout()

    # 保存 PDF
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)

    print(f"Saved to  {pdf_path}")
    return pdf_path


def turntable_to_event_df(table_tp, table_velocities, vmax, min_dur_sec, tolerable_drop_sec, event_df):
    valid_indices = np.where(np.abs(table_velocities) < 10)[0]
    event_name = "turntable_velocity_~"+str(vmax)+"mm_per_s"
    if len(valid_indices) > 0:
        start, end = valid_indices[0], valid_indices[0]
        tp_width = table_tp[1]-table_tp[0]
        for i in range(1, len(valid_indices)):
            if valid_indices[i] <= valid_indices[i - 1] + 1 + tolerable_drop_sec/tp_width:
                end = valid_indices[i]
            else:
                if (end + 1 - start) * tp_width > min_dur_sec and start * tp_width >= 0:
                    event_df.loc[len(event_df)] = [table_tp[start], table_tp[end],event_name]
                start, end = valid_indices[i], valid_indices[i]
        if (end + 1 - start) * tp_width > min_dur_sec and start * tp_width > 0:
            event_df.loc[len(event_df)] = [table_tp[start], table_tp[min(end, len(table_tp) - 1)], event_name]

    return event_df


def process_file(file_path, index, dlc_type,dlc_dir, EEG_ch_dict, EMG_ch_dict, analog_dict, cont_time,result_list, lock):
    """ 各 .ns3 / .ns2 ファイルを処理する関数（並列処理可能） """
    root_dir = os.path.dirname(file_path)
    print(root_dir)
    output_dir = os.path.join(root_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    exp_dur = (cont_time[1]-cont_time[0])*60 #sec
    # EMG / EEG / 呼吸データを抽出 #cont_timeにしたがってcut済み
    emg, eeg, sampling_rate, breathe, tem, channel_times, digital_input_data = extract_raw_data(file_path, EEG_ch_dict, EMG_ch_dict, analog_dict, cont_time)
    analog_tp = cont_time[0]*60 + np.arange(len(emg[0]))/sampling_rate

    epoch_length = 2
    nperseg = int(epoch_length * sampling_rate)
    f_stft, t_stft, Zxx = stft(eeg, fs=sampling_rate, nperseg=nperseg, noverlap=nperseg // 2)
    # stftは隣接する(Hann)窓同士が50%オーバーラップされていてスムージングされているらしい。
    t_stft=t_stft[:-1] #各実験を連結していくときに両端があるとかぶってしまうので削除
    #TODO 計算正しく直す。cf. Memorandoms 2025.09.22 /  EEG_analysis_20250922.py
    Zxx=Zxx[:,:,:-1]

    cutoff = np.argmax(f_stft > 100)
    f_stft = f_stft[:cutoff]
    Zxx = Zxx[:,:cutoff,:]
    t_stft +=cont_time[0] * 60
    linear_power = np.abs(Zxx) ** 2
    dB_power = 10 * np.log10(linear_power + 1e-10)

    breathing_rate = pupil_size = pupil_tp = table_velocities = table_tp = OF_tp = velocity = cumulative_distance = distance_to_center = velocity_boundary = OF_likelihood =None

    event_df = pd.DataFrame(columns=["start_time", "end_time", "event_name"])

    # Rotary encoder data
    if digital_input_data is not None:
        Timer_channel, A_channel, B_channel, Z_channel = 0, 4, 6, 8
        digital_timer, A, B, Z = (
            channel_times[f"channel_{Timer_channel}"],
            channel_times[f"channel_{A_channel}"],
            channel_times[f"channel_{B_channel}"],
            channel_times[f"channel_{Z_channel}"],
        )
        # light_timer = light_timer[::2]
        print(digital_timer)
        print(digital_timer.shape)

        table_velocities, table_tp = rotary_analysis(A, B, Z, exp_duration=len(eeg[0]) / sampling_rate)
        table_tp += cont_time[0]*60

        event_df = turntable_to_event_df(table_tp, table_velocities, 10, 10, 0, event_df)

    # TODO 2025.11.03現在Light (digital) timerはEEG_Analysis.pyではlight_timerは活用されていない？videoとEEG記録開始がそろっているという仮定になっている？
    # すると、PETHもグラフもずれてくるはずだが、40sec 区画でだす場合には、0.5sくらいのずれは問題にならない？
    # ただし、supp movieでは0.5 secが問題になる。ここは厳密に合わせる必要がある。
    # print(light_timer)
    # print(light_timer.shape)
    # Pupillometry
    if dlc_type == "Pupillometry":
        pupil_size, video_frame_time, likelihood = pupillometry(os.path.join(root_dir, "raw_video"))
        pupil_size = pupil_size[:int(exp_dur/video_frame_time)]
        likelihood = likelihood[:int(exp_dur / video_frame_time)]
        pupil_tp = cont_time[0]*60 + np.arange(len(pupil_size)) * video_frame_time
        #こうすると、concat timeseriesをつくるときに、最大0.05*5(実験数)=0.25秒ずれてしまう。問題になる場合は考える

        # 呼吸数計算
        time_stamp, breathing_rate = compute_breathing_frequency(breathe, sampling_rate)

    elif dlc_type == "Openfield":
        ##################
        velocity_boundary =[10] #mm/s
        video_frame_time, velocity, cumulative_distance, distance_to_center,event_df, likelihood= Openfield(index, os.path.dirname(file_path), dlc_dir, event_df, cont_time[0]*60, velocity_boundary)
        velocity = velocity[:int(exp_dur / video_frame_time)]
        likelihood = likelihood[:int(exp_dur / video_frame_time)]
        distance_to_center = distance_to_center[:int(exp_dur / video_frame_time)]
        OF_tp = cont_time[0] * 60 + np.arange(len(velocity)) * video_frame_time
        #TODO 本当はevent_dfも各実験時間からはみでるものはカットしたほうがいい

    ecg = emg[1] if len(emg) > 1 else None
    heartrate = None
    hr_tp = None
    if ecg is not None:
        bin_size = 0.25 #sec
        heartrate, hr_tp = calculate_heartrate(ecg, sampling_rate, bin_size)
        hr_tp += cont_time[0]*60

    with lock:
        result_list.append((index, analog_tp, eeg, emg, sampling_rate, breathe, tem, breathing_rate, pupil_size, pupil_tp ,table_velocities, table_tp,
                            OF_tp, velocity, cumulative_distance, distance_to_center, t_stft, f_stft, linear_power,dB_power,event_df, velocity_boundary, likelihood,
                            heartrate, hr_tp, digital_timer))

    mouse_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    if eeg is not None and emg is not None:
        for i, electrode in enumerate(EEG_ch_dict):
            pdf_name = mouse_name + "_____" + electrode + ".pdf"
            save_eeg_analysis_results(dlc_type,
                                      analog_tp, eeg[i], t_stft, f_stft, linear_power[i],dB_power[i],
                                      emg[0], ecg,sampling_rate, heartrate, hr_tp, breathe, tem, breathing_rate, pupil_size, pupil_tp,
                                      table_velocities, table_tp,
                                      OF_tp, velocity,cumulative_distance,distance_to_center,velocity_boundary, likelihood, event_df,manual_event=None,
                                      output_dir=output_dir, pdf_name=pdf_name, figsize=(15,25)  )


def process_folder(data_folder):
    output_dir = os.path.join(data_folder, "_Combined")
    os.makedirs(output_dir, exist_ok=True)

    dlc_type, dlc_dir, EEG_ch_dict, EMG_ch_dict, analog_dict, cont_time= extract_params(data_folder)

    #extract manual_event.csv
    csv_path = os.path.join(output_dir, "manual_event.csv")
    if os.path.exists(csv_path):
        print("manual_event.csv exists")
        manual_event = pd.read_csv(csv_path)
    else:
        manual_event = None

    # .ns3 / .ns2 ファイルのリストを取得
    file_list = []
    for root_dir, _, files in os.walk(data_folder):
        if not os.path.basename(root_dir).startswith("_"):  # "_" で始まるディレクトリを除外
            file_list.extend([os.path.join(root_dir, f) for f in files if f.endswith(('.ns3', '.ns2'))])
    print(file_list)


    if not file_list:
        print("No .ns3 or .ns2 files found.")
        return


    else:
        # 結果を格納するリスト（スレッドセーフ）
        manager = multiprocessing.Manager()
        result_list = manager.list()
        lock = manager.Lock()  # 競合を防ぐためのロック

        num_workers = min(multiprocessing.cpu_count(), len(file_list))  # CPU コア数に合わせる
        with multiprocessing.Pool(num_workers) as pool:
            pool.starmap(
                process_file,
                zip(file_list, range(len(file_list)), [dlc_type] * len(file_list), [dlc_dir] * len(file_list), [EEG_ch_dict] * len(file_list),
                    [EMG_ch_dict] * len(file_list), [analog_dict] * len(file_list), cont_time,
                    [result_list] * len(file_list), [lock] * len(file_list))
            )
            pool.close()
            pool.join()

        #Concatenate timeseries
        sorted_results = sorted(result_list, key=lambda x: x[0])


    all_analog_tp = np.concatenate([res[1] for res in sorted_results], axis=0)
    all_eeg = np.concatenate([res[2] for res in sorted_results], axis=1)
    all_emg = np.concatenate([res[3] for res in sorted_results], axis=1)
    sampling_rate = sorted_results[0][4]  # 同じなので1つだけ取得
    all_breathe = None if any(res[5] is None for res in sorted_results) else np.concatenate([res[5] for res in sorted_results], axis=0)
    all_tem = None if any(res[6] is None for res in sorted_results) else np.concatenate([res[6] for res in sorted_results], axis=0)
    all_b_rate = None if any(res[7] is None for res in sorted_results) else np.concatenate([res[7] for res in sorted_results], axis=0)
    all_pupil = None if any(res[8] is None for res in sorted_results) else np.concatenate([res[8] for res in sorted_results], axis=0)
    all_pupil_tp = None if any(res[9] is None for res in sorted_results) else np.concatenate([res[9] for res in sorted_results], axis=0)
    all_table_v = None if any(res[10] is None for res in sorted_results) else np.concatenate([res[10] for res in sorted_results], axis=0)
    all_table_tp = None if any(res[11] is None for res in sorted_results) else np.concatenate([res[11] for res in sorted_results], axis=0)
    all_OF_tp = None if any(res[12] is None for res in sorted_results) else np.concatenate([res[12] for res in sorted_results], axis=0)
    all_v = None if any(res[13] is None for res in sorted_results) else np.concatenate([res[13] for res in sorted_results], axis=0)
    all_cum_d = None if any(res[14] is None for res in sorted_results) else np.concatenate([res[14] for res in sorted_results], axis=0)
    all_distance = None if any(res[15] is None for res in sorted_results) else np.concatenate([res[15] for res in sorted_results], axis=0)
    all_t_stft = None if any(res[16] is None for res in sorted_results) else np.concatenate([res[16] for res in sorted_results], axis=0)
    f_stft = sorted_results[0][17]
    all_linear_power = np.concatenate([res[18] for res in sorted_results], axis=2)
    all_dB_power = np.concatenate([res[19] for res in sorted_results], axis=2)
    all_event_df = None if any(res[20] is None for res in sorted_results) else pd.concat([res[20] for res in sorted_results], axis=0, ignore_index=True)
    velocity_boundary = sorted_results[0][21]
    all_OF_likelihood = None if any(res[22] is None for res in sorted_results) else np.concatenate([res[22] for res in sorted_results], axis=0)
    all_hr = None if any(res[23] is None for res in sorted_results) else np.concatenate([res[23] for res in sorted_results], axis=0)
    all_hr_tp = None if any(res[24] is None for res in sorted_results) else np.concatenate([res[24] for res in sorted_results], axis=0)
    # print(sorted_results[])

    all_digital_timer = None if any(res[25] is None for res in sorted_results) else [res[25] for res in sorted_results]


    all_event_df.to_csv (os.path.join(output_dir, "event_combined.csv"))
    h5_name = os.path.join(output_dir, "data.h5")
    with h5py.File(h5_name, "w") as f:
        f.create_dataset("all_analog_tp", data=all_analog_tp)
        f.create_dataset("all_eeg", data=all_eeg)
        f.create_dataset("all_emg", data=all_emg)
        f.create_dataset("sampling_rate", data=sampling_rate)

        f.create_dataset("all_breathe", data=all_breathe if all_breathe is not None else np.array([np.nan]))
        f.create_dataset("all_tem", data=all_tem if all_tem is not None else np.array([np.nan]))
        f.create_dataset("all_b_rate", data=all_b_rate if all_b_rate is not None else np.array([np.nan]))
        f.create_dataset("all_pupil", data=all_pupil if all_pupil is not None else np.array([np.nan]))
        f.create_dataset("all_pupil_tp", data=all_pupil_tp if all_pupil_tp is not None else np.array([np.nan]))
        f.create_dataset("all_hr", data=all_hr if all_hr is not None else np.array([np.nan]))
        f.create_dataset("all_hr_tp", data=all_hr_tp if all_hr_tp is not None else np.array([np.nan]))
        f.create_dataset("all_table_v", data=all_table_v if all_table_v is not None else np.array([np.nan]))
        f.create_dataset("all_table_tp", data=all_table_tp if all_table_tp is not None else np.array([np.nan]))
        f.create_dataset("all_OF_tp", data=all_OF_tp if all_OF_tp is not None else np.array([np.nan]))
        f.create_dataset("all_v", data=all_v if all_v is not None else np.array([np.nan]))
        f.create_dataset("all_cum_d", data=all_cum_d if all_cum_d is not None else np.array([np.nan]))
        f.create_dataset("all_distance", data=all_distance if all_distance is not None else np.array([np.nan]))
        f.create_dataset("all_t_stft", data=all_t_stft if all_t_stft is not None else np.array([np.nan]))

        f.create_dataset("f_stft", data=f_stft)

        f.create_dataset("all_linear_power", data=all_linear_power)
        f.create_dataset("all_dB_power", data=all_dB_power)
        # f.create_dataset("velocity_boundary", data=velocity_boundary)
        f.create_dataset("velocity_boundary", data=velocity_boundary) if velocity_boundary is not None and len(velocity_boundary) > 0 else None

        if all_digital_timer is None:
            f.create_dataset("all_digital_timer", data=np.array([np.nan]))
        else:
            dt = h5py.vlen_dtype(np.float64)
            # list of 1D arrays/ lists of float なら OK
            all_digital_timer_obj = np.empty(len(all_digital_timer), dtype=object)
            for i, arr in enumerate(all_digital_timer):
                all_digital_timer_obj[i] = arr
            f.create_dataset("all_digital_timer", data=all_digital_timer_obj, dtype=dt)
            # f.create_dataset("all_digital_timer", data=all_digital_timer, dtype=dt)


        # DataFrameをHDF5に保存
        if all_event_df is not None:
            all_event_df.to_hdf(h5_name, key="event_df", mode="a", format="table")

    # 5つのデータをまとめた結果を保存
    #TODO multiprocess
    mouse_name = os.path.basename(data_folder)
    all_ecg = all_emg[1] if len(all_emg) > 1 else None
    for i, electrode in enumerate(EEG_ch_dict):
        pdf_name = mouse_name + "_____" + electrode + "_COMBINED.pdf"
        save_eeg_analysis_results(dlc_type,
                                  all_analog_tp, all_eeg[i], all_t_stft, f_stft, all_linear_power[i], all_dB_power[i],
                                  all_emg[0], all_ecg,sampling_rate, all_hr, all_hr_tp, all_breathe, all_tem, all_b_rate, all_pupil, all_pupil_tp,
                                  all_table_v, all_table_tp,
                                  all_OF_tp, all_v, all_cum_d, all_distance,velocity_boundary, all_OF_likelihood, all_event_df,manual_event,
                                  output_dir = output_dir, pdf_name=pdf_name, figsize = (45,25)
                                  )


def main():
    data_folder = select_folder()
    # data_folder = r"X:\Behavior\Openfield_EEG\Pup-IRES-dGAP\20250618_z249-3_Pup-IRES-dGAP-2x-2p-P-4ul(8w-No-eeg)_Openfield"
    process_folder(data_folder)

if __name__ == "__main__":
    main()
