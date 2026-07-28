import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, cwt, morlet
from neo.io import BlackrockIO

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select the 'data' directory")
    return folder_path

def extract_emg_data(file_path):
    try:
        reader = BlackrockIO(filename=file_path)
        block = reader.read_block()
        raw_signals = [seg.analogsignals[0] for seg in block.segments]
        ch1_signal = raw_signals[0][:, 0]  # EEG 数据
        ch2_signal, ch3_signal = raw_signals[0][:, 1], raw_signals[0][:, 2]  # EMG 数据
        emg = (ch2_signal - ch3_signal).magnitude.flatten()
        eeg = ch1_signal.magnitude.flatten()
        sampling_rate = int(ch2_signal.sampling_rate.magnitude)
        return emg, eeg, sampling_rate
    except Exception as e:
        print(f"Failed to extract data from {file_path}: {e}")
        return None, None, None

def sliding_window_rms(signal, sampling_rate, center_time, window_pre=1, window_post=4, window_size=0.25):
    start_time = center_time - window_pre
    end_time = center_time + window_post
    start_idx = int(start_time * sampling_rate)
    end_idx = int(end_time * sampling_rate)
    window_samples = int(window_size * sampling_rate)

    time_points = []
    rms_values = []

    for idx in range(start_idx, end_idx - window_samples + 1):
        window_signal = signal[idx:idx + window_samples]
        rms = np.sqrt(np.mean(window_signal ** 2))
        time_points.append((idx + window_samples / 2) / sampling_rate - center_time)
        rms_values.append(rms)

    return time_points, rms_values

def calculate_delta_rms(rms_values, time_points, baseline_duration=1.0):
    baseline_end_idx = next(i for i, t in enumerate(time_points) if t >= 0)
    baseline_start_idx = next(i for i, t in enumerate(time_points) if t >= -baseline_duration)
    baseline_mean = np.mean(rms_values[baseline_start_idx:baseline_end_idx])
    delta_rms = [rms - baseline_mean for rms in rms_values]
    return delta_rms

def calculate_wavelet_spectrum(signal, sampling_rate):
    sampling_rate = 2000
    # 计算尺度，使得最高频率不超过low_pass_freq
    widths = np.arange(40, 250)

    # 进行小波变换，这里使用Haar小波
    wavelet_coefficients = cwt(signal, morlet, widths)

    # 将尺度转换为频率
    freqs = sampling_rate / widths

    # 计算功率谱
    power_spectrum = np.abs(wavelet_coefficients) ** 2
    return freqs, power_spectrum

def plot_combined_rms_and_wavelet(time_points_list, rms_values_list, delta_rms_values_list, labels, wavelet_results, time_points, output_pdf):
    n_plots = len(time_points)
    fig = plt.figure(figsize=(12, 18))
    gs = gridspec.GridSpec(2 + n_plots, 1, height_ratios=[1]*2 + [1]*n_plots)
    gs.update(wspace=0.3, hspace=0.3)

    # RMS 图像
    ax1 = fig.add_subplot(gs[0, 0])
    for time_points, rms_values, label in zip(time_points_list, rms_values_list, labels):
        smoothed_values = savgol_filter(rms_values, window_length=11, polyorder=2)
        ax1.plot(time_points, smoothed_values, label=label)
    ax1.set_title('RMS of EMG at Different Time Points')
    ax1.set_xlabel('Time after tone onset (s)')
    ax1.set_xlim(-1, 4)
    ax1.set_ylabel('RMS of EMG (mV)')
    ax1.grid(True)
    ax1.legend()

    # ΔRMS 图像
    ax2 = fig.add_subplot(gs[1, 0])
    for time_points, delta_rms_values, label in zip(time_points_list, delta_rms_values_list, labels):
        smoothed_values = savgol_filter(delta_rms_values, window_length=11, polyorder=2)
        ax2.plot(time_points, smoothed_values, label=label)
    ax2.set_title('ΔRMS of EMG at Different Time Points')
    ax2.set_xlabel('Time after tone onset (s)')
    ax2.set_xlim(-1, 4)
    ax2.set_ylabel('ΔRMS of EMG (mV)')
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.grid(True)
    ax2.legend()

    # 小波变换结果热图
    for i, (center_time, (freqs, power_spectrum)) in enumerate(zip(time_points, wavelet_results)):
        ax = fig.add_subplot(gs[2 + i, 0])
        time_range = np.linspace(-1, 4, power_spectrum.shape[1])
        img = ax.imshow(
            10 * np.log10(power_spectrum + 1e-10),  # 加上一个小常数避免对数为负无穷
            aspect='auto',
            origin='lower',
            extent=[time_range[0], time_range[-1], freqs[-1], freqs[0]],
            cmap='turbo',
            vmin=np.percentile(10 * np.log10(power_spectrum + 1e-10), 5),
            vmax=np.percentile(10 * np.log10(power_spectrum + 1e-10), 95)
        )
        ax.set_title(f'Tone at {center_time}s')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim(freqs[-1], freqs[0])  # 设置纵轴的频率范围
        # fig.colorbar(img, ax=ax3, label='Power (dB)')

    # 保存图像到 PDF
    fig.tight_layout()
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Combined plot saved as {output_pdf}")

def process_file(file_path, time_points, window_pre=1, window_post=4, window_size=0.2):
    emg, eeg, sampling_rate = extract_emg_data(file_path)
    if emg is None or eeg is None or sampling_rate is None:
        return

    time_points_list = []
    rms_values_list = []
    delta_rms_values_list = []
    labels = []

    wavelet_results = []

    for center_time in time_points:
        time_pts, rms_vals = sliding_window_rms(emg, sampling_rate, center_time, window_pre, window_post, window_size)
        delta_rms_vals = calculate_delta_rms(rms_vals, time_pts)

        time_points_list.append(time_pts)
        rms_values_list.append(rms_vals)
        delta_rms_values_list.append(delta_rms_vals)
        labels.append(f'Tone at {center_time}s')

        # 计算小波变换
        start_idx = int((center_time - window_pre) * sampling_rate)
        end_idx = int((center_time + window_post) * sampling_rate)
        signal_window = eeg[start_idx:end_idx]
        freqs, power_spectrum = calculate_wavelet_spectrum(signal_window, sampling_rate)

        wavelet_results.append((freqs, power_spectrum))

    # 定义输出 PDF 的路径
    output_pdf = os.path.join(os.path.dirname(file_path),
                              f"{os.path.splitext(os.path.basename(file_path))[0]}_Tone_analysis_cwt.pdf")

    # 绘制并保存图像
    plot_combined_rms_and_wavelet(
        time_points_list, rms_values_list, delta_rms_values_list, labels, wavelet_results, time_points, output_pdf
    )

def main():
    folder_path = select_folder()
    if not folder_path:
        print("No directory selected.")
        return

    time_points = [1530, 1620, 1720]
    for root_dir, _, files in os.walk(folder_path):
        ns3_files = [file for file in files if file.endswith('.ns3')]
        for ns3_file in ns3_files:
            file_path = os.path.join(root_dir, ns3_file)
            print(f"Processing file: {file_path}")
            process_file(file_path, time_points)

if __name__ == "__main__":
    main()
