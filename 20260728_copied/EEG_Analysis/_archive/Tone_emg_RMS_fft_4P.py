import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from neo.io import BlackrockIO
import pandas as pd

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select the 'data' directory")
    return folder_path


def extract_emg_data(file_path, change_threshold=1000):  # 设置变化阈值
    try:
        reader = BlackrockIO(filename=file_path)
        block = reader.read_block()
        raw_signals = [seg.analogsignals[0] for seg in block.segments]

        # 读取 anp1 数据
        analog_input_signals = [seg.analogsignals[1] for seg in block.segments]
        anp1_signal = analog_input_signals[0][:, 0]

        # 转换为一维数组并获取采样率
        anp1 = anp1_signal.magnitude.flatten()
        sampling_rate = int(anp1_signal.sampling_rate.magnitude)

        # 计算 anp1 的剧烈下降位置
        time_list = []
        for i in range(1, len(anp1)):
            if anp1[i - 1] - anp1[i] > change_threshold:
                time_list.append(round(i / sampling_rate, 1))  # 保留小数点后1位

        # 去重并排序
        time_list = sorted(set(time_list))

        # 保存 time_list 到 CSV 文件
        time_list_csv_path = file_path.replace('.ns3', '_time_list.csv')
        pd.DataFrame(time_list, columns=['time_points']).to_csv(time_list_csv_path, index=False)

        # 返回提取的其他数据
        ch1_signal = raw_signals[0][:, 0]  # EEG 数据
        ch2_signal, ch3_signal = raw_signals[0][:, 1], raw_signals[0][:, 2]  # EMG 数据
        emg = (ch2_signal - ch3_signal).magnitude.flatten()
        eeg = ch1_signal.magnitude.flatten()

        return emg, eeg, sampling_rate, time_list_csv_path
    except Exception as e:
        print(f"Failed to extract data from {file_path}: {e}")
        return None, None, None, None

def sliding_window_rms(signal, sampling_rate, center_time, window_pre=15, window_post=20, window_size=0.5):
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

def calculate_spectrum(signal, sampling_rate, window_size=0.25, freq_bin=0.5):
    n = int(sampling_rate / freq_bin)
    mean = np.mean(signal)
    signal -= mean
    signal_padded = np.pad(signal, (0, n - len(signal)), 'constant')
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)
    fft_vals = np.fft.fft(signal_padded, n=n)
    power_spectrum = np.abs(fft_vals) ** 2 / n
    return freqs[:n // 2], power_spectrum[:n // 2]

def plot_combined_rms_and_spectrum(time_points_list, rms_values_list, delta_rms_values_list, labels, freqs, power_spectrums, time_points, output_pdf):
    n_plots = len(time_points)
    basic_figsize = (12, 12)
    additional_height_per_plot = 3
    total_height = basic_figsize[1] + (additional_height_per_plot * n_plots)
    figsize = (basic_figsize[0], total_height)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2 + n_plots, 1, height_ratios=[1]*2 + [1]*n_plots)
    gs.update(wspace=0.3, hspace=0.3)

    # RMS 图像
    ax1 = fig.add_subplot(gs[0, 0])
    for time_points, rms_values, label in zip(time_points_list, rms_values_list, labels):
        smoothed_values = savgol_filter(rms_values, window_length=11, polyorder=2)
        ax1.plot(time_points, smoothed_values, label=label)
    ax1.set_title('RMS of EMG at Different Time Points')
    ax1.set_xlabel('Time after tone onset (s)')
    ax1.set_xlim(-15, 20)
    ax1.set_ylabel('RMS of EMG (mV)')
    ax1.set_ylim(-100, 400)
    ax1.grid(True)
    ax1.legend()

    # ΔRMS 图像
    ax2 = fig.add_subplot(gs[1, 0])
    for time_points, delta_rms_values, label in zip(time_points_list, delta_rms_values_list, labels):
        smoothed_values = savgol_filter(delta_rms_values, window_length=11, polyorder=2)
        ax2.plot(time_points, smoothed_values, label=label)
    ax2.set_title('ΔRMS of EMG at Different Time Points')
    ax2.set_xlabel('Time after tone onset (s)')
    ax2.set_xlim(-15, 20)
    ax2.set_ylabel('ΔRMS of EMG (mV)')
    ax2.set_ylim(-100, 400)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.grid(True)
    ax2.legend()

    # 频谱热图
    for i, center_time in enumerate(time_points):
        if i < len(power_spectrums):
            ax3 = fig.add_subplot(gs[2 + i, 0])
            img = ax3.imshow(
                10 * np.log10(power_spectrums[i]),
                aspect='auto',
                origin='lower',
                extent=[time_points[0], time_points[-1], freqs[0], freqs[-1]],
                cmap='turbo',
                vmin=30,
                vmax=60
            )
            # fig.colorbar(img, ax=ax3, label='Power (dB)')
            ax3.set_title(f'Tone at {center_time}s')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Frequency (Hz)')
            ax3.set_ylim(0, 100)

    # 保存图像到 PDF
    fig.tight_layout()
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Combined plot saved as {output_pdf}")

def process_file(file_path, time_points, window_pre=15, window_post=20, rms_window_size=0.5, spectrum_window_size=0.5, freq_bin=0.5):
    emg, eeg, sampling_rate, _ = extract_emg_data(file_path)
    if emg is None or eeg is None or sampling_rate is None:
        return

    time_points_list = []
    rms_values_list = []
    delta_rms_values_list = []
    labels = []
    power_spectrums = []

    # 计算 RMS
    for center_time in time_points:
        time_pts, rms_vals = sliding_window_rms(emg, sampling_rate, center_time, window_pre, window_post, rms_window_size)
        delta_rms_vals = calculate_delta_rms(rms_vals, time_pts)

        time_points_list.append(time_pts)
        rms_values_list.append(rms_vals)
        delta_rms_values_list.append(delta_rms_vals)
        labels.append(f'Tone at {center_time}s')

    # 计算频谱
    for center_time in time_points:
        spectrum_start_idx = int((center_time - window_pre) * sampling_rate)
        spectrum_end_idx = int((center_time + window_post) * sampling_rate)
        window_samples = int(spectrum_window_size * sampling_rate)

        spectrum = []
        for idx in range(spectrum_start_idx, spectrum_end_idx - window_samples + 1, window_samples):
            freqs, power_spectrum = calculate_spectrum(eeg[idx:idx + window_samples], sampling_rate, freq_bin=freq_bin)
            spectrum.append(power_spectrum)

        power_spectrums.append(np.array(spectrum).T if spectrum else None)

    # 定义输出 PDF 的路径
    output_pdf = os.path.join(os.path.dirname(file_path),
                              f"{os.path.splitext(os.path.basename(file_path))[0]}_Tone_analysis.pdf")

    # 绘制并保存图像
    plot_combined_rms_and_spectrum(
        time_points_list, rms_values_list, delta_rms_values_list, labels, freqs, power_spectrums,
        time_points, output_pdf
    )


def main():
    folder_path = select_folder()
    if not folder_path:
        print("No directory selected.")
        return

    for root_dir, _, files in os.walk(folder_path):
        ns3_files = [file for file in files if file.endswith('.ns3')]

        for ns3_file in ns3_files:
            file_path = os.path.join(root_dir, ns3_file)
            print(f"Processing file: {file_path}")

            # 提取数据并生成 time_list.csv
            emg, eeg, sampling_rate, time_list_csv_path = extract_emg_data(file_path)
            if emg is None or eeg is None or sampling_rate is None:
                print("Data extraction failed, skipping this file.")
                continue

            # 从 CSV 文件读取 time_points
            time_points_df = pd.read_csv(time_list_csv_path)
            time_points = time_points_df['time_points'].tolist()

            # 使用 time_points 进行后续分析
            process_file(file_path, time_points)


if __name__ == "__main__":
    main()