import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from neo.io import BlackrockIO
import tkinter as tk
from tkinter import filedialog
import glob
import os
import pandas as pd
from scipy.signal import cheby1, bessel, butter, filtfilt


# TODO 前四秒还有瑕疵。
def plot_signal(ax, signal, sampling_rate, ylabel, ylim, time_points, time_window=8, highlight_duration=1):
    time_window_samples = int(time_window * sampling_rate)  # 默认是8秒窗口
    highlight_samples = int(highlight_duration * sampling_rate)  # 高亮区域的样本数

    for time_point in time_points:
        if time_point <= 4:
            # For the first 5 seconds, use the window from 0 to 8 seconds
            start_sample = 0
            end_sample = time_window_samples
        else:
            # From the 6th second onward, start scrolling
            start_sample = max(0, int((time_point - 4) * sampling_rate))
            end_sample = start_sample + time_window_samples

        # 绘制信号
        ax.plot(signal[start_sample:end_sample], lw=1.0, color='red')

        # 高亮显示time_point位置的背景颜色
        highlight_start = int(time_point * sampling_rate)
        highlight_end = highlight_start + highlight_samples

        # 转换为相对于绘图区域的x位置
        ax.axvspan(highlight_start - start_sample, highlight_end - start_sample, color='yellow', alpha=0.3)

        # 设置x轴刻度，xticks以秒为单位，xtick_labels从time_point-4到time_point+4
        xticks = np.arange(0, time_window_samples, sampling_rate)
        if time_point <= 4:
            xtick_labels = np.arange(0, 8, 1)
        else:
            xtick_labels = np.arange(time_point - 4, time_point + 4, 1)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel + " (uV)")
        ax.set_ylim(-ylim, ylim)
        ax.margins(x=0)


def calculate_spectrum(signal, sampling_rate):
    n = len(signal)
    mean = np.mean(signal)
    signal -= mean  # 去除直流偏移
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)
    fft_vals = np.fft.fft(signal)
    power_spectrum = np.abs(fft_vals) ** 2 / n
    return freqs[:n // 2], power_spectrum[:n // 2]


def cheby1_bandpass_filter(data, lowcut, highcut, sampling_rate, order=4, ripple=0.5):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = cheby1(order, ripple, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

"""def bessel_bandpass_filter(data, lowcut, highcut, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = bessel(order, [low, high], btype='band', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def butter_bandpass_filter(data, lowcut, highcut, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data"""

def butter_lowpass_filter(data, cutoff, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def spectrum_graph(signal, sampling_rate, time_points, freq_max, freq_min, ax):
    # Low-pass filter parameters
    lowpass_cutoff = 50  # Cutoff frequency in Hz
    order = 4  # Filter order

    # Apply the low-pass filter to the signal
    signal = butter_lowpass_filter(signal, lowpass_cutoff, sampling_rate, order)

    freq_bin = 0.5
    num_bins = int(np.ceil((freq_max - freq_min) / freq_bin))
    bins = np.linspace(freq_min, freq_max, num_bins + 1)
    vertical_lines = [1, 4, 8, 12, 30]

    for time_point in time_points:
        start_sample = max(0, int((time_point - 1.5) * sampling_rate))
        end_sample = min(len(signal), int((time_point + 1.5) * sampling_rate))
        signal_segment = signal[start_sample:end_sample]

        freqs, power_spectrum = calculate_spectrum(signal_segment, sampling_rate)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_indices = np.digitize(freqs, bins)

        power_spectrum_means = np.array([np.mean(power_spectrum[bin_indices == i]) for i in range(1, num_bins + 1)])
        power_spectrum_means_norm = np.array([power_spectrum[bin_indices == i].sum() for i in range(1, num_bins + 1)]) / power_spectrum.sum() * 100

        label_with_time_window = f'Time {time_point:.2f}s ±1.5 sec' # Time window range

        """
        ax1.plot(bin_centers, power_spectrum_means_norm, lw=1.5, linestyle='-', label=label_with_time_window)
        ax1.set_xlim(freq_min, freq_max)
        ax1.set_ylim(0, 12)
        ax1.set_xlabel("Hz")
        ax1.set_ylabel("Normalized power (%)")
        ax1.legend()
        """

        ax.plot(bin_centers, power_spectrum_means, lw=1.5, linestyle='-', label=label_with_time_window)
        ax.set_xlim(freq_min, freq_max)
        ax.set_xlabel("Hz")
        ax.set_ylim(0, 2000000)  # When Y-scale use log10, ylim(0.1, 1000000)
        ax.set_ylabel("Power (μV²)")
        # ax2.set_yscale('log')  # Y-scale log10
        ax.legend()

        for line in vertical_lines:
            # ax1.axvline(x=line, color='grey', linestyle='--')
            ax.axvline(x=line, color='grey', linestyle='--')


def dB_heatmap(signal, sampling_rate, time_point, freq_max, freq_min, ax, title):
    # Adjust the time window based on the time_point and the scrolling behavior
    if time_point < 4:
        start_time = 0
        end_time = 8
    else:
        start_time = time_point - 4
        end_time = time_point + 4

    time_bin = 1
    freq_bin = 1
    start_sample = int(start_time * sampling_rate)
    end_sample = int(end_time * sampling_rate)
    signal_segment = signal[start_sample:end_sample]

    series_list = []

    time_bin_num = int(len(signal_segment) / (sampling_rate * time_bin))
    for t in range(time_bin_num):
        bin_signal = signal_segment[int(sampling_rate * time_bin * t):int(sampling_rate * time_bin * (t + 1))]
        freqs, power_spectrum = calculate_spectrum(bin_signal, sampling_rate)
        power_spectrum_db = 10 * np.log10(power_spectrum + 1e-10)

        num_bins = int(np.ceil((freq_max - freq_min) / freq_bin))
        bins = np.linspace(freq_min, freq_max, num_bins + 1)
        bin_indices = np.digitize(freqs, bins)
        power_spectrum_means_db = [power_spectrum_db[bin_indices == i].mean() if i in bin_indices else np.nan
                                   for i in range(1, num_bins + 1)]
        series_list.append(pd.Series(power_spectrum_means_db))
    df = pd.concat(series_list, axis=1).T.reset_index(drop=True)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    df = df.loc[:, (freq_min / freq_bin):(freq_max / freq_bin)]

    # The setting for limit of heatmap, to avoid the data over-range
    # vmin_db = np.percentile(df.values, 1)  # 选择 1% 分位数作为 vmin
    # vmax_db = np.percentile(df.values, 99)  # 选择 99% 分位数作为 vmax

    # Adjust extent to reflect the correct time window based on time_point
    im = ax.imshow(df.T, aspect='auto', cmap='rainbow', origin='lower', vmin=30, vmax=60,
                         extent=[start_time, end_time, freq_min, freq_max])

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    # Set x-axis ticks based on the selected time range
    # zhou: I deleted the xticks setting, while it looks no big problems


    yticks = np.arange(0, freq_max / freq_bin + 1, int(freq_max / freq_bin / 5))
    ytick_labels = np.arange(0, freq_max + 1, int(freq_max / 5))
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    ax.set_ylim = (1, )

    # 检查df中的数据格式，以及是否包含数据
    # print(f"DataFrame shape: {df.shape}")
    # print(df.head())



def combined_spectrum_image(signal, sampling_rate, time_point, freq_max, freq_min, output_dir):
    # 在输出路径下创建名为 "combined_analysis_video" 的新文件夹
    output_dir = os.path.join(output_dir, "combined_analysis_video")
    os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在则创建

    time_points = np.arange(0, len(signal) // sampling_rate, 1)
    image_paths = []

    for time_point in time_points:
        fig = plt.figure(figsize=(30, 20))
        gs = gridspec.GridSpec(6, 3, height_ratios=[0.6, 0.4, 0.4, 0.4, 0.4, 0.4])
        # 调整子图间隔
        plt.subplots_adjust(hspace=0.5)

        # 绘制功率谱图
        ax = fig.add_subplot(gs[0, 0])
        spectrum_graph(signal, sampling_rate, [time_point], freq_max, freq_min, ax)
        ax_lfp = fig.add_subplot(gs[0, 1])
        spectrum_graph(lfp, sampling_rate, [time_point], freq_max, freq_min, ax_lfp)

        # 绘制9-16Hz滤波后的信号
        filtered_signal_9_16 = cheby1_bandpass_filter(signal, 9, 16, sampling_rate)
        ax_filtered_9_16 = fig.add_subplot(gs[3, 0])
        plot_signal(ax_filtered_9_16, filtered_signal_9_16, sampling_rate, "9-16Hz V_EEG Signal", 500, [time_point])
        filtered_lfp_9_16 = cheby1_bandpass_filter(lfp, 9, 16, sampling_rate)
        ax_filtered_lfp_9_16 = fig.add_subplot(gs[3, 1])
        plot_signal(ax_filtered_lfp_9_16, filtered_lfp_9_16, sampling_rate, "9-16Hz M_EEG Signal", 500, [time_point])

        # 绘制0-50Hz滤波后的信号
        filtered_signal_0_100 = butter_lowpass_filter(signal, 70, sampling_rate)
        ax_filtered_0_50 = fig.add_subplot(gs[4, 0])
        plot_signal(ax_filtered_0_50, filtered_signal_0_100, sampling_rate, "0-70Hz V_EEG Signal", 500, [time_point])

        # LFP
        filtered_lfp_0_100 = butter_lowpass_filter(lfp, 70, sampling_rate)
        ax_filtered_lfp_0_100 = fig.add_subplot(gs[4, 1])
        plot_signal(ax_filtered_lfp_0_100, filtered_lfp_0_100, sampling_rate, "0-70Hz M_EEG Signal", 500, [time_point])

        # 绘制EMG信号
        emg = (ch3_signal - ch4_signal).magnitude.flatten()
        ax_emg = fig.add_subplot(gs[5, :])
        plot_signal(ax_emg, emg, sampling_rate, "EMG", 500, [time_point])

        # heatmap
        ax_db = fig.add_subplot(gs[1, 0])
        dB_heatmap(signal, sampling_rate, time_point, 70, freq_min, ax=ax_db, title = 'V_EEG_Power_dB')
        ax_db_lfp = fig.add_subplot(gs[1, 1])
        dB_heatmap(lfp, sampling_rate, time_point, 70, freq_min, ax=ax_db_lfp, title = 'M_EEG_Power_dB')
        # heatmap_20
        ax_db_20 = fig.add_subplot(gs[2, 0])
        dB_heatmap(signal, sampling_rate, time_point, 20, 1, ax=ax_db_20, title = 'V_EEG_Power_dB')
        ax_db_lfp_20 = fig.add_subplot(gs[2, 1])
        dB_heatmap(lfp, sampling_rate, time_point, 20, 1, ax=ax_db_lfp_20, title = 'M_EEG_Power_dB')

        # M1 - V1
        ax_m1v1 = fig.add_subplot(gs[0, 2])
        spectrum_graph(m1v1, sampling_rate, [time_point], freq_max, freq_min, ax_m1v1)

        ax_db_m1v1 = fig.add_subplot(gs[1, 2])
        dB_heatmap(m1v1, sampling_rate, time_point, 70, freq_min, ax=ax_db_m1v1, title='M1-V1_EEG_Power_dB')

        ax_db_m1v1_20 = fig.add_subplot(gs[2, 2])
        dB_heatmap(m1v1, sampling_rate, time_point, 20, 1, ax=ax_db_m1v1_20, title='M1-V1_EEG_Power_dB')

        filtered_m1v1_9_16 = cheby1_bandpass_filter(m1v1, 9, 16, sampling_rate)
        ax_filtered_m1v1_9_16 = fig.add_subplot(gs[3, 2])
        plot_signal(ax_filtered_m1v1_9_16, filtered_m1v1_9_16, sampling_rate, "9-16Hz M1-V1_EEG Signal", 500, [time_point])

        filtered_m1v1_0_100 = butter_lowpass_filter(m1v1, 70, sampling_rate)
        ax_filtered_m1v1_0_100 = fig.add_subplot(gs[4, 2])
        plot_signal(ax_filtered_m1v1_0_100, filtered_m1v1_0_100, sampling_rate, "0-70Hz M1-V1_EEG Signal", 500, [time_point])

        # 保存图像到新创建的文件夹中
        combined_image_path = os.path.join(output_dir, f"combined_analysis_{time_point}.png")
        plt.savefig(combined_image_path, bbox_inches='tight')
        plt.close(fig)

        image_paths.append(combined_image_path)

    return image_paths


# 创建一个隐藏的根窗口
root = tk.Tk()
root.withdraw()

# 打开一个文件夹选择对话框，让用户选择上级文件夹
mouse_dir = filedialog.askdirectory(title="Select the 'data' directory")

# 确保用户选择了目录
if mouse_dir:
    # 遍历所选目录下的所有子文件夹和文件
    for root_dir, sub_dirs, files in os.walk(mouse_dir):
        # 只处理 .ns3 文件
        ns3_files = [file for file in files if file.endswith('.ns3')]

        # 如果找到了 .ns3 文件
        for ns3_file in ns3_files:
            # 生成完整的文件路径并赋值给 file_path
            file_path = os.path.join(root_dir, ns3_file)

            # 打印当前正在处理的文件路径
            print(f"Processing file: {file_path}")

            # 在此处添加对 file_path 的处理逻辑，例如调用分析函数
            # 调用函数并生成图片
            dir = os.path.dirname(file_path)
            output_dir = dir

            freq_max, freq_min = 100, 0
            # 读取数据
            reader = BlackrockIO(filename=file_path)
            block = reader.read_block()
            raw_signals = [seg.analogsignals[0] for seg in block.segments]
            raw_signal = raw_signals[0][:, 1]
            sampling_rate = int(raw_signal.sampling_rate.magnitude)
            signal = raw_signal.magnitude.flatten()
            ch1_signal = raw_signals[0][:, 0]
            lfp = ch1_signal.magnitude.flatten()
            m1v1 = (ch1_signal-raw_signal).magnitude.flatten()
            ch3_signal, ch4_signal = raw_signals[0][:, 2], raw_signals[0][:, 3]

            combined_spectrum_image(signal, sampling_rate, 0, freq_max, freq_min, output_dir)
else:
    print("No directory selected.")
