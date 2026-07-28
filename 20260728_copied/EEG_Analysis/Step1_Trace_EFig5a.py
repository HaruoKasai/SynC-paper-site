import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from neo.io import BlackrockIO
import tkinter as tk
from tkinter import filedialog
import os
from scipy.signal import butter, filtfilt

def lowpass_filter(signal, fs, cutoff=150, order=4):
    """
    Butterworth 低通滤波
    signal : np.array, 原始信号
    fs     : int, 采样率 Hz
    cutoff : float, 截止频率 Hz
    order  : int, 滤波器阶数
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal



def plot_signal(ax, signal, sampling_rate, ylabel, ylim, start_time, end_time):
    """绘制指定时间段的信号，不含高亮区域"""
    start_sample = int(start_time * sampling_rate)
    end_sample = int(end_time * sampling_rate)

    # 截取信号
    segment = signal[start_sample:end_sample]

    # 绘制信号
    ax.plot(segment, lw=1.0, color='red')

    # 设置 x 轴
    total_duration = end_time - start_time
    xticks = np.arange(0, len(segment), sampling_rate)  # 每秒一个刻度
    xtick_labels = np.arange(start_time, end_time, 1)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel + " (uV)")
    ax.set_ylim(-ylim, ylim)
    ax.margins(x=0)


def combined_spectrum_image(m1_signal, sampling_rate, start_time, end_time, ch3_signal, ch4_signal, output_dir):
    """绘制指定时间段的EEG和EMG图"""
    output_dir = os.path.join(output_dir, "_EFig_Trace")
    os.makedirs(output_dir, exist_ok=True)

    # 对 EEG 信号做低通滤波
    m1_signal_filtered = lowpass_filter(m1_signal, sampling_rate, cutoff=150)

    # 创建画布
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.4, 0.4])
    plt.subplots_adjust(hspace=0.5)

    # EEG信号
    ax_eeg = fig.add_subplot(gs[0, 0])
    plot_signal(ax_eeg,
                m1_signal_filtered, # 4p signal; 6p m1_signal
                sampling_rate, "EEG Signal", 500, start_time, end_time)

    # EMG信号
    emg = (ch2_signal - ch3_signal).magnitude.flatten() # 4p
    # emg = (ch3_signal - ch4_signal).magnitude.flatten() # 6p
    print(ch3_signal, ch4_signal)
    ax_emg = fig.add_subplot(gs[1, 0])
    plot_signal(ax_emg, emg, sampling_rate, "EMG", 1000, start_time, end_time)

    # 保存图像
    save_path = os.path.join(output_dir, f"EEG_Raw_Trace_{start_time}-{end_time}s.pdf")
    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.close(fig)

    print(f"Saved figure: {save_path}")
    return save_path

"""
def combined_spectrum_image(m1_signal, sampling_rate, start_time, end_time, ch3_signal, ch4_signal, output_dir):
    # 绘制EEG、EMG及FFT频谱图
    output_dir = os.path.join(output_dir, "_EFig_Trace")
    os.makedirs(output_dir, exist_ok=True)

    # 创建画布
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.4, 0.4, 0.4])
    plt.subplots_adjust(hspace=0.6)

    # EEG信号
    ax_eeg = fig.add_subplot(gs[0, 0])
    plot_signal(ax_eeg, m1_signal, sampling_rate, "EEG Signal", 500, start_time, end_time)

    # EMG信号
    emg = (ch3_signal - ch4_signal).magnitude.flatten()
    ax_emg = fig.add_subplot(gs[1, 0])
    plot_signal(ax_emg, emg, sampling_rate, "EMG", 1000, start_time, end_time)

    # ===== FFT 频谱 =====
    start_sample = int(start_time * sampling_rate)
    end_sample = int(end_time * sampling_rate)
    eeg_segment = m1_signal[start_sample:end_sample]

    # 计算FFT
    N = len(eeg_segment)
    freqs = np.fft.rfftfreq(N, d=1/sampling_rate)  # 只取正频部分
    fft_values = np.fft.rfft(eeg_segment)

    # 计算功率谱 (单位：uV²)
    power = (np.abs(fft_values) ** 2) / N
    # 转换为 dB（避免 log(0)）
    power_db = 10 * np.log10(power + 1e-12)

    # 限制显示范围：0–100 Hz
    freq_mask = freqs <= 100
    freqs = freqs[freq_mask]
    power_db = power_db[freq_mask]

    # 绘制频谱
    ax_spec = fig.add_subplot(gs[2, 0])
    ax_spec.plot(freqs, power_db, color='blue', lw=1.2)
    ax_spec.set_xlim(0, 100)
    ax_spec.set_xlabel("Frequency (Hz)")
    ax_spec.set_ylabel("Power (uV²)")
    ax_spec.set_title("EEG Power Spectrum (FFT)")
    ax_spec.grid(True, linestyle='--', alpha=0.5)

    # 保存图像
    save_path = os.path.join(output_dir, f"EEG_Raw_Trace_{start_time}-{end_time}s.pdf")
    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.close(fig)

    print(f"Saved figure: {save_path}")
    return save_path
"""

# ===== 主程序 =====
root = tk.Tk()
root.withdraw()

# 选择数据目录
mouse_dir = filedialog.askdirectory(title="Select the 'data' directory")

if mouse_dir:
    for root_dir, sub_dirs, files in os.walk(mouse_dir):
        valid_files = [file for file in files if file.endswith(('.ns3', '.ns2'))]

        for valid_file in valid_files:
            file_path = os.path.join(root_dir, valid_file)
            print(f"Processing file: {file_path}")


            """
            # 读取数据 4p
            reader = BlackrockIO(filename=file_path)
            block = reader.read_block()
            raw_signals = [seg.analogsignals[0] for seg in block.segments]
            raw_signal = raw_signals[0][:, 0]  # EEG
            sampling_rate = int(raw_signal.sampling_rate.magnitude)
            signal = raw_signal.magnitude.flatten()
            ch2_signal, ch3_signal = raw_signals[0][:, 1], raw_signals[0][:, 2]
            """

            # 读取数据 6p
            reader = BlackrockIO(filename=file_path)
            block = reader.read_block()
            raw_signals = [seg.analogsignals[0] for seg in block.segments]
            ch1_signal = raw_signals[0][:, 0]
            ch2_signal = raw_signals[0][:, 1]
            ch3_signal, ch4_signal = raw_signals[0][:, 2], raw_signals[0][:, 3]
            sampling_rate = int(ch1_signal.sampling_rate.magnitude)
            m1_signal = ch1_signal.magnitude.flatten()
            v1_signal = ch2_signal.magnitude.flatten()
            m1v1 = (ch1_signal - ch2_signal).magnitude.flatten()


            # ===== 在这里指定时间范围 =====
            start_time = 1423  # 单位：秒
            end_time = 1433    # 单位：秒

            # 调用函数生成图片
            combined_spectrum_image(m1_signal, sampling_rate, start_time, end_time, ch3_signal, ch4_signal, os.path.dirname(file_path))
else:
    print("No directory selected.")
