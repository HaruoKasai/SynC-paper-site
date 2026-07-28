import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from neo.io import BlackrockIO
import os
from moviepy.editor import ImageClip, concatenate_videoclips
import pandas as pd


# 假设file_path是您的数据文件路径
# ns2文件路径
file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\20240627_z155-1_Pup-Ctrl_1stAC\Spec_temp\z155-1_afterR_006.ns2"
dir = os.path.dirname(file_path)


# 读取数据
reader = BlackrockIO(filename=file_path)
block = reader.read_block()
raw_signals = [seg.analogsignals[0] for seg in block.segments]
raw_signal = raw_signals[0][:, 0]
sampling_rate = int(raw_signal.sampling_rate.magnitude)
signal = raw_signal.magnitude.flatten()
ch2_signal, ch3_signal = raw_signals[0][:, 1], raw_signals[0][:, 2]
emg = (ch2_signal - ch3_signal).magnitude.flatten()


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


def spectrum_graph(signal, sampling_rate, time_points, freq_max, freq_min, ax1, ax2):
    freq_bin = 0.5
    num_bins = int(np.ceil((freq_max - freq_min) / freq_bin))
    bins = np.linspace(freq_min, freq_max, num_bins + 1)
    vertical_lines = [1, 4, 8, 12, 30]

    for time_point in time_points:
        start_sample = max(0, int((time_point - 4) * sampling_rate))
        end_sample = min(len(signal), int((time_point + 4) * sampling_rate))
        signal_segment = signal[start_sample:end_sample]

        freqs, power_spectrum = calculate_spectrum(signal_segment, sampling_rate)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_indices = np.digitize(freqs, bins)

        power_spectrum_means = np.array([np.mean(power_spectrum[bin_indices == i]) for i in range(1, num_bins + 1)])
        power_spectrum_means_norm = np.array([power_spectrum[bin_indices == i].sum() for i in range(1, num_bins + 1)]) / power_spectrum.sum() * 100

        ax1.plot(bin_centers, power_spectrum_means_norm, lw=1.5, linestyle='-', label=f'Time {time_point:.2f}s')
        ax1.set_xlim(freq_min, freq_max)
        ax1.set_ylim(0, 12)
        ax1.set_xlabel("Hz")
        ax1.set_ylabel("Normalized power (%)")

        ax2.plot(bin_centers, power_spectrum_means, lw=1.5, linestyle='-', label=f'Time {time_point:.2f}s')
        ax2.set_xlim(freq_min, freq_max)
        ax2.set_xlabel("Hz")
        ax2.set_ylim(0, 1000000)
        ax2.set_ylabel("Power (μV²)")

        for line in vertical_lines:
            ax1.axvline(x=line, color='grey', linestyle='--')
            ax2.axvline(x=line, color='grey', linestyle='--')


def dB_heatmap(signal, sampling_rate, time_point, freq_max, freq_min, ax):
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

    df = pd.DataFrame()

    time_bin_num = int(len(signal_segment) / (sampling_rate * time_bin))
    for t in range(time_bin_num):
        bin_signal = signal_segment[int(sampling_rate * time_bin * t):int(sampling_rate * time_bin * (t + 1))]
        freqs, power_spectrum = calculate_spectrum(bin_signal, sampling_rate)
        power_spectrum_db = 10 * np.log10(power_spectrum)

        num_bins = int(np.ceil((freq_max - freq_min) / freq_bin))
        bins = np.linspace(freq_min, freq_max, num_bins + 1)
        bin_indices = np.digitize(freqs, bins)
        power_spectrum_means_db = [power_spectrum_db[bin_indices == i].mean() if i in bin_indices else np.nan
                                   for i in range(1, num_bins + 1)]
        df = df.append(pd.Series(power_spectrum_means_db), ignore_index=True)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    df = df.loc[:, (freq_min / freq_bin):(freq_max / freq_bin)]

    # The setting for limit of heatmap, to avoid the data over-range
    # vmin_db = np.percentile(df.values, 1)  # 选择 1% 分位数作为 vmin
    # vmax_db = np.percentile(df.values, 99)  # 选择 99% 分位数作为 vmax

    # Adjust extent to reflect the correct time window based on time_point
    im = ax.imshow(df.T, aspect='auto', cmap='rainbow', origin='lower', vmin=30, vmax=50,
                         extent=[start_time, end_time, freq_min, freq_max])

    ax.set_title('Power Spectrum in dB')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    # Set x-axis ticks based on the selected time range
    # zhou: I deleted the xticks setting, while it looks no big problems


    yticks = np.arange(0, freq_max / freq_bin + 1, int(freq_max / freq_bin / 5))
    ytick_labels = np.arange(0, freq_max + 1, int(freq_max / 5))
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)

    # 检查df中的数据格式，以及是否包含数据
    # print(f"DataFrame shape: {df.shape}")
    # print(df.head())


def combined_spectrum_image(signal, sampling_rate, freq_max, freq_min, output_dir):
    time_points = np.arange(0, len(signal) // sampling_rate, 1)
    image_paths = []

    for time_point in time_points:
        fig = plt.figure(figsize=(10, 15))
        gs = gridspec.GridSpec(4, 2, height_ratios=[1.0, 0.6, 0.6, 0.6])
        # 调整子图间隔
        plt.subplots_adjust(hspace=0.5)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        spectrum_graph(signal, sampling_rate, [time_point], freq_max, freq_min, ax1, ax2)

        ax = fig.add_subplot(gs[1, :])
        plot_signal(ax, signal, sampling_rate, "EEG", 500, [time_point])

        ax_emg = fig.add_subplot(gs[2, :])
        plot_signal(ax_emg, emg, sampling_rate, "EMG", 300, [time_point])

        ax_db = fig.add_subplot(gs[3, :])
        dB_heatmap(signal, sampling_rate, time_point, freq_max, freq_min, ax=ax_db)

        combined_image_path = os.path.join(output_dir, f"combined_spectrum_{time_point}.png")
        plt.savefig(combined_image_path, bbox_inches='tight')
        plt.close(fig)

        image_paths.append(combined_image_path)

    return image_paths


def create_clip(signal, sampling_rate, freq_max, freq_min, output_path):
    output_dir = os.path.dirname(output_path)

    # Check if all images already exist
    image_paths = []
    time_points = np.arange(0, len(signal) // sampling_rate, 1)
    all_images_exist = True

    for time_point in time_points:
        image_path = os.path.join(output_dir, f"combined_spectrum_{time_point}.png")
        image_paths.append(image_path)
        if not os.path.exists(image_path):
            all_images_exist = False

    # Generate images if they do not exist
    if not all_images_exist:
        image_paths = combined_spectrum_image(signal, sampling_rate, freq_max, freq_min, output_dir)

    # Create video clip
    clips = []
    frame_duration = 1  # Set according to your needs

    for temp_file_path in image_paths:
        clip = ImageClip(temp_file_path).set_duration(frame_duration)
        clips.append(clip)

    spectrum_clip = concatenate_videoclips(clips, method="compose")
    spectrum_clip.write_videofile(output_path, codec="libx264", fps=24)
    return spectrum_clip


# 使用create_clip函数创建频谱视频
freq_max, freq_min = 50, 0
spectrum_output_path = os.path.join(dir, "_spectrum_video.mp4")
spectrum_clip = create_clip(signal, sampling_rate, freq_max, freq_min, spectrum_output_path)