import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter, stft, find_peaks
from matplotlib.backends.backend_pdf import PdfPages
from neo.io import BlackrockIO


def select_folder():
    """
    弹出文件夹选择对话框，供用户选择数据目录。
    """
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select the 'data' directory")
    return folder_path

def rotary_analysis(A, B, Z, exp_duration, time_bin=1, radius=50, resolution=100):
    """
    分析 Rotary Encoder 数据。
    """
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
    """
    绘制时间序列数据。
    """
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


def extract_emg_data(file_path):
    """
    从指定文件中提取 EEG 和 EMG 数据。
    Parameters:
        file_path (str): 数据文件的路径。
    Returns:
        tuple: EMG 数据、EEG 数据和采样率。
    """
    try:
        reader = BlackrockIO(filename=file_path)
        block = reader.read_block()
        raw_signals = [seg.analogsignals[0] for seg in block.segments]

        # 读取 anp1 数据
        analog_input_signals = [seg.analogsignals[1] for seg in block.segments]
        anp2_signal = analog_input_signals[0][:, 1]
        anp3_signal = analog_input_signals[0][:, 2]

        # 转换为一维数组并获取采样率
        breathe = anp2_signal.magnitude.flatten()
        tem = anp3_signal.magnitude.flatten()

        ch1_signal = raw_signals[0][:, 1]  # EEG 数据
        ch0_signal = raw_signals[0][:, 0]
        ch2_signal, ch3_signal = raw_signals[0][:, 2], raw_signals[0][:, 3]  # EMG 数据
        sampling_rate = int(ch1_signal.sampling_rate.magnitude)

        emg = (ch2_signal - ch3_signal).magnitude.flatten()
        eeg1 = ch0_signal.magnitude.flatten()  # Motor
        eeg2 = ch1_signal.magnitude.flatten()  # Visual
        eeg3 = (ch0_signal - ch1_signal).magnitude.flatten()  # M-V

        start_time = raw_signals[0].t_start.magnitude
        digital_input_data = None

        for segment in block.segments:
            for event_array in segment.events:
                if event_array.name == 'digital_input_port':
                    digital_input_data = event_array
        if digital_input_data is not None:
            times = digital_input_data.times.rescale('s').magnitude
            labels = digital_input_data.labels.astype(int)
            num_channels = 16
            bitwise_states = np.array([[(label >> i) & 1 for i in range(num_channels)] for label in labels])
            channel_times = {f'channel_{i}': [] for i in range(num_channels)}
            for i in range(num_channels):
                changes = np.where(np.diff(bitwise_states[:, i]) != 0)[0] + 1
                channel_times[f'channel_{i}'] = times[changes] - start_time
        else:
            channel_times = {}
            print(f"Failed to extract digital_data")

        return emg, [eeg1, eeg2, eeg3], sampling_rate, breathe, tem, channel_times, digital_input_data
    except Exception as e:
        print(f"Failed to extract data from {file_path}: {e}")
        return None, None, None


def calculate_band_power(freqs, power_spectrum, lower_bound, upper_bound, normalize=False, to_db=False):
    """
    计算指定频段的功率。
    Parameters:
        freqs (numpy.ndarray): 频率数组。
        power_spectrum (numpy.ndarray): 功率谱数组。
        lower_bound (float): 频段下限。
        upper_bound (float): 频段上限。
        normalize (bool): 是否归一化。
        to_db (bool): 是否转换为分贝。
    Returns:
        float: 频段功率。
    """
    band_mask = (freqs >= lower_bound) & (freqs < upper_bound)
    band_power = np.sum(power_spectrum[band_mask])

    if normalize:
        band_power = band_power / np.sum(power_spectrum) * 100
    if to_db:
        band_power = 10 * np.log10(band_power + 1e-10)
    return band_power


def plot_timeseries_power(eeg, sampling_rate, time_bin, output_dir, ax, lw):
    """
    绘制时间序列功率图。
    Parameters:
        eeg (numpy.ndarray): EEG 数据。
        sampling_rate (int): 采样率。
        time_bin (int): 时间窗口大小（秒）。
        output_dir (str): 保存数据的目录。
        ax (matplotlib.axes.Axes): 子图。
    """
    freqs, power_spectrum = calculate_spectrum(eeg, sampling_rate)
    columns = ['Delta (0.5-4 Hz)', 'Theta (4-8 Hz)', 'Alpha (8-12 Hz)', 'Beta (12-30 Hz)', 'Gamma (30-100 Hz)']
    powers = {col: [] for col in columns}
    time_bins = int(len(eeg) / (sampling_rate * time_bin))

    for t in range(time_bins):
        start = t * sampling_rate * time_bin
        end = start + sampling_rate * time_bin
        segment = eeg[start:end]
        freqs, power_spectrum = calculate_spectrum(segment, sampling_rate)

        powers['Delta (0.5-4 Hz)'].append(calculate_band_power(freqs, power_spectrum, 0.5, 4, to_db=True))
        powers['Theta (4-8 Hz)'].append(calculate_band_power(freqs, power_spectrum, 4, 8, to_db=True))
        powers['Alpha (8-12 Hz)'].append(calculate_band_power(freqs, power_spectrum, 8, 12, to_db=True))
        powers['Beta (12-30 Hz)'].append(calculate_band_power(freqs, power_spectrum, 12, 30, to_db=True))
        powers['Gamma (30-100 Hz)'].append(calculate_band_power(freqs, power_spectrum, 30, 100, to_db=True))

    df = pd.DataFrame(powers)
    df.to_csv(os.path.join(output_dir, "power_time_series.csv"), index=False)

    # 绘制时间序列图
    for col in columns:
        ax.plot(range(1, len(df) + 1), df[col], label=col)

    record_time = len(eeg) / sampling_rate  # sec
    ax.set_title("Band Power Time Series")
    xticks = np.arange(0, int(record_time / time_bin) + 1, int(5 * 60 / time_bin))
    xtick_labels = np.arange(0, int(record_time / 60) + 1, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Power (dB)")
    ax.set_ylim(55, 85)
    ax.legend(loc="upper right")


def calculate_spectrum(eeg, sampling_rate):
    """
    计算信号的频谱和功率谱。
    Parameters:
        signal (numpy.ndarray): 输入信号。
        sampling_rate (int): 信号的采样率。
    Returns:
        tuple: 频率和功率谱。
    """
    n = len(eeg)
    eeg -= np.mean(eeg)  # 去掉直流偏移
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)
    fft_vals = np.fft.fft(eeg)
    power_spectrum = np.abs(fft_vals) ** 2 / n
    return freqs[:n // 2], power_spectrum[:n // 2]


def plot_heatmap(ax, t, f, power, title, ylabel, freq_limit, cmap, power_range):
    """
    绘制热图的通用函数。
    Parameters:
        ax (matplotlib.axes.Axes): 要绘制的子图。
        t (numpy.ndarray): 时间数据。
        f (numpy.ndarray): 频率数据。
        power (numpy.ndarray): 功率谱数据。
        title (str): 图表标题。
        ylabel (str): Y 轴标签。
        freq_limit (int): 频率上限。
        cmap (str): 配色方案。
        power_range (tuple): 功率范围 (vmin, vmax)。
    """
    pcm = ax.pcolormesh(
        t, f, power, shading='gouraud', rasterized=True, cmap=cmap,
        vmin=power_range[0], vmax=power_range[1]
    )
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, freq_limit)
    # cbar = plt.colorbar(pcm, ax=ax)
    # cbar.set_label('Power' + (' (µV²)' if 'Linear' in title else ' (dB)'))


def compute_breathing_frequency(breathe, sampling_rate, segment_duration=1):
    """
    使用峰值和谷值检测计算每秒的呼吸频率，并过滤掉小波动。

    Parameters:
        breathe (numpy.ndarray): 输入信号（呼吸波动信号）。
        sampling_rate (int): 信号采样率（Hz）。
        segment_duration (int): 每段的时长（秒）。
        amplitude_threshold (float): 峰值与谷值之间的最小振幅差。

    Returns:
        numpy.ndarray: 每秒的呼吸频率（单位：次/秒）。
    """
    print(breathe)
    amplitude_threshold = 5
    # 平滑信号
    smoothed_signal = savgol_filter(breathe, window_length=51, polyorder=3)

    segment_length = segment_duration * sampling_rate
    breathing_frequencies = []
    amplitude_differences = []

    for start in range(0, len(breathe), segment_length):
        segment = smoothed_signal[start:start + segment_length]
        if len(segment) < segment_length:
            break

        # 检测峰值（吸气高峰）和谷值（呼气低谷）
        peaks, _ = find_peaks(segment, distance=sampling_rate // 2)
        troughs, _ = find_peaks(-segment, distance=sampling_rate // 2)

        # 合并峰值和谷值，并排序
        all_extremes = np.sort(np.concatenate([peaks, troughs]))

        # 计算相邻极值的差值
        valid_cycles = 0
        total_amplitude_diff = 0
        for i in range(1, len(all_extremes)):
            peak_idx, trough_idx = all_extremes[i - 1], all_extremes[i]
            amplitude_diff = abs(segment[peak_idx] - segment[trough_idx])

            # 如果振幅差大于阈值，计为一个有效周期
            if amplitude_diff >= amplitude_threshold:
                valid_cycles += 1
                total_amplitude_diff += amplitude_diff

        # 呼吸频率 = 有效周期数 / 时间段长度
        breathing_frequencies.append(valid_cycles / segment_duration)

        # 计算一秒内所有呼吸周期的振幅差的平均值
        if valid_cycles > 0:
            average_amplitude_diff = total_amplitude_diff / valid_cycles
        else:
            average_amplitude_diff = 0
        amplitude_differences.append(average_amplitude_diff)

    return np.array(breathing_frequencies), np.array(amplitude_differences)


def save_eeg_analysis_results(eeg, emg, sampling_rate, breathe, tem, breathing_frequencies, amplitude_differences, output_dir, pdf_name, velocities=None, time_centers=None):
    """
    分析 EEG 和 EMG 数据，并将结果保存为 PDF。
    Parameters:
        eeg (numpy.ndarray): EEG 数据。
        emg (numpy.ndarray): EMG 数据。
        sampling_rate (int): 信号采样率。
        output_dir (str): 结果保存的目录。
    Returns:
        str: PDF 文件路径。
    """
    pdf_path = os.path.join(output_dir, pdf_name)
    fig = plt.figure(figsize=(15, 25))
    gs = gridspec.GridSpec(10, 1, height_ratios=[0.4, 1, 1, 1, 1, 0.4, 1, 1, 1, 1])
    plt.subplots_adjust(hspace=0.5)

    # STFT 参数
    epoch_length = 2
    nperseg = int(epoch_length * sampling_rate)
    f_stft, t_stft, Zxx = stft(eeg, fs=sampling_rate, nperseg=nperseg, noverlap=nperseg // 2)
    linear_power = np.abs(Zxx) ** 2
    dB_power = 10 * np.log10(linear_power + 1e-10)

    # 绘制图表
    # 1. Raw EEG Signal
    ax1 = fig.add_subplot(gs[0, 0])
    time_eeg = np.arange(len(eeg)) / sampling_rate  # 转换为秒
    ax1.plot(time_eeg, eeg, lw=0.1)
    ax1.set_title("Raw EEG Signal")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (µV)")
    ax1.set_ylim(-500, 500)
    ax1.set_xlim(0, len(eeg) / sampling_rate)

    # 2. STFT Linear Power
    ax2 = fig.add_subplot(gs[1, 0])
    plot_heatmap(ax2, t_stft, f_stft, linear_power, "STFT Linear Power", "Frequency (Hz)", 100, "rainbow", [-10, 100])

    # 3. STFT dB Power
    ax3 = fig.add_subplot(gs[2, 0])
    plot_heatmap(ax3, t_stft, f_stft, dB_power, "STFT dB Power", "Frequency (Hz)", 100, "rainbow", [-10, 33])

    # 4. STFT Low-Frequency dB Power
    ax4 = fig.add_subplot(gs[3, 0])
    plot_heatmap(ax4, t_stft, f_stft, dB_power, "STFT dB Power (0-20 Hz)", "Frequency (Hz)", 20, "rainbow", [-10, 33])

    # 5. Band Power Time Series
    ax5 = fig.add_subplot(gs[4, 0])
    plot_timeseries_power(eeg, sampling_rate, time_bin=2, output_dir=output_dir, ax=ax5, lw=0.1)

    # 6. Raw EMG Signal
    ax6 = fig.add_subplot(gs[5, 0])
    time_emg = np.arange(len(emg)) / sampling_rate  # 转换为秒
    ax6.plot(time_emg, emg, lw=0.1, color='orange')
    ax6.set_title("Raw EMG Signal")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Amplitude (µV)")
    ax6.set_xlim(0, len(emg) / sampling_rate)
    ax6.set_ylim(-1000, 1000)

    # 7. Breathe
    ax7 = fig.add_subplot(gs[6, 0])
    time_breathe = np.arange(len(breathe)) / sampling_rate  # 转换为秒
    ax7.plot(time_breathe, breathe, lw=0.1, color='red')
    ax7.set_title("Breathe")
    ax7.set_xlabel("Time (s)")
    ax7.set_ylabel("Amplitude (µV)")
    ax7.set_xlim(0, len(breathe) / sampling_rate)
    ax7.set_ylim(1900, 2400)

    # 8. Velocity
    ax8 = fig.add_subplot(gs[7, 0])
    if velocities is not None and time_centers is not None:
        plot_binned_timeseries(velocities, time_centers, ylabel="Velocity (mm/s)", ax=ax8)
    else:
        ax8.set_title("No Rotary Encoder Data Available")
        ax8.set_xlabel("Time (s)")
        ax8.set_ylabel("Velocity (mm/s)")

    # 9. Breathe
    ax9 = fig.add_subplot(gs[8, 0])

    # 时间轴
    time_avg = np.arange(len(breathing_frequencies))

    # 创建第一个纵轴
    ax9.plot(time_avg, breathing_frequencies, label="Average Frequency", color="purple", linewidth=0.1)
    ax9.set_title("Average Frequency and Amplitude Over Time", fontsize=14)
    ax9.set_xlabel("Time (s)", fontsize=12)
    ax9.set_ylabel("Average Frequency", fontsize=12, color="purple")
    ax9.tick_params(axis='y', labelcolor="purple")
    ax9.set_ylim([0, 5])  # 设置第一个纵轴的范围

    # 创建第二个纵轴
    ax9_2 = ax9.twinx()  # 创建共享 x 轴的新纵轴
    ax9_2.plot(time_avg, amplitude_differences, label="Average Amplitude", color="orange", linewidth=0.1)
    ax9_2.set_ylabel("Average Amplitude", fontsize=12, color="orange")
    ax9_2.tick_params(axis='y', labelcolor="orange")
    ax9_2.set_ylim([0, 200])  # 设置第二个纵轴的范围

    # 设置图例
    lines1, labels1 = ax9.get_legend_handles_labels()
    lines2, labels2 = ax9_2.get_legend_handles_labels()
    ax9.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)


    # 10. Temperature
    ax10 = fig.add_subplot(gs[9, 0])
    time_tem = np.arange(len(tem)) / sampling_rate  # 转换为秒
    ax10.plot(time_tem, tem, lw=0.1, color='red')
    ax10.set_title("Rectal Temperature")
    ax10.set_xlabel("Time (s)")
    ax10.set_ylabel("Amplitude (µV)")
    ax10.set_xlim(0, len(tem) / sampling_rate)
    ax10.set_ylim(1750, 3500)

    plt.tight_layout()

    # 保存 PDF
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)

    print(f"Analysis results saved to {pdf_path}")
    return pdf_path


def main():
    """
    主程序：选择文件夹，提取和分析所有 .ns3 或 .ns2 文件的数据。
    """
    # 选择数据文件夹
    data_folder = select_folder()

    # 遍历根目录及其子目录中的 .ns3 或 .ns2 文件
    for root_dir, _, files in os.walk(data_folder):
        # 过滤出 .ns3 和 .ns2 文件
        valid_files = [file for file in files if file.endswith(('.ns3', '.ns2'))]

        if valid_files:  # 如果当前目录有有效文件
            # 创建当前目录下的 results 文件夹
            output_dir = os.path.join(root_dir, "results")
            os.makedirs(output_dir, exist_ok=True)

            # 遍历找到的文件
            for valid_file in valid_files:
                # 生成完整的文件路径
                file_path = os.path.join(root_dir, valid_file)

                # 提取并处理数据
                emg, eeg_list, sampling_rate, breathe, tem, channel_times, digital_input_data = extract_emg_data(file_path)

                # 定义每个 EEG 数据对应的 PDF 名称
                eeg_pdf_names = {
                    0: "Motor_Respiration-Rectal_eeg_analysis_results.pdf",
                    1: "Visual_Respiration-Rectal_eeg_analysis_results.pdf",
                    2: "M-V_Respiration-Rectal_analysis_results.pdf",
                }

                if digital_input_data is not None:
                    A_channel, B_channel, Z_channel = 4, 6, 8
                    A,B,Z = channel_times["channel_"+str(A_channel)], channel_times["channel_"+str(B_channel)], channel_times["channel_"+str(Z_channel)]
                    velocities, time_centers = rotary_analysis(A, B, Z, exp_duration=len(eeg_list[0]) / sampling_rate)
                else:
                    velocities, time_centers = None, None

                # 计算校正信号和其他分析数据
                breathing_frequencies, amplitude_differences = compute_breathing_frequency(breathe, sampling_rate)

                if eeg_list is not None and emg is not None:
                    for i, eeg in enumerate(eeg_list):
                        # 根据索引找到对应的 PDF 名称
                        pdf_name = eeg_pdf_names.get(i, f"default_eeg_analysis_{i}.pdf")
                    # 保存分析结果到当前目录的 results 文件夹
                        save_eeg_analysis_results(
                            eeg, emg, sampling_rate, breathe, tem, breathing_frequencies, amplitude_differences,
                            output_dir, pdf_name=pdf_name, velocities=velocities, time_centers=time_centers
                        )


if __name__ == "__main__":
    main()
