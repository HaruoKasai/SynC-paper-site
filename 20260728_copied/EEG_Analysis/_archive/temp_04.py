import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from neo.io import BlackrockIO
import pandas as pd
import os

# TODO binの切り方とプロットの仕方をちゃんと合わせる。
fig = plt.figure(figsize=(40, 12))
gs = gridspec.GridSpec(5, 3, height_ratios=[2, 0.4, 1, 1, 0.4])
freq_bin = 0.5  # Hz #Fourier plot時のbin
time_bin = 15  # sec #heatmap
freq_max, freq_min = 50, 0
time_list = [  # sec

    # 提取某段时间的设置进行波形分布的分析。单位：秒 sec
    # Time for awake

    [[200, 360], [840, 960], [1560, 1680], [1760, 1920], [3300, 3420]],
    # [[730, 840]],
    # [[1515, 1740]],
    # #[[720, 820]],
    #[[3500, 3600]],

    # Time for REM

    # [[1260, 1320]],
    # [[915, 945]],

    # Time for NREM

    [[1980, 2760], [1380, 1500], [3000, 3180]],
    #[[1660, 1740]]
]

# ns2ファイルのパス
file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\20240710_z161_ROI-ctrl_002\z161_beforeR_004.ns2"
# file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\20240627_z155-1_Pup-Ctrl_1stAC\z155-1_beforeR_003.ns2"
# file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\20240624_impedanec_test_new_cage\66.1kOhm001.ns2"
# file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\20240625_z153-4_Pup-low-2x_2ndAC\afterR_003.ns2"
# file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\20240604_impedance_test_\10kOhm_001.ns2"
# file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\20240605_z146_ctrl_mouse\z147_anesthesia4%_001.ns2"
# file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\20240605_z146_ctrl_mouse\z146_003.ns2"
# file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\z129_SYNCit-C_high_titer_ROI\20240530_z129_SYNCit-C_3rd-ip_before&after_rapa\z129_before_R_002.ns2"
# file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\z151-1_SYNCit-C_pup_20240528_\z151-1_before_rapa_001.ns2"
# file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\20240527_z129_SYNCit-C\z129_after_rapa004.ns2"
# file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\20240529_z129_SYNCit-C_before_rapa_only\z129_before_rapa_001.ns2"
# file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\20240529_z129_SYNCit-C_before_rapa_only\z129_before_rapa_003.ns2"
# file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\20240523_z147-z146_ctrl\z147_002.ns2"
# file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\20240523_z147-z146_ctrl\z146_001.ns2"

dir = os.path.dirname(file_path)


# 生データの周波数スペクトルを計算するための関数
def calculate_spectrum(signal, sampling_rate):
    n = len(signal)
    mean = np.mean(signal)
    signal -= mean #DC offset
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)
    fft_vals = np.fft.fft(signal)
    power_spectrum = np.abs(fft_vals)**2/ n
    return freqs[:n // 2], power_spectrum[:n // 2]


def spectrum_graph(signal, sampling_rate, time_list, freq_max, freq_min, dir, gs, ax1, ax2):  # 全体の平均でnormalizeしている
    ax = fig.add_subplot(gs[ax1])
    ax2 = fig.add_subplot(gs[ax2])
    for status in range(len(time_list)):
        signal_status = np.arange(0)
        time_list_status = time_list[status]
        status_name = ""
        for epoch in range(len(time_list_status)):
            signal_within_epoch = signal[sampling_rate * time_list[status][epoch][0]:sampling_rate * time_list[status][epoch][1]]
            signal_status = np.concatenate(
                (signal_status, signal_within_epoch))  # TODO 厳密には、異なるepochのデータを直接無理やりつなげてからfourier解析するのはよくないだろうが、とりあえず。
            status_name += str(time_list[status][epoch][0]) + "-" + str(time_list[status][epoch][1]) + "s_"
        freqs, power_spectrum = calculate_spectrum(signal_status, sampling_rate)

        # ビンの数を計算
        num_bins = int(np.ceil((freq_max - freq_min) / freq_bin))
        # ビンの境界を設定
        bins = np.linspace(freq_min, freq_max, num_bins + 1)
        # 各ビンに対応するインデックスを取得
        bin_indices = np.digitize(freqs, bins)

        # 各ビンごとのfft_valsの平均を計算
        power_spectrum_means_norm = [power_spectrum[bin_indices == i].sum() for i in range(1, num_bins + 1)] / power_spectrum[bin_indices < np.max(bin_indices)].sum() * 100
        power_spectrum_means = [power_spectrum[bin_indices == i].mean() for i in range(1, num_bins + 1)]
        # ビンの中心を計算
        bin_centers = (bins[:-1] + bins[1:]) / 2

        ax.plot(bin_centers, power_spectrum_means_norm, marker=None, lw=1.5, linestyle='-',color=plt.get_cmap("tab10")(status),label=status_name)  # label=f'Status {status+1}
        ax2.plot(bin_centers, power_spectrum_means, marker=None, lw=1.5, linestyle='-', color=plt.get_cmap("tab10")(status),label=status_name)
    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(0, 12)
    ax.set_xlabel("Hz")
    ax.set_ylabel("Normalized power (%)")
    ax.legend()

    ax2.set_xlim(freq_min, freq_max)
    ax2.set_xlabel("Hz")
    ax2.set_ylabel("Power (μV²)")
    ax2.legend()

    # uV2 = 0 position add line
    horizontal_lines = [0]
    for ax in fig.axes:
        for line in horizontal_lines:
            ax.axhline(y=line, color='grey', linestyle='--')

    # Hz = 1, 4, 8, 12, 30 position add line, to separate δ, θ, α, β, ɤ wave.
    vertical_lines = [1, 4, 8, 12, 30]
    for ax in fig.axes:
        for line in vertical_lines:
            ax.axvline(x=line, color='grey', linestyle='--')


def heatmap(signal, sampling_rate, time_bin, freq_max, freq_min, dir, gs, ax, iv=[0.5, 4], rep="Power(μV2)"): #iv: どの範囲のpowerでvmaxを決めるか
    df = pd.DataFrame()

    time_bin_num = int(len(signal) / sampling_rate / time_bin)
    for t in range(time_bin_num):
        bin_signal = signal[sampling_rate * time_bin * t:sampling_rate * time_bin * (t + 1)]
        freqs, power_spectrum = calculate_spectrum(bin_signal, sampling_rate)
        num_bins = int(np.ceil((freqs.max() - freqs.min()) / freq_bin))
        bins = np.linspace(freqs.min(), freqs.max(), num_bins + 1)
        bin_indices = np.digitize(freqs, bins)
        power_spectrum_means = [power_spectrum[bin_indices == i].mean() for i in range(1, num_bins + 1)]
        power_spectrum_means_norm = [power_spectrum[bin_indices == i].sum() for i in range(1, num_bins + 1)] / power_spectrum[bin_indices < np.max(bin_indices)].sum() * 100
        if rep=="Normalized power (%)":
            df = df.append(pd.Series(power_spectrum_means_norm), ignore_index=True)
        else:
            df = df.append(pd.Series(power_spectrum_means), ignore_index=True)

        bin_centers = (bins[:-1] + bins[1:]) / 2
    df = df.loc[:, freq_min / freq_bin:freq_max / freq_bin]

    ax = fig.add_subplot(gs[ax])
    # vmax適当に決める
    range_mean, range_std, range_median = df.loc[:, iv[0]:iv[1]].values.mean(), df.loc[:, iv[0]:iv[1]].values.std(), np.median(df.loc[:, iv[0]:iv[1]].values)
    print(range_median)
    print(range_mean)
    print(range_std)
    # vmax = range_median + range_std * 1.5
    vmax = range_median*2

    im = ax.imshow(df.T, aspect='auto', cmap='magma', origin='lower', vmin=0, vmax=vmax)

    xticks = np.arange(0, int(60 * 60 / time_bin) + 1, int(1 * 60 / time_bin))
    # record_time = len(signal)/sampling_rate/60 #min
    ax.set_title(rep)
    xtick_labels = np.arange(0, 60 * 60 + 1, 60)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)


    yticks = np.arange(0, freq_max / freq_bin + 1, int(freq_max / freq_bin / 5))
    ytick_labels = np.arange(0, freq_max + 1, int(freq_max / 5))
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)

    ax.set_xlabel("sec")
    ax.set_ylabel("Hz")
    # fig.colorbar(im, ax=ax)


def plot_raw(raw, sampling_rate, ylabel, ylim, gs, ax, time_list=None):
    ax = fig.add_subplot(gs[ax])
    ax.plot(raw, lw=0.1)
    n = len(raw)
    xticks = np.arange(0, 60 * 60 * sampling_rate + 1, 1 * 60 * sampling_rate)
    # record_time = n / sampling_rate / 60  # min
    xtick_labels = np.arange(0, 61, 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("min")
    ax.set_ylabel(ylabel + " (uV)")
    ax.set_ylim(-ylim, ylim)
    ax.margins(x=0)

    if time_list:
        for status, time_list_status in enumerate(time_list):
            for start, end in time_list_status:
                ax.axvspan(start * sampling_rate, end * sampling_rate, color=plt.get_cmap("tab10")(status), alpha=0.3)


# Neoを使用してns2ファイルを読み込む
reader = BlackrockIO(filename=file_path)
block = reader.read_block()

# 生データの取得（RawSignalChannel）
raw_signals = [seg.analogsignals[0] for seg in block.segments]

# EEG (Ch1)
raw_signal = raw_signals[0][:, 0]
sampling_rate = int(raw_signal.sampling_rate.magnitude)
signal = raw_signal.magnitude.flatten()
spectrum_graph(signal, sampling_rate, time_list, freq_max, freq_min, dir, gs, ax1=(0, 0), ax2=(0, 1))
heatmap(signal, sampling_rate, time_bin, freq_max, freq_min, dir, gs, ax=(2, slice(0, 3)),iv = [0.5, 4])
heatmap(signal, sampling_rate, time_bin, freq_max, freq_min, dir, gs, ax=(3, slice(0, 3)), iv=[12, 30], rep="Normalized power (%)")
plot_raw(signal, sampling_rate, ylabel="EEG", ylim=500, gs=gs, ax=(1, slice(0, 3)), time_list=time_list)

# EMG (Ch2, CH3)
ch2_signal, ch3_signal = raw_signals[0][:, 1], raw_signals[0][:, 2]
emg = ch2_signal - ch3_signal
plot_raw(emg, sampling_rate, ylabel="EMG", ylim=300, gs=gs, ax=(4, slice(0, 3)))

plt.tight_layout()
file_num = os.path.basename(file_path)[-7:-4]
fig.savefig(os.path.join(dir, "_graph_" + file_num + ".pdf"), dpi=300, transparent=True)

# 新しい図を作成し、選択した時間段的时序数据を绘制 #TODO 周さん追加部分。simpleに直す
fig2, axs = plt.subplots(len(time_list) * 2, 1, figsize=(20, 5 * len(time_list)))
for status, time_list_status in enumerate(time_list):
    for start, end in time_list_status:
        start_time = max(end - 5, start)
        eeg_ax = axs[2 * status] if len(time_list) > 1 else axs
        emg_ax = axs[2 * status + 1] if len(time_list) > 1 else axs
        eeg_ax.plot(signal[start_time * sampling_rate:end * sampling_rate], lw=0.5, color=plt.get_cmap("tab10")(status))
        eeg_ax.set_title(f"EEG from {start_time}s to {end}s")
        eeg_ax.set_xlabel("Time (samples)")
        eeg_ax.set_ylabel("Amplitude (uV)")
        eeg_ax.margins(x=0)
        eeg_ax.set_ylim(-500, 500)
        emg_ax.plot(emg[start_time * sampling_rate:end * sampling_rate], lw=0.5, color=plt.get_cmap("tab10")(status))
        emg_ax.set_title(f"EMG from {start_time}s to {end}s")
        emg_ax.set_xlabel("Time (samples)")
        emg_ax.set_ylabel("Amplitude (uV)")
        emg_ax.margins(x=0)
        emg_ax.set_ylim(-200, 200)
fig2.tight_layout()
fig2.savefig(os.path.join(dir, "_selected_times_" + file_num + ".pdf"), dpi=300, transparent=True)