import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from neo.io import BlackrockIO
import pandas as pd
import os

#TODO binの切り方とプロットの仕方をちゃんと合わせる。
fig = plt.figure(figsize=(40,10))
gs = gridspec.GridSpec(4, 3, height_ratios=[2, 0.4, 1, 0.4])
freq_bin = 0.5 #Hz #Fourier plot時のbin
time_bin = 15 #sec #heatmap
freq_max, freq_min = 50,0
time_list = [           #sec
    # [[0,180]],
    # [[300,500]],
    # [[630, 750]],
    # [[780, 860]],
    # [[870, 980]],
    # [[990, 1020]],
    # [[1020, 1290]],
    # [[1860, 2460]],
    # [[3200, 3240]],
    # [[3300, 3400]],


    # [[1080, 1500]],
    # [[2220, 2580]],
    # [[2700, 3200]],

    # [[1620, 2580]],
    # [[3360, 3420]],

    # 提取某段时间的设置进行波形分布的分析。单位：秒 sec
    [[279, 291], [311, 323], [381, 399]],
    [[476, 498], [555, 583], [586, 637]],
    #[[703, 720], [780, 790], [795, 810]],
    #[[3060, 3240]],
    #[[590, 660]],
    #[[730, 740]]
]

# ns2ファイルのパス
# file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\20240604_impedance_test_\10kOhm_001.ns2"
file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\20240605_z146_ctrl_mouse\z147_anesthesia4%_001.ns2"
# file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\20240523_z147-z146_ctrl\z146_001.ns2"
# file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\z129_SYNCit-C_high_titer_ROI\20240530_z129_SYNCit-C_3rd-ip_before&after_rapa\z129_after_R_003.ns2"
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
    print (n)
    freqs = np.fft.fftfreq(n, d=1/sampling_rate)
    fft_vals = np.fft.fft(signal)
    fft_vals = np.abs(fft_vals)
    return freqs[:n//2], fft_vals[:n//2]

def spectrum_graph (signal, sampling_rate, time_list, freq_max, freq_min, dir, gs, ax1, ax2): #全体の平均でnormalizeしている
    ax = fig.add_subplot(gs[ax1])
    ax2 = fig.add_subplot(gs[ax2])
    for status in range(len(time_list)):
        signal_status = np.arange(0)
        time_list_status = time_list[status]
        status_name =""
        for epoch in range (len(time_list_status)):
            signal_within_epoch = signal[sampling_rate*time_list[status][epoch][0]:sampling_rate*time_list[status][epoch][1]]
            signal_status = np.concatenate((signal_status, signal_within_epoch)) #TODO 厳密には、異なるepochのデータを直接無理やりつなげてからfourier解析するのはよくないだろうが、とりあえず。
            status_name += str(time_list[status][epoch][0]) + "-" + str(time_list[status][epoch][1])+"s_"
        freqs, fft_vals = calculate_spectrum(signal_status, sampling_rate)

        # ビンの数を計算
        num_bins = int(np.ceil((freq_max - freq_min) / freq_bin))
        # ビンの境界を設定
        bins = np.linspace(freq_min, freq_max, num_bins + 1)
        # 各ビンに対応するインデックスを取得
        bin_indices = np.digitize(freqs, bins)

        # 各ビンごとのfft_valsの平均を計算
        fft_vals_means_norm = [fft_vals[bin_indices == i].sum() for i in range(1, num_bins + 1)] / fft_vals[bin_indices < np.max(bin_indices)].sum() * 100
        fft_vals_means = [fft_vals[bin_indices == i].mean() for i in range(1, num_bins + 1)]
        # ビンの中心を計算
        bin_centers = (bins[:-1] + bins[1:]) / 2

        ax.plot(bin_centers, fft_vals_means_norm, marker=None, lw=0.7, linestyle='-', color=plt.get_cmap("tab10")(status), label=status_name)  #label=f'Status {status+1}
        ax2.plot(bin_centers, fft_vals_means, marker=None, lw=0.7, linestyle='-', color=plt.get_cmap("tab10")(status), label=status_name)
    ax.set_xlim(freq_min,freq_max)
    ax.set_ylim(0, 4.5)
    ax.set_xlabel("Hz")
    ax.set_ylabel("Normalized power (%)")
    ax.legend()

    ax2.set_xlim(freq_min, freq_max)
    ax2.set_xlabel("Hz")
    ax2.set_ylabel("a.u.")
    ax2.legend()
def heatmap(signal, sampling_rate, time_bin, freq_max, freq_min, dir, gs, ax):
    df = pd.DataFrame()

    time_bin_num = int(len(signal) / sampling_rate / time_bin)
    for t in range(time_bin_num):
        bin_signal = signal[sampling_rate * time_bin * t:sampling_rate * time_bin * (t + 1)]
        freqs, fft_vals = calculate_spectrum(bin_signal, sampling_rate)
        print(freqs)
        num_bins = int(np.ceil((freqs.max() - freqs.min()) / freq_bin))
        bins = np.linspace(freqs.min(), freqs.max(), num_bins + 1)
        bin_indices = np.digitize(freqs, bins)
        fft_vals_means = [fft_vals[bin_indices == i].mean() for i in range(1, num_bins + 1)]
        df = df.append(pd.Series(fft_vals_means), ignore_index=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
    print(df.shape)
    df = df.loc[:, freq_min/freq_bin:freq_max/freq_bin]

    ax = fig.add_subplot(gs[ax])
    # vmax適当に決める
    delta_mean, delta_std, delta_median = df.loc[:, 0:4].values.mean(), df.loc[:, 0:4].values.std(), np.median(df.loc[:, 0:4].values)
    vmax = delta_median + delta_std * 3
    im = ax.imshow(df.T, aspect='auto', cmap='magma', origin='lower', vmin=0, vmax=vmax)

    xticks = np.arange (0, int(60*60/time_bin)+1, int(1*60/time_bin))
    # record_time = len(signal)/sampling_rate/60 #min
    xtick_labels = np.arange(0,60*60+1,60)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    yticks = np.arange(0, freq_max/freq_bin +1, int(freq_max/freq_bin/5))
    ytick_labels = np.arange(0, freq_max+1, int(freq_max/5))
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)


    ax.set_xlabel("sec")
    ax.set_ylabel("Hz")
    # fig.colorbar(im, ax=ax)

def plot_raw (raw, sampling_rate, ylabel, ylim, gs, ax):
    ax = fig.add_subplot(gs[ax])
    ax.plot(raw, lw=0.1)
    n = len(raw)
    xticks = np.arange(0, 60*60*sampling_rate+1, 1*60*sampling_rate)
    # record_time = n / sampling_rate / 60  # min
    xtick_labels = np.arange(0, 61, 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("min")
    ax.set_ylabel(ylabel + " (uV)")
    ax.set_ylim(-ylim, ylim)
    ax.margins(x=0)



# Neoを使用してns2ファイルを読み込む
reader = BlackrockIO(filename=file_path)
block = reader.read_block()

# 生データの取得（RawSignalChannel）
raw_signals = [seg.analogsignals[0] for seg in block.segments]

#EEG (Ch1)
raw_signal = raw_signals[0][:,0]
sampling_rate = int(raw_signal.sampling_rate.magnitude)
signal = raw_signal.magnitude.flatten()
spectrum_graph(signal, sampling_rate, time_list, freq_max, freq_min, dir, gs, ax1=(0,0), ax2=(0,1))
heatmap(signal, sampling_rate, time_bin, freq_max, freq_min, dir, gs, ax = (2, slice(0,6)))
plot_raw(signal, sampling_rate , ylabel="EEG", ylim=500, gs = gs, ax = (1, slice(0,6)))

#EMG (Ch2, CH3)
ch2_signal, ch3_signal = raw_signals[0][:,1], raw_signals[0][:,2]
emg = ch2_signal - ch3_signal
plot_raw(emg, sampling_rate, ylabel="EMG", ylim=300, gs = gs, ax = (3, slice(0,6)))


plt.tight_layout()
file_num = os.path.basename(file_path)[-7:-4]
fig.savefig(os.path.join(dir, "_graph_"+file_num+".pdf"), dpi=300, transparent=True)

