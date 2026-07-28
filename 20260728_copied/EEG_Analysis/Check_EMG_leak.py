import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from neo.io import BlackrockIO
import pandas as pd
from tkinter import filedialog
import glob


# 生データの周波数スペクトルを計算するための関数
def calculate_spectrum(signal, sampling_rate):
    n = len(signal)
    mean = np.mean(signal)
    signal -= mean #DC offset #TODO　まず全体で引いてから
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)
    fft_vals = np.fft.fft(signal)
    power_spectrum = np.abs(fft_vals)**2/ n


    return freqs[:n // 2], power_spectrum[:n // 2] #ナイキスト周波数まで

def spectrum_graph(signal, sampling_rate, freq_max, freq_min, gs, ax1, ax2, ax3, color, label):  # 全体の平均でnormalizeしている
    ax = fig.add_subplot(gs[ax1])
    ax2 = fig.add_subplot(gs[ax2])
    ax3 = fig.add_subplot(gs[ax3])

    num_bins = int(np.ceil((freq_max - freq_min) / freq_bin))  # ビンの数を計算

    bins = np.linspace(freq_min, freq_max, num_bins + 1)  # ビンの境界を設定
    bin_centers = (bins[:-1] + bins[1:]) / 2

    #全体平均
    freqs, power_spectrum = calculate_spectrum(signal, sampling_rate)
    bin_indices = np.digitize(freqs, bins)  # 各ビンに対応するインデックスを取得
    binned_power_spectrum = [power_spectrum[bin_indices == i].mean() for i in range(1, num_bins + 1)]
    binned_power_spectrum_norm = binned_power_spectrum / np.sum(binned_power_spectrum) * 100
    ax.plot(bin_centers, binned_power_spectrum_norm, marker=None, lw=1, linestyle='-', color=color,label=label)
    ax2.plot(bin_centers, binned_power_spectrum, marker=None, lw=1, linestyle='-', color=color,label=label)
    ax3.plot(bin_centers, binned_power_spectrum, marker=None, lw=1, linestyle='-', color=color,label=label)

    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(0, 12)
    ax.set_xlabel("Hz")
    ax.set_ylabel("Normalized power (%)")
    ax.legend()

    ax2.set_xlim(freq_min, freq_max)
    ax2.set_xlabel("Hz")
    ax2.set_ylabel("Power (μV²)")
    ax2.set_ylim(0, 600000)
    ax2.legend()

    ax3.set_xlim(freq_min, freq_max)
    ax3.set_xlabel("Hz")
    ax3.set_ylabel("Power (μV²)")
    ax3.set_yscale('log')
    ax3.set_ylim(1000, 1500000)
    ax3.legend()

    # uV2 = 0 position add line
    horizontal_lines = [0]
    vertical_lines = [0.5, 4, 8, 12, 30]  # to separate δ, θ, α, β, ɤ wave.
    ax_list = [ax, ax2, ax3]
    for ax in ax_list:
        for line in horizontal_lines:
            ax.axhline(y=line, color='grey', linestyle='--')
        for line in vertical_lines:
            ax.axvline(x=line, color='grey', linestyle='--')
        ax.legend(ncol=2,fontsize =5)

def heatmap(signal, sampling_rate, time_bin, freq_max, freq_min, dir, gs, ax, vrange=[0,100], normalization=False, dB=False, label=None):
    df = pd.DataFrame()

    time_bin_num = int(len(signal) / sampling_rate / time_bin)
    for t in range(time_bin_num):
        bin_signal = signal[sampling_rate * time_bin * t:sampling_rate * time_bin * (t + 1)]
        freqs, power_spectrum = calculate_spectrum(bin_signal, sampling_rate)
        title_dB = ""
        if dB:
            power_spectrum = 10 * np.log10(power_spectrum)  # TODO 確認 power spectrumの平均(or sum)は常に一定？(その値は何を意味？)それを基準にdB計算しているということでok?
            title_dB = " (dB)"
        num_bins = int(np.ceil((freq_max - freq_min) / freq_bin))
        bins = np.linspace(freq_min, freq_max, num_bins + 1)
        bin_indices = np.digitize(freqs, bins)
        power_spectrum_means = [power_spectrum[bin_indices == i].mean() for i in range(1, num_bins + 1)]
        if normalization:
            power_spectrum_means_norm = [power_spectrum[bin_indices == i].sum() for i in range(1, num_bins + 1)] / \
                                        power_spectrum[bin_indices < np.max(bin_indices)].sum() * 100
            df = pd.concat([df, pd.DataFrame([power_spectrum_means_norm])], ignore_index=True)
                # df.append(pd.Series(power_spectrum_means_norm), ignore_index=True))
            title_norm=" Normalized"
            title_unit = " (%)"
        else:
            df = pd.concat([df, pd.DataFrame([power_spectrum_means])], ignore_index=True)
            title_norm=""
            title_unit = " (uV2)"
    df = df.loc[:, (freq_min / freq_bin):(freq_max / freq_bin)]
    ax = fig.add_subplot(gs[ax])

    ax.imshow(df.T, aspect='auto', cmap='rainbow', origin='lower', vmin=vrange[0], vmax=vrange[1],
                   extent = [0, 0+df.shape[0], 0, 0+df.shape[1]]) #extentを指定しないと、binの半分だけずれて表示されてしまう
    record_time = len(signal) / sampling_rate #sec
    xticks = np.arange(0, int(record_time / time_bin) + 1, int(5 * 60 / time_bin))
    ax.set_title("Power"+title_norm+title_unit+title_dB)
    xtick_labels = np.arange(0, int(record_time/60) + 1, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    yticks = np.arange(0, freq_max / freq_bin + 1, int(freq_max / freq_bin / 5))
    ytick_labels = np.arange(0, freq_max + 1, int(freq_max / 5))
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)

    ax.set_xlabel("min")
    ax.set_ylabel(label + " (Hz)", fontsize = 18)


def plot_raw(raw, sampling_rate, ylabel, ylim, gs, ax, color):
    ax = fig.add_subplot(gs[ax])
    ax.plot(raw, lw=0.4, c=color)
    n = len(raw)
    xticks = np.arange(0, n + 1, 2 * sampling_rate)
    xtick_labels = np.arange(0, int(n/sampling_rate)+1, 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("sec")
    ax.set_ylabel(ylabel + " (uV)", fontsize=18)
    ax.set_ylim(-ylim, ylim)
    ax.margins(x=0)

################################################################################
# ns2ファイルのパス
freq_bin = 0.5  # Hz #Fourier plot時のbin
time_bin = 3  # sec #heatmap
freq_max, freq_min = 100, 0

mouse_list=[
    # r"X:\Behavior\EEG\SYNCit-C_Pup\20250122_z212-Pup-Cript-IRES-5x-2p-Occipital-2uL_(9w-M1V1-Ce_male)-6P",
    r"X:\Behavior\EEG\_Fuction-Generator-Test\20250205_Function-Generator-Test",
]
for mouse_dir in mouse_list:

    # exp =glob.glob(os.path.join(mouse_dir, "[!_]*"))[1] #各マウス、はじめの30分は安定していないかもしれないからとりあえず、そのあとの実験を使う。Before A/C
    exp_list = glob.glob(os.path.join(mouse_dir, "[!_]*"))
    for exp in exp_list:
        mouse_name, exp_name= os.path.basename(mouse_dir), os.path.basename(exp)


        fig = plt.figure(figsize=(40, 40))
        gs = gridspec.GridSpec(12, 4, width_ratios=[1,1,1,3], height_ratios=[1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        fig.suptitle(mouse_name + "  " + exp_name, fontsize=20, y=1.05)

        #EEG analysis
        file_path = glob.glob(os.path.join(exp, "*.ns3"))[0]
        dir = os.path.dirname(file_path)
        reader = BlackrockIO(filename=file_path)
        block = reader.read_block()

        # 生データの取得（RawSignalChannel）
        raw_signals = [seg.analogsignals[0] for seg in block.segments]
        sampling_rate = int(raw_signals[0][:, 0].sampling_rate.magnitude)
        arr = np.empty((8, len(raw_signals[0][:, 0].magnitude.flatten())))

        # label_list = ["M1", "V1", "M1-V1", "M1 (low pass 70Hz)", "M1 (ICA)", "EMG1-2", "EMG1", "EMG2", ]
        label_list = ["CH1", "CH2", "CH1-CH2", "CH1", "CH1", "EMG1-2", "EMG1", "EMG2"]
        arr[0] = raw_signals[0][:, 0].magnitude.flatten()#ch0
        arr[1] = raw_signals[0][:, 1].magnitude.flatten() #ch1
        arr[2] = arr[0]-arr[1] #ch0-ch1
        arr[3] = arr[0] #TODO low pass filter
        arr[4] = arr[0] #TODO ICA??
        arr[6] = raw_signals[0][:, 2].magnitude.flatten() #ch2
        arr[7] = raw_signals[0][:, 3].magnitude.flatten()  # ch3
        arr[5] = arr[6]-arr[7]
        print(arr[6])
        print(arr[5])

        for i in range(len(arr)):
            print(i)
            signal = arr[i]
            # signal = raw_signal.magnitude.flatten()
            exp_duration = len(signal) / sampling_rate  # sec

            spectrum_graph(signal, sampling_rate, freq_max, freq_min, gs, ax1=(0, 0), ax2=(0, 1), ax3=(0, 2), color=plt.get_cmap("tab10")(i), label = label_list[i])
            print("done")
            plot_raw(signal[sampling_rate*1:sampling_rate*9], sampling_rate, ylabel=label_list[i], ylim=3000, gs=gs, ax=(i+1, 3), color=plt.get_cmap("tab10")(i))
            heatmap(signal, sampling_rate, time_bin, freq_max, freq_min, dir, gs, ax=(i+1, 0), vrange=[0, 350000], #[:60*10*sampling_rate]
                    normalization=False, dB=False, label=label_list[i])
            heatmap(signal, sampling_rate, time_bin, freq_max, freq_min, dir, gs, ax=(i+1, 1), vrange=[0, 1.7], #[:60*10*sampling_rate]
                    normalization=True, dB=False, label=label_list[i])
            heatmap(signal, sampling_rate, time_bin, freq_max, freq_min, fig, gs, ax=(i+1, 2), vrange=[30, 60], #[:60*10*sampling_rate]
                    normalization=False, dB=True, label=label_list[i])


        plt.tight_layout()
        file_num = os.path.basename(file_path)[-7:-4]

        fig.savefig(os.path.join("X:\Behavior\EEG\_Check_EMG_leak", mouse_name+"_"+exp_name+".pdf"), dpi=150, transparent=True)