import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from neo.io import BlackrockIO
import pandas as pd
import os
from tkinter import filedialog
import glob




# 生データの周波数スペクトルを計算するための関数
def calculate_spectrum(signal, sampling_rate):
    n = len(signal)
    mean = np.mean(signal)
    signal -= mean #DC offset
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)
    fft_vals = np.fft.fft(signal)
    power_spectrum = np.abs(fft_vals)**2/ n

    # power_spectrum_db = 10 * np.log10(power_spectrum)

    return freqs[:n // 2], power_spectrum[:n // 2] #ナイキスト周波数まで

def spectrum_graph(signal, sampling_rate, freq_max, freq_min, gs, ax1, ax2, event_df):  # 全体の平均でnormalizeしている
    ax = fig.add_subplot(gs[ax1])
    ax2 = fig.add_subplot(gs[ax2])

    num_bins = int(np.ceil((freq_max - freq_min) / freq_bin))  # ビンの数を計算

    bins = np.linspace(freq_min, freq_max, num_bins + 1)  # ビンの境界を設定

    event_list = event_df["event_name"].unique().tolist()
    for e, event in enumerate(event_list):
        df = event_df[event_df['event_name'] == event]
        df = df.reset_index(drop=True)
        ave_power_spectrum = np.empty((0, num_bins))

        n_event = 0 #event全体のサンプル数
        for ep in range (len(df)):
            epoch_signal = signal[int(sampling_rate * df.loc[ep, "start_time"]): int(sampling_rate * df.loc[ep, "end_time"])]
            n_epoch = len(epoch_signal)
            freqs, power_spectrum = calculate_spectrum(epoch_signal, sampling_rate)

            # 各ビンごとのfft_valsの平均を計算
            bin_indices = np.digitize(freqs, bins)  # 各ビンに対応するインデックスを取得

            binned_power_spectrum = [power_spectrum[bin_indices == i].mean()*n_epoch for i in range(1, num_bins + 1)] #後でepochの長さで加重平均するためにn_epochをかけておく
            n_event += n_epoch
            ave_power_spectrum = np.vstack([ave_power_spectrum, binned_power_spectrum])

        ave_power_spectrum = np.sum(ave_power_spectrum, axis =0) / n_event #サンプルの長さで割って標準化
        ave_power_spectrum_norm = ave_power_spectrum / np.sum(ave_power_spectrum) *100

        #ビンの中心を計算
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.plot(bin_centers, ave_power_spectrum_norm, marker=None, lw=1.5, linestyle='-',color=plt.get_cmap("tab10")(e),label=event)  # label=f'Status {status+1}
        ax2.plot(bin_centers, ave_power_spectrum, marker=None, lw=1.5, linestyle='-', color=plt.get_cmap("tab10")(e),label=event)
    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(0, 12)
    ax.set_xlabel("Hz")
    ax.set_ylabel("Normalized power (%)")
    ax.legend()

    ax2.set_xlim(freq_min, freq_max)
    ax2.set_xlabel("Hz")
    ax2.set_ylabel("Power (μV²)")
    ax.set_ylim(0, 6000000)
    ax2.legend()

    # uV2 = 0 position add line
    horizontal_lines = [0]
    for ax in fig.axes:
        for line in horizontal_lines:
            ax.axhline(y=line, color='grey', linestyle='--')

    vertical_lines = [0.5, 4, 8, 12, 30] #to separate δ, θ, α, β, ɤ wave.
    for ax in fig.axes:
        for line in vertical_lines:
            ax.axvline(x=line, color='grey', linestyle='--')


def heatmap(signal, sampling_rate, time_bin, freq_max, freq_min, dir, gs, ax, vrange=[0,100], normalization=False, dB=False): #iv=[0.5, 4] #iv: どの範囲のpowerでvmaxを決めるか
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
    # vmax適当に決める
    # range_mean, range_std, range_median = df.loc[:, iv[0]:iv[1]].values.mean(), df.loc[:, iv[0]:iv[1]].values.std(), np.median(df.loc[:, iv[0]:iv[1]].values)
    # vmax = range_median + range_std * 1.5
    # vmax = range_median*2
    im = ax.imshow(df.T, aspect='auto', cmap='rainbow', origin='lower', vmin=vrange[0], vmax=vrange[1],
                   extent = [0, 0+df.shape[0], 0, 0+df.shape[1]]) #extentを指定しないと、binの半分だけずれて表示されてしまう
    # fig.colorbar(im_db, ax=ax_db)
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
    ax.set_ylabel("Hz")
    # fig.colorbar(im, ax=ax)

def calculate_band_power(freqs, power_spectrum, lower_bound, upper_bound, normalization = None, dB = None):
    # Filter frequencies and power within the specified band
    band_mask = (freqs >= lower_bound) & (freqs < upper_bound)
    if normalization == None:
        band_power = np.sum(power_spectrum[band_mask])
        if dB == True:
            band_power = 10 * np.log10(band_power) #TODO 確認 power spectrumの平均(or sum)は常に一定？(その値は何を意味？)それを基準にdB計算しているということでok?
    else: #normalized power (%)
        band_power = np.sum(power_spectrum[band_mask]) /  np.sum(power_spectrum) *100
        #この場合dB化はどういう順番ですべき？
    return band_power


def plot_timeseries_power (signal, sampling_rate, time_bin, dir, gs, ax1):
    columns = pd.MultiIndex.from_tuples([
        ('Power(dB)', 'delta'),('Power(dB)', 'theta'),('Power(dB)', 'alpha'),('Power(dB)', 'beta'),('Power(dB)', 'gamma'),
        ('Normalized Power (%)', 'delta'),('Normalized Power (%)', 'theta'),('Normalized Power (%)', 'alpha'),('Normalized Power (%)', 'beta'),('Normalized Power (%)', 'gamma'),
    ])

    df = pd.DataFrame(columns=columns)

    time_bin_num = int(len(signal) / sampling_rate / time_bin)
    for t in range(time_bin_num): #TODO binごとの計算をheatmapでもここでも二重に繰り返してるのは無駄。関数分けて共通にする
        bin_signal = signal[sampling_rate * time_bin * t:sampling_rate * time_bin * (t + 1)]
        freqs, power_spectrum = calculate_spectrum(bin_signal, sampling_rate)

        data = {
            ('Power(dB)', 'delta'):calculate_band_power(freqs, power_spectrum, 0.5, 4, normalization = None, dB = True),
            ('Power(dB)', 'theta'):calculate_band_power(freqs, power_spectrum, 4, 8, normalization=None, dB = True),
            ('Power(dB)', 'alpha'):calculate_band_power(freqs, power_spectrum, 8, 12, normalization=None,dB = True),
            ('Power(dB)', 'beta'):calculate_band_power(freqs, power_spectrum, 12, 30, normalization=None, dB = True),
            ('Power(dB)', 'gamma'):calculate_band_power(freqs, power_spectrum, 30, 100, normalization=None, dB = True),
            ('Normalized Power (%)', 'delta'):calculate_band_power(freqs, power_spectrum, 0.5, 4, normalization=True),
            ('Normalized Power (%)', 'theta'):calculate_band_power(freqs, power_spectrum, 4, 8, normalization=True),
            ('Normalized Power (%)', 'alpha'):calculate_band_power(freqs, power_spectrum, 8, 12, normalization=True),

            ('Normalized Power (%)', 'beta'):calculate_band_power(freqs, power_spectrum, 12, 30, normalization=True),
            ('Normalized Power (%)', 'gamma'):calculate_band_power(freqs, power_spectrum, 30, 100, normalization=True),
        }
        new_row = pd.DataFrame([data])
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(os.path.join(dir, "power_time-series.csv"))
    ax1, ax2= fig.add_subplot(gs[ax1]), fig.add_subplot(gs[ax2])
    df1 = df[[col for col in df.columns if col[0] == 'Power(dB)']]
    df1.columns = [col[1] for col in df1.columns]
    ax1.plot(df1.index + 0.5, df1[['delta', 'theta', 'alpha', 'beta', 'gamma']])
    ax1.legend(df1.columns, loc='best')

    record_time = len(signal) / sampling_rate  # sec
    xticks = np.arange(0, int(record_time / time_bin) + 1, int(5 * 60 / time_bin))
    xtick_labels = np.arange(0, int(record_time / 60) + 1, 5)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels)
    ax1.set_xlim(xticks[0], xticks[-1])
    ax1.set_ylim(55,85)
    ax1.set_xlabel("min")
    ax1.set_ylabel("Power (dB)")



def plot_raw(raw, sampling_rate, ylabel, ylim, gs, ax, event_df=None):
    ax = fig.add_subplot(gs[ax])
    ax.plot(raw, lw=0.1)
    n = len(raw)
    xticks = np.arange(0, n + 1, 5 * 60 * sampling_rate)
    xtick_labels = np.arange(0, int(n/sampling_rate/60)+1, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("min")
    ax.set_ylabel(ylabel + " (uV)")
    ax.set_ylim(-ylim, ylim)
    ax.margins(x=0)

    if event_df is not None:
        event_list = event_df["event_name"].unique().tolist()
        for e, event in enumerate(event_list):
            df = event_df[event_df['event_name'] == event]
            df = df.reset_index(drop=True)
            for ep in range(len(df)):
                ax.axvspan(int(sampling_rate * df.loc[ep, "start_time"]), int(sampling_rate * df.loc[ep, "end_time"]), color=plt.get_cmap("tab10")(e), alpha=0.3)

############################################################################
mouse_dir = filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG")
exp_list =glob.glob(os.path.join(mouse_dir, "[!_]*"))

freq_bin = 0.5  # Hz #Fourier plot時のbin
time_bin = 3  # sec #heatmap
freq_max, freq_min = 100, 0

for exp in exp_list:
    file_path = glob.glob(os.path.join(exp, "*.ns3"))[0]
    # DLC_dir = filedialog.askdirectory(initialdir=r"Z:\ProbeG")
    # event_df = pd.read_csv(os.path.join(DLC_dir, "event.csv"))
    event_df = pd.read_csv(r"Z:\ProbeG\cond_Pup-Ctrl\z155-2(10w)\day1_TEST2\event_test.csv")
    dir = os.path.dirname(file_path)

    fig = plt.figure(figsize=(40, 20))
    gs = gridspec.GridSpec(8, 3, height_ratios=[2, 0.4, 1, 1, 1, 1, 1.8,0.4])

    # Neoを使用してns2ファイルを読み込む
    reader = BlackrockIO(filename=file_path)
    block = reader.read_block()

    # 生データの取得（RawSignalChannel）
    raw_signals = [seg.analogsignals[0] for seg in block.segments]

    # EEG (Ch1)
    raw_signal = raw_signals[0][:, 0]
    sampling_rate = int(raw_signal.sampling_rate.magnitude)
    signal = raw_signal.magnitude.flatten()
    spectrum_graph(signal, sampling_rate, freq_max, freq_min, gs, ax1=(0, 0), ax2=(0, 1), event_df=event_df)
    plot_raw(signal, sampling_rate, ylabel="EEG", ylim=500, gs=gs, ax=(1, slice(0, 3)), event_df=event_df)
    heatmap(signal, sampling_rate, time_bin, freq_max, freq_min, dir, gs, ax=(2, slice(0, 3)),vrange = [0, 300000], normalization=False,dB=False)
    heatmap(signal, sampling_rate, time_bin, freq_max, freq_min, dir, gs, ax=(3, slice(0, 3)), vrange=[0, 1.8], normalization=True,dB=False)
    heatmap(signal, sampling_rate, time_bin, freq_max, freq_min, fig, gs, ax=(4, slice(0, 3)), vrange=[30, 56], normalization=False,dB=True)
    heatmap(signal, sampling_rate, time_bin, 20, freq_min, fig, gs, ax=(5, slice(0, 3)), vrange=[30, 70],normalization=False, dB=True)
    plot_timeseries_power (signal, sampling_rate, time_bin, dir=dir, gs=gs, ax1=(6, slice(0, 3))) #,ax2=(6, slice(0, 3))

    # EMG (Ch2, CH3)
    ch2_signal, ch3_signal = raw_signals[0][:, 1], raw_signals[0][:, 2]
    emg = ch2_signal - ch3_signal
    plot_raw(emg, sampling_rate, ylabel="EMG", ylim=1000, gs=gs, ax=(7, slice(0, 3)))

    plt.tight_layout()
    file_num = os.path.basename(file_path)[-7:-4]
    fig.savefig(os.path.join(dir, "EEG_graph_" + file_num + ".pdf"), dpi=300, transparent=True)