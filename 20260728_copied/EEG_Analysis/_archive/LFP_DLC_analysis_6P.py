import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from neo.io import BlackrockIO
import pandas as pd
import os
from tkinter import filedialog
import glob
from lib import DLCAnalysis
import json




# 生データの周波数スペクトルを計算するための関数
def calculate_spectrum(signal, sampling_rate):
    n = len(signal)
    mean = np.mean(signal)
    signal -= mean #DC offset #TODO　まず全体で引いてから
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)
    fft_vals = np.fft.fft(signal)
    power_spectrum = np.abs(fft_vals)**2/ n

    # power_spectrum_db = 10 * np.log10(power_spectrum)

    return freqs[:n // 2], power_spectrum[:n // 2] #ナイキスト周波数まで

def spectrum_graph(signal, sampling_rate, freq_max, freq_min, gs, ax1, ax2, ax3, event_df):  # 全体の平均でnormalizeしている
    ax = fig.add_subplot(gs[ax1])
    ax2 = fig.add_subplot(gs[ax2])
    ax3 = fig.add_subplot(gs[ax3])

    num_bins = int(np.ceil((freq_max - freq_min) / freq_bin))  # ビンの数を計算

    bins = np.linspace(freq_min, freq_max, num_bins + 1)  # ビンの境界を設定
    bin_centers = (bins[:-1] + bins[1:]) / 2

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

        ax.plot(bin_centers, ave_power_spectrum_norm, marker=None, lw=1, linestyle='-',color=plt.get_cmap("tab10")(e),label=event)  # label=f'Status {status+1}
        ax2.plot(bin_centers, ave_power_spectrum, marker=None, lw=1, linestyle='-', color=plt.get_cmap("tab10")(e),label=event)
        ax3.plot(bin_centers, ave_power_spectrum, marker=None, lw=1, linestyle='-', color=plt.get_cmap("tab10")(e),label=event)

    #全体平均
    freqs, power_spectrum = calculate_spectrum(signal, sampling_rate)
    bin_indices = np.digitize(freqs, bins)  # 各ビンに対応するインデックスを取得
    binned_power_spectrum = [power_spectrum[bin_indices == i].mean() for i in range(1, num_bins + 1)]
    binned_power_spectrum_norm = binned_power_spectrum / np.sum(binned_power_spectrum) * 100
    ax.plot(bin_centers, binned_power_spectrum_norm, marker=None, lw=1, linestyle='-', color="k",label="All-time Average")
    ax2.plot(bin_centers, binned_power_spectrum, marker=None, lw=1, linestyle='-', color="k",label="All-time Average")
    ax3.plot(bin_centers, binned_power_spectrum, marker=None, lw=1, linestyle='-', color="k",label="All-time Average")

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
    ax1= fig.add_subplot(gs[ax1]) #, ax2#, fig.add_subplot(gs[ax2])
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

################################################################################
# ns2ファイルのパス
freq_bin = 0.5  # Hz #Fourier plot時のbin
time_bin = 3  # sec #heatmap
freq_max, freq_min = 100, 0

mouse_dir = filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG")
exp_list =glob.glob(os.path.join(mouse_dir, "[!_]*"))
dlc_json = os.path.join(mouse_dir, "_DLC_dir.json")
with open(dlc_json, 'r', encoding='utf-8') as file:
    data = json.load(file)
    dlc_dir = data["DLC"]["DLC_dir"]
    print(dlc_dir)

for e, exp in enumerate(exp_list):
    fig = plt.figure(figsize=(40, 40))
    gs = gridspec.GridSpec(12, 3, height_ratios=[2, 0.4, 1, 1, 1, 1, 1.8, 0.4, 1, 1, 1, 1])


    #EEG analysis
    file_path = glob.glob(os.path.join(exp, "*.ns3"))[0]
    dir = os.path.dirname(file_path)


    # Neoを使用してns2ファイルを読み込む
    reader = BlackrockIO(filename=file_path)
    block = reader.read_block()

    # 生データの取得（RawSignalChannel）
    raw_signals = [seg.analogsignals[0] for seg in block.segments]

    # LFP (Ch1)
    raw_signal = raw_signals[0][:, 0]
    sampling_rate = int(raw_signal.sampling_rate.magnitude)
    signal = raw_signal.magnitude.flatten()
    exp_duration = len(signal)/sampling_rate #sec


    # DLC_analysis
    arena_mm_per_pix = 0.6

    dlc_exp_dir = glob.glob(os.path.join(dlc_dir, "day*"))[e]
    print("#################")
    print(exp)
    print(dlc_exp_dir)
    dlc_h5_path = os.path.join(dlc_exp_dir, "dlc_raw.h5")
    param_ind = os.path.join(dlc_exp_dir, "param_individual.json")

    df = pd.read_hdf(dlc_h5_path, key='dlc_data')
    dlc_output_dir = os.path.join(exp, "_DLC_analysis")
    if not os.path.exists(dlc_output_dir):
        os.makedirs(dlc_output_dir)
    df.to_csv(os.path.join(dlc_output_dir, "dlc_data.csv"))
    real_frame_time = (df["time"].iloc[-1] - df["time"].iloc[0]) / (len(df) - 1)  # real frame time
    real_frame_time = round(pd.to_timedelta(real_frame_time).total_seconds(), 4)
    event_df = pd.DataFrame(columns=["start_time", "end_time", "event_name"])

    velocity_boundary = [5,25,200] #mm/s
    frames_extracted_by_velocity = DLCAnal.time_series_velocity(df, real_frame_time=real_frame_time, arena_mm_per_pix=arena_mm_per_pix, exp_duration=exp_duration, fig=fig, ax=(8, slice(0, 3)), gs=gs, body_part="body_center", velocity_boundary=velocity_boundary)
    for v in range(len(frames_extracted_by_velocity)):
        event_name = "~"+str(velocity_boundary[v]) + " mm/s"
        event_df = DLCAnal.frame_to_sec(frames_extracted_by_velocity[v], real_frame_time, event_df, event_name, tolerable_frame_drop = 0, min_duration=3, exp_duration=exp_duration)


    coordinate = DLCAnal.get_roi_coordinate("Object", param_ind=param_ind)
    # frame_around_roi = DLCAnal.extract_frame_around_roi(df, coordinate, "head_center", 50)
    # event_df = DLCAnal.frame_to_sec(frame_around_roi, real_frame_time, event_df, "around_object")
    frame_approaching, frame_leaving = DLCAnal.time_series_distance_to_object(df, object_coordinate=coordinate, real_frame_time=real_frame_time, arena_mm_per_pix=arena_mm_per_pix, exp_duration=exp_duration, fig=fig, ax=(9, slice(0, 3)), gs=gs, body_part="snout", distance_to_boundary_mm = 100, plot=True)
    event_df = DLCAnal.frame_to_sec_v2(frame_approaching, real_frame_time, event_df, event_name="Approaching (-5~0 sec)", before_sec= -5, after_sec=0, exp_duration=exp_duration)
    event_df = DLCAnal.frame_to_sec_v2(frame_approaching, real_frame_time, event_df, event_name="Before approaching(-10 ~ -5 sec)", before_sec=-10, after_sec=-5, exp_duration=exp_duration)
    event_df = DLCAnal.frame_to_sec_v2(frame_approaching, real_frame_time, event_df, event_name="0-5 sec after approach", before_sec=0, after_sec=5, exp_duration=exp_duration)
    event_df = DLCAnal.frame_to_sec_v2(frame_approaching, real_frame_time, event_df, event_name="-5 ~ 15 sec after approach)", before_sec=-5, after_sec=15, exp_duration=exp_duration)
    #TODO　interaction frameは、approach, leaveでなく、距離から単純に定義するべき "frame_around_roi"
    event_df = DLCAnal.frame_to_sec_v2(frame_leaving, real_frame_time, event_df, event_name="Leaving (0-5 sec)", before_sec=0, after_sec=5, exp_duration=exp_duration)
    event_df = DLCAnal.frame_to_sec_v2(frame_leaving, real_frame_time, event_df, event_name="Left (5-10 sec)", before_sec=5, after_sec=10, exp_duration=exp_duration)



    DLCAnal.time_series_angle_to_object_direction(df, object_coordinate=coordinate, real_frame_time=real_frame_time, exp_duration=exp_duration, fig=fig, ax=(10, slice(0, 3)), gs=gs)

    coordinate_ctrl = DLCAnal.get_roi_coordinate("Diagonal_position", param_ind=param_ind)
    frame_approaching_c, frame_leaving_c = DLCAnal.time_series_distance_to_object(df, object_coordinate=coordinate_ctrl, real_frame_time=real_frame_time, arena_mm_per_pix=arena_mm_per_pix, exp_duration=exp_duration, fig=fig, ax=(9, slice(0, 3)), gs=gs, body_part="snout", distance_to_boundary_mm = 100, plot=False)
    # event_df = DLCAnal.frame_to_sec_v2(frame_approaching_c, real_frame_time, event_df, event_name="[Diagonal] Approaching (0-5 sec)", before_sec= -5, after_sec=0, exp_duration=exp_duration)
    # event_df = DLCAnal.frame_to_sec_v2(frame_approaching_c, real_frame_time, event_df, event_name="[Diagonal] Approaching(5-10 sec)",before_sec=-10, after_sec=-5, exp_duration=exp_duration)
    # event_df = DLCAnal.frame_to_sec_v2(frame_approaching_c, real_frame_time, event_df, event_name="[Diagonal] Object interaction (5 sec after approach)",before_sec=0, after_sec=5, exp_duration=exp_duration)
    # event_df = DLCAnal.frame_to_sec_v2(frame_leaving_c, real_frame_time, event_df, event_name="[Diagonal] Leaving (0-5 sec)",before_sec=0, after_sec=5, exp_duration=exp_duration)
    # event_df = DLCAnal.frame_to_sec_v2(frame_leaving_c, real_frame_time, event_df, event_name="[Diagonal] Leaving (5-10 sec)",before_sec=5, after_sec=10, exp_duration=exp_duration)


    # frame_around_roi = DLCAnal.extract_frame_around_roi(df, coordinate, "head_center", 50)
    # event_df = DLCAnal.frame_to_sec(frame_around_roi, real_frame_time, event_df, "diagonal_control")

    event_df.to_csv(os.path.join(dlc_output_dir, "event.csv"))

    # EEG (Ch1)
    spectrum_graph(signal, sampling_rate, freq_max, freq_min, gs, ax1=(0, 0), ax2=(0, 1), ax3=(0, 2), event_df=event_df)
    plot_raw(signal, sampling_rate, ylabel="EEG", ylim=500, gs=gs, ax=(1, slice(0, 3)), event_df=event_df)
    heatmap(signal, sampling_rate, time_bin, freq_max, freq_min, dir, gs, ax=(2, slice(0, 3)), vrange=[0, 350000],
            normalization=False, dB=False)
    heatmap(signal, sampling_rate, time_bin, freq_max, freq_min, dir, gs, ax=(3, slice(0, 3)), vrange=[0, 1.7],
            normalization=True, dB=False)
    heatmap(signal, sampling_rate, time_bin, freq_max, freq_min, fig, gs, ax=(4, slice(0, 3)), vrange=[30, 60],
            normalization=False, dB=True)
    heatmap(signal, sampling_rate, time_bin, 20, freq_min, fig, gs, ax=(5, slice(0, 3)), vrange=[30, 70],
            normalization=False, dB=True)
    plot_timeseries_power(signal, sampling_rate, time_bin, dir=dir, gs=gs,
                          ax1=(6, slice(0, 3)))  # ,ax2=(6, slice(0, 3))

    # EMG (Ch2, CH3)
    ch2_signal, ch3_signal = raw_signals[0][:, 2], raw_signals[0][:, 3]
    emg = ch2_signal - ch3_signal
    plot_raw(emg, sampling_rate, ylabel="EMG", ylim=1000, gs=gs, ax=(7, slice(0, 3)))

    plt.tight_layout()
    file_num = os.path.basename(file_path)[-7:-4]
    mouse_name, exp_name= os.path.basename(mouse_dir), os.path.basename(exp)

    fig.savefig(os.path.join(dir, mouse_name+"_"+exp_name+"_LFP_timeseries.pdf"), dpi=150, transparent=True)