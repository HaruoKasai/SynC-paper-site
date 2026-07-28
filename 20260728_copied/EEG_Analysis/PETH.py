import h5py
import numpy as np
import pandas as pd
from EEG_Analysis import select_folder, plot_timeseries_power, plot_heatmap, plot_timeseries, extract_params
import os
import matplotlib.pyplot as plt
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12   })
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import binary_dilation





def load_dataset(name, file):
    if name not in file:
        return None  # データセットが存在しない場合
    data = file[name][:]  # データ取得
    if np.issubdtype(np.array(data).dtype, np.number):
        return data if not np.isnan(data).all() else None
    else:
        # 数値型でない場合（例: 文字列やobject配列）はそのまま返す
        return data

def open_h5 (h5_path):
    with h5py.File(h5_path, "r") as f:
        analog_tp = f["all_analog_tp"][:]
        eeg = f["all_eeg"][:]
        emg = f["all_emg"][:]
        sampling_rate = f["sampling_rate"][()]

        breathe = load_dataset("all_breathe", f)
        tem = load_dataset("all_tem", f)
        breathing_rate = load_dataset("all_b_rate", f)
        pupil_size = load_dataset("all_pupil", f)
        pupil_tp = load_dataset("all_pupil_tp", f)
        table_v = load_dataset("all_table_v", f)
        table_tp = load_dataset("all_table_tp", f)
        OF_tp = load_dataset("all_OF_tp", f)
        velocity = load_dataset("all_v", f)
        cum_d = load_dataset("all_cum_d", f)
        distance = load_dataset("all_distance", f)
        t_stft = load_dataset("all_t_stft", f)
        f_stft = f["f_stft"][:]
        linear_power = f["all_linear_power"][:]
        dB_power = f["all_dB_power"][:]
        hr_tp = load_dataset("all_hr_tp", f)
        heartrate = load_dataset("all_hr", f)

    try:
        event_df = pd.read_hdf(h5_path, key="event_df")
    except (KeyError, FileNotFoundError):
        event_df = None

    return (h5_path, analog_tp, eeg, t_stft, f_stft, linear_power,dB_power,emg, sampling_rate, breathe, tem, breathing_rate, pupil_size, pupil_tp,
            table_v, table_tp, OF_tp, velocity,cum_d,distance,event_df, hr_tp, heartrate)

def emg_rms(emg: np.ndarray, sr: int, window_sec: float) -> np.ndarray:
    N = int(sr * window_sec)
    num_windows = len(emg) // N
    emg_trimmed = emg[:num_windows * N].reshape(num_windows, N)
    rms_values = np.sqrt(np.mean(emg_trimmed ** 2, axis=1))
    return rms_values


def plot_PETH(valid_t,time_array,time_array2, t_pre,t_post,OF_tp,f_stft, t_stft, time_bin_emg, time_bin_power,sampling_rate,eeg, emg, velocity, linear_power, analog_tp,
              table_v, table_tp,
              pupil_tp, pupil_size, breathing_rate, tem,
              hr_tp, heartrate,
              fig, gs, column, output_dir, ch_name, epoch_name, epoch_type,time_type, contime, dlc_type):

    print("###############")
    print(epoch_name)
    print(epoch_type)



    #pupillometryの場合は、OF_tp, velocityにpupil_tp, pupil_sizeを代入して計算
    if dlc_type=="Pupillometry": #if OF_tp is None and pupil_tp is not None:
        OF_tp = pupil_tp
        velocity = pupil_size
        ax0_title = "Pupil_size"
        ax0_ylable = "Δ(%)"
        ax0_ylim = (-50,150)
    else:
        ax0_title = "Velocity"
        ax0_ylable = "mm/s"
        ax0_ylim = (0, 50)




    print(epoch_name)
    print(epoch_type)
    axes = [fig.add_subplot(gs[i, column]) for i in range(11)]
    ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10= axes

    n_event = len(valid_t)

    linear_power_array = np.empty([n_event, len(f_stft), int((t_post - t_pre) / (t_stft[1] - t_stft[0]))])
    emg_rms_array = np.empty([n_event, int((t_post - t_pre) / time_bin_emg)])
    delta_array = np.empty([n_event, int((t_post - t_pre) / time_bin_power)])
    theta_array = np.empty([n_event, int((t_post - t_pre) / time_bin_power)])
    alpha_array = np.empty([n_event, int((t_post - t_pre) / time_bin_power)])
    beta_array = np.empty([n_event, int((t_post - t_pre) / time_bin_power)])
    gamma_array = np.empty([n_event, int((t_post - t_pre) / time_bin_power)])
    highgamma_array = np.empty([n_event, int((t_post - t_pre) / time_bin_power)])
    lowgamma_array = np.empty([n_event, int((t_post - t_pre) / time_bin_power)])
    t_emg_tp = np.arange(t_pre, t_post, time_bin_emg)
    t_t_stft = np.arange(t_pre, t_post, t_stft[1] - t_stft[0])
    t_OF_tp = np.arange(t_pre, t_post, OF_tp[1] - OF_tp[0])
    t_table_tp = np.arange(t_pre, t_post, table_tp[1] - table_tp[0])
    breath_window = 2 #sec 本当はh5に入れてから取得してくるのがいんだろうけど、
    hr_window = 0.25
    t_breath_tp = np.arange(t_pre+breath_window/2, t_post, breath_window)
    t_hr_tp = np.arange(t_pre + hr_window / 2, t_post, hr_window)
    breath_tp = np.arange(contime[0][0]*60 + breath_window/2 ,contime[-1][1]*60, breath_window)
    hr_tp = np.arange(contime[0][0] * 60 + hr_window / 2, contime[-1][1] * 60, hr_window) #元のhr_tpは使わず再定義(breath_tpと同型になって楽なので)
    if breathing_rate is None: #Openfield etc
        breathing_rate = np.zeros_like(breath_tp)
    if heartrate is None: #Openfield etc
        heartrate = np.zeros_like(hr_tp)

    print("heartrate")
    print(heartrate)
    # velocity_array = np.zeros([n_event, int((t_post - t_pre) / (OF_tp[1] - OF_tp[0]))], dtype=np.float64)
    velocity_array = np.zeros([n_event, len(t_OF_tp)], dtype=np.float64)
    table_v_array = np.zeros([n_event, len(t_table_tp)], dtype=np.float64)
    breath_array = np.zeros([n_event, len(t_breath_tp)], dtype=np.float64)
    hr_array = np.zeros([n_event, len(t_hr_tp)], dtype=np.float64)
    i = 0
    print(t_pre)
    for t in valid_t:
        tmin, tmax = time_array[t] + t_pre, time_array[t] + t_post
        analog_tp_mask = (analog_tp >= tmin) & (analog_tp <= tmax)
        t_stft_mask = (t_stft >= tmin) & (t_stft < tmax)
        table_tp_mask = (table_tp >= tmin) & (table_tp < tmax)
        OF_tp_mask = (OF_tp >= tmin) & (OF_tp < tmax)
        breath_tp_mask = (breath_tp >= tmin) & (breath_tp < tmax)
        hr_tp_mask = (hr_tp >= tmin) & (hr_tp < tmax)
        t_linear_power = linear_power[:, t_stft_mask]
        t_analog_tp = analog_tp[analog_tp_mask] - time_array[t]

        # print("Event:", t, " Time window:", tmin, "to", tmax)
        # print("OF_tp_mask sum:", OF_tp_mask.sum())
        # print("analog_tp_mask sum:", analog_tp_mask.sum())
        # print("t_stft_mask sum:", t_stft_mask.sum())

        t_velocity = velocity[OF_tp_mask]
        t_emg = emg_rms(emg[analog_tp_mask], sampling_rate, time_bin_emg)
        t_eeg = eeg[analog_tp_mask]
        t_table_v = table_v[table_tp_mask]
        t_breath = breathing_rate[breath_tp_mask]
        t_hr = heartrate[hr_tp_mask]

        if time_type=="all":
            Otp = OF_tp[OF_tp_mask]
            ttp = table_tp[table_tp_mask]
            atp = analog_tp[analog_tp_mask]
            stfttp = t_stft[t_stft_mask]
            emtp = np.arange(atp[0], atp[-1], time_bin_emg)
            btp = breath_tp[breath_tp_mask]
            htp = hr_tp[hr_tp_mask]
            if len(emtp) > len(t_emg):
                emtp = emtp[:len(t_emg)]  # 余分な要素を削除
            elif len(emtp) < len(t_emg):
                emtp = np.pad(emtp, (0, len(t_emg) - len(emtp)), mode='edge')

            OF_tp_mask2 = np.zeros_like(Otp, dtype=bool)
            analog_tp_mask2 = np.zeros_like(atp, dtype=bool)
            t_stft_mask2 = np.zeros_like(stfttp, dtype=bool)
            emg_tp_mask2 = np.zeros_like(emtp, dtype=bool)
            table_tp_mask2 = np.zeros_like(ttp, dtype=bool)
            breath_tp_mask2 =np.zeros_like(btp, dtype=bool)
            hr_tp_mask2 = np.zeros_like(htp, dtype=bool)


            for (start, end) in contime:
                condition1 = (Otp >= start*60) & (Otp <= end*60)  # Otpがcontime[i]の範囲にある
                a_condition1 = (atp >= start*60) & (atp <= end*60)
                s_condition1 = (stfttp >= start*60) & (stfttp <= end*60)
                em_condition1 = (emtp >= start * 60) & (emtp <= end * 60)
                t_condition1 = (ttp >= start * 60) & (ttp <= end * 60)
                b_condition1 = (btp >= start * 60) & (btp <= end * 60)
                h_condition1 = (htp >= start * 60) & (htp <= end * 60)
                if epoch_type=="start":
                    condition2 = t == 0 or Otp > time_array2[t - 1]
                    condition3 = Otp < time_array2[t]  # Otpがtime_array2[t]より小さい

                    a_condition2 = t == 0 or atp > time_array2[t - 1]
                    a_condition3 = atp < time_array2[t]

                    s_condition2 = t == 0 or stfttp > time_array2[t - 1]
                    s_condition3 = stfttp < time_array2[t]

                    em_condition2 = t == 0 or emtp > time_array2[t - 1]
                    em_condition3 = emtp < time_array2[t]

                    t_condition2 = t == 0 or ttp > time_array2[t - 1]
                    t_condition3 = ttp < time_array2[t]

                    b_condition2 = t == 0 or btp > time_array2[t - 1]
                    b_condition3 = btp < time_array2[t]

                    h_condition2 = t == 0 or htp > time_array2[t - 1]
                    h_condition3 = htp < time_array2[t]
                else: #end
                    condition2 = Otp>time_array2[t]
                    condition3 = t == len(time_array)-1 or Otp < time_array2[t+1]  # Otpがtime_array2[t]より小さい

                    a_condition2 = atp > time_array2[t]
                    a_condition3 = t == len(time_array) - 1 or atp < time_array2[t + 1]

                    s_condition2 = stfttp > time_array2[t]
                    s_condition3 = t == len(time_array) - 1 or stfttp < time_array2[t + 1]

                    em_condition2 = emtp > time_array2[t]
                    em_condition3 = t == len(time_array) - 1 or emtp < time_array2[t + 1]

                    t_condition2 = ttp > time_array2[t]
                    t_condition3 = t == len(time_array) - 1 or ttp < time_array2[t + 1]

                    b_condition2 = btp > time_array2[t]
                    b_condition3 = t == len(time_array) - 1 or btp < time_array2[t + 1]

                    h_condition2 = htp > time_array2[t]
                    h_condition3 = t == len(time_array) - 1 or htp < time_array2[t + 1]

                OF_tp_mask2 |= condition1 & condition2 & condition3  # 3つの条件を満たすものをマスク
                analog_tp_mask2 |= a_condition1 & a_condition2 & a_condition3
                t_stft_mask2 |= s_condition1 & s_condition2 & s_condition3
                emg_tp_mask2 |= em_condition1 & em_condition2 & em_condition3
                table_tp_mask2 |= t_condition1 & t_condition2 & t_condition3
                breath_tp_mask2 |= b_condition1 & b_condition2 & b_condition3
                hr_tp_mask2 |= h_condition1 & h_condition2 & h_condition3

            print("OF_tp_mask2 sum:", OF_tp_mask2.sum())
            print("analog_tp_mask2 sum:", analog_tp_mask2.sum())
            print("t_stft_mask2 sum:", t_stft_mask2.sum())
            print("emg_tp_mask2 sum:", emg_tp_mask2.sum())
            print("hr_tp_mask2 sum:", hr_tp_mask2.sum())

            t_velocity =  np.where(OF_tp_mask2, t_velocity, np.nan)
            t_linear_power = np.where(t_stft_mask2, t_linear_power, np.nan)
            t_emg = np.where(emg_tp_mask2, t_emg, np.nan)
            t_eeg = np.where(analog_tp_mask2, t_eeg, np.nan)
            t_table_v = np.where(table_tp_mask2, t_table_v, np.nan)
            t_breath = np.where(breath_tp_mask2, t_breath, np.nan)
            t_hr = np.where(hr_tp_mask2, t_hr, np.nan)
            print("t_velocity:", t_velocity)
            print("t_hr:", t_hr)

        if len(t_velocity) > velocity_array.shape[1]:
            t_velocity = t_velocity[:velocity_array.shape[1]]  # 余分な要素を削除
        elif len(t_velocity) < velocity_array.shape[1]:
             t_velocity = np.pad(t_velocity, (0, velocity_array.shape[1] - len(t_velocity)), mode='edge')

        if dlc_type=="Pupillometry": #delta化
            basemask = t_OF_tp<0
            base = np.nanmean(t_velocity[basemask])
            if np.isnan(base):  # 有効なデータが1つもなかった場合
                t_velocity[:] = np.nan
            else:
                t_velocity = t_velocity / base * 100 - 100


        # plot
        print(t_OF_tp)
        print(t_velocity)
        plot_timeseries(t_OF_tp, t_velocity, 1, ax0, color="gray", lw=0.2, title="", ylabel="", ylim=ax0_ylim,label=None)
        plot_timeseries(t_emg_tp, t_emg, 1, ax1, color="gray", lw=0.2, title="", ylabel="", ylim=(0, 200), label=None)
        powers = plot_timeseries_power(t_eeg, t_analog_tp, sampling_rate, time_bin_power,
                                       [ax3, ax4, ax5, ax6, ax7], 0.2, legend=False)
        plot_timeseries(t_table_tp, t_table_v, 1, ax8, color="gray", lw=0.2, title="", ylabel="", ylim=(-50, 250), label=None)
        plot_timeseries(t_breath_tp, t_breath, 4, ax9, color="gray", lw=0.2, title="", ylabel="", ylim=(50, 450), label=None)
        # print(t_hr_tp)
        # print(t_hr)
        plot_timeseries(t_hr_tp, t_hr, 4, ax10, color="gray", lw=0.2, title="", ylabel="", ylim=(200, 1000),label=None)

        # concatenate for merge
        linear_power_array[i] = t_linear_power
        velocity_array[i] = t_velocity #.copy()
        table_v_array[i] = t_table_v
        breath_array[i] = t_breath
        hr_array[i] = t_hr

        emg_rms_array[i] = t_emg
        delta_array[i] = np.array(powers.get('Delta (0.5-4 Hz)', []))
        theta_array[i] = np.array(powers.get('Theta (4-8 Hz)', []))
        alpha_array[i] = np.array(powers.get('Alpha (8-12 Hz)', []))
        beta_array[i] = np.array(powers.get('Beta (12-30 Hz)', []))
        gamma_array[i] = np.array(powers.get('Gamma (30-80 Hz)', []))
        highgamma_array[i] = np.array(powers.get('High gamma (60-100Hz)', []))
        lowgamma_array[i] = np.array(powers.get('Low gamma (30-60Hz)', []))
        i += 1

    # plot average
    average_power =np.nanmean(linear_power_array, axis=0)
    average_then_db = 10 * np.log10(average_power + 1e-10)
    plot_heatmap(ax2, t_t_stft, f_stft, average_then_db, "STFT dB Power", "Frequency (Hz)", 100, "rainbow", [-10, 33])
    average_v = np.nanmean(velocity_array, axis=0)
    average_table_v = np.nanmean(table_v_array, axis=0)
    average_breath = np.nanmean(breath_array, axis=0)
    average_hr = np.nanmean(hr_array, axis=0)
    plot_timeseries(t_OF_tp, average_v, 1, ax0, color='#1f77b4', lw=3, title=ax0_title,ylabel=ax0_ylable, ylim=ax0_ylim, label=None, alpha=0.3)
    plot_timeseries(t_table_tp, average_table_v, 1, ax8, color='#1f77b4', lw=3, title="Velocity on table",ylabel="mm/s", ylim=(-50, 250), label=None)
    plot_timeseries(t_breath_tp, average_breath, 4, ax9, color='#1f77b4', lw=3, title="Breathing rate",ylabel="BPM", ylim=(50, 450), label=None)
    plot_timeseries(t_hr_tp, average_hr, 4, ax10, color='#1f77b4', lw=3, title="Heart rate", ylabel="BPM",
                    ylim=(200, 1000), label=None)
    average_emg_rms = np.nanmean(emg_rms_array, axis=0)
    plot_timeseries(t_emg_tp, average_emg_rms, 1, ax1, color='#1f77b4', lw=3, title="EMG-RMS",ylabel=None, ylim=(0, 200), label=None)


    delta_linear=np.nanmean(delta_array, axis=0)
    theta_linear=np.nanmean(theta_array, axis=0)
    alpha_linear=np.nanmean(alpha_array, axis=0)
    beta_linear=np.nanmean(beta_array, axis=0)
    gamma_linear=np.nanmean(gamma_array, axis=0)
    high_gamma_linear = np.nanmean(highgamma_array, axis=0)
    low_gamma_linear = np.nanmean(lowgamma_array, axis=0)
    delta = 10 * np.log10(delta_linear + 1e-10)
    theta = 10 * np.log10(theta_linear + 1e-10)
    alpha = 10 * np.log10(alpha_linear + 1e-10)
    beta = 10 * np.log10(beta_linear + 1e-10)
    gamma = 10 * np.log10(gamma_linear + 1e-10)
    power_time_array = np.arange(t_pre + time_bin_power / 2, t_post + time_bin_power / 2, time_bin_power)
    plot_timeseries(power_time_array, delta, 1, ax3, color='#1f77b4', lw=4, title="delta power", ylabel="(dB)",ylim=(55, 85), label=None, alpha=0.3)
    plot_timeseries(power_time_array, theta, 1, ax4, color='#1f77b4', lw=4, title="theta power", ylabel="(dB)",ylim=(55, 85), label=None, alpha=0.3)
    plot_timeseries(power_time_array, alpha, 1, ax5, color='#1f77b4', lw=4, title="alpha power", ylabel="(dB)",ylim=(55, 85), label=None, alpha=0.3)
    plot_timeseries(power_time_array, beta, 1, ax6, color='#1f77b4', lw=4, title="beta power", ylabel="(dB)",ylim=(55, 85), label=None, alpha=0.3)
    plot_timeseries(power_time_array, gamma, 1, ax7, color='#1f77b4', lw=4, title="gamma power", ylabel="(dB)",ylim=(55, 85), label=None, alpha=0.3)

    #store data in h5
    """
    #TODO
    timeblockごとに保存できる形には直す。
    """

    h5_name = os.path.join(output_dir, epoch_name+"_"+epoch_type+"_"+str(t_pre)+"s_"+str(t_post)+"s_PETH_average_"+time_type+"_"+ch_name+".h5")
    with h5py.File(h5_name, "w") as f:
        f.create_dataset("OF_tp", data=t_OF_tp) #or pupil_tp
        f.create_dataset("velocity", data=average_v) #or pupil_size

        f.create_dataset("emg_tp", data=t_emg_tp)
        f.create_dataset("emg_rms", data=average_emg_rms)

        f.create_dataset("t_stft", data=t_t_stft)
        f.create_dataset("f_stft", data=f_stft)
        f.create_dataset("power_spectrum", data=average_power) #not decibel

        f.create_dataset("power_time_array", data=power_time_array)
        f.create_dataset("delta", data=delta_linear) #not decibel
        f.create_dataset("theta", data=theta_linear) #not decibel
        f.create_dataset("alpha", data=alpha_linear) #not decibel
        f.create_dataset("beta", data=beta_linear) #not decibel
        f.create_dataset("gamma", data=gamma_linear) #not decibel
        f.create_dataset("high_gamma", data=high_gamma_linear)  # not decibel
        f.create_dataset("low_gamma", data=low_gamma_linear)  # not decibel

        f.create_dataset("breath_tp", data=t_breath_tp)
        f.create_dataset("breathing_rate", data=average_breath)
        f.create_dataset("hr_tp", data=t_hr_tp)
        f.create_dataset("heartrate", data=average_hr)

        f.create_dataset("table_tp", data=t_table_tp)
        f.create_dataset("table_velocity", data=average_table_v)


def process_events( h5_path, analog_tp, eegs, t_stft, f_stft, linear_powers,dB_power,emg, sampling_rate, breathe, tem, breathing_rate, pupil_size, pupil_tp,
            table_v, table_tp, OF_tp, velocity,cum_d,distance,event_df,hr_tp, heartrate, EEG_ch_dict, manual_event, contime, dlc_type):

    # PETH_time_list = [[-20,600],[-20,300],[-600,20],[-180,20],[-60,60],[-30,30],[-12,12]]
    # time_type_list = ["rough", "rough", "rough", "rough", "rough", "strict", "strict"]
    """
    "all":　各epoch前後をすべて抽出。他のepochと重なる部分だけNanとして無視して平均する
    "strict": 
    "rough": 
    """
    PETH_time_list = [[-180, 180]] #,[-120, 120],[-300, 300],[-60,60]
    time_type_list = ["all","all", "all", "all"]
    time_bin_power = 2  # for plot_timeseries_power
    time_bin_emg = 0.5
    time_blocks = [[0, 300]] #min
    output_dir = os.path.dirname(h5_path)
    emg =emg[0]

    for ch in range (len(eegs)):
        eeg=eegs[ch]
        linear_power=linear_powers[ch]
        keys=list(EEG_ch_dict.keys())
        ch_name = keys[ch]

        # 衝突アーチファクトのぞく awakeタイミングだけでのぞく
        low_freq_mask = (f_stft >= 0) & (f_stft < 2)
        lf_power = np.sum(linear_power[low_freq_mask], axis=0)

        # threshold = np.mean(lf_power) + 3 * np.std(lf_power)
        # above_threshold = lf_power > threshold
        lf_power_db = 10 * np.log10(lf_power)
        threshold_db = np.mean(lf_power_db) + 3 * np.std(lf_power_db)

        # 閾値超えの箇所（dBベース）
        above_threshold = lf_power_db > threshold_db

        x = 5  # 3sdをこえたところの前後x秒もnanにする
        artifact_mask = binary_dilation(above_threshold, structure=np.ones(2 * x + 1))

        n_seconds = linear_power.shape[1]
        start_time_sec = contime[0][0] * 60 + 0.5  # 例: -1000 * 60 = -60000
        times_sec = np.arange(n_seconds) + start_time_sec  # 1Hzごとの時間（秒）
        immotive_mask = np.zeros(n_seconds, dtype=bool)

        for _, row in event_df.iterrows():
            start, end = row['start_time'], row['end_time']
            immotive_mask |= (times_sec >= start) & (times_sec <= end)


        final_mask = artifact_mask & (~immotive_mask)
        artifact_indices = np.where(final_mask)[0]
        print(contime[0][0] * 60 + artifact_indices)
        linear_power[:, final_mask] = np.nan
        eeg_artifact_mask = np.repeat(final_mask, 2000)
        eeg_artifact_mask = eeg_artifact_mask[:len(eeg)]
        eeg[eeg_artifact_mask] = np.nan




        for p, PETH_time in enumerate(PETH_time_list):
            t_pre, t_post = PETH_time[0], PETH_time[1]# sec plot
            time_type = time_type_list[p]
            for event_df in [manual_event]: #[event_df, manual_event]:
                start_array = event_df["start_time"].to_numpy()
                end_array = event_df["end_time"].to_numpy()

                for event_name in event_df["event_name"].unique().tolist():
                    fig = plt.figure(figsize=(len(time_blocks)*8, 24))
                    gs = gridspec.GridSpec(11, len(time_blocks)*2, height_ratios=[1,1,1,1,0.1,0.1,0.1,1,1,1,1])
                    plt.subplots_adjust(hspace=0.5)

                    event_name_mask = event_df["event_name"] == event_name
                    event_index_list = event_df.index[event_name_mask].tolist()

                    for b in range (len(time_blocks)):
                        if time_type=="rough":
                            start_valid_t = [t for t in range(len(start_array))
                                if
                                np.maximum(0, np.minimum(end_array, start_array[t]) - np.maximum(start_array, start_array[t]+t_pre)).sum() < abs(t_pre)*0.1
                                and
                                np.maximum(0, np.minimum(end_array, start_array[t]+t_post) - np.maximum(start_array,start_array[t])).sum() > abs(t_post)*0.75
                                and
                                (start_array[t]>= time_blocks[b][0]*60) and (start_array[t] < time_blocks[b][1]*60)
                                and
                                (t in event_index_list)
                                and not any(
                                    t_pre + start_array[t] <= value*60 <= t_post + start_array[t]
                                    for sublist in contime for value in sublist     )]

                            end_valid_t = [t for t in range(len(end_array))
                                if
                                np.maximum(0, np.minimum(end_array, end_array[t]) - np.maximum(start_array, end_array[t]+t_pre)).sum() > abs(t_pre)*0.75
                                and
                                np.maximum(0, np.minimum(end_array, end_array[t]+t_post) - np.maximum(start_array,end_array[t])).sum() < abs(t_post)*0.1
                                and
                                (end_array[t]>= time_blocks[b][0]*60) and (end_array[t] < time_blocks[b][1]*60)
                                and
                                (t in event_index_list)
                                and not any(
                                    t_pre + end_array[t] <= value * 60 <= t_post + end_array[t]
                                    for sublist in contime for value in sublist)]

                        elif time_type=="strict":
                            start_valid_t = [t for t in range(len(start_array))
                                if
                                 (t == 0 or end_array[t - 1] - start_array[t] < t_pre )
                                and
                                end_array[t] - start_array[t] > t_post
                                and
                                (start_array[t]>= time_blocks[b][0]*60) and (start_array[t] < time_blocks[b][1]*60)
                                and
                                (t in event_index_list)
                                and not any(
                                    t_pre + start_array[t] <= value*60 <= t_post + start_array[t]
                                    for sublist in contime for value in sublist     )]

                            end_valid_t = [t for t in range(len(end_array))
                                if
                                (t == len(end_array)-1 or start_array[t+1] - end_array[t] > t_post )
                                and
                                start_array[t]-end_array[t]<t_pre
                                and
                                (end_array[t]>= time_blocks[b][0]*60) and (end_array[t] < time_blocks[b][1]*60)
                                and
                                (t in event_index_list)
                                and not any(
                                    t_pre + end_array[t] <= value * 60 <= t_post + end_array[t]
                                    for sublist in contime for value in sublist)]

                        else:  # time_type=="all":
                            start_valid_t = [t for t in range(len(start_array))
                                             if t in event_index_list
                                             and
                                             start_array[t]+t_pre>contime[0][0]*60
                                             and
                                             start_array[t]+t_post<contime[-1][1]*60]
                            end_valid_t = [t for t in range(len(end_array))
                                           if t in event_index_list
                                           and
                                           end_array[t]+t_post<contime[-1][1]*60
                                           and
                                           end_array[t]+t_pre>contime[0][0]*60]

                        plot_PETH(start_valid_t, start_array, end_array, t_pre, t_post, OF_tp, f_stft, t_stft, time_bin_emg,
                                  time_bin_power,
                                  sampling_rate, eeg, emg, velocity, linear_power, analog_tp, table_v, table_tp,pupil_tp, pupil_size, breathing_rate, tem,
                                  hr_tp, heartrate,
                                  fig, gs, b, output_dir, ch_name, event_name, "start", time_type, contime, dlc_type)

                        plot_PETH(end_valid_t, end_array, start_array, t_pre, t_post, OF_tp, f_stft, t_stft, time_bin_emg,
                                  time_bin_power,
                                  sampling_rate, eeg, emg, velocity, linear_power, analog_tp, table_v, table_tp,pupil_tp, pupil_size, breathing_rate, tem,
                                  hr_tp, heartrate,
                                  fig, gs, len(time_blocks) + b, output_dir, ch_name, event_name, "end", time_type, contime, dlc_type)



                    plt.tight_layout()

                    pdf_path = os.path.join (os.path.dirname(h5_path), "PETH_"+event_name+"_"+str(t_pre)+"s_"+str(t_post)+"s_"+time_type+"__"+ch_name+".pdf")
                    with PdfPages(pdf_path) as pdf:
                        pdf.savefig(fig, dpi=300)
                    plt.close(fig)
                    # print(f"Saved to  {pdf_path}")

def process_folder(data_folder):
    print(data_folder)
    data = open_h5(os.path.join(data_folder, "_Combined", "data.h5"))
    event_path = os.path.join(data_folder, "_Combined", "manual_event.csv")
    if os.path.exists(event_path):
        manual_event = pd.read_csv(event_path)
        dlc_type, dlc_dir, EEG_ch_dict, EMG_ch_dict, analog_dict, contime = extract_params(data_folder)
        process_events(*data, EEG_ch_dict, manual_event, contime,dlc_type)

def main():
    data_folder = select_folder()
    # data_folder = r"X:\Behavior\Openfield_EEG\Pup-IRES-Parietal-5x\20250108_z199-2_Pup-IRES(A170-1)-5x-2p-parietal_12w-M1V1-Ce_male_6P"  # for development
    process_folder(data_folder)

if __name__ == "__main__":
    main()