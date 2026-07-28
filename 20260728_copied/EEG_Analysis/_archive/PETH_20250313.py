import h5py
import numpy as np
import pandas as pd
from _archive.EEG_Analysis import select_folder, plot_timeseries_power, plot_heatmap, plot_timeseries, extract_params
import os
import matplotlib.pyplot as plt
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12   })
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages





def load_dataset(name, file):
    if name not in file:
        return None  # データセットが存在しない場合
    data = file[name][:]  # データ取得
    return data if not np.isnan(data).all() else None

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

    try:
        event_df = pd.read_hdf(h5_path, key="event_df")
    except (KeyError, FileNotFoundError):
        event_df = None

    return (h5_path, analog_tp, eeg, t_stft, f_stft, linear_power,dB_power,emg, sampling_rate, breathe, tem, breathing_rate, pupil_size, pupil_tp,
            table_v, table_tp, OF_tp, velocity,cum_d,distance,event_df)

def emg_rms(emg: np.ndarray, sr: int, window_sec: float) -> np.ndarray:
    N = int(sr * window_sec)
    num_windows = len(emg) // N
    emg_trimmed = emg[:num_windows * N].reshape(num_windows, N)
    rms_values = np.sqrt(np.mean(emg_trimmed ** 2, axis=1))
    return rms_values




def plot_PETH(valid_t,time_array,time_array2, t_pre,t_post,OF_tp,f_stft, t_stft, time_bin_emg, time_bin_power,sampling_rate,eeg, emg, velocity, linear_power, analog_tp, table_tp,
              fig, gs, column, output_dir, ch_name, epoch_name, epoch_type,time_type, contime):
    print(epoch_name)
    print(epoch_type)
    axes = [fig.add_subplot(gs[i, column]) for i in range(8)]
    ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7= axes

    n_event = len(valid_t)
    velocity_array = np.zeros([n_event, int((t_post - t_pre) / (OF_tp[1] - OF_tp[0]))], dtype=np.float64)
    linear_power_array = np.empty([n_event, len(f_stft), int((t_post - t_pre) / (t_stft[1] - t_stft[0]))])
    # band_power_array = np.empty([n_event, band_num, int((t_post-t_pre)/time_bin_power)])
    emg_rms_array = np.empty([n_event, int((t_post - t_pre) / time_bin_emg)])
    delta_array = np.empty([n_event, int((t_post - t_pre) / time_bin_power)])
    theta_array = np.empty([n_event, int((t_post - t_pre) / time_bin_power)])
    alpha_array = np.empty([n_event, int((t_post - t_pre) / time_bin_power)])
    beta_array = np.empty([n_event, int((t_post - t_pre) / time_bin_power)])
    gamma_array = np.empty([n_event, int((t_post - t_pre) / time_bin_power)])
    t_emg_tp = np.arange(t_pre, t_post, time_bin_emg)
    t_t_stft = np.arange(t_pre, t_post, t_stft[1] - t_stft[0])
    t_OF_tp = np.arange(t_pre, t_post, OF_tp[1] - OF_tp[0])
    i = 0
    print(t_pre)
    print(valid_t)
    for t in valid_t:
        print(t)
        tmin, tmax = time_array[t] + t_pre, time_array[t] + t_post
        analog_tp_mask = (analog_tp >= tmin) & (analog_tp <= tmax)
        t_stft_mask = (t_stft >= tmin) & (t_stft < tmax)
        table_tp_mask = (table_tp >= tmin) & (table_tp < tmax)
        OF_tp_mask = (OF_tp >= tmin) & (OF_tp < tmax)
        t_linear_power = linear_power[:, t_stft_mask]
        t_analog_tp = analog_tp[analog_tp_mask] - time_array[t]

        t_velocity = velocity[OF_tp_mask]
        t_emg = emg_rms(emg[analog_tp_mask], sampling_rate, time_bin_emg)
        t_eeg = eeg[analog_tp_mask]

        if time_type=="all":
            Otp = OF_tp[OF_tp_mask]
            atp = analog_tp[analog_tp_mask]
            stfttp = t_stft[t_stft_mask]
            emtp = np.arange(atp[0], atp[-1], time_bin_emg)
            if len(emtp) > len(t_emg):
                emtp = emtp[:len(t_emg)]  # 余分な要素を削除
            elif len(emtp) < len(t_emg):
                emtp = np.pad(emtp, (0, len(t_emg) - len(emtp)), mode='edge')

            OF_tp_mask2 = np.zeros_like(Otp, dtype=bool)
            analog_tp_mask2 = np.zeros_like(atp, dtype=bool)
            t_stft_mask2 = np.zeros_like(stfttp, dtype=bool)
            emg_tp_mask2 = np.zeros_like(emtp, dtype=bool)

            for (start, end) in contime:
                condition1 = (Otp >= start*60) & (Otp <= end*60)  # Otpがcontime[i]の範囲にある
                a_condition1 = (atp >= start*60) & (atp <= end*60)
                s_condition1 = (stfttp >= start*60) & (stfttp <= end*60)
                em_condition1 = (emtp >= start * 60) & (emtp <= end * 60)
                if epoch_type=="start":
                    condition2 = t == 0 or Otp > time_array2[t - 1]
                    condition3 = Otp < time_array2[t]  # Otpがtime_array2[t]より小さい

                    a_condition2 = t == 0 or atp > time_array2[t - 1]
                    a_condition3 = atp < time_array2[t]

                    s_condition2 = t == 0 or stfttp > time_array2[t - 1]
                    s_condition3 = stfttp < time_array2[t]

                    em_condition2 = t == 0 or emtp > time_array2[t - 1]
                    em_condition3 = emtp < time_array2[t]

                else: #end
                    condition2 = Otp>time_array2[t]
                    condition3 = t == len(time_array)-1 or Otp < time_array2[t+1]  # Otpがtime_array2[t]より小さい

                    a_condition2 = atp > time_array2[t]
                    a_condition3 = t == len(time_array) - 1 or atp < time_array2[t + 1]

                    s_condition2 = stfttp > time_array2[t]
                    s_condition3 = t == len(time_array) - 1 or stfttp < time_array2[t + 1]

                    em_condition2 = emtp > time_array2[t]
                    em_condition3 = t == len(time_array) - 1 or emtp < time_array2[t + 1]
                OF_tp_mask2 |= condition1 & condition2 & condition3  # 3つの条件を満たすものをマスク
                analog_tp_mask2 |= a_condition1 & a_condition2 & a_condition3
                t_stft_mask2 |= s_condition1 & s_condition2 & s_condition3
                emg_tp_mask2 |= em_condition1 & em_condition2 & em_condition3

            t_velocity =  np.where(OF_tp_mask2, t_velocity, np.nan)
            t_linear_power = np.where(t_stft_mask2, t_linear_power, np.nan)
            t_emg = np.where(emg_tp_mask2, t_emg, np.nan)
            t_eeg = np.where(analog_tp_mask2, t_eeg, np.nan)

        if len(t_velocity) > velocity_array.shape[1]:
            t_velocity = t_velocity[:velocity_array.shape[1]]  # 余分な要素を削除
        elif len(t_velocity) < velocity_array.shape[1]:
             t_velocity = np.pad(t_velocity, (0, velocity_array.shape[1] - len(t_velocity)), mode='edge')

        # plot
        plot_timeseries(t_OF_tp, t_velocity, 1, ax0, color="gray", lw=0.2, title="", ylabel="", ylim=(0, 50),
                        label=None)
        plot_timeseries(t_emg_tp, t_emg, 1, ax1, color="gray", lw=0.2, title="", ylabel="", ylim=(0, 200), label=None)
        powers = plot_timeseries_power(t_eeg, t_analog_tp, sampling_rate, time_bin_power,
                                       [ax3, ax4, ax5, ax6, ax7], 0.2, legend=False)

        # concatenate for merge
        linear_power_array[i] = t_linear_power
        velocity_array[i] = t_velocity #.copy()

        emg_rms_array[i] = t_emg
        delta_array[i] = np.array(powers.get('Delta (0.5-4 Hz)', []))
        theta_array[i] = np.array(powers.get('Theta (4-8 Hz)', []))
        alpha_array[i] = np.array(powers.get('Alpha (8-12 Hz)', []))
        beta_array[i] = np.array(powers.get('Beta (12-30 Hz)', []))
        gamma_array[i] = np.array(powers.get('Gamma (30-100 Hz)', []))
        i += 1

    # plot average
    average_power =np.nanmean(linear_power_array, axis=0)
    average_then_db = 10 * np.log10(average_power + 1e-10)
    plot_heatmap(ax2, t_t_stft, f_stft, average_then_db, "STFT dB Power", "Frequency (Hz)", 100, "rainbow", [-10, 33])
    average_v = np.nanmean(velocity_array, axis=0)
    plot_timeseries(t_OF_tp, average_v, 1, ax0, color='#1f77b4', lw=3, title="Velocity",ylabel="mm/s", ylim=(0, 50), label=None)
    average_emg_rms = np.nanmean(emg_rms_array, axis=0)
    plot_timeseries(t_emg_tp, average_emg_rms, 1, ax1, color='#1f77b4', lw=3, title="EMG-RMS",ylabel=None, ylim=(0, 200), label=None)


    delta_linear=np.nanmean(delta_array, axis=0)
    theta_linear=np.nanmean(theta_array, axis=0)
    alpha_linear=np.nanmean(alpha_array, axis=0)
    beta_linear=np.nanmean(beta_array, axis=0)
    gamma_linear=np.nanmean(gamma_array, axis=0)
    delta = 10 * np.log10(delta_linear + 1e-10)
    theta = 10 * np.log10(theta_linear + 1e-10)
    alpha = 10 * np.log10(alpha_linear + 1e-10)
    beta = 10 * np.log10(beta_linear + 1e-10)
    gamma = 10 * np.log10(gamma_linear + 1e-10)
    power_time_array = np.arange(t_pre + time_bin_power / 2, t_post + time_bin_power / 2, time_bin_power)
    plot_timeseries(power_time_array, delta, 1, ax3, color='#1f77b4', lw=4, title="delta power", ylabel="(dB)",ylim=(55, 85), label=None)
    plot_timeseries(power_time_array, theta, 1, ax4, color='#1f77b4', lw=4, title="theta power", ylabel="(dB)",ylim=(55, 85), label=None)
    plot_timeseries(power_time_array, alpha, 1, ax5, color='#1f77b4', lw=4, title="alpha power", ylabel="(dB)",ylim=(55, 85), label=None)
    plot_timeseries(power_time_array, beta, 1, ax6, color='#1f77b4', lw=4, title="beta power", ylabel="(dB)",ylim=(55, 85), label=None)
    plot_timeseries(power_time_array, gamma, 1, ax7, color='#1f77b4', lw=4, title="gamma power", ylabel="(dB)",ylim=(55, 85), label=None)

    #store data in h5
    """
    #TODO
    timeblockごとに保存できる形には直す。
    """

    h5_name = os.path.join(output_dir, epoch_name+"_"+epoch_type+"_"+str(t_pre)+"s_"+str(t_post)+"s_PETH_average_"+time_type+"_"+ch_name+".h5")
    with h5py.File(h5_name, "w") as f:
        f.create_dataset("OF_tp", data=t_OF_tp)
        f.create_dataset("velocity", data=average_v)

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


def process_events( h5_path, analog_tp, eegs, t_stft, f_stft, linear_powers,dB_power,emg, sampling_rate, breathe, tem, breathing_rate, pupil_size, pupil_tp,
            table_v, table_tp, OF_tp, velocity,cum_d,distance,event_df, EEG_ch_dict, manual_event, contime):

    # PETH_time_list = [[-20,600],[-20,300],[-600,20],[-180,20],[-60,60],[-30,30],[-12,12]]
    # time_type_list = ["rough", "rough", "rough", "rough", "rough", "strict", "strict"]
    """
    "all":　各epoch前後をすべて抽出。他のepochと重なる部分だけNanとして無視して平均する
    "strict": 
    "rough": 
    """
    PETH_time_list = [[-180, 180],[-120, 120],[-300, 300],[-60,60]]
    time_type_list = ["all","all", "all", "all"]
    time_bin_power = 2  # for plot_timeseries_power
    time_bin_emg = 0.5
    time_blocks = [[0, 180]] #min
    output_dir = os.path.dirname(h5_path)
    emg =emg[0]

    for ch in range (len(eegs)):
        eeg=eegs[ch]
        linear_power=linear_powers[ch]
        keys=list(EEG_ch_dict.keys())
        ch_name = keys[ch]

        for p, PETH_time in enumerate(PETH_time_list):
            t_pre, t_post = PETH_time[0], PETH_time[1]# sec plot
            time_type = time_type_list[p]
            for event_df in [manual_event]: #[event_df, manual_event]:
                start_array = event_df["start_time"].to_numpy()
                end_array = event_df["end_time"].to_numpy()

                for event_name in event_df["event_name"].unique().tolist():
                    fig = plt.figure(figsize=(len(time_blocks)*8, 21))
                    gs = gridspec.GridSpec(8, len(time_blocks)*2, height_ratios=[1, 1, 1, 1,1,1,1,1])
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

                        else:  # all
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
                                  sampling_rate, eeg, emg, velocity, linear_power, analog_tp, table_tp,
                                  fig, gs, b, output_dir, ch_name, event_name, "start", time_type, contime)

                        plot_PETH(end_valid_t, end_array, start_array, t_pre, t_post, OF_tp, f_stft, t_stft, time_bin_emg,
                                  time_bin_power,
                                  sampling_rate, eeg, emg, velocity, linear_power, analog_tp, table_tp,
                                  fig, gs, len(time_blocks) + b, output_dir, ch_name, event_name, "end", time_type, contime)



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
        process_events(*data, EEG_ch_dict, manual_event, contime)

def main():
    data_folder = select_folder()
    # data_folder = r"X:\Behavior\Openfield_EEG\Pup-IRES-Parietal-1x\20240925_z178-4_temp"  # for development
    process_folder(data_folder)

if __name__ == "__main__":
    main()