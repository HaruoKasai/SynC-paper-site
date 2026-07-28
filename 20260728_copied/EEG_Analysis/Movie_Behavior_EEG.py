import os
import glob
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12   })
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import stft

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from moviepy.editor import VideoFileClip, CompositeVideoClip, VideoClip
from moviepy.video.fx.all import rotate
from PIL import Image, ImageDraw
from EEG_Analysis import extract_params, plot_timeseries_power, plot_timeseries, plot_heatmap
from PETH import emg_rms, load_dataset
import h5py

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
        digital_timer = load_dataset("all_digital_timer", f)

    try:
        event_df = pd.read_hdf(h5_path, key="event_df")
    except (KeyError, FileNotFoundError):
        event_df = None

    return (h5_path, analog_tp, eeg, t_stft, f_stft, linear_power,dB_power,emg, sampling_rate, breathe, tem, breathing_rate, pupil_size, pupil_tp,
            table_v, table_tp,  digital_timer, OF_tp, velocity,cum_d,distance,event_df, hr_tp, heartrate)


def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select the 'data' directory", initialdir=r"X:\Behavior")
    root.destroy()
    return folder_path


def safe_path(path):
    # UNCパスなどすでに始まっていたら何もしない
    if not path.startswith('\\\\?\\'):
        path = '\\\\?\\' + os.path.abspath(path)
    return path


def make_even(x):
    return x if x % 2 == 0 else x + 1

def process_folder(data_folder):
    output_dir = os.path.join(data_folder, "_movies")
    os.makedirs(output_dir, exist_ok=True)

    #　近くにstate changeや実験切れ目がない時点を抽出
    event_path = os.path.join(data_folder, "_Combined", "manual_event_beh.csv")
    c_event_path = os.path.join(data_folder, "_Combined", "manual_event.csv")
    # event_path = os.path.join(data_folder, "_Combined", "manual_event.csv")
    # event_path = os.path.join(data_folder, "_Combined", "manual_event_laser.csv")
    event_df = pd.read_csv(event_path) if os.path.exists(event_path) else pd.DataFrame()
    c_event_df = pd.read_csv(c_event_path)

    _,dlc_dir, EEG_ch_dict, *_, cont_time = extract_params(data_folder)
    boundary_list = [60 * t for pair in cont_time for t in pair] #experiment boundary

    melted = pd.DataFrame({
        'time': event_df['start_time'].tolist() + event_df['end_time'].tolist(),
        'label': [f"{n}_start" for n in event_df['event_name']] + [f"{n}_end" for n in event_df['event_name']]
    }).sort_values('time').reset_index(drop=True)

    before = 20 #before秒前からの間に、実験切れ目や、前のstate_Changeが入らないという条件で抽出
    after = 20
    def keep_row(idx):
        t = melted.loc[idx, 'time']
        if idx > 0 and t - melted.loc[idx - 1, 'time'] < before:
            return False
        if idx < len(melted) - 1 and melted.loc[idx + 1, 'time'] - t < after:
            return False
        for tp in boundary_list:
            if t - before <= tp <= t + after:
                return False
        return True
    filtered_df = melted[[keep_row(i) for i in range(len(melted))]].reset_index(drop=True)
    print(filtered_df)

    # XXsec間データを抽出して画像保存　EEGraw, EMG raw, gammma power
    _, analog_tp,eeg,_,_,_,_,emg,sampling_rate,_,_,_,_,_,_,_,digital_timer,OF_tp,velocity,*_= open_h5(os.path.join(data_folder, "_Combined", "data.h5"))
    key = "M1-Ce" if "M1-Ce" in EEG_ch_dict else "M1-V1"
    eeg = eeg[list(EEG_ch_dict.keys()).index(key)]
    t_pre = -5
    t_post = 5
    time_bin_emg = 0.5 #PETH.pyとそろえた　2025.05.23
    time_bin_power =2 #PETH.pyとそろえた　2025.05.23
    video_rate = 1/np.diff(OF_tp)[0]
    print(video_rate)
    figsize = (12, 5)
    # figsize = (25, 10)
    dpi = 100
    for _, row in filtered_df.iterrows():

        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = gridspec.GridSpec(4, 1,
                               height_ratios=[1.5, 1.5, 1.5, 1] # height_ratios=[1.5, 1.5, 1, 1, 1.5, 1]
                               )
        plt.subplots_adjust(hspace=0.05)
        axes= [fig.add_subplot(gs[i]) for i in range(4)]
        ax0, ax1, ax4, ax5 = axes # ax0, ax1, ax2, ax3, ax4, ax5 = axes




        time = float(row['time'])
        label = row['label']
        print(label + "_" + str(time) + "s")

        exp_idx = next((i for i, (start, end) in enumerate(cont_time) if start <= time / 60 < end), None)
        exp_dir_in_Z = glob.glob(os.path.join(data_folder, "[!_]*"))[exp_idx]
        df_top = pd.read_csv(os.path.join(exp_dir_in_Z, "_DLC_analysis", "light_rising_frames_top.csv"))
        top_light_frame = df_top["rising_frame"].to_numpy()

        #videoとEEGの時間合わせ
        print("time", time)
        frame = (time - cont_time[exp_idx][0]*60)* video_rate
        last_frame_idx = np.searchsorted(top_light_frame, frame) - 1
        last_frame = top_light_frame[last_frame_idx] if last_frame_idx >= 0 else None
        print("frame", frame)
        print("last_frame", last_frame)
        print("last_index", last_frame_idx)
        last_frame_time = last_frame/video_rate
        dig_timer_exp = digital_timer[exp_idx]
        closest_in_digital_time =dig_timer_exp[np.argmin(np.abs(dig_timer_exp - last_frame_time))]
        time_in_Bl = closest_in_digital_time + (time - last_frame_time)
        print("time_in_Bl",time_in_Bl)

        # tp_mask = (analog_tp >= time+t_pre) & (analog_tp <= time+t_post)
        tp_mask = (analog_tp >= time_in_Bl + t_pre) & (analog_tp <= time_in_Bl + t_post)  # TODO ここの取り方をかえればよさそう。
        t_tp = analog_tp[tp_mask]
        t_emg = emg[0][tp_mask]
        t_emg_rms = emg_rms(t_emg, sampling_rate, time_bin_emg)
        t_eeg = eeg[tp_mask]
        t_emg_rms_tp = np.arange(time+t_pre, time+t_post, time_bin_emg)
        OF_tp_mask = (OF_tp >= time+t_pre) & (OF_tp <= time+t_post)

        t_OF_tp = OF_tp[OF_tp_mask]
        t_velocity = velocity[OF_tp_mask]

        epoch_length = 2
        nperseg = int(epoch_length * sampling_rate)
        f_stft, t_stft, Zxx = stft(t_eeg, fs=sampling_rate, nperseg=nperseg, noverlap=nperseg // 2)
        # stftは隣接する(Hann)窓同士が50%オーバーラップされていてスムージングされているらしい。
        # t_stft = t_stft[:-1]  # 各実験を連結していくときに両端があるとかぶってしまうので削除
        # print(f_stft)
        # print(Zxx.shape)
        # Zxx = Zxx[:,:-1]

        cutoff = np.argmax(f_stft > 80)
        f_stft = f_stft[:cutoff]
        Zxx = Zxx[:cutoff, :]
        t_stft += time+t_pre
        linear_power = np.abs(Zxx) ** 2
        dB_power = 10 * np.log10(linear_power + 1e-10)
        plot_heatmap(ax1, t_stft, f_stft, dB_power, None, "Frequency (Hz)", 80, "rainbow", [-10, 33])
        ax1.set_yticks([0, 40, 80])
        plot_timeseries(t_tp- (time_in_Bl-time), t_eeg, 1, ax0, color="gray", lw=0.2, title="", ylabel="EEG", ylim=(-500,500), label=None)
        plot_timeseries(t_tp- (time_in_Bl-time), t_emg, 1, ax4, color="gray", lw=0.2, title="", ylabel="EMG", ylim=(-1000, 1000), label=None)
        # plot_timeseries(t_emg_rms_tp, t_emg_rms, 1, ax2, color="gray", lw=1, title="", ylabel="EMG-RMS", ylim=(0, 200), label=None)

        plot_timeseries(t_OF_tp, t_velocity, 1, ax5, color="gray", lw=1, title="", ylabel="Velocity", ylim=(0, 100),
                        label=None)
        for ep in range(len(c_event_df)):
            color = "#d52e80" if c_event_df.loc[ep, "event_name"] == "StateC" else "#727171"
            ax5.axvspan(c_event_df.loc[ep, "start_time"], c_event_df.loc[ep, "end_time"], color=color, alpha=0.3,
                        linewidth=0, label=label)

        """
        powers = plot_timeseries_power(t_eeg, t_tp, sampling_rate, time_bin_power,
                                       [ax3, None, None, None, ax2], 1, legend=False, dB=False)
        
        ax2.set_ylabel("Gamma power")
        ax2.set_title(None)
        ax2.set_ylim(0, 3e6)
        ax2.axhline(y=1e6, linestyle='--', linewidth=0.3, color ="gray")
        ax2.axhline(y=2e6, linestyle='--', linewidth=0.3, color="gray")

        ax3.set_ylabel("Delta power")
        ax3.set_title(None)
        ax3.set_ylim(0, 3e7)
        ax3.axhline(y=1e7,linestyle='--', linewidth=0.3, color="gray")
        ax3.axhline(y=2e7, linestyle='--', linewidth=0.3, color="gray")
        """

        for ax in axes[:-1]:  # ax0～ax4
            ax.set_xticklabels([])
            ax.set_xlabel(None)
        for ax in axes:
            ax.set_xlim(time+t_pre, time+t_post)
            ax.margins(x=0)

        plt.tight_layout()
        pdf_path = os.path.join(output_dir, label+"_"+str(time)+"s.pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, dpi=300)
        # fig.savefig(os.path.join(output_dir, label+"_"+str(time)+"_s.png"), dpi=300)

        canvas = FigureCanvas(fig)
        canvas.draw()
        fig.set_dpi(dpi)  # 明示的に解像度を高めに指定
        fig.set_size_inches(figsize)

        fig_width, fig_height = canvas.get_width_height()
        base_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        base_img = base_img.reshape((fig_height, fig_width, 3))
        # plt.close(fig)

        # x軸共有：例として ax4 から描画範囲取得
        bbox_x = ax4.get_position()
        x0 = int(bbox_x.x0 * fig_width)
        x1 = int(bbox_x.x1 * fig_width)

        # 縦にまたがる範囲（ax0〜ax4の上下）
        # bbox_top = ax0.get_position()
        # bbox_bottom = ax4.get_position()
        # y_top = int(bbox_top.y0 * fig_height)
        # y_bottom = int(bbox_bottom.y1 * fig_height)
        duration = t_post - t_pre
        # 赤線描画関数
        def make_waveform_frame(t):
            img = base_img.copy()
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)

            # 時間に対応するx位置（x0〜x1の中で動かす）
            x = int(x0 + (x1 - x0) * (t / duration))

            # 縦断する赤線（ax0〜ax4の範囲内だけに）
            draw.line([(x, 0), (x, fig_height)], fill=(255, 0, 0), width=2)
            return np.array(img_pil)
        waveform_clip = VideoClip(make_frame=make_waveform_frame, duration=duration).set_fps(20)

        #Behavior Videoの抽出
        # print("exp_idx="+str(exp_idx))

        video_dir = dlc_dir.replace("/", "\\").replace(r"Z:\ProbeG", r"Y:\raw_ProbeG\prj1\OFT")
        # video_dir = dlc_dir.replace(r"Z:\ProbeG", r"Y:\raw_ProbeG\prj1\OFT")
        exp_dir = glob.glob(os.path.join(video_dir, "[!_]*"))[exp_idx]
        video_path =glob.glob(os.path.join(exp_dir,"raw_video", "*avi"))[0]
        print(video_path)

        start_in_video = (time+t_pre-cont_time[exp_idx][0]*60) * video_rate/20 #TODO EEGと正確に合わせるにはここの設定の仕方を変えなくては
        end_in_video = (time+t_post-cont_time[exp_idx][0]*60) * video_rate/20
        video = VideoFileClip(video_path).subclip(start_in_video, end_in_video).fx(rotate, 90)


        #light timerを使って。上記top videoに対応する時間のside videoを抽出
        #top-EEGさえあえば、この部分は変えなくていいはず
        side_video_path =safe_path(glob.glob(os.path.join(exp_dir_in_Z,"raw_video", "*avi"))[0])
        df_side = pd.read_csv(os.path.join(exp_dir_in_Z, "_DLC_analysis", "light_rising_frames_side.csv"))
        side_light_frame = df_side["rising_frame"].to_numpy()

        # top_fps = (top_light_frame[50]-top_light_frame[0])/500 #EEGのclockを使用せず、light間隔を10秒としてひとまず計算に利用する
        # side_fps = (side_light_frame[50]-side_light_frame[0])/500
        # ***動画は”20Hzとして”記録されている
        def top_to_side (sec_in_top, top_light_frame, side_light_frame):
            print("video_rate", video_rate)
            frame_top = sec_in_top * 20
            x = np.searchsorted(top_light_frame, frame_top, side='right') - 1
            print(x)
            if x < -1 or x >= len(top_light_frame):
                raise ValueError("frame_top is out of the range of top_light_frame values")

            if x>=0:
                y = frame_top - top_light_frame[x]
                frame_side = side_light_frame[x] + y
            if x==-1:
                y = frame_top + top_light_frame[0]
                frame_side = side_light_frame[0] - y

            return frame_side/20

        start_in_side_video = top_to_side(start_in_video, top_light_frame, side_light_frame)
        print("start", start_in_video, start_in_side_video)
        end_in_side_video = top_to_side(end_in_video, top_light_frame, side_light_frame)
        side_video =VideoFileClip(side_video_path).subclip(start_in_side_video, end_in_side_video)
        side_video = side_video.resize(height=video.h)

        #長い方の動画から、randomにframeを間引く・同じ長さになるように。再生は20Hz
        fps=20
        len_v = video.duration
        len_s = side_video.duration
        # 長い方を短い方に合わせる
        if len_v > len_s:
            n_v = int(len_v * fps)
            n_s = int(len_s * fps)
            idx = sorted(np.random.choice(np.arange(n_v), size=n_s, replace=False))

            def time_map_v(t):
                i = int(t * fps)
                if i >= len(idx):
                    i = len(idx) - 1
                return idx[i] / fps

            video = video.fl_time(time_map_v, apply_to=['mask', 'video']).set_duration(len_s)

        else:
            n_v = int(len_v * fps)
            n_s = int(len_s * fps)
            idx = sorted(np.random.choice(np.arange(n_s), size=n_v, replace=False))

            def time_map_s(t):
                i = int(t * fps)
                if i >= len(idx):
                    i = len(idx) - 1
                return idx[i] / fps

            side_video = side_video.fl_time(time_map_s, apply_to=['mask', 'video']).set_duration(len_v)

        # total_width = max(video.w + side_video.w, fig_width)  # W: figの幅
        # total_height = video.h + fig_height
        total_width = make_even(max(video.w + side_video.w, fig_width))
        total_height = make_even(video.h + fig_height)

        video_width, video_height = video.size
        side_width, side_height = side_video.size

        # 中央に並べるための X 座標（左右に並べて中央寄せ）
        video_x = (total_width - video_width - side_width) // 2
        side_x = video_x + video_width
        video_y = 0
        side_y = 0
        waveform_x = (total_width - waveform_clip.w) // 2
        waveform_y = total_height - waveform_clip.h
        # Composite
        final = CompositeVideoClip([
            video.set_position((video_x, video_y)),
            side_video.set_position((side_x, side_y)),
            waveform_clip.set_position((waveform_x, waveform_y))
        ], size=(total_width, total_height))

        # final = CompositeVideoClip([
        #     video.set_position(("left", "top")),
        #     side_video.set_position(("right", "top")),
        #     waveform_clip.set_position(("center", "bottom"))
        # ], size=(total_width, total_height))

        # 6. 書き出し
        final.write_videofile(os.path.join(output_dir, label+"_"+str(time)+"s.mp4"),
                              fps=20,
                              codec="libx264",
                              audio=False,
                              preset="ultrafast",
                              ffmpeg_params=[
                                  "-pix_fmt", "yuv420p"
                              ]
                              )




def main():
    data_folder = select_folder()
    # data_folder = r"X:\Behavior\Openfield_EEG\Ctrl_mouse\20250430_z240_WT(11w-MVC-8p-ecg)_Openfield_ECG-CH5" #for development
    process_folder(data_folder)

if __name__ == "__main__":
    main()
