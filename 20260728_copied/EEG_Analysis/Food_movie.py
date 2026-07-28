import os
import glob
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
from _archive.EEG_Analysis import extract_params, plot_timeseries_power, plot_timeseries, plot_heatmap
from PETH import open_h5, emg_rms
from _archive import Movie_Behavior_EEG_all_state_changes_20251103 as MB


def process_folder(data_folder):
    group_name = os.path.basename(os.path.dirname(data_folder))
    output_dir = os.path.join(r"X:\Behavior\Openfield_EEG\_Food_intake_movie", group_name, os.path.basename(data_folder))
    os.makedirs(output_dir, exist_ok=True)

    food_path = os.path.join(data_folder, "_Combined", "food_approach.csv")
    food_df = pd.read_csv(food_path) if os.path.exists(food_path) else pd.DataFrame()

    event_path = os.path.join(data_folder, "_Combined", "manual_event.csv")
    event_df = pd.read_csv(event_path) if os.path.exists(event_path) else pd.DataFrame()

    _,dlc_dir, EEG_ch_dict, *_, cont_time = extract_params(data_folder)


    # XXsec間データを抽出して画像保存　EEGraw, EMG raw, gammma power
    _, analog_tp,eeg,_,_,_,_,emg,sampling_rate,_,_,_,_,_,_,_,OF_tp,velocity,*_= open_h5(os.path.join(data_folder, "_Combined", "data.h5"))
    key = "M1-Ce" if "M1-Ce" in EEG_ch_dict else "M1-V1"
    eeg = eeg[list(EEG_ch_dict.keys()).index(key)]
    t_pre = -5
    t_post = 10
    time_bin_emg = 0.5 #PETH.pyとそろえた　2025.05.23
    time_bin_power =2 #PETH.pyとそろえた　2025.05.23
    video_rate = 1/np.diff(OF_tp)[0]
    print(video_rate)
    figsize = (25, 10)
    dpi = 100
    for idx, row in food_df.iterrows():
        start = float(row['start_time'])
        end = float(row['end_time'])
        food_event_name = row['event_name']

        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = gridspec.GridSpec(6, 1, height_ratios=[1.5, 1.5, 1, 1, 1.5, 1])
        plt.subplots_adjust(hspace=0.05)
        axes= [fig.add_subplot(gs[i]) for i in range(6)]
        ax0, ax1, ax2, ax3, ax4, ax5 = axes

        # time = float(row['time'])
        # label = row['label']
        # print(label + "_" + str(time) + "s")
        tp_mask = (analog_tp >= start+t_pre) & (analog_tp <= end+t_post)
        t_tp = analog_tp[tp_mask]
        t_emg = emg[0][tp_mask]
        t_emg_rms = emg_rms(t_emg, sampling_rate, time_bin_emg)
        t_eeg = eeg[tp_mask]
        t_emg_rms_tp = np.arange(start+t_pre, end+t_post, time_bin_emg)
        OF_tp_mask = (OF_tp >= start+t_pre) & (OF_tp <= end+t_post)

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

        cutoff = np.argmax(f_stft > 100)
        f_stft = f_stft[:cutoff]
        Zxx = Zxx[:cutoff, :]
        t_stft += start+t_pre
        linear_power = np.abs(Zxx) ** 2
        dB_power = 10 * np.log10(linear_power + 1e-10)
        plot_heatmap(ax1, t_stft, f_stft, dB_power, None, "Frequency (Hz)", 100, "rainbow", [-10, 33])

        plot_timeseries(t_tp, t_eeg, 1, ax0, color="gray", lw=0.1, title="", ylabel="EEG", ylim=(-500,500), label=None)
        plot_timeseries(t_tp, t_emg, 1, ax4, color="gray", lw=0.1, title="", ylabel="EMG", ylim=(-1000, 1000), label=None)
        # plot_timeseries(t_emg_rms_tp, t_emg_rms, 1, ax2, color="gray", lw=1, title="", ylabel="EMG-RMS", ylim=(0, 200), label=None)
        powers = plot_timeseries_power(t_eeg, t_tp, sampling_rate, time_bin_power,
                                       [ax3, None, None, None, ax2], 1, legend=False, dB=False)
        plot_timeseries(t_OF_tp, t_velocity, 1, ax5, color="gray", lw=1, title="", ylabel="Velocity", ylim=(0,75), label=None)

        for ep in range(len(event_df)):
            label = event_df.iloc[ep]["event_name"]
            color = plt.get_cmap("tab10")(0) if event_df.loc[ep, "event_name"]=="StateC" else plt.get_cmap("tab10")(1)
            ax5.axvspan(event_df.loc[ep, "start_time"], event_df.loc[ep, "end_time"], color=color, alpha=0.3,
                    linewidth=0, label=label)
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

        for ax in axes[:-1]:  # ax0～ax4
            ax.set_xticklabels([])
            ax.set_xlabel(None)
        for ax in axes:
            ax.set_xlim(start+t_pre, end+t_post)
            ax.margins(x=0)

        plt.tight_layout()
        pdf_path = os.path.join(output_dir, food_event_name+"_"+str(start+t_pre)+"s_"+str(end+t_post)+"s.pdf")
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
        duration = end+t_post - (start+t_pre)
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
        exp_idx = next((i for i, (cont_starttime, cont_endtime) in enumerate(cont_time) if cont_starttime <= start/60 < cont_endtime), None)
        # print("exp_idx="+str(exp_idx))

        video_dir = dlc_dir.replace("/", "\\").replace(r"Z:\ProbeG", r"Y:\raw_ProbeG\prj1\OFT")
        # video_dir = dlc_dir.replace(r"Z:\ProbeG", r"Y:\raw_ProbeG\prj1\OFT")
        exp_dir = glob.glob(os.path.join(video_dir, "[!_]*"))[exp_idx]
        video_path =glob.glob(os.path.join(exp_dir,"raw_video", "*avi"))[0]
        print(video_path)

        start_in_video = (start+t_pre-cont_time[exp_idx][0]*60) * video_rate/20 #TODO EEGと正確に合わせるにはここの設定の仕方を変えなくては
        end_in_video = (end+t_post-cont_time[exp_idx][0]*60) * video_rate/20
        video = VideoFileClip(video_path).subclip(start_in_video, end_in_video).fx(rotate, 90)

        side_video = video  # いったん入れておく。sidevideoがないときに下のsize等の計算が大丈夫なように


        #light timerを使って。上記top videoに対応する時間のside videoを抽出
        #top-EEGさえあえば、この部分は変えなくていいはず
        exp_dir_in_Z = glob.glob(os.path.join(data_folder, "[!_]*"))[exp_idx]
        side_video_path =MB.safe_path(glob.glob(os.path.join(exp_dir_in_Z,"raw_video", "*avi"))[0])
        df_top = pd.read_csv(os.path.join(exp_dir_in_Z, "_DLC_analysis", "light_rising_frames_top.csv"))
        top_light_frame = df_top["rising_frame"].to_numpy()
        if os.path.exists(os.path.join(exp_dir_in_Z, "_DLC_analysis", "light_rising_frames_side.csv")):
            df_side = pd.read_csv(os.path.join(exp_dir_in_Z, "_DLC_analysis", "light_rising_frames_side.csv"))
            side_light_frame = df_side["rising_frame"].to_numpy()



            def top_to_side (sec_in_top, top_light_frame, side_light_frame):
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
            end_in_side_video = top_to_side(end_in_video, top_light_frame, side_light_frame)
            side_video =VideoFileClip(side_video_path).subclip(start_in_side_video, end_in_side_video)
            side_video = side_video.resize(height=video.h)

        # total_width = max(video.w + side_video.w, fig_width)  # W: figの幅
        # total_height = video.h + fig_height
        total_width = MB.make_even(max(video.w + side_video.w, fig_width))
        total_height = MB.make_even(video.h + fig_height)

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

        # 6. 書き出し
        final.write_videofile(os.path.join(output_dir, food_event_name+"_"+str(start+t_pre)+"s_"+str(end+t_post)+"s.mp4"),
                              fps=20,
                              codec="libx264",
                              audio=False,
                              preset="ultrafast",
                              ffmpeg_params=[
                                  "-pix_fmt", "yuv420p"
                              ]
                              )


def main():
    data_folder = MB.select_folder()
    # data_folder = r"X:\Behavior\Openfield_EEG\Ctrl_mouse\20250430_z240_WT(11w-MVC-8p-ecg)_Openfield_ECG-CH5" #for development
    # data_folder = r"X:\Behavior\Openfield_EEG\Pup-IRES-Parietal-1x\20250311_z178-4_Pup-IRES-1x-2p-P(27w-M-C-4p)_openfield_2ndAC_Food-cut"
    process_folder(data_folder)

if __name__ == "__main__":
    main()
