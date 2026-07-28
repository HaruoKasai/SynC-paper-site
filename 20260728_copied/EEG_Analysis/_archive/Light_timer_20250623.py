import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import pandas as pd
import json
import shutil
from _archive.EEG_Analysis import select_folder, extract_params
from pathlib import Path

# ---------- 動画情報取得（FPSなど） ----------

def extract_light_timing(data_folder, camera, w, h, position):
    _, dlc_dir, *_= extract_params(data_folder)
    for e, exp_dir in enumerate(glob.glob(os.path.join(data_folder, "[!_]*"))):
        print(exp_dir)

        if camera =="top":
            dlc_exp_dir = glob.glob(os.path.join(dlc_dir, "[!_]*"))[e]
            json_path = os.path.join(dlc_exp_dir, "raw_video", "video_info.json")
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                path = data["raw_video_list"][0]
            video_path = os.path.join(r"Y:", path)
        elif camera=="side":
            video_path = glob.glob(os.path.join(exp_dir, "raw_video", "202*orizontal*.avi"))[0]

        video_path = Path(video_path).as_posix()
        # ffmpegを入れるようにtemp. 解析後消す
        temp_dir = os.path.join(r"C:\_temp_ffmpeg",os.path.basename(data_folder), os.path.basename(exp_dir), camera)
        os.makedirs(temp_dir, exist_ok=True)
        output_dir = os.path.join(exp_dir, "_DLC_analysis")
        os.makedirs(output_dir, exist_ok=True)

        #
        #
        # in_path = Path(video_path)
        # ffmpeg_in = "\\\\?\\" + str(in_path)


        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,nb_frames",
            "-of", "default=noprint_wrappers=1", video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        fps = 20  # fallback
        for line in result.stdout.splitlines():
            if "r_frame_rate=" in line:
                num, denom = map(int, line.split('=')[1].split('/'))
                fps = num / denom
            if "nb_frames=" in line:
                total_frames = int(line.split('=')[1])

        # print(f"FPS: {fps:.2f}, Estimated total frames: {total_frames}")

        # ---------- ffmpegで画像化（左下60x60をトリミング） ----------
        print("\n--- Extracting ROI images using ffmpeg ---")
        if position == "bottom-left":
            x = 0
            y = f"ih-{h}"  # 動画の高さから height を引いた位置
            crop_filter = f"crop={w}:{h}:{x}:{y}"
        if position == "bottom-right":
            x = f"iw-{w}"
            y = f"ih-{h}"  # 動画の高さから height を引いた位置
            crop_filter = f"crop={w}:{h}:{x}:{y}"

        extract_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", crop_filter,
            os.path.join(temp_dir, "frame_%05d.png")
        ]
        subprocess.run(extract_cmd, check=True)
        print("Frame extraction complete.")

        # ---------- 輝度計算 ----------
        print("\n--- Calculating brightness ---")
        img_files = sorted(glob.glob(os.path.join(temp_dir, "frame_*.png")))
        brightness = [np.mean(cv2.imread(f, cv2.IMREAD_GRAYSCALE)) for f in img_files]
        print(f"Processed {len(brightness)} frames.")

        # ---------- プロット ----------
        time_axis = [i / fps for i in range(len(brightness))]
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, brightness)
        plt.xlabel("Time (s)")
        plt.ylabel("Mean Brightness (60x60 bottom-left)")
        plt.title("Brightness Over Time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "light_timeseries_"+camera+".png")
        plt.savefig(plot_path, dpi=150)



        diff = np.diff(brightness)

        # 閾値 = 平均 + 3×標準偏差
        threshold = np.mean(diff) + 3 * np.std(diff)

        # 立ち上がりとみなせるフレームを取得（+1 は diff のずれ補正）
        rising_frames = np.where(diff > threshold)[0] + 1

        # ノイズで近接するフレームが複数立ち上がりとして検出されることがあるので、最初の1個だけ取る
        min_interval = 100  # 最小間隔（フレーム数）を空ける
        final_risings = []
        last_frame = -min_interval
        for f in rising_frames:
            if f - last_frame >= min_interval:
                final_risings.append(f)
                last_frame = f

        # CSV出力
        df = pd.DataFrame({'rising_frame': final_risings})
        csv_path = os.path.join(output_dir, "light_rising_frames_"+camera+".csv")
        df.to_csv(csv_path, index=False)

        shutil.rmtree(temp_dir)

        print(f"✓ Detected {len(final_risings)} rising edges")
        print(f"✓ Saved to {csv_path}")


def main():
    data_folder = select_folder()
    # data_folder =r"X:\Behavior\Openfield_EEG\Ctrl_mouse\20250430_z240_WT(11w-MVC-8p-ecg)_Openfield_ECG-CH5"
    extract_light_timing(data_folder, "top", 60,60, "bottom-left")
    extract_light_timing(data_folder, "side", 200, 400, "bottom-right")

if __name__ == "__main__":
    main()