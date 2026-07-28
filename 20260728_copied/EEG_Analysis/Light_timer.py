import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import pandas as pd
import json
import shutil
from EEG_Analysis import select_folder, extract_params
from pathlib import Path

# ---------- 長いパス対応ユーティリティ ----------

def to_long_path(path):
    path = os.path.abspath(path)
    if not path.startswith("\\\\?\\"):
        return "\\\\?\\" + path
    return path

# ---------- 動画情報取得と処理 ----------

def extract_light_timing(data_folder, camera, w, h, position):
    _, dlc_dir, *_ = extract_params(data_folder)
    for e, exp_dir in enumerate(glob.glob(os.path.join(data_folder, "[!_]*"))):
        print(exp_dir)

        if camera == "top":
            dlc_exp_dir = glob.glob(os.path.join(dlc_dir, "[!_]*"))[e]
            json_path = os.path.join(dlc_exp_dir, "raw_video", "video_info.json")
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                path = data["raw_video_list"][0]
            video_path = os.path.join(r"Y:", path)
        elif camera == "side":
            video_path = glob.glob(os.path.join(exp_dir, "raw_video", "202*orizontal*.avi"))[0]

        video_path_unix = Path(video_path).as_posix()  # ffmpeg系にはUNIX形式が安全
        video_path_long = to_long_path(video_path_unix)  # 長パス対応

        temp_dir = os.path.join(r"C:\_temp_ffmpeg", os.path.basename(data_folder), os.path.basename(exp_dir), camera)
        os.makedirs(temp_dir, exist_ok=True)
        output_dir = os.path.join(exp_dir, "_DLC_analysis")
        os.makedirs(output_dir, exist_ok=True)

        # ---------- ffprobeで動画情報取得 ----------
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,nb_frames",
            "-of", "default=noprint_wrappers=1", video_path_long
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        fps = 20  # fallback
        for line in result.stdout.splitlines():
            if "r_frame_rate=" in line:
                num, denom = map(int, line.split('=')[1].split('/'))
                fps = num / denom
            if "nb_frames=" in line:
                total_frames = int(line.split('=')[1])

        # ---------- ffmpegで画像化 ----------
        print("\n--- Extracting ROI images using ffmpeg ---")
        if position == "bottom-left":
            x = 0
            y = f"ih-{h}"
        elif position == "bottom-right":
            x = f"iw-{w}"
            y = f"ih-{h}"
        else:
            raise ValueError(f"Unknown position: {position}")
        crop_filter = f"crop={w}:{h}:{x}:{y}"

        extract_cmd = [
            "ffmpeg", "-y",
            "-i", video_path_long,
            "-vf", crop_filter,
            to_long_path(os.path.join(temp_dir, "frame_%05d.png"))
        ]
        subprocess.run(extract_cmd, check=True)
        print("Frame extraction complete.")

        # ---------- 輝度計算 ----------
        print("\n--- Calculating brightness ---")
        img_files = sorted(glob.glob(to_long_path(os.path.join(temp_dir, "frame_*.png"))))
        brightness = [np.mean(cv2.imread(f.replace('\\\\?\\', ''), cv2.IMREAD_GRAYSCALE)) for f in img_files]
        print(f"Processed {len(brightness)} frames.")

        # ---------- プロット ----------
        time_axis = [i / fps for i in range(len(brightness))]
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, brightness)
        plt.xlabel("Time (s)")
        plt.ylabel("Mean Brightness ({}x{} {})".format(w, h, position))
        plt.title("Brightness Over Time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "light_timeseries_" + camera + ".png")
        plt.savefig(plot_path, dpi=150)

        # ---------- 立ち上がり検出 ----------
        diff = np.diff(brightness)
        threshold = np.mean(diff) + 3 * np.std(diff)
        rising_frames = np.where(diff > threshold)[0] + 1

        min_interval = 150
        final_risings = []
        last_frame = -min_interval
        for f in rising_frames:
            if f - last_frame >= min_interval:
                final_risings.append(f)
                last_frame = f

        # ---------- CSV出力 ----------
        df = pd.DataFrame({'rising_frame': final_risings})
        csv_path = os.path.join(output_dir, "light_rising_frames_" + camera + ".csv")
        df.to_csv(csv_path, index=False)

        # 一時フォルダ削除
        shutil.rmtree(temp_dir)

        print(f"✓ Detected {len(final_risings)} rising edges")
        print(f"✓ Saved to {csv_path}")

# ---------- main ----------

def main():
    data_folder = select_folder()
    extract_light_timing(data_folder, "top", 60, 60, "bottom-left")
    extract_light_timing(data_folder, "side", 200, 400, "bottom-right")

if __name__ == "__main__":
    main()
