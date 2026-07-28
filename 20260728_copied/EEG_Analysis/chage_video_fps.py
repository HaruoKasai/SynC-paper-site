#TODO crawlして全動画を変換するように

import subprocess
import glob
import os

folder = r"X:\Behavior\Openfield_EEG\Pup-IRES-Parietal-1x\20240925_z178-4_temp"
exp_list = glob.glob(os.path.join(folder, "[!_]*"))
for exp in exp_list:
    for avi in glob.glob(os.path.join(exp, "20*avi")):
        input_path = avi
        output_path = os.path.join(exp, "fps_adjusted.avi")
        new_fps = 19.95  # 変更したいFPS

        if not os.path.exists(output_path):
            # FFmpeg コマンドを実行
            subprocess.run([
                "ffmpeg", "-i", input_path, "-r", str(new_fps), "-c:v", "copy", "-c:a", "copy", output_path
            ])


# input_path = r"X:\Behavior\Openfield_EEG\Pup-IRES-Parietal-1x\20240925_z178-4_temp\06-post-60min_006\20240925-203836-C16980_1_OFT_Pup-IRES-low_178-4(5w-M1-Ce)_day1_06-post-60min__000.avi"
# output_path = r"X:\Behavior\Openfield_EEG\Pup-IRES-Parietal-1x\20240925_z178-4_temp\06-post-60min_006\20240925-203836-C16980_1_OFT_Pup-IRES-low_178-4(5w-M1-Ce)_day1_06-post-60min__000_fps.avi"
