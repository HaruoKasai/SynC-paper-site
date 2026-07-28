from tkinter import filedialog, messagebox
import glob
import os
import deeplabcut

messagebox.showinfo("Choose mouse folder containing raw videos to analyze")
mouse_dir = filedialog.askdirectory(initialdir=r"X:\Behavior\EEG\Turntable")


messagebox.showinfo("Choose Model folder")
config_dir = filedialog.askdirectory(initialdir=r"Z:\DeepLabCutModels")
config_path = os.path.join(config_dir, "config.yaml")

# output_dir =
exp_list = glob.glob(os.path.join(mouse_dir, "[!_]*"))

for exp in exp_list:
    input_path = glob.glob(os.path.join(exp,"raw_video", "*.avi"))[0]
    deeplabcut.analyze_videos(config_path, input_path, save_as_csv=True, batchsize=128)
    deeplabcut.filterpredictions(config_path, [input_path])
    deeplabcut.plot_trajectories(config_path, [input_path])
    deeplabcut.create_labeled_video(config_path, [input_path]) #, draw_skeleton = True
        #If you want to create high-quality videos, please add save_frames=True