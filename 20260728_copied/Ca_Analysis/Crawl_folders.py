import os
import tkinter as tk
from tkinter import filedialog
import glob
import F_Correlation
import EEG_Ca_treadmill_analysis

def main():
    root = tk.Tk()
    root.withdraw()
    parent_folder_path = filedialog.askdirectory(
        title="Select the parent directory",
        # initialdir=r"X:\Behavior"
        initialdir=r"X:\Behavior\Ca_imaging"
    )
    mouse_list = glob.glob(os.path.join(parent_folder_path, "[!_]*"))
    for mouse in mouse_list:
        # EEG_Ca_treadmill_analysis.process_folder(mouse)
        if os.path.exists(os.path.join(mouse, "_GCaMP", "_spks_cell.npy")):
            F_Correlation.process(mouse)

if __name__ == "__main__":
    main()


