from suite2p.gui.select_folder import gui as run_gui

if __name__ == '__main__':
    run_gui()
# import bioformats



# import javabridge
# import tifffile
# import numpy as np
# import os
# import tkinter as tk
# from tkinter import filedialog
# import glob



# def select_folder():
#     root = tk.Tk()
#     root.withdraw()
#     folder_path = filedialog.askdirectory(title="Select the 'data' directory", initialdir=r"X:\Behavior\Ca_imaging")
#     root.destroy()
#     return folder_path
#
# def main():
#     data_folder = select_folder()
#     # data_folder = r"X:\Behavior\Ca_imaging\20250707_z251-2_SynC-GCaMP"
#     db = {
#         'data_path': os.path.join(data_folder, "_GCaAMP"),
#         'save_path0': os.path.join(data_folder, "_GCaAMP"),
#         'subfolders': [''],
#         'fast_disk': r"C:/Users/you/AppData/Local/Temp/"
#     }
#     ops = np.load(r"X:\Behavior\Ca_imaging\ops_terada_2025-07-08.npy", allow_pickle=True).item()
#     ops['fast_disk'] = r'C:/Users/you/AppData/Local/Temp/'
#     ops['save_path0'] = '/new/output/path'
#
#
# if __name__ == "__main__":
#     main()