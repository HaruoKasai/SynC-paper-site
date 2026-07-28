import pandas as pd
import numpy as np
import os
import tifffile as tiff
from scipy import stats
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from read_roi import read_roi_zip, read_roi_file
import glob
import tkinter.filedialog
import tkinter.messagebox
import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/Lib')
from ImageJRoiReader import * #original


#入力：img, roi
#出力：dataframe(roiname, area, mean_ch1, mean_ch2)

root = tkinter.Tk()
root.withdraw()
tkinter.messagebox.showinfo('File Chooser','Please choose date folder')
dir = tkinter.filedialog.askdirectory(initialdir=r"\\Synology")
# tkinter.messagebox.showinfo('File Chooser','Please choose an spine roi file')
# spine_roi_fname = tkinter.filedialog.askopenfilename()
# tkinter.messagebox.showinfo('File Chooser','Please choose an spine_back roi file')
# spine_back_roi_fname = tkinter.filedialog.askopenfilename()
# tkinter.messagebox.showinfo('Directory Chooser','Please choose an csv output directory')
# output_dir = tkinter.filedialog.askdirectory()
output_dir = os.path.join(dir, "csv")

img_list = glob.glob(os.path.join(dir, "*.tif"))
for img_fname in img_list:
    basename_wo_ext = os.path.splitext(os.path.basename(img_fname))[0]
    print(basename_wo_ext)
    spine_roi_fname = img_fname[:-4]+"_ROIspine.zip"
    spine_back_roi_fname = img_fname[:-4] + "_ROIback.zip"
    spine_den_roi_fname = img_fname[:-4] + "_ROIden.zip"
    results = ImageJRoiReader(img_fname, spine_roi_fname)
    results_back = ImageJRoiReader(img_fname, spine_back_roi_fname)
    results.to_csv(os.path.join(output_dir, basename_wo_ext + "_spine.csv"), index=False)
    results_back.to_csv(os.path.join(output_dir, basename_wo_ext + "_back.csv"), index=False)
    if os.path.exists(spine_den_roi_fname):
        results_den = ImageJRoiReader(img_fname, spine_den_roi_fname)
        results_den.to_csv(os.path.join(output_dir, basename_wo_ext + "_den.csv"), index=False)
    else:
        print(f"\033[91mNo_den_roi_file\033[0m")


