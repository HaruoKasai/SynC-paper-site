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
sys.path.append("C:/Users/h_uki/Documents/GitHub/as/ImageAnalysis/Lib")
sys.path.append("C:/Users/h_uki/Documents/GitHub/as/ImageAnalysis/IALib")
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../Lib')
sys.path.append(str(current_dir) + '/../IALib')
from ImageJRoiReader import * #original


#入力：img, roi
#出力：dataframe(roiname, area, mean_ch1, mean_ch2)

root = tkinter.Tk()
root.withdraw()
# tkinter.messagebox.showinfo('File Chooser','Please choose date folder')
# dir = tkinter.filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data\sawada\CID_Analysis")
dir = tkinter.filedialog.askdirectory(initialdir=r"X:\SYNCit-C")
output_dir = os.path.join(dir, "csv_by_dend")

img_list = glob.glob(os.path.join(dir, "*.tif"))
for img_fname in img_list:
    basename_wo_ext = os.path.splitext(os.path.basename(img_fname))[0]
    print(basename_wo_ext)
    dend_list = glob.glob(os.path.join(dir, "ROI_by_dend", basename_wo_ext+"*_ROIspine.zip"))
    for dend in dend_list:
        spine_roi_fname = dend
        back_roi_fname = dend[:-13]+"_ROIback.zip"
        results = ImageJRoiReader(img_fname, spine_roi_fname)
        results_back = ImageJRoiReader(img_fname, back_roi_fname)
        results.to_csv(os.path.join(output_dir, basename_wo_ext + "_dend"+str(dend_list.index(dend)+1)+"_spine.csv"), index=False)
        results_back.to_csv(os.path.join(output_dir, basename_wo_ext + "_dend"+str(dend_list.index(dend)+1)+"_back.csv"), index=False)





