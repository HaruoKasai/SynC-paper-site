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
import csv
sys.path.append("C:/Users/h_uki/Documents/GitHub/as/ImageAnalysis/Lib")
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../Lib')
sys.path.append(str(current_dir) + '/../IALib')
from ImageJRoiReader import * #original package
from get_flat_area import * #original package


def unmixing_coef_calc(imgs, img_dir):
    print(imgs.shape)
    FP = os.path.basename(img_dir)
    Y = imgs[0, :, :]
    R = imgs[1,:,:]
    BaseR = get_flat_area(R)[0]
    BaseY = get_flat_area(Y)[0]
    if FP=="mScarlet":
        coef = (Y.mean()-BaseY) / (R.mean()-BaseR)
    else:
        coef = (R.mean() - BaseR) / (Y.mean() - BaseY)
    return coef


###############crawl####################################
root = tkinter.Tk()
root.withdraw()
img_dir = tkinter.filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data\sawada\CID_Analysis\unmixing_coef_calc")
output_dir = img_dir
img_files = glob.glob(os.path.join(img_dir, "*.tif*"))
print("number of files: %s" % len(img_files))
coef_list = []
for file in img_files:
    print("filename: %s" % file)
    # basename_wo_ext = os.path.splitext(os.path.basename(file))[0]
    imgs = tiff.imread(file)
    coef_list.append(os.path.basename(file))
    coef_list.append(unmixing_coef_calc(imgs, img_dir))

output_fname = os.path.join(output_dir, "coef.csv")
with open(output_fname, 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(coef_list)
########################################################






print("Finished")

