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
from ImageJRoiReader import *  # original package
from get_flat_area import *  # original package

coefficient_list = [
    [0, 0.1576 * 2 / 3],  # 0 2023.01.10
    [0, 0.2431],  # 1 2023.02.13
    [0, 0.1626],  # 2 2023.01.17 cLTP
    [0, 0.0541],  # 3 2023.02.20 GEF-dead
    [0, 0.0264],  # 4 2023.02.21 PSD95 UTR
    [0, 0.0541 * 3 / 2],  # 5 2023.02.27 woRapa
    [0, 0.0264 * 3 / 2 * 2],  # 6 2023.03.06 Rapa->cLTP
    [0, 0.0264 / 2],  # 7 2023.03.07 PSD95-UTR , 2023.03.13PSD95-UTR_Vehicle
    [0, 0.0541 * 3 * 2 / 3],  # 8 2023.03.07 w_wo_APV,   20230313_cL100_APV
    [0, 0.0541 * 2 / 3],  # 9 2023.03.14 cLTP
    [0, 0.0541 * 3],  # 10 2023.04.10 cLTD control
    [0, 0.05],  # 11 temp
    [0.00081, 0],  # 12 230514 c-fos imaging mVenus
    [0.00291, 0],  # 13 230514 c-fos imaging 633
    [0.000154, 0],  # 14 240429 ~ paCaMKII imaging_zoom5
    [0.00044, 0],  # 15 240429~ soma_spine_zoom0.75
    [0.00047, 0],  # 16 240624~_A144_conc_zoom0.75_514nm_1.0%
    [0.0023, 0],  # 17 zoom3 514nm 1.0%
    [0.00116, 0],  # 18 zoom3 514nm 0.5%
    [0.0011, 0],  # 19 230514 c-fos imaging mVenus_strong
    [0.000696, 0],  #20 zoom3 514nm 0.5%, 561 nm laser 3.0%
    [0.000232, 0],   #21 zoom3 514nm 0.5%, 561 nm laser 1.0%
    [0.00084, 0],   #22 zoom3 lipofection target protein 561nm 5.0%, 514nm 0.5%
    [0.00129, 0],   #23 zoom3 lipofection target protein 561 nm 3.0%, 514 nm 0.5%
    [0.001756, 0],  #24 zoom3 lipofection target protein 561 nm 2.0%, 514 nm 0.5%
    [0.00318, 0],   #25 zoom3 lipofection target protein 561 nm 1.0%, 514 nm 0.5%
    [0.000104, 0],   #26 zoom3 lipofection target protein 561nm 5.0%, 514nm 0.3%
    [0.069487, 0]   #27 2P imaging zoom8 A163 2.0E13, A140 6.7E8, HV500, HV520
]


def unmixing(imgs, param=20):  # for 2 colors
    print("Input image shape:", imgs.shape)
    print("Input image dtype:", imgs.dtype)

    # Convert to float if not already
    if not np.issubdtype(imgs.dtype, np.floating):
        print("Converting image to float32")
        imgs = imgs.astype(np.float32)

    timepoints_n = imgs.shape[0]
    imgs_unmixed = np.empty_like(imgs)
    [a, b] = coefficient_list[param]
    for i in range(timepoints_n):
        Y = imgs[i, 0, :, :]
        R = imgs[i, 1, :, :]
        BaseR = get_flat_area(R)[0]
        BaseY = get_flat_area(Y)[0]
        V = (Y - b * R + b * BaseR - BaseY) / (1 - b * a)
        imgs_unmixed[i, 0, :, :] = V
        imgs_unmixed[i, 1, :, :] = R - a * V - BaseR
    return imgs_unmixed


###############crawl####################################
root = tkinter.Tk()
root.withdraw()
img_dir = tkinter.filedialog.askdirectory(initialdir=r"\\Synology\arima\raw\2photon")
output_dir = img_dir
img_files = glob.glob(os.path.join(img_dir, "*.tif"))
print("Number of files: %s" % len(img_files))

for file in img_files:
    print("Filename: %s" % file)
    basename_wo_ext = os.path.splitext(os.path.basename(file))[0]
    imgs = tiff.imread(file)

    # Check and display image dtype before processing
    print(f"Data type of input image {file}: {imgs.dtype}")

    # Apply unmixing after converting to float if necessary
    imgs_unmixed = unmixing(imgs)

    output_fname = os.path.join(output_dir, basename_wo_ext + "_unmix.tif")
    tiff.imwrite(output_fname, imgs_unmixed, imagej=True, metadata={'axes': 'TCYX'})

########################################################

print("Finished")
