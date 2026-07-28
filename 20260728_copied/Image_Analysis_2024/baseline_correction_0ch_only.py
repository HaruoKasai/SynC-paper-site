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
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/Lib')
from ImageJRoiReader import * #original package
from get_flat_area import * #original package

#back値と画像全体の輝度でpixel値をnormalize
#入力：img (xyt画像(時系列がstackになっている))
#出力：img

def baseline_correction(imgs, ref_img_n=0):
    print(imgs.shape)
    if imgs.ndim == 3:
        imgs = np.expand_dims(imgs, axis=1)
    print(imgs.shape)
    timepoints_n = imgs.shape[0]
    imgs_normalized = np.empty_like(imgs)

    for c in range(imgs.shape[1]):
        if c == 0:
            ref_img = imgs[ref_img_n, c, :, :]
            print(ref_img.shape)
            ref_base = get_flat_area(ref_img)[0]
            ref_fore = ref_img.mean()

            for i in range(timepoints_n):
                img = imgs[i, c, :, :]
                baseline = get_flat_area(img)[0]
                print(baseline)
                img_normalized = (img[:, :] - baseline) * (ref_fore - ref_base) / (img.mean() - baseline) + ref_base
                imgs_normalized[i, c, :, :] = img_normalized
        else:
            imgs_normalized[:, c, :, :] = imgs[:, c, :, :]

    return imgs_normalized

###############crawl####################################
root = tkinter.Tk()
root.withdraw()
# tkinter.messagebox.showinfo('Directory Chooser','Please choose an directory with imgs')
img_dir = tkinter.filedialog.askdirectory(initialdir=r"\\Synology\arima\raw\2photon")
# tkinter.messagebox.showinfo('Directory Chooser','Please choose an output directory for the normalized image')
# output_dir = tkinter.filedialog.askdirectory()
output_dir = os.path.dirname(img_dir)

#img_files = glob.glob(os.path.join(img_dir, "*_al.tif"))
img_files = glob.glob(os.path.join(img_dir, "*_al_cr*.tif"))
# img_files = glob.glob(os.path.join(img_dir, "**.tif"))
print("number of files: %s" % len(img_files))

for file in img_files:
    print("filename: %s" % file)
    basename_wo_ext = os.path.splitext(os.path.basename(file))[0]
    imgs = tiff.imread(file)
    imgs_normalized = baseline_correction(imgs, 0)
    output_fname = os.path.join(output_dir, basename_wo_ext + "_normch0.tif")
    # tiff.imsave(output_fname, imgs_normalized )
    tiff.imwrite(output_fname, imgs_normalized, imagej=True, metadata={'axes': 'TCYX'})
########################################################

print("Finished")
