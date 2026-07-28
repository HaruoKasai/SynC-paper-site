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
import random

#入力：img, roi
#出力：dataframe(roiname, area, mean_ch1, mean_ch2)
def ImageJRoiReader(img_fname, roi_fname):
    img = tiff.imread(img_fname)
    img = img.astype(np.float32)#np.nanを使うためにdataをfloat32で扱う
    # print("img shape : %s" % (img.shape,))
    #ImageJのROI fileはROIの数により拡張子が異なる(ROIが一つの時:.roi, ROIが複数の時：.zip)
    if os.path.splitext(roi_fname)[1] == ".zip":
        dict_roi = read_roi_zip(roi_fname)
    elif os.path.splitext(roi_fname)[1] == ".roi":
        dict_roi = read_roi_file(roi_fname)

    # print("img shape : %s" % (img.shape,))
    # print(len(dict_roi))

    # results = pd.DataFrame(columns=["name", "area_in_pixel", "mean_ch1", "mean_ch2", "significance_ch1"])
    results = pd.DataFrame(columns=["name", "area_in_pixel"])
    slice_n = img.shape[0]
    if img.ndim ==2:
        results["mean_0"] = 0
        for roi_id in dict_roi:
            if dict_roi[roi_id]['type'] == 'oval':
                roi_name, roi_area, roi_mean = measure_roi_oval(img, dict_roi[roi_id])
            if dict_roi[roi_id]['type'] == 'rectangle':
                roi_name, roi_area, roi_mean = measure_roi_rectangle(img, dict_roi[roi_id])
            result = pd.DataFrame({'name': [roi_name], 'area_in_pixel': roi_area})
            result["mean_0"] = roi_mean
            results = pd.concat([results, result],sort=False)

    if img.ndim ==3:
        for i in range(slice_n):
            results["mean%s" % (i)] = 0

        for roi_id in dict_roi:
            if dict_roi[roi_id]['type'] == 'oval':
                roi_name, roi_area, roi_mean = measure_roi_oval(img, dict_roi[roi_id])
            if dict_roi[roi_id]['type'] == 'rectangle':
                roi_name, roi_area, roi_mean = measure_roi_rectangle(img, dict_roi[roi_id])
            result = pd.DataFrame({'name': [roi_name], 'area_in_pixel': roi_area})
            for i in range(slice_n):
                print(i)
                print(roi_mean)
                result["mean%s" % (i)] = roi_mean[i]
            results = pd.concat([results, result],sort=False)

    if img.ndim == 4:
        print("img.ndim==4")
        for c in range(img.shape[1]):
            for i in range(slice_n):
                col_name = "mean"+str(i)+"_c"+str(c)
                results[col_name] = 0

        for roi_id in dict_roi:
            if dict_roi[roi_id]['type'] == 'oval':
                roi_name, roi_area, roi_mean = measure_roi_oval(img, dict_roi[roi_id])
            result = pd.DataFrame({'name': [roi_name], 'area_in_pixel': roi_area})
            for c in range(img.shape[1]):
                for i in range(slice_n):
                    col_name = "mean" + str(i) + "_c" + str(c)
                    result[col_name] = roi_mean[i][c]
            results = pd.concat([results, result],sort=False)

    return results


def measure_roi_oval(img, dict_roi):
    # imgと単一のROIのdictをもらう
    # 定量結果[roiname, Area_in_pixel, 各Chのmean(array)]
    # ImageJは楕円の中にそのpixelの中心が入ってたら定量に含めるアルゴリズムになってる
    x1 = dict_roi['left']
    y1 = dict_roi["top"]
    h1 = dict_roi["height"]
    w1 = dict_roi["width"]
    cx = w1 / 2
    cy = h1 / 2


    if img.ndim == 3:
        sy = img.shape[1]
        sx = img.shape[2]
        y, x = np.ogrid[0:sy, 0:sx]  # x and y indices of pixels

        mask = ((x + 0.5 - cx - x1) ** 2 / (w1 / 2) ** 2 + (y + 0.5 - cy - y1) ** 2 / (h1 / 2) ** 2) > 1

        roi_img = img.copy()

        roi_img[:, mask] = np.nan
        roi_name = dict_roi["name"]
        roi_area = np.count_nonzero(~np.isnan(roi_img[1]))
        roi_mean = np.nansum(roi_img, axis=(1, 2)) / roi_area

    elif img.ndim == 4:
        sy = img.shape[2]
        sx = img.shape[3]
        y, x = np.ogrid[0:sy, 0:sx]  # x and y indices of pixels
        mask = ((x + 0.5 - cx - x1) ** 2 / (w1 / 2) ** 2 + (y + 0.5 - cy - y1) ** 2 / (h1 / 2) ** 2) > 1

        roi_img = img.copy()

        roi_img[:,:, mask] = np.nan
        roi_name = dict_roi["name"]
        roi_area = np.count_nonzero(~np.isnan(roi_img[0][0]))
        roi_mean = np.nansum(roi_img, axis=(2, 3)) / roi_area

    elif img.ndim == 2:
        sy = img.shape[0]
        sx = img.shape[1]
        y, x = np.ogrid[0:sy, 0:sx]  # x and y indices of pixels
        mask = ((x + 0.5 - cx - x1) ** 2 / (w1 / 2) ** 2 + (y + 0.5 - cy - y1) ** 2 / (h1 / 2) ** 2) > 1

        roi_img = img.copy()
        roi_img[mask] = np.nan

        roi_name = dict_roi["name"]
        roi_area = np.count_nonzero(~np.isnan(roi_img))
        roi_mean = np.nansum(roi_img) / roi_area
        # roi_mean = roi_img.sum(axis=(1, 2)) / roi_area

    else:
        print("error in imageJRoiReader, the dimension of the image")

    return roi_name, roi_area, roi_mean

def measure_roi_rectangle(img, dict_roi):
    # imgと単一のROIのdictをもらう
    # 定量結果[roiname, Area_in_pixel, 各Chのmean(array)]
    x1 = dict_roi['left']
    y1 = dict_roi["top"]
    h1 = dict_roi["height"]
    w1 = dict_roi["width"]

    roi_img = img[:, y1:y1 + h1, x1:x1 + w1]
    roi_name = dict_roi["name"]
    roi_area = h1 * w1
    roi_mean = roi_img.sum(axis=(1, 2)) / roi_area

    return roi_name, roi_area, roi_mean


def measure_random_roi_oval(img, dict_roi):
    # imgと単一のROIのdictをもらう
    # 楕円の中心をランダムにする（ただしROIが画像からはみ出ない範囲で)
    # 定量結果[roiname, Area_in_pixel, 各Chのmean(array)]
    # ImageJは楕円の中にそのpixelの中心が入ってたら定量に含めるアルゴリズムになってる
    sy = img.shape[1]
    sx = img.shape[2]
    h1 = dict_roi["height"]
    w1 = dict_roi["width"]

    x1 = random.randint(w1, sx-w1)
    y1 = random.randint(h1, sy-h1)
    cx = w1 / 2
    cy = h1 / 2

    y, x = np.ogrid[0:sy, 0:sx]  # x and y indices of pixels
    mask = ((x + 0.5 - cx - x1) ** 2 / (w1 / 2) ** 2 + (y + 0.5 - cy - y1) ** 2 / (h1 / 2) ** 2) > 1
    roi_img = img.copy()
    roi_img[:, mask] = np.nan

    roi_area = np.count_nonzero(~np.isnan(roi_img[1]))
    roi_mean = np.nansum(roi_img, axis=(1, 2)) / roi_area

    return roi_mean



