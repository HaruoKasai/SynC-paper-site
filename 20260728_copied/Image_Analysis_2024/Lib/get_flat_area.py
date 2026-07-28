import pandas as pd
import numpy as np
import os
import tifffile as tiff
from scipy import stats
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from read_roi import read_roi_zip, read_roi_file
import glob
import tkinter.filedialog
import tkinter.messagebox
from skimage import io, color
import re

def get_flat_area(img):
    mean_val = img.mean()
    std_val = img.std()
    (x_size, y_size) = img.shape
    rect_x = round(x_size / 10)
    rect_y = round(y_size / 10)
    step = round(x_size / 100)
    x_block = 0
    y_block = 0

    for i in range(0, x_size - rect_x, step):
        for j in range(0, y_size - rect_y, step):
            s = img[i:i + rect_x, j:j + rect_y].std()
            if s < std_val:
                std_val = s
                mean_val = img[i:i + rect_x, j:j + rect_y].mean()
                x_block = i
                y_block = j

    return mean_val, std_val, x_block, y_block


# def get_flat_area(img, ch):
#     mean_val = img[ch].mean()
#     std_val = img[ch].std()
#     (x_size, y_size) = img[ch].shape
#     rect_x = round(x_size / 10)
#     rect_y = round(y_size / 10)
#     step = round(x_size / 100)
#     x_block = 0
#     y_block = 0
#
#     for i in range(0, x_size - rect_x, step):
#         for j in range(0, y_size - rect_y, step):
#             s = img[ch, i:i + rect_x, j:j + rect_y].std()
#             if s < std_val:
#                 std_val = s
#                 mean_val = img[ch, i:i + rect_x, j:j + rect_y].mean()
#                 x_block = i
#                 y_block = j
#
#     return mean_val, std_val, x_block, y_block
