
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
from ImageJRoiReader import * #original package
from get_flat_area import * #original package


#黒田さんが作ってくれた回転stackに対してunmixingしてみる

#back値と画像全体の輝度でpixel値をnormalize
#入力：img (xyt画像(時系列がstackになっている))
#出力：img

"""
unmixing
R = S + a*V + BR
Y = b*S + V + BY

Y = b*(R-a*V-BR) + V + BY
Y-b*R+b*BR-BY = (1-b*a*)V
V = (Y-b*R+b*BR-BY) / ( (1-b*a)
S = R - a*V - BR
"""





# file = r"\\DESKTOP-WS2\data\sawada\raw\LSFM\20240524_142s_mStayGold_mScarlet\temp\3D-Projection_142s_B6J-Male-8w_mStayGold_mScarlet_1.60X.tif"
file = r"\\DESKTOP-WS2\data\sawada\raw\LSFM\20240524_143s_mStayGold_mScarlet\temp\3D-Projection_143s_B6J-Male-8w_mStayGold_mScarlet_1.52X.tif"
img = np.array(tiff.imread(file))
print(type(img))
print(img.shape)

unmixed = np.empty_like(img[:, 1])
print(unmixed.shape)
for i in range(img.shape[0]):
    print(i)
    iimg = img[i]
    R = iimg[0]
    G = iimg[1]
    FR = iimg[2]
    GFP= G-0.5*R #-1.4*FR #超適当
    unmixed[i]=GFP
print(unmixed)
unmixed = unmixed.astype(np.int16)
print(unmixed)
dir = os.path.dirname (file)

output_fname = os.path.join(dir, "_unmix.tif")
tiff.imwrite(output_fname, unmixed)
# tiff.imwrite(output_fname, imgs_unmixed, imagej=True, metadata={'axes': 'TCYX'})
########################################################






print("Finished")

