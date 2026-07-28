
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

coefficient_list = [
    [0, 0.08],    #0 s142s 脳がないところをBとして適当に係数だしたら、0.15だったが、それだとunmixedで黒抜けが目立ったので、適当な変数にした。黒抜けはおそらく、色収差で位置がずれていることが原因かも？これもちゃんと補正しなくては。
    [0, 0.2431]    #1
]

def unmix(G, R, param=0): #for 2 colors
    [a,b] = coefficient_list[param]
    # BaseR = get_flat_area(R)[0]
    # BaseG = get_flat_area(G)[0]
    BaseR = 123
    BaseG = 123 #脳がないところのシグナル値をひとまずつかったが、いいのか？
    GFP = (G-b*R+b*BaseR-BaseG) /(1-b*a)
    return GFP




###############crawl####################################
# root = tkinter.Tk()
# root.withdraw()
# img_dir = tkinter.filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data\sawada\CID_Analysis")

# imgG_dir = r"\\DESKTOP-WS2\data\sawada\raw\LSFM\20240524_142s_mStayGold_mScarlet\1.60X_488nm_100mW_20ms_Z6.5um_G515-30"
# imgR_dir = r"\\DESKTOP-WS2\data\sawada\raw\LSFM\20240524_142s_mStayGold_mScarlet\1.60X_532nm_25mW_10ms_Z6.5um_Y593-40"
# output_dir = r"\\DESKTOP-WS2\data\sawada\raw\LSFM\20240524_142s_mStayGold_mScarlet\unmixed_G"

imgG_dir = r"\\DESKTOP-WS2\data\sawada\raw\LSFM\20240524_143s_mStayGold_mScarlet\1.60X_488nm_100mW_20ms_Z6.5um_G515-30"
imgR_dir = r"\\DESKTOP-WS2\data\sawada\raw\LSFM\20240524_143s_mStayGold_mScarlet\1.60X_532nm_25mW_10ms_Z6.5um_Y593-40"
output_dir = r"\\DESKTOP-WS2\data\sawada\raw\LSFM\20240524_143s_mStayGold_mScarlet\unmixed_G"


imgG_files = glob.glob(os.path.join(imgG_dir, "*.tif"))
imgR_files = glob.glob(os.path.join(imgR_dir, "*.tif"))

for i in range(len(imgG_files)):
    print(i)
    G =tiff.imread(imgG_files[i])
    R = tiff.imread(imgR_files[i])
    unmixed = unmix (G,R)
    unmixed = unmix (G,R).astype(np.int16)
    output_fname = os.path.join(output_dir, os.path.basename(imgG_files[i])[:-4]+"_unmix.tif")
    tiff.imwrite(output_fname, unmixed )
    # tiff.imwrite(output_fname, imgs_unmixed, imagej=True, metadata={'axes': 'TCYX'})
########################################################






print("Finished")

