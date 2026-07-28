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
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams["font.family"]= "Arial"


root = tkinter.Tk()
root.withdraw()
dir_name = tkinter.filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data\arima\conditions\AAVAS\A101")

# tp_before = [0,2] #このプログラム内では計算に使わない
tp_after = [3,5]
axis_list=[["c2_AU","c0_delta(%)"]
    ,["c1_AU","c0_delta(%)"]
           ] #[y,x]

file_list = glob.glob(os.path.join(dir_name, "*.csv"))
img_list = list(set(["_".join(string.split("_")[:-2]) for string in file_list]))

for img in img_list:
    fig, axes = plt.subplots(nrows=len(axis_list), ncols=1, figsize=(3, 3 * len(axis_list)))
    for axis in axis_list:
        df_y = pd.read_csv(img+"_"+axis[0]+".csv")
        df_x = pd.read_csv(img+"_"+axis[1]+".csv")
        y = df_y.iloc[tp_after[0]:tp_after[1]].mean(axis=0)
        x = df_x.iloc[tp_after[0]:tp_after[1]].mean(axis=0)
        # y = y.apply(lambda a: 10 if a > 10 else a) #大きすぎる値はtemporaryに丸め込んで表示
        x = x.apply(lambda a: 200 if a > 200 else a) #大きすぎる値はtemporaryに丸め込んで表示
        df_plot = pd.concat([x,y], axis=1)
        df_plot.columns = ["x", "y"]
        print(df_plot)
        for i, row in df_plot.iterrows():
            axes[axis_list.index(axis)].scatter(row["x"], row["y"], label=i, s=5)  # 2行以上になる場合は、axes[axis_list.index(axis)]
        # axes.scatter(df_plot["x"], df_plot["y"], s=5) #2行以上になる場合は、axes[axis_list.index(axis)]
        axes[axis_list.index(axis)].set_xlabel(axis[1])
        axes[axis_list.index(axis)].set_ylabel(axis[0])
        axes[axis_list.index(axis)].set_ylim(0,)
        axes[axis_list.index(axis)].legend(fontsize=2, loc="upper right", ncol=3)

    fig.tight_layout()
    fig.savefig(img+"_scatter.pdf", dpi=300, transparent=True)





