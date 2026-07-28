import pandas as pd
import numpy as np
import os
import tifffile as tiff
from scipy import stats
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams["font.family"]= "Arial"
import matplotlib.pyplot as plt
from read_roi import read_roi_zip, read_roi_file
import glob
import tkinter.filedialog
import tkinter.messagebox
import re
import sys
import pathlib

color_list=["dodgerblue", "orangered","forestgreen", "dimgray", "purple", "gold", "lime", "pink"]

def delta_data_collection(dir, color_num=1, channel_list=["c0", "c1", "c2"],  tp=2): #c2: mVenus/mScarlet ratio
    cond_list = glob.glob(os.path.join(dir, "[!_]*"))
    print("number of conditions: %s" % len(cond_list))
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3, 3* 3))
    for cond in cond_list:
        date_list = glob.glob(os.path.join(cond, "[!_]*"))

        for c in range(color_num):
            df = pd.DataFrame(index=[])
            for date in date_list:
                csv_list=glob.glob(os.path.join(date,"graph", "*"+channel_list[c]+"*delta(%).csv")) #TODO timeseries_graph.pyで作成されるｃｓｖ名をcolumns_listと対応させる必要あり
                print("number of cells: %s" % len(csv_list))
                for csv in csv_list:
                    tp = pd.read_csv(csv).loc[:,["min"]]
                    df_ind=pd.read_csv(csv).iloc[:, 1:-3]
                    print("#############df ind")
                    print(df_ind)
                    df = pd.concat([df,df_ind], axis=1)
            average = df.mean(axis = "columns").values.reshape(-1,1)
            sem = df.sem(axis="columns").values.reshape(-1, 1)
            df = pd.concat([df, tp], axis=1)
            df["average"] = average
            df["sem"] = sem
            df.to_csv(os.path.join(dir, "_summary", os.path.basename(cond)+"_"+channel_list[c]+"_"+"delta(%).csv"))
            # ax = df.plot(x="min", y="average", ax = axes[c], legend=False, linewidth=1.5, color=color_list[cond_list.index(cond)])
            ax = df.plot(x="min", y="average", yerr="sem", elinewidth=0.6 ,ax=axes[c], legend=False, linewidth=1.2,color=color_list[cond_list.index(cond)])
            ax.axvspan(0, 2, color="lightblue", alpha=0.3, linewidth=0)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.axhline(y=0, color="gray", ls="dotted", lw=1)
            x = df["min"].values
            # y1 = (average - sem).reshape(average.shape[0])
            # y2 = (average + sem).reshape(average.shape[0])
            # ax.fill_between(x, y1, y2, color=color_list[cond_list.index(cond)], alpha=0.3, linewidth=0)
            ax.set_xlim(-80, 210)
            ax.set_xticks(np.arange(-80, 210, 80))
            ax.set_ylim(-50, 50)
            ax.set_yticks(np.arange(-50, 50, 25))





    fig.tight_layout()
    fig.savefig(os.path.join(dir, "_summary", "timeseries.pdf"), dpi=300, transparent=True)

    return df

# def bargraph(df):


root = tkinter.Tk()
root.withdraw()
dir = tkinter.filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data")
df=delta_data_collection(dir)
# bargraph(df)


