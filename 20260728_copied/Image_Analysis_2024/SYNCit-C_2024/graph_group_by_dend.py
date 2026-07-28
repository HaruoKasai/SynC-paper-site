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
import sys
import pathlib
import re

color_list=["deeppink","forestgreen", "dimgray", "orangered"]

#下記二つのフォルダでだけtemporaryにtp_listをここで指定した(2024.02.13)
# tp_list = [-65, -45, -25, -5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400] #cLTP-25min-Rapa_entire_time_graph
# tp_list = [-65, -45, -25, -5, 15, 25, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260] #Rapa-30min-cLTP_entire_time_graph


def delta_data_collection(dir, channel_list=["c0", "c1", "c2"],  tp=2): #c2: mVenus/mScarlet ratio
    cond_list = glob.glob(os.path.join(dir, "[!_]*"))
    print("number of conditions: %s" % len(cond_list))
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3, 3* 3))
    for cond in cond_list:
        print("cond!!!!   "+cond)
        date_list = glob.glob(os.path.join(cond, "[!_]*"))
        for c in range(len(channel_list)):
            df = pd.DataFrame(index=[])  # , columns=columns_list
            for date in date_list:
                day = os.path.basename(date)[-4:]
                csv_list=glob.glob(os.path.join(date,"timeseries_by_dend", "*"+channel_list[c]+"*delta(%).csv")) #TODO timeseries_graph.pyで作成されるｃｓｖ名をcolumns_listと対応させる必要あり
                print("number of dendrites: %s" % len(csv_list))
                for csv in csv_list:
                    #dend番号
                    pattern =r'dend(\d+)'
                    match = re.search(pattern, csv)
                    dend_index = match.group(1)
                    # print(dend_index)

                    #cell番号 #"series", "cell"どちらかで番号が振られている
                    pattern_se = r'series(\d+)'
                    pattern_ce = r'cell(\d+)'
                    pattern_ro = r'ROI(\d+)'
                    match_se = re.search(pattern_se, csv)
                    match_ce = re.search(pattern_ce, csv)
                    match_ro = re.search(pattern_ro, csv)
                    if match_se:
                        cell_ind = match_se.group(1)
                    elif match_ce:
                        cell_ind =  match_ce.group(1)
                    elif match_ro:
                        cell_ind = match_ro.group(1)
                    else:
                        print("cell番号抽出の問題")

                    tp = pd.read_csv(csv).loc[:,["min"]]
                    df_mean = pd.read_csv(csv).loc[:,["average"]]
                    df_mean.rename(columns={'average': day+"_"+cell_ind+"_"+dend_index}, inplace=True)
                    # print(df_mean)
                    # df_ind=pd.read_csv(csv).iloc[:, 1:-3]
                    df = pd.concat([df,df_mean], axis=1)

            average = df.mean(axis = "columns").values.reshape(-1,1)
            sem = df.sem(axis="columns").values.reshape(-1, 1)
            df = pd.concat([df, tp], axis=1)
            df["average"] = average
            df["sem"] = sem

            # df = df.head(len(tp_list))
            # df["min"] = tp_list
            # print("######df############")

            print(df)
            df.to_csv(os.path.join(dir, "_summary_by_dend", os.path.basename(cond)+"_"+channel_list[c]+"_"+"delta(%).csv"))
            # ax = df.plot(x="min", y="average", ax = axes[c], legend=False, linewidth=1.5, color=color_list[cond_list.index(cond)])
            ax = df.plot(x="min", y="average", yerr="sem", elinewidth=0.6 ,ax=axes[c], legend=False, linewidth=1.2,color=color_list[cond_list.index(cond)])
            ax.axvspan(0, 10, color="skyblue", alpha=0.3, linewidth=0)
            ax.axvspan(57, 72, color="gold", alpha=0.3, linewidth=0)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.axhline(y=0, color="gray", ls="dotted", lw=1)
            x = df["min"].values
            # y1 = (average - sem).reshape(average.shape[0])
            # y2 = (average + sem).reshape(average.shape[0])
            # ax.fill_between(x, y1, y2, color=color_list[cond_list.index(cond)], alpha=0.3, linewidth=0)
            ax.set_xlim(-60, 275)
            ax.set_xticks(np.arange(-50, 270, 50))
            ax.set_ylim(-55, 160)
            ax.set_yticks(np.arange(-50, 155, 50))

    fig.tight_layout()
    fig.savefig(os.path.join(dir, "_summary_by_dend", "timeseries.pdf"), dpi=300, transparent=True)

    return df

# def bargraph(df):


root = tkinter.Tk()
root.withdraw()
# dir = tkinter.filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data\sawada\CID_Analysis")
dir = tkinter.filedialog.askdirectory(initialdir=r"X:\SYNCit-C")
df=delta_data_collection(dir)
# bargraph(df)


