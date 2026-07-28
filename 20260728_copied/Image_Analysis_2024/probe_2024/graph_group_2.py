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

color_list=["dodgerblue", "orangered","forestgreen", "dimgray"]

#def delta_data_collection(dir, channel_list=["c0", "c1", "c2"], sp_cond_list= ["stim", "neighbor"]): #c2: mVenus/mScarlet ratio
def delta_data_collection(dir, channel_list=["c0", "c1", "c2"],sp_cond_list=["stim"]):  # c2: mVenus/mScarlet ratio

    cond_list = glob.glob(os.path.join(dir, "[!_]*"))
    print("number of conditions: %s" % len(cond_list))
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3, 3* 3))
    for cond in cond_list:
        print("cond!!!!   "+cond)
        for sp_cond in sp_cond_list:
            date_list = glob.glob(os.path.join(cond, "[!_]*"))
            for i in range (len(sp_cond)):
                for c in range(len(channel_list)):
                    df = pd.DataFrame(index=[])  # , columns=columns_list
                    for date in date_list:
                        day = os.path.basename(date)[2:6]
                        csv_list=glob.glob(os.path.join(date,"timeseries_ind", sp_cond, "*"+channel_list[c]+"*delta(%).csv")) #TODO timeseries_graph.pyで作成されるｃｓｖ名をcolumns_listと対応させる必要あり
                        print("number of dendrites: %s" % len(csv_list))
                        for csv in csv_list:
                            # #dend番号
                            # pattern =r'dend(\d+)'
                            # match = re.search(pattern, csv)
                            # dend_index = match.group(1)
                            # # print(dend_index)
                            #
                            # #cell番号 #"series", "cell"どちらかで番号が振られている
                            # pattern_se = r'series(\d+)'
                            # pattern_ce = r'cell(\d+)'
                            # pattern_ro = r'ROI(\d+)'
                            # match_se = re.search(pattern_se, csv)
                            # match_ce = re.search(pattern_ce, csv)
                            # match_ro = re.search(pattern_ro, csv)
                            # if match_se:
                            #     cell_ind = match_se.group(1)
                            # elif match_ce:
                            #     cell_ind =  match_ce.group(1)
                            # elif match_ro:
                            #     cell_ind = match_ro.group(1)
                            # else:
                            #     print("cell番号抽出の問題")

                            tp = pd.read_csv(csv).loc[:,["min"]]


                            df_ind = pd.read_csv(csv).loc[:,["average"]] #TODO 一般性をもたせる
                            df_ind.rename(columns={'average': day}, inplace=True)

                            df = pd.concat([df,df_ind], axis=1)

                    average = df.mean(axis = "columns").values.reshape(-1,1)
                    sem = df.sem(axis="columns").values.reshape(-1, 1)
                    df = pd.concat([df, tp], axis=1)
                    df["average"] = average
                    df["sem"] = sem

                    # df = df.head(len(tp_list))
                    # df["min"] = tp_list
                    # print("######df############")

                    print(df)
                    df.to_csv(os.path.join(dir, "_summary", os.path.basename(cond)+"_"+channel_list[c]+"_"+"delta(%).csv"))
                    # ax = df.plot(x="min", y="average", ax = axes[c], legend=False, linewidth=1.5, color=color_list[cond_list.index(cond)])
                    ax = df.plot(x="min", y="average", yerr="sem", elinewidth=0.6 ,ax=axes[c], legend=False, linewidth=1.2,color=color_list[cond_list.index(cond)])
                    ax.axvspan(0, 10, color="blue", alpha=0.3, linewidth=0)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.axhline(y=0, color="gray", ls="dotted", lw=1)
                    x = df["min"].values
                    # y1 = (average - sem).reshape(average.shape[0])
                    # y2 = (average + sem).reshape(average.shape[0])
                    # ax.fill_between(x, y1, y2, color=color_list[cond_list.index(cond)], alpha=0.3, linewidth=0)
                    ax.set_xlim(-60, 360)
                    ax.set_xticks(np.arange(-60, 360, 60))
                    if channel_list[c] == 'c0':
                        ax.set_ylabel('ΔV (%)')
                    elif channel_list[c] == 'c1':
                        ax.set_ylabel('ΔiAS (%)')
                    elif channel_list[c] == 'c2':
                        ax.set_ylabel('ΔiAS (%)/ΔV (%)')
                    #ax.set_ylim(-40, 200)
                    #ax.set_yticks(np.arange(-40, 200, 40))

    fig.tight_layout()
    fig.savefig(os.path.join(dir, "_summary", "_timeseries_group.pdf"), dpi=300, transparent=True)

    return df

# def bargraph(df):


root = tkinter.Tk()
root.withdraw()
dir = tkinter.filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate")
df=delta_data_collection(dir)
# bargraph(df)


