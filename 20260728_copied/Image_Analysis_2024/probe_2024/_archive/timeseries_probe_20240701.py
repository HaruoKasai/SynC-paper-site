import pandas as pd
import numpy as np
import os
import tifffile as tiff
from scipy import stats
from sklearn import linear_model
import seaborn as sns
# sns.set()
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
from matplotlib.colors import Normalize
import random


def Individual_timeseries(spine_csv, back_csv, tp_base=[0, 1], color_list=["orangered", "forestgreen", "forestgreen"]):
    print(spine_csv)
    dir = os.path.dirname(os.path.dirname(spine_csv))
    fname = os.path.basename(spine_csv)
    file_name  = os.path.join(dir, "graph", fname[:-9])
    spine_df = pd.read_csv(spine_csv)
    back_df = pd.read_csv(back_csv)
    area = spine_df['area_in_pixel'].values.reshape(-1,1)
    sp_minus_back_x_area = (spine_df.iloc[:, 2:] - back_df.iloc[:, 2:])*area
    tp = int(sp_minus_back_x_area.shape[1]/2) #2 color用

    #全時系列データをがっちゃんこして、Ransacをとる(temporary)
    X = sp_minus_back_x_area.iloc[:, :tp].values.reshape(-1, 1)  # filler: ch0
    y = sp_minus_back_x_area.iloc[:,tp:].values.reshape(-1, 1) #Probe:ch1
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)

    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)
    fig, axes = plt.subplots(nrows=int(tp / 2) + 1, ncols=2, figsize=(10, (int(tp / 2) + 1) * 3))
    for t in range(tp):
        X_t = sp_minus_back_x_area.iloc[:, t].values.reshape(-1, 1)  # filler: ch0
        y_t = sp_minus_back_x_area.iloc[:,tp+t].values.reshape(-1, 1)
        ransac_score = y_t / ransac.predict(X_t)
        sp_minus_back_x_area["mean" + str(t) + "_ransac"] = ransac_score

        #plot
        axes[int(t / 2), t % 2].scatter(X_t, y_t, color="yellowgreen", marker=".") #,label="Inliers")
        axes[int(t/2),t%2].plot(line_X,line_y_ransac,color="cornflowerblue",linewidth=2) #,label="RANSAC regressor")
    fig.tight_layout()
    fig.savefig(file_name + "ransac.pdf", dpi=300, transparent=True)


#各tpでRansac regressionする場合
    # fig, axes = plt.subplots(nrows=int(tp/2)+1, ncols=2, figsize=(10, (int(tp/2)+1) * 3))
    # for t in range(tp):
    #     #Ransac regression
    #     X = sp_minus_back_x_area.iloc[:,t].values.reshape(-1, 1) #filler: ch0
    #     y = sp_minus_back_x_area.iloc[:,tp+t].values.reshape(-1, 1) #Probe:ch1
    #     ransac = linear_model.RANSACRegressor()
    #     ransac.fit(X, y)
    #     inlier_mask = ransac.inlier_mask_
    #     outlier_mask = np.logical_not(inlier_mask)
    #
    #     # Predict data of estimated models
    #     line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    #     line_y_ransac = ransac.predict(line_X)
    #     ransac_score = y/ransac.predict(X)
    #     sp_minus_back_x_area["mean"+str(t)+"_ransac"] = ransac_score
    #
    #     #plot
    #     lw = 2
    #     axes[int(t/2),t%2].scatter(X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers")
    #     axes[int(t/2),t%2].scatter(X[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers")
    #     axes[int(t/2),t%2].plot(line_X,line_y_ransac,color="cornflowerblue",linewidth=lw,label="RANSAC regressor")
    #     # axes[int(t/2),t%2].legend(loc="lower right")
    #     axes[int(t / 2), t % 2].xaxis.set_ticklabels([])
    #     axes[int(t / 2), t % 2].yaxis.set_ticklabels([])
    # fig.tight_layout()
    # fig.savefig(file_name + "ransac.pdf", dpi=300, transparent=True)


    #n spineずつ分けてプロットしてみる
    spine_num = len(spine_df)
    n = 100
    calc = ["AU", "delta(%)"]
    interval = 1
    ncols = int(spine_num/n)+1
    fig, axes = plt.subplots(nrows=len(color_list)*2, ncols=ncols, figsize=(3*ncols, len(color_list) *2*3))
    for c in range(len(color_list)):
        df = sp_minus_back_x_area.iloc[:,tp*c:tp*(c+1)]
        for cal in range(len(calc)):
            if calc[cal]=="delta(%)":
                base = df.iloc[:, tp_base[0]:tp_base[1]].mean(axis='columns').values.reshape(-1, 1)
                df = df / base * 100 - 100
            df_T=df.T
            df_T.to_csv(file_name + "c" + str(c) +"_"+calc[cal]+ ".csv")
            tp_list =np.arange(-interval * (tp_base[1] - 0.8), interval * (tp - tp_base[1] + 0.8), interval).reshape(-1, 1)

            for i in range(ncols):
                df_split =  df_T.iloc[:, i*n:(i+1)*n]
                if len(df_split.columns)>0:
                    average = df_split.mean(axis=1)
                    sem=df_split.sem(axis=1)
                    df_split["hour"] = tp_list
                    if ncols>1:
                        ax = df_split.plot(x="hour", ax=axes[2*c+cal][i], linewidth=0.5, legend = True, cmap="Paired", marker=".", markersize=2)
                    if ncols==1:
                        ax = df_split.plot(x="hour", ax=axes[2*c+cal], linewidth=0.5, legend=True, cmap="Paired", marker=".", markersize=2)
                    df_split["average"] = average
                    df_split["sem"] = sem
                    print(df_split)
                    ax.plot(df_split["hour"], df_split["average"], linewidth =1.5)
                    ax.axvspan(0, 0.5, color="blue", alpha=0.3, linewidth=0)
                    ax.legend(fontsize=3, loc="upper right", ncol=2)
                    # ax.set_ylabel(calc[cal])
                    # ax.axvspan(0,rapa_wash_min,color = "blue", alpha=0.3, linewidth=0)
                    # ax.axvspan(700, 710, color="blue", alpha=0.3, linewidth=0)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.axhline(y=0, color="gray", ls="dotted", lw=1)
                    ax.set_xlim(-3, 10)
                    ax.set_xticks(np.arange(0, 10, 5))
                    print("!!!!!!!!!!!!!c="+str(c))
                    if cal==0: #"AU"
                        if c==0: #filler
                            # ax.set_ylim(0, 250000)
                            # ax.set_yticks(np.arange(0, 250000, 50000))
                            ax.set_ylabel("Volume (a.u)")
                        if c == 1: #mVenus
                            # ax.set_ylim(0, 2000)
                            # ax.set_yticks(np.arange(0, 2000, 500))
                            ax.set_ylabel("Probe (a.u)")
                        if c==2: #ransac
                            # ax.set_ylim(0, 8)
                            # ax.set_yticks(np.arange(0, 8, 2))
                            ax.set_ylabel("Ransac score")
                    elif cal==1:
                        if c==0: #filler
                            ax.set_ylabel("ΔV (%)")
                        if c == 1: #mVenus
                            ax.set_ylabel("ΔProbe (%)")
                        if c==2: #ransac
                            ax.set_ylabel("ΔRansac(%)")






    fig.tight_layout()
    # fig.savefig(file_name+ "timeseries.png", format="png", dpi=300)
    fig.savefig(file_name + "timeseries.pdf", dpi=300,  transparent=True)




root = tkinter.Tk()
root.withdraw()
# dir = tkinter.filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data\sawada\CID_Analysis\Timseries")
dir = tkinter.filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data\arima\conditions\AAVAS\A101\230305-7\csv")
spine_csv_files = glob.glob(os.path.join(dir, "*spine.csv"))
print("number of files: %s" % len(spine_csv_files))

for spine_csv in spine_csv_files:
    back_csv = spine_csv[:-9]+"back.csv"
    Individual_timeseries(spine_csv, back_csv)