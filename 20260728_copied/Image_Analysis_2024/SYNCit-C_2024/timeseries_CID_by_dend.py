import pandas as pd
import numpy as np
import os
import tifffile as tiff
from scipy import stats
from sklearn.linear_model import LinearRegression
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


# def CID_timeseries(spine_csv, back_csv, tp_base=[0,8], color_list=["forestgreen", "orangered","forestgreen"], interval=20, rapa_wash_min = 10):
# def CID_timeseries(spine_csv, back_csv, tp_base=[0,8], color_list=["forestgreen", "orangered","forestgreen"], interval=20, rapa_wash_min = 10, ms="NikonA1R"): #Vol_timeseries
#     tp_list = [-145, -125, -105, -85, -65, -45, -25, -5, 15, 35, 55, 75,95,115,135,155,175,195,215,235,255,275,295,315,335,355,375,395,415,435,455,475,495]

# def CID_timeseries(spine_csv, back_csv, tp_base=[0,4], color_list=["forestgreen", "orangered","forestgreen"], interval=20, rapa_wash_min = 10, ms="NikonA1R"): #mDlx


# def CID_timeseries(spine_csv, back_csv, tp_base=[8,11], color_list=["forestgreen", "orangered","forestgreen"], interval=20, rapa_wash_min = 10, ms="NikonA1R"): #Rapa_60min_cLTP (cLTP comparison)
#     tp_list = [-45,-25, -5, 18, 38, 58, 78, 98, 118, 138]

# def CID_timeseries(spine_csv, back_csv, tp_base=[4,8], color_list=["forestgreen", "orangered","forestgreen"], interval=20, rapa_wash_min = 10, ms="NikonA1R"): #Rapa_60min_cLTP
#     # tp_list = [-145, -125, -105, -85, -65, -45,-25, -5, 15, 35, 55, 78, 98, 118, 138, 158, 178, 198, 218, 238, 258, 278, 298, 318, 338, 358, 378, 398, 418, 438, 458, 478, 498]
#     tp_list = [-65, -45, -25, -5, 15, 35, 55, 78, 98, 118, 138, 158, 178, 198, 218, 238, 258,278, 298, 318, 338, 358, 378, 398, 418, 438, 458, 478, 498]

# def CID_timeseries(spine_csv, back_csv, tp_base=[4, 6], color_list=["forestgreen", "orangered", "forestgreen"], interval=20, rapa_wash_min=10, ms="NikonA1R"):  # Rapa_30min_cLTP (cLTP comparison)
#     tp_list = [-15, -5, 10, 30, 50, 70, 90, 110, 130]

# def CID_timeseries(spine_csv, back_csv, tp_base=[1, 4], color_list=["forestgreen", "orangered", "forestgreen"], interval=20, rapa_wash_min=10, ms="NikonA1R"):  # cLTPonly
#     tp_list = [-45, -25, -5, 18, 38, 58, 78, 98, 118, 138]


# def CID_timeseries(spine_csv, back_csv, tp_base=[1,4], color_list=["forestgreen", "orangered","forestgreen"], interval=20, rapa_wash_min = 10, ms="NikonA1R"): #SEP-GLuA2
#     tp_list = [-80, -45, -10, 25, 60, 95, 130, 165, 200, 235]

# def CID_timeseries(spine_csv, back_csv, tp_base=[0, 1], color_list=["orangered","forestgreen", "forestgreen"], rapa_wash_min=10, ms = "Olympus2P"):  # vivo
#     tp_list = [-10, 30, 220]

# def CID_timeseries(spine_csv, back_csv, tp_base=[0, 3], color_list=["orangered", "forestgreen", "forestgreen"], rapa_wash_min=10, ms="Olympus2P"):  # vivo SEP
#    tp_list = [-10, 30]

# def CID_timeseries(spine_csv, back_csv, tp_base=[0, 3], color_list=["forestgreen", "orangered", "forestgreen"],interval=60, rapa_wash_min=10, ms="NikonA1R"): #PSD95
#     tp_list = [-140, -80, -20, 40, 100, 160, 220, 280, 340, 400, 460]

def CID_timeseries(spine_csv, back_csv, tp_base=[0, 3], color_list=["forestgreen", "orangered", "forestgreen"], interval=20, rapa_wash_min=10, ms="NikonA1R"):  # cLTPonly
    tp_list = [-45, -25, -5, 15, 35, 55, 78, 98, 118, 138, 158, 178, 198, 218, 238, 258]

    print(spine_csv)
    dir = os.path.dirname(os.path.dirname(spine_csv))
    fname = os.path.basename(spine_csv)
    file_name  = os.path.join(dir, "timeseries_by_dend", fname[:-9])
    print(file_name)
    spine_df = pd.read_csv(spine_csv)
    back_df = pd.read_csv(back_csv)
    area = spine_df['area_in_pixel'].values.reshape(-1,1)
    print(spine_df)
    sp_minus_back_x_area = (spine_df.iloc[:, 2:] - back_df.iloc[:, 2:])*area
    tp = int(sp_minus_back_x_area.shape[1]/2) #2 color用
    calc = ["AU","delta(%)"]
    fig, axes = plt.subplots(nrows=len(color_list), ncols=2, figsize=(6, len(color_list)*3))
    col_list = []
    for i in range(tp):
        col_list.append("mean"+str(i)+"_ratio")
    if ms=="NikonA1R":
        df_ratio = sp_minus_back_x_area.iloc[:,tp*0:tp*1].set_axis(col_list, axis='columns') / sp_minus_back_x_area.iloc[:,tp*1:tp*2].set_axis(col_list, axis='columns')
    elif ms=="Olympus2P":
        df_ratio = sp_minus_back_x_area.iloc[:,tp*1:tp*2].set_axis(col_list, axis='columns') / sp_minus_back_x_area.iloc[:,tp*0:tp*1].set_axis(col_list, axis='columns')
    df_concat = pd.concat([sp_minus_back_x_area, df_ratio], axis=1)
    for c in range(len(color_list)):
        df = df_concat.iloc[:,tp*c:tp*(c+1)] #/10**6
        for cal in range(len(calc)):
            if calc[cal]=="delta(%)":
                base = df.iloc[:, tp_base[0]:tp_base[1]].mean(axis='columns').values.reshape(-1, 1)
                df = df/base*100-100
            df_T=df.T
            df_T.columns = df_T.columns+1 #imageJで対応を見やすいように、spine番号を0,1,2,,,から1,2,3,,,に変更
            df_T = df_T.iloc[tp_base[0]:tp_base[0]+len(tp_list)]
            # average = df.mean(axis='index').values.reshape(-1,1)
            # sem = df.sem(axis='index').values.reshape(-1, 1)
            average = df_T.mean(axis = "columns").values
            sem = df_T.sem(axis='columns').values




            # tp_list =np.arange(-interval * (tp_base[1] - 0.8), interval * (tp - tp_base[1] + 0.8), interval).reshape(-1, 1)

            df_T["min"] = tp_list
            ax = df_T.plot(x="min", ax=axes[c, cal], legend=False, linewidth=0.5, cmap="bone")
            ax.set_ylabel(calc[cal])
            ax.axvspan(0,rapa_wash_min,color = "blue", alpha=0.3, linewidth=0)
            ax.axvspan(57, 72, color="yellow", alpha=0.3, linewidth=0)
            # ax.axvspan(700, 710, color="blue", alpha=0.3, linewidth=0)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.axhline(y=0, color="gray", ls="dotted", lw=1)
            df_T["average"]=average
            df_T["sem"] = sem
            df_T.to_csv(file_name+"c"+str(c)+"_"+calc[cal]+".csv")
            ax.plot(tp_list, average, color=color_list[c],linewidth=1.5)
            # x=tp_list.reshape(tp_list.shape[0])
            x = tp_list
            y1 = (average-sem).reshape(average.shape[0])
            y2 = (average +sem).reshape(average.shape[0])
            ax.fill_between(x,y1,y2, color=color_list[c], alpha=0.3, linewidth=0)
            ax.set_xlim(-60, 310)
            ax.set_xticks(np.arange(-50, 300, 50))

            if calc[cal] == "delta(%)" and c<2:
                ax.set_ylim(-100,300)
                ax.set_yticks(np.arange(-100, 310, 100))
            if calc[cal] == "delta(%)" and c == 2:
                ax.set_ylim(-100, 300)
                ax.set_yticks(np.arange(-100, 310, 100))


    fig.tight_layout()
    # fig.savefig(file_name+ "timeseries.png", format="png", dpi=300)
    fig.savefig(file_name + "timeseries.pdf", dpi=300,  transparent=True)

def Scatter_deltaV_V(spine_csv, back_csv, tp_base=8, tp_post=10, color_num=2): #TODO tp_baseを途中のtpで採れるように直していない (e.g. cLTP_after_CID)
    dir = os.path.dirname(os.path.dirname(spine_csv))
    fname = os.path.basename(spine_csv)
    spine_df = pd.read_csv(spine_csv)
    back_df = pd.read_csv(back_csv)
    area = spine_df['area_in_pixel'].values.reshape(-1, 1)
    sp_minus_back_x_area = (spine_df.iloc[:, 2:] - back_df.iloc[:, 2:]) * area
    tp = int(sp_minus_back_x_area.shape[1] / color_num )
    df = sp_minus_back_x_area.iloc[:, tp*(color_num-1):tp*(color_num-1)+tp_post]
    df["base"] = df.iloc[:, :tp_base].mean(axis='columns').values.reshape(-1, 1)
    v_max = df["base"].max()
    df["V_base(normalized)"] = df["base"] / v_max
    df["delta V"] = df.iloc[:, tp_post-1]-df["base"]
    df["ΔV (a.u.)"] = df["delta V"]/v_max
    df["ΔV(%)"] =df["delta V"]/df["base"]*100
    file_name = os.path.join(dir, "graph", fname[:-9])
    df.plot.scatter(x="V_base(normalized)", y="ΔV (a.u.)")
    plt.savefig(file_name+"Scatter_deltaV.png", format="png", dpi=300)
    df.plot.scatter(x="V_base(normalized)", y="ΔV(%)")
    plt.savefig(file_name+"Scatter_deltaV_percent.png", format="png", dpi=300)


root = tkinter.Tk()
root.withdraw()
# dir = tkinter.filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data\sawada\CID_Analysis\Timseries")
dir = tkinter.filedialog.askdirectory(initialdir=r"X:\SYNCit-C")
spine_csv_files = glob.glob(os.path.join(dir,"csv_by_dend", "*spine.csv"))
print("number of files: %s" % len(spine_csv_files))

for spine_csv in spine_csv_files:
    back_csv = spine_csv[:-9]+"back.csv"
    CID_timeseries(spine_csv, back_csv)
    # Scatter_deltaV_V(spine_csv, back_csv)