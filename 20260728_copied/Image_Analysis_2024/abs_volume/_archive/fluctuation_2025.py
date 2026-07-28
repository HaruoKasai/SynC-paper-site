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
import scipy.stats
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../Lib')
sys.path.append(str(current_dir) + '/../IALib')
from ImageJRoiReader import * #original
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams["font.family"]= "Arial"
import re


# dir_name = r"N:\SynC_invitro_test\EDF1c_fluctuation\Control\zstack" #"\\DESKTOP-WS2\data\sawada\CID_Analysis\Vol_timseries\FK-Kal7_Rapa\_deltaV-V"
# ref_dir =r"N:\SynC_invitro_test\EDF1c_fluctuation\Control"
# output_dir = r"N:\SynC_invitro_test\EDF1c_fluctuation\Control\fluctuation"

dir_name = r"N:\SynC_invitro_test\EDF1c_fluctuation\SynC1\zstack" #"\\DESKTOP-WS2\data\sawada\CID_Analysis\Vol_timseries\FK-Kal7_Rapa\_deltaV-V"
ref_dir =r"N:\SynC_invitro_test\EDF1c_fluctuation\SynC1"
output_dir = r"N:\SynC_invitro_test\EDF1c_fluctuation\SYnC1\fluctuation"



tp_list = [
[[0,3], [3,10]],
[[0,3], [3,6]],
[[0,3], [4,7]],
[[0,3], [5,8]],
[[0,3], [6,9]]
           ]

#            ]

for b in range(len(tp_list)):
    tp = tp_list[b]
# tp = [[0,2], # before #tp0,1,,,6 #tp7には7→8のchangeを入れているので入れない
#     [9,11]]  #After 60分以降 , sigmaを計算するので、before, afterのtp数はそろえておく

    df_abs_all = pd.DataFrame(columns=range(tp[1][1]))
    df_change_all = pd.DataFrame(columns=range(tp[1][1]))

    img_files = glob.glob(os.path.join(dir_name, "*_binary_sum.tif*"))
    for img in img_files:
        exp_name = os.path.basename(img)[:5]
        ser_name = re.findall(r"series.", os.path.basename(img))[0]
        print(exp_name, ser_name)
        roi_list = glob.glob(os.path.join(ref_dir, "ROI_by_dend", "*"+exp_name+"*"+ ser_name + "*ROIspine.zip"))
        au_csv_list = glob.glob(os.path.join(ref_dir, "timeseries_by_dend", "*"+exp_name+"*"+ ser_name + "*c1_AU.csv"))
        for r, spine_roi in enumerate(roi_list):
            print("####################", spine_roi)
            df = ImageJRoiReader(img, spine_roi)
            df["V_t2"] = df["area_in_pixel"]*df["mean_0"]*0.3*0.069*0.069
            au_csv_num=0

            df_au = pd.read_csv(au_csv_list[r])

            # print("df",df)
            # print("df_ay",df_au)
            ########
            zero_cols = [col for col in df_au.columns[:-2] if (df_au[col] == 0).all()]
            cols_as_str = df_au.columns.astype(str)

            # 列番号（インデックス位置）を取得
            zero_col_indices = [cols_as_str.get_loc(str(c)) if str(c) in cols_as_str else i
                                for i, c in enumerate(df_au.columns) if c in zero_cols]
            # X-1 行目を削除対象に
            rows_to_drop = [i - 1 for i in zero_col_indices]
            # 対象列を df_au から削除
            df_au = df_au.drop(columns=zero_cols)
            # df の行を削除
            df = df.reset_index(drop=True)
            df = df.drop(index=rows_to_drop, errors="ignore")

            print(df.shape)
            print(df_au.shape)


            # print("df", df)
            # print("df_ay", df_au)


            df_abs = df_au.iloc[:, 1:-3].div(df_au.iloc[2][1:-3], axis = 1)
            df_abs.loc["V_abs_t2"] = df["V_t2"].values #別のデータフレームの列を掛け算するのがなぜかうまくいかないので、一度df_abs内に列足して後から消す。
            df_abs = df_abs.T
            df_abs = df_abs*df_abs["V_abs_t2"].values.reshape(-1,1)
            df_abs  = df_abs.drop("V_abs_t2", axis=1)
            df_abs.to_csv(os.path.join(output_dir, exp_name+"_"+ser_name + "_abs.csv"), index=True)

            ########################################体積絶対値の時系列グラフを出しておく
            df_plot = df_abs.T
            average = df_abs.mean(axis='index').values.reshape(-1,1)
            sem = df_abs.sem(axis='index').values.reshape(-1,1)
            df_plot["min"] = df_au["min"]

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
            ax = df_plot.plot(x="min", ax=axes[0], legend=False, linewidth=0.5, cmap="bone")
            ax.axvspan(0, 10, color="blue", alpha=0.3, linewidth=0) #rapa_wash_min = 10
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.axhline(y=0, color="gray", ls="dotted", lw=1)
            df_plot["average"] = average
            df_plot["sem"] = sem

            ax.plot(df_plot["min"], df_plot["average"], color="red", linewidth=1.5)
            x = df_plot["min"].values
            y1 = (average - sem).reshape(average.shape[0])
            y2 = (average + sem).reshape(average.shape[0])
            ax.fill_between(x, y1, y2, color="red", alpha=0.3, linewidth=0)
            ax.set_xlim(-144, 496)
            ax.set_xticks(np.arange(-120, 490, 120))
            # ax.set_ylim(0,0.5)
            fig.tight_layout()
            output_dir2 = os.path.join(output_dir, "Vol_abs_ts_graph")
            os.makedirs(output_dir2, exist_ok=True)
            fig.savefig(os.path.join(output_dir2, exp_name+"_"+ser_name + ".pdf"), dpi=300, transparent=True)
            plt.close(fig)
            ######################################################################

            df_change = df_abs.shift(-1, axis=1) -df_abs
            df_abs_all = pd.concat([df_abs_all, df_abs.iloc[:,:tp[1][1]]])
            df_change_all = pd.concat([df_change_all, df_change.iloc[:,:tp[1][1]]])


    vol_bin = 8
    output =  pd.DataFrame(columns=["time","bin", "vol_mean", "mu","se", "sigma", "CI"])
    t_list=["before", "after"]
    for t in range(len(tp)):

        abs = df_abs_all.iloc[:, tp[t][0]:tp[t][1]].values.reshape(-1,1)
        print(t_list[t],"abs", abs)
        change = df_change_all.iloc[:, tp[t][0]:tp[t][1]].values.reshape(-1,1)
        concat = np.concatenate((abs, change), axis=1)
        concat = concat[np.argsort(concat[:,0])] #vが小さい順に並び替え
        n = int(concat.shape[0]/vol_bin)
        for i in range(vol_bin):
            # print("i="+str(i))
            concat_bin = concat[n*i:n*(i+1)]
            vol_mean = np.mean(concat_bin[:,0])
            m = np.mean(concat_bin[:,1])
            sd = np.std(concat_bin[:,1])
            se = se = sd / np.sqrt(n)

            # calculate t-value and confidence interval
            alpha = 0.05
            df = n - 1
            t_value = scipy.stats.t.ppf(1 - alpha / 2, df)
            CI = t_value * se
            # output = output.append({"time":t_list[t], "bin":i, "vol_mean":vol_mean, "mu":m,"se":se, "sigma":sd, "CI":CI},ignore_index=True)
            new_row = pd.DataFrame([{
                "time": t_list[t],
                "bin": i,
                "vol_mean": vol_mean,
                "mu": m,
                "se": se,
                "sigma": sd,
                "CI": CI
            }])
            output = pd.concat([output, new_row], ignore_index=True)



    output.to_csv(os.path.join(output_dir,"summary_tp"+str(tp[0][0])+"-"+str(tp[0][1])+"-"+str(tp[1][0])+"-"+str(tp[1][1])+".csv"), index=False)


    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9,6))
    color_list = ["k", "red"]
    for t in range(len(tp)):
        data = output.loc[output['time'] == t_list[t]]

        axes[0][t].scatter(data["vol_mean"], data["mu"], s=5,color=color_list[t])
        axes[0][t].errorbar(data["vol_mean"], data["mu"], yerr=data['se'], fmt='none', ecolor=color_list[t], capsize=3)
        axes[0][t].set_xlim(0, 0.7)
        axes[0][t].set_ylim(-0.4, 0.4)
        axes[0][t].set_yticks(np.arange(-0.4, 0.4, 0.1))
        axes[0][t].axhline(y=0, color="k", lw=1) #ls = "dotted"

        axes[1][t].scatter(data["vol_mean"], data["sigma"], s=5, color=color_list[t])
        axes[1][t].errorbar(data["vol_mean"], data["sigma"], yerr=data['CI'], fmt='none', ecolor=color_list[t], capsize=3)
        axes[1][t].set_ylim(0, 0.3)
        axes[1][t].set_xlim(0, 0.7)
        axes[1][t].set_yticks(np.arange(0, 0.3, 0.05))

        #重ねたグラフも作る
        axes[0][2].scatter(data["vol_mean"], data["mu"], s=5, color=color_list[t])
        axes[0][2].errorbar(data["vol_mean"], data["mu"], yerr=data['se'], fmt='none', ecolor=color_list[t], capsize=3)
        axes[0][2].set_xlim(0, 0.7)
        axes[0][2].set_ylim(-0.4, 0.4)
        axes[0][2].set_yticks(np.arange(-0.4, 0.4, 0.1))
        axes[0][2].axhline(y=0, color="k", lw=1)  # ls = "dotted"

        axes[1][2].scatter(data["vol_mean"], data["sigma"], s=5, color=color_list[t])
        axes[1][2].errorbar(data["vol_mean"], data["sigma"], yerr=data['CI'], fmt='none', ecolor=color_list[t],
                            capsize=3)
        axes[1][2].set_ylim(0, 0.3)
        axes[1][2].set_xlim(0, 0.7)
        axes[1][2].set_yticks(np.arange(0, 0.3, 0.05))
        #############################

        for i in range(2):
            axes[i][t].spines['right'].set_visible(False)
            axes[i][t].spines['top'].set_visible(False)
            axes[i][2].spines['right'].set_visible(False)
            axes[i][2].spines['top'].set_visible(False)

        #V^(2/3)にfittingする
        x=data[["vol_mean"]]
        y=data["sigma"]

        reg = LinearRegression(fit_intercept=False).fit(x**(2/3),y)
        x_plot = np.linspace(0,0.7,300)
        axes[1][t].plot(x_plot, reg.coef_[0]*x_plot**(2/3)+reg.intercept_)
        axes[1][2].plot(x_plot, reg.coef_[0] * x_plot ** (2 / 3) + reg.intercept_)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,"summary_tp"+str(tp[0][0])+"-"+str(tp[0][1])+"-"+str(tp[1][0])+"-"+str(tp[1][1])+".pdf"), dpi=300, transparent=True)






