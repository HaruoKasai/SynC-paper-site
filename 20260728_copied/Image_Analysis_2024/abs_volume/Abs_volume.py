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
import re

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../Lib')
sys.path.append(str(current_dir) + '/../IALib')
from ImageJRoiReader import * #original
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams["font.family"]= "Arial"


tp = 2 #zstack_registration_cropで抽出したtp
tp_before = [0,3] #5,6,7
tp_after =[3,11]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(3,6))

dir_name_list = [r"N:\SynC_invitro_test\EDF1c_fluctuation\SynC1\zstack"
                 ,r"N:\SynC_invitro_test\EDF1c_fluctuation\Control\zstack"] #,r"\\DESKTOP-WS2\data\sawada\CID_Analysis\Vol_timseries\FK-Kal7_Vehicle\_deltaV-V", r"\\DESKTOP-WS2\data\sawada\CID_Analysis\Vol_timseries\FK-Kal7_Rapa\_deltaV-V"
ref_dir_list  =[r"N:\SynC_invitro_test\EDF1c_fluctuation\SynC1"
                ,r"N:\SynC_invitro_test\EDF1c_fluctuation\Control"] #r"\\DESKTOP-WS2\data\sawada\CID_Analysis\Vol_timseries\FK-Kal7_Vehicle", r"\\DESKTOP-WS2\data\sawada\CID_Analysis\Vol_timseries\FK-Kal7_Rapa"
color_list = ["gray", "red"]
ctrl_sd=0


for d in range(len(dir_name_list)):
    dir_name = dir_name_list[d]
    ref_dir = ref_dir_list[d]
    cond = os.path.basename(dir_name)
    output_dir = os.path.join(dir_name, "abs_vol_csv")
    os.makedirs(output_dir, exist_ok=True)
    img_files = glob.glob(os.path.join(dir_name, "*_binary_sum.tif*"))

    df_sum = pd.DataFrame()
    for img in img_files:
        print("####",img)
        exp_name = os.path.basename(img)[:5]
        ser_name = re.findall(r"series.", os.path.basename(img))[0]

        ROI_dir =  os.path.join(ref_dir, "ROI_by_dend")
        roi_list = glob.glob(os.path.join(ROI_dir, "SUM_"+exp_name+"*"+ser_name+ "*ROIspine.zip"))
        delta_list = glob.glob(os.path.join(os.path.join(ref_dir, "timeseries_by_dend"), "SUM_"+exp_name+"*"+ser_name+ "*c1*delta*.csv"))
        if len(roi_list) != len(delta_list):
            print("Couldn't find files well")
        for r, roi_file in enumerate(roi_list):
            print(img)
            print(roi_file)
            results = ImageJRoiReader(img, roi_file)
            # results_back = ImageJRoiReader(img, back_roi)
            # img = tiff.imread(img)
            results["V"] = results["area_in_pixel"]*results["mean_0"]*0.3*0.069*0.069
            # results_back.to_csv(os.path.join(output_dir, dendname + "_back.csv"), index=False)

            delta_csv = delta_list[r]
            df = results
            df_delta = pd.read_csv(delta_csv)

            #deltaを出すときにバグで計算されないROIがあるらしい（Ohtsuka）→そのスパインを消して計算する
            nan_cols = df_delta.columns[df_delta.isna().any()].tolist()
            cols_as_str = df_delta.columns.astype(str)

            nan_col_indices = [cols_as_str.get_loc(str(c)) if str(c) in cols_as_str else i
                               for i, c in enumerate(df_delta.columns) if c in nan_cols]
            rows_to_drop = [i - 1 for i in nan_col_indices]
            df_delta = df_delta.drop(columns=nan_cols)
            df = df.reset_index(drop=True)
            df = df.drop(index=rows_to_drop, errors="ignore")
            ####


            df["delta(%)_"+str(tp)] = df_delta.iloc[tp].values[1:-3]
            df["delta(%)_after"] = df_delta.iloc[tp_after[0]:tp_after[1]].mean().values[:-3]
            df["delta(%)_before"] = df_delta.iloc[tp_before[0]:tp_before[1]].mean().values[:-3]
            df["delta(%)_after-before"] = df["delta(%)_after"] - df["delta(%)_before"]
            df["delta(%)_after/before"] = (df["delta(%)_after"]+100) / (df["delta(%)_before"]+100)*100-100
            # df["V_before_ave"] = df["V"] / (df["delta(%)_"+str(tp)]+100)*100
            df["V_before_ave"] = df["V"] / (df["delta(%)_" + str(tp)] + 100) * (df["delta(%)_before"]+100 )
            # df["delta_abs"] = df["V_before_ave"] * df["delta(%)_after"] / 100
            df["delta_abs_after-before"] = df["V_before_ave"] * df["delta(%)_after-before"] / 100
            # print(df)
            # axes[0].scatter(df["V_before_ave"], df["delta(%)_after/before"], s=15,marker = ".",lw=0,label=dendname, color =color_list[d])
            # axes[1].scatter(df["V_before_ave"], df["delta_abs_after-before"],s=15,marker = ".",lw=0)


                # for a in range(2):
                #     axes[a].spines['right'].set_visible(False)
                #     axes[a].spines['top'].set_visible(False)

            dend_name = re.findall(r"dend.", os.path.basename(roi_file))[0]
            df.to_csv(os.path.join(output_dir, exp_name+"_"+ser_name+"_"+dend_name+"_spine_abs.csv"), index=False)
            df.insert(0, "dend_name", dend_name)
            df.insert(0, "ser_name", ser_name)
            df.insert(0, "exp_name", exp_name)
            df_sum = pd.concat([df_sum,df], axis=0)
            # if d==0:
            #     ctrl_sd = df_sum["delta(%)_after/before"].std(ddof=1)
            # if d==1:
            #     spine_num = len(df_sum)
            #     enlarged_spine_num = len(df_sum[df_sum["delta(%)_after/before"] > 2*ctrl_sd])
            #     print("spine num:" + str(spine_num))
            #     print("enlarged spine num:" + str(enlarged_spine_num))
            #     print("ratio:")
            #     print(enlarged_spine_num/spine_num)


    df_sum.to_csv(os.path.join(output_dir,  "integrated_spine_abs.csv"), index=False)
    weights = np.ones(len(df_sum)) / len(df_sum)  # 各サンプルが1/Nの確率

    bins = np.arange(0, 1.2 + 0.15, 0.1)
    ax=axes[d]
    ax.set_ylim([0,0.4])
    ax.hist(df_sum["V_before_ave"], bins=bins,  weights=weights, edgecolor='black', alpha=0.7) #density=True,
    ax.set_xlabel("V")
    ax.set_ylabel("Probability")
    ax.set_title(f"Histogram of V (subplot {d})")


plt.tight_layout()

# PDF保存
fig.savefig(r"N:\SynC_invitro_test\EDF1c_fluctuation\V_histograms.pdf", bbox_inches="tight")
plt.close(fig)

# axes[0].axhline(y=0, color="gray", lw=1)
# axes[0].axhline(y=2*ctrl_sd, color="gray", lw=1)
# axes[0].axhline(y=-2 * ctrl_sd, color="gray", lw=1)
# axes[0].set_xticks(np.arange(0, 0.7, 0.1))
# axes[0].set_ylim(-40,170)
# axes[0].set_xlim(0,0.77)
# axes[0].legend(fontsize=3, loc="upper right")
# axes[1].set_xticks(np.arange(0, 0.7, 0.1))
# axes[1].set_ylim(-0.4,0.4)
# fig.tight_layout()
# fig.savefig(os.path.join(r"\\DESKTOP-WS2\data\sawada\CID_Analysis\Vol_timseries\FK-Kal7_Rapa\_deltaV-V", "abs_vol_csv","_scatter_deltaV(%).pdf"), dpi=300, transparent=True)






