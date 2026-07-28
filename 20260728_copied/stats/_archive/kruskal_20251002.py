import pandas as pd
import numpy as np
import os
os.environ['R_HOME'] = r"C:/Users/SIQI ZHOU/anaconda3/envs/analysis_v1/Lib/R"
from scipy import stats
import glob
import tkinter.filedialog
import tkinter.messagebox
import sys
import pathlib
import itertools
from scikit_posthocs import posthoc_dunn
# import rpy2.robjects as ro
# from rpy2.robjects import pandas2ri



root = tkinter.Tk()
root.withdraw()
# csv = tkinter.filedialog.askopenfilename(initialdir=r"\\DESKTOP-WS2\data\sawada\CID_Analysis\_stats")
csv = tkinter.filedialog.askopenfilename(initialdir=r"\\Synology\zhou\\SynC_Fig\_stats")
df = pd.read_csv(csv)
print(df)
df_long = pd.melt(df, var_name='group', value_name='values') #　Data_form_change　
df_long = df_long.dropna(subset=['values']) # remove the NaN data
print(df_long.head())
columns = df.columns
print(columns[0])


#kruskal wallis
df_tp =np.transpose(df.values)
groups = [column for column in df_tp]
for c in range(len(groups)):
    arr = groups[c]
    arr_wo_nan = arr[np.logical_not(np.isnan(arr))]
    groups[c] = arr_wo_nan

statistic, p_value = stats.kruskal(*groups)

print("############## Kruskal Wallis ########################")
print("statistic")
print("p value")
print(statistic)
print(p_value)


#post hoc Dunn
print("##############Dunn's test results:")
new_data = {
    'group': [],
    'values': []
}
for column in df.columns:
    new_data['group'].extend([column] * len(df))
    new_data['values'].extend(df[column].tolist())
new_df = pd.DataFrame(new_data)
dunn_results = posthoc_dunn(df_long, # Long_form変換しなかった時、new_df　を使用してください
                            group_col='group', val_col='values') #,p_adjust='bonferroni'→適応するとP値に掛け算して出してくる。普通は有意水準を割るので、ここでは適応しない。
print(dunn_results)

#post hoc steel
# print("##############Steel-Dwass test results:")
# # Rへ変換
# pandas2ri.activate()
# r_df_long = pandas2ri.py2rpy(df_long)
# # Rパッケージ読み込み
# ro.r('library(NSM3)')
# # Steel-Dwass実行
# ro.globalenv['rdf'] = r_df_long
# res = ro.r('pSteelDwassTest(rdf$value, rdf$group)')
# print(res)

