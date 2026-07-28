import pandas as pd
import numpy as np
import os
from scipy import stats
import glob
import tkinter.filedialog
import tkinter.messagebox
import sys
import pathlib
import itertools
import scikit_posthocs as sp
from lib.steel_test_v2 import steel_test




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
dunn_results = sp.posthoc_dunn(df_long, # Long_form変換しなかった時、new_df　を使用してください
                            group_col='group', val_col='values') #,p_adjust='bonferroni'→適応するとP値に掛け算して出してくる。普通は有意水準を割るので、ここでは適応しない。
print(dunn_results)

# post hoc steel-Dwass
# Used DSCF (Dwass–Steel–Critchlow–Fligner) analysis, better than Steel-Dwass for different numbers of samples
print("##############Steel-Dwass test results:")
steel_results = sp.posthoc_dscf(df_long,
                                group_col='group', val_col='values')
print(steel_results)

# posthoc steel.test
# Set control
control_group_name = df.columns[0]
print("##############Steel test results:")
# Use steel_test function from lib
steel_result = steel_test(
    data=df_long['values'],
    group=df_long['group'],
    control=control_group_name,
    alternative="two.sided"
)
print(steel_result)


# #すべての組み合わせに対してMann-Whitney
# print("############## Mann Whitney U ########################")
# combinations = itertools.combinations(columns, 2)
# for combination in combinations:
#     print(combination)
#     group1, group2 = df[combination[0]].values,df[combination[1]].values
#     group1 = group1[np.logical_not(np.isnan(group1))]
#     group2 = group2[np.logical_not(np.isnan(group2))]
#
#     statistic, p_value = stats.mannwhitneyu(group1, group2)
#     print("statistic")
#     print("p value")
#     print(statistic)
#     print(p_value)