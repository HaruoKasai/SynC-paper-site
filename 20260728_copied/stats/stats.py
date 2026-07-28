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
from scikit_posthocs import posthoc_dunn


def Kruskal (df):
    print(df)
    columns = df.columns
    # print(columns[0])

    #kruskal wallis
    df_tp =np.transpose(df.values)
    groups = [column for column in df_tp]
    for c in range(len(groups)):
        arr = groups[c]
        arr_wo_nan = arr[np.logical_not(np.isnan(arr))]
        groups[c] = arr_wo_nan

    statistic, p_value = stats.kruskal(*groups)

    kruskal_result = [
        ["Kruskal Wallis", ""],
        ["statistic", statistic],
        ["p value", p_value]
    ]

    df_results = pd.DataFrame(kruskal_result)


    #post hoc Dunn
    new_data = {
        'group': [],
        'values': []
    }
    for column in df.columns:
        new_data['group'].extend([column] * len(df))
        new_data['values'].extend(df[column].tolist())
    new_df = pd.DataFrame(new_data)
    dunn = posthoc_dunn(new_df, group_col='group', val_col='values') #,p_adjust='bonferroni'→適応するとP値に掛け算して出してくる。普通は有意水準を割るので、ここでは適応しない。
    #output csvに見やすく組み込むために、行名と列名も要素として扱う
    group = dunn.index.tolist()
    values = dunn.values
    dunn_result = pd.concat([pd.DataFrame(group).T, pd.DataFrame(values)], ignore_index=True)
    dfg = pd.DataFrame(group)
    dfg.loc[-1] = "Dunn's test"
    dfg.index = dfg.index + 1
    dfg = dfg.sort_index()
    dunn_result = pd.concat([dfg, dunn_result], axis=1)
    dunn_result = dunn_result.transpose().reset_index(drop=True)

    df_results = pd.concat([df_results, dunn_result], ignore_index = True)
    df_results.columns =[''] * len(df_results.columns)
    return df_results


def Mann_Whitney (df):
    columns = df.columns
    combinations = itertools.combinations(columns, 2)
    for combination in combinations:
        # print(combination)
        group1, group2 = df[combination[0]].values, df[combination[1]].values
        group1 = group1[np.logical_not(np.isnan(group1))]
        group2 = group2[np.logical_not(np.isnan(group2))]

        statistic, p_value = stats.mannwhitneyu(group1, group2)

        result = [
            ["Mann-Whitney", ""],
            ["statistic", statistic],
            ["p value", p_value]
        ]

        df_results = pd.DataFrame(result)
        df_results.columns = [''] * len(df_results.columns)

    return df_results











