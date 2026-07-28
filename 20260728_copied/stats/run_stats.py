import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from statsmodels.stats.multicomp import MultiComparison
from scipy import stats  # 他の統計手法で利用

# 設定
# output_dir = r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_stats"
# method_csv = r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_stats\_stats_method.csv"
output_dir = r"\\Synology\arima\rotarod_movie\_stats"
method_csv = r"\\Synology\arima\rotarod_movie\_stats\_stats_method.csv"
df = pd.read_csv(method_csv)

for i in range(df.shape[0]):
    fig_name = df.iloc[i, 0]
    data_dir = df.iloc[i, 1]
    group_list = df.iloc[i, 2].split('\n')
    fname_list = df.iloc[i, 3].split('\n')

    if isinstance(df.iloc[i, 4], str):
        data_range = df.iloc[i, 4].split(',')
        print(f"Row {i} - Data range: {data_range}, Length: {len(data_range)}")  # デバッグ用
        data_range = [int(x) for x in data_range]
    else:
        print(f"Skipping row {i} due to missing or invalid data range")
        continue

    stats_method = df.iloc[i, 5]
    control_group = df.iloc[i, 6] if len(df.columns) > 6 else None  # コントロール群の指定

    summary = pd.DataFrame(np.nan, index=range(2000), columns=group_list)
    summary = summary.apply(pd.to_numeric, errors='coerce')
    print(summary.dtypes)
    max_len = 0
    data_dict = {}
    print(summary.dtypes)
    print(summary.head())  # どのグループに有効データがあるかチェック
    print(summary.isna().sum())

    for k in range(len(group_list)):
        group_csv = os.path.join(data_dir.rstrip(os.sep), str(fname_list[k]).strip())

        if not os.path.isfile(group_csv):
            print(f"Warning: File {group_csv} does not exist! Skipping...")
            continue

        group_df = pd.read_csv(group_csv)
        print(group_df.dtypes)
        print(group_df.head())
        # 指定範囲の平均値を取得
        values = group_df.iloc[data_range[0]:data_range[1], 1:].mean().values
        values = values.astype(float)
        max_len = max(max_len, len(values))
        summary[group_list[k]].iloc[:len(values)] = values
        data_dict[group_list[k]] = values  # Dunnett検定のため辞書に格納

    summary = summary.iloc[:max_len]

    # 統計検定
    if stats_method.lower() == "dunnett":
        if not control_group or control_group not in group_list:
            print(f"Skipping {fig_name}: Control group not specified or not found.")
            continue

        # Dunnett検定を実施
        data = []
        groups = []

        for group, values in data_dict.items():
            for value in values:
                data.append(value)
                groups.append(group)

        mc = MultiComparison(data, groups)
        dunnett_result = mc.allpairtest(sm.stats.ttest_ind, method="holm")[0]  # Holm補正付き Dunnett検定

        # 結果を DataFrame に変換
        dunnett_result_df = pd.DataFrame(dunnett_result.data[1:], columns=dunnett_result.data[0])
        print(dunnett_result_df)

        # 結果を CSV に保存
        dunnett_result_df.to_csv(os.path.join(output_dir, fig_name + "_dunnett.csv"), index=False)

    else:
        if hasattr(stats, stats_method.lower()):
            stat_func = getattr(stats, stats_method.lower())
            stat_result = stat_func(summary)
            summary = pd.concat([summary, pd.DataFrame({'': [np.nan] * len(summary)}), stat_result], axis=1)
            summary.to_csv(os.path.join(output_dir, fig_name + ".csv"), index=False)
        else:
            print(f"Error: {stats_method} is not a valid method in scipy.stats")
