import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.colors import Normalize
import os

#Swansonの値から”expression level”を1次元で表す


def flatten(data_csv, volume_csv):
    df = pd.read_csv(data_csv)
    df_v = pd.read_csv(volume_csv, index_col=0)

    # Cortex行の体積(%)
    v_percent = df_v.loc["Cortex"][2:] / 100

    # 共通列だけを使って加重平均
    common_cols = df.columns.intersection(v_percent.index)
    weighted = df[common_cols] * v_percent[common_cols]

    # 元のMouse列を保持
    if "Mouse" in df.columns:
        weighted.insert(0, "Mouse", df["Mouse"])  # 先頭列に追加

    # 行ごとの合計
    weighted["Weighted_sum"] = weighted[common_cols].sum(axis=1)

    # 出力
    output_file = os.path.join(os.path.dirname(data_csv), "Weighted_sum_output_FP+.csv")
    weighted.to_csv(output_file, index=False)


def main():
    csv_path = r"P:\Histological_analysis\Pup_expression_summary.csv"
    volume_csv = r"P:\Histological_analysis\Swanson_provisional_area_volume_FP+.csv"
    flatten(csv_path, volume_csv)

if __name__ == "__main__":
    main()