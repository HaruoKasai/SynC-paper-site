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
import matplotlib.cm as cm
import glob
import tkinter.filedialog
import tkinter.messagebox
import sys
import pathlib
import re




def data_collection(dir, channel_list=["c0", "c1"], sp_cond_list=["stim"]):  # c2: mVenus/mScarlet ratio

    cond_list = glob.glob(os.path.join(dir, "[!_]*"))
    print("number of conditions: %s" % len(cond_list))
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3, 3* 3))
    #格納する4つのデータフレームの初期化
    combined_data = {
        "delta": {ch: pd.DataFrame() for ch in channel_list},
        "AU": {ch: pd.DataFrame() for ch in channel_list},
        "initial_V": {ch: pd.DataFrame() for ch in channel_list}
    }

    for cond in cond_list:
        print("cond!!!!   "+cond)
        for sp_cond in sp_cond_list:
            date_list = glob.glob(os.path.join(cond, "[!_]*"))
            for c in range(len(channel_list)):
                channel = channel_list[c]  # , columns=columns_list
                for date in date_list:
                    day = os.path.basename(date)[2:6]

                    # delta(%) file processing
                    delta_csv_list = glob.glob(os.path.join(date, "timeseries_ind", sp_cond, "*" + channel_list[c] + "*delta(%).csv"))
                    for csv in delta_csv_list:
                        df = pd.read_csv(csv)
                        df = df.drop(columns=['min', 'average', 'sem'], errors='ignore')
                        if df.columns[0] == df.index.name or df.columns[0].lower().startswith('label'):
                            df = df.iloc[:, 1:]
                        df.reset_index(drop=True, inplace=True)  # インデックスをリセット
                        first_row = df.iloc[0]
                        numeric_columns = first_row[first_row.apply(lambda x: pd.api.types.is_numeric_dtype(type(x)))].index
                        extracted_rows = df.loc[3:5, numeric_columns].T  # .iloc を .loc に変更
                        extracted_rows.columns = [f"{day}_row_{i + 3}" for i in range(3)]
                        combined_data["delta"][channel] = pd.concat([combined_data["delta"][channel], extracted_rows], axis=1)

                    # AU file processing
                    AU_csv_list = glob.glob(os.path.join(date, "timeseries_ind", sp_cond, "*" + channel_list[c] + "*AU.csv"))
                    for csv in AU_csv_list:
                        df = pd.read_csv(csv)
                        df = df.drop(columns=['min', 'average', 'sem'], errors='ignore')
                        if df.columns[0] == df.index.name or df.columns[0].lower().startswith('label'):
                            df = df.iloc[:, 1:]
                        df.reset_index(drop=True, inplace=True)  # インデックスをリセット
                        first_row = df.iloc[0]
                        numeric_columns = first_row[first_row.apply(lambda x: pd.api.types.is_numeric_dtype(type(x)))].index
                        extracted_rows = df.loc[3:5, numeric_columns].T  # .iloc を .loc に変更
                        extracted_rows.columns = [f"{day}_row_{i + 3}" for i in range(3)]
                        combined_data["AU"][channel] = pd.concat([combined_data["AU"][channel], extracted_rows], axis=1)
                        # 初期V (最初の3行の平均値)
                        first_row = df.iloc[0]
                        numeric_columns = first_row[
                            first_row.apply(lambda x: pd.api.types.is_numeric_dtype(type(x)))].index
                        initial_values = df.loc[0:2, numeric_columns].mean(axis=0)


                        # 細胞ごとの正規化 (各細胞の平均値で割る)
                        normalized_initial_values = initial_values / initial_values.mean()

                        normalized_initial_values.name = day

                        # 結果を格納
                        combined_data["initial_V"][channel] = pd.concat(
                            [combined_data["initial_V"][channel], normalized_initial_values], axis=1)

                        # 中間結果を保存する
                        initial_v_output_file = os.path.join(dir, f"_{channel}_initial_V.csv")
                        combined_data["initial_V"][channel].to_csv(initial_v_output_file)

    for dtype in ["delta", "AU"]:
        for channel in channel_list:
            print(f"Data for {dtype} in channel {channel}:")
            print(combined_data[dtype][channel])
            # 必要であればCSVとして保存
            output_file = os.path.join(dir, f"_{dtype}_{channel}_combined_data.csv")
            combined_data[dtype][channel].to_csv(output_file)


    # CSVファイルを読み込む際にヘッダーを無視する
    for channel in channel_list:
        delta_file = os.path.join(dir, f"_delta_{channel}_combined_data.csv")
        au_file = os.path.join(dir, f"_AU_{channel}_combined_data.csv")
        initial_V_file = os.path.join(dir,  f"_{channel}_initial_V.csv")

        # ヘッダーを無視してデータを読み込む

        combined_data["delta"][channel] = pd.read_csv(delta_file, header=0).iloc[1:, 1:]
        combined_data["AU"][channel] = pd.read_csv(au_file, header=0).iloc[1:, 1:]
        combined_data["initial_V"]["c0"] =pd.read_csv(initial_V_file, header=0).iloc[1:, 1:]

    # 列を3つのグループに分ける
    def split_columns_by_modulo(df):
        group_0 = df.iloc[:, [i for i in range(df.shape[1]) if i % 3 == 0]]
        group_1 = df.iloc[:, [i for i in range(df.shape[1]) if i % 3 == 1]]
        group_2 = df.iloc[:, [i for i in range(df.shape[1]) if i % 3 == 2]]
        return group_0, group_1, group_2

    initial_V_c0 = combined_data["initial_V"]["c0"]

    # デルタとAUのそれぞれのチャンネルで3グループに分ける
    delta_c0_group_0, delta_c0_group_1, delta_c0_group_2 = split_columns_by_modulo(combined_data["delta"]["c0"])
    delta_c1_group_0, delta_c1_group_1, delta_c1_group_2 = split_columns_by_modulo(combined_data["delta"]["c1"])
    au_c0_group_0, au_c0_group_1, au_c0_group_2 = split_columns_by_modulo(combined_data["AU"]["c0"])
    au_c1_group_0, au_c1_group_1, au_c1_group_2 = split_columns_by_modulo(combined_data["AU"]["c1"])

    # 散布図を作成する
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

    total_cells = max(delta_c0_group_0.shape[1], delta_c1_group_0.shape[1], au_c0_group_0.shape[1],
                      au_c1_group_0.shape[1])
    colors = cm.get_cmap('tab20', total_cells)  # 色を自動生成 (最大20色まで一度に使える)
    print(combined_data["delta"]["c0"].duplicated().sum())
    combined_data["delta"]["c0"] = combined_data["delta"]["c0"].drop_duplicates()

    # プロットするデータ
    plots = [
        (delta_c0_group_0, delta_c1_group_0, axes[0, 0], 'Group 0', 'delta_c0', 'delta_c1'),
        (delta_c0_group_1, delta_c1_group_1, axes[0, 1], 'Group 1', 'delta_c0', 'delta_c1'),
        (delta_c0_group_2, delta_c1_group_2, axes[0, 2], 'Group 2', 'delta_c0', 'delta_c1'),
        (initial_V_c0, delta_c0_group_0, axes[1, 0], 'Group 0', 'initial_V_c0', 'delta_c0'),
        (initial_V_c0, delta_c0_group_1, axes[1, 1], 'Group 1', 'initial_V_c0', 'delta_c0'),
        (initial_V_c0, delta_c0_group_2, axes[1, 2], 'Group 2', 'initial_V_c0', 'delta_c0'),
        (initial_V_c0, delta_c1_group_0, axes[2, 0], 'Group 0', 'initial_V_c0', 'delta_c1'),
        (initial_V_c0, delta_c1_group_1, axes[2, 1], 'Group 1', 'initial_V_c0', 'delta_c1'),
        (initial_V_c0, delta_c1_group_2, axes[2, 2], 'Group 2', 'initial_V_c0', 'delta_c1')
    ]

    for group_x, group_y, ax, group_label, x_label, y_label in plots:
        for i, (col_x, col_y) in enumerate(zip(group_x.columns, group_y.columns)):
            x_valid = pd.to_numeric(group_x[col_x], errors='coerce').dropna()
            y_valid = pd.to_numeric(group_y[col_y], errors='coerce').dropna()
            print(f"Column {col_x} (x) length: {len(x_valid)}")
            print(f"Column {col_y} (y) length: {len(y_valid)}")
            ax.scatter(group_x[col_x], group_y[col_y], color=colors(i), label=f'Cell {i + 1}')

        ax.set_xlabel(x_label)
        if ax == axes[0,0]:
             ax.set_xlim(-100, 300)
             ax.set_ylim(-100, 600)
        elif ax == axes[0,1]:
             ax.set_xlim(-100, 300)
             ax.set_ylim(-100, 600)
        elif ax == axes[0,2]:
            ax.set_xlim(-100, 300)
            ax.set_ylim(-100, 600)
        elif ax == axes[1,0]:
             ax.set_xlim(0, 3.5)
             ax.set_ylim(-100, 300)
        elif ax == axes[1,1]:
             ax.set_xlim(0, 3.5)
             ax.set_ylim(-100, 300)
        elif ax == axes[1,2]:
             ax.set_xlim(0, 3.5)
             ax.set_ylim(-100, 300)
        elif ax == axes[2, 0]:
             ax.set_xlim(0, 3.5)
             ax.set_ylim(-100, 600)
        elif ax == axes[2, 1]:
             ax.set_xlim(0, 3.5)
             ax.set_ylim(-100, 600)
        elif ax == axes[2, 2]:
             ax.set_xlim(0, 3.5)
             ax.set_ylim(-100, 600)
        if group_label == 'Group 0':
            title = '25 min'
        elif group_label == 'Group 1':
            title = '45 min'
        elif group_label == 'Group 2':
            title = '65 min'

        ax.set_ylabel(y_label)
        ax.set_title(f'{x_label} vs {y_label} ({title})')
        #ax.grid(True)
        ax.legend()

    # 全体をPDFに保存
    pdf_output_file = os.path.join(dir, "_scatter_plots_by_group.pdf")
    plt.tight_layout()
    plt.savefig(pdf_output_file)

#def initial_V_collection(dir, channel_list=["c0", "c1"], sp_cond_list=["stim"]):


# ディレクトリを選択して処理を開始
root = tkinter.Tk()
root.withdraw()
dir = tkinter.filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate")
df = data_collection(dir)


