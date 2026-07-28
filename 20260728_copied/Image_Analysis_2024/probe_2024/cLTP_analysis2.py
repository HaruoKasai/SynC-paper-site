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
                        extracted_rows = df.loc[3:14, numeric_columns].T  # .iloc を .loc に変更
                        extracted_rows.columns = [f"{day}_row_{i + 3}" for i in range(12)]
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
                        extracted_rows = df.loc[3:14, numeric_columns].T  # .iloc を .loc に変更
                        extracted_rows.columns = [f"{day}_row_{i + 3}" for i in range(12)]
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
        group_0 = df.iloc[:, [i for i in range(df.shape[1]) if i % 12 == 0]]
        group_1 = df.iloc[:, [i for i in range(df.shape[1]) if i % 12 == 1]]
        group_2 = df.iloc[:, [i for i in range(df.shape[1]) if i % 12 == 2]]
        group_3 = df.iloc[:, [i for i in range(df.shape[1]) if i % 12 == 3]]
        group_4 = df.iloc[:, [i for i in range(df.shape[1]) if i % 12 == 4]]
        group_5 = df.iloc[:, [i for i in range(df.shape[1]) if i % 12 == 5]]
        group_6 = df.iloc[:, [i for i in range(df.shape[1]) if i % 12 == 6]]
        group_7 = df.iloc[:, [i for i in range(df.shape[1]) if i % 12 == 7]]
        group_8 = df.iloc[:, [i for i in range(df.shape[1]) if i % 12 == 8]]
        group_9 = df.iloc[:, [i for i in range(df.shape[1]) if i % 12 == 9]]
        group_10 = df.iloc[:, [i for i in range(df.shape[1]) if i % 12 == 10]]
        group_11 = df.iloc[:, [i for i in range(df.shape[1]) if i % 12 == 11]]
        return group_0, group_1, group_2, group_3, group_4, group_5, group_6, group_7, group_8, group_9, group_10, group_11

    initial_V_c0 = combined_data["initial_V"]["c0"]

    # デルタとAUのそれぞれのチャンネルで3グループに分ける
    delta_c0_group_0, delta_c0_group_1, delta_c0_group_2, delta_c0_group_3, delta_c0_group_4, delta_c0_group_5, delta_c0_group_6, delta_c0_group_7, delta_c0_group_8, delta_c0_group_9, delta_c0_group_10, delta_c0_group_11 = split_columns_by_modulo(combined_data["delta"]["c0"])
    delta_c1_group_0, delta_c1_group_1, delta_c1_group_2, delta_c1_group_3, delta_c1_group_4, delta_c1_group_5, delta_c1_group_6, delta_c1_group_7, delta_c1_group_8, delta_c1_group_9, delta_c1_group_10, delta_c1_group_11 = split_columns_by_modulo(combined_data["delta"]["c1"])
    au_c0_group_0, au_c0_group_1, au_c0_group_2, au_c0_group_3, au_c0_group_4, au_c0_group_5, au_c0_group_6, au_c0_group_7, au_c0_group_8, au_c0_group_9, au_c0_group_10, au_c0_group_11 = split_columns_by_modulo(combined_data["AU"]["c0"])
    au_c1_group_0, au_c1_group_1, au_c1_group_2, au_c1_group_3, au_c1_group_4, au_c1_group_5, au_c1_group_6, au_c1_group_7, au_c1_group_8, au_c1_group_9, au_c1_group_10, au_c1_group_11 = split_columns_by_modulo(combined_data["AU"]["c1"])

    # 散布図を作成する
    fig, axes = plt.subplots(nrows=3, ncols=12, figsize=(36, 9))

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
        (delta_c0_group_3, delta_c1_group_3, axes[0, 3], 'Group 3', 'delta_c0', 'delta_c1'),
        (delta_c0_group_4, delta_c1_group_4, axes[0, 4], 'Group 4', 'delta_c0', 'delta_c1'),
        (delta_c0_group_5, delta_c1_group_5, axes[0, 5], 'Group 5', 'delta_c0', 'delta_c1'),
        (delta_c0_group_6, delta_c1_group_6, axes[0, 6], 'Group 6', 'delta_c0', 'delta_c1'),
        (delta_c0_group_7, delta_c1_group_7, axes[0, 7], 'Group 7', 'delta_c0', 'delta_c1'),
        (delta_c0_group_8, delta_c1_group_8, axes[0, 8], 'Group 8', 'delta_c0', 'delta_c1'),
        (delta_c0_group_9, delta_c1_group_9, axes[0, 9], 'Group 9', 'delta_c0', 'delta_c1'),
        (delta_c0_group_10, delta_c1_group_10, axes[0, 10], 'Group 10', 'delta_c0', 'delta_c1'),
        (delta_c0_group_11, delta_c1_group_11, axes[0, 11], 'Group 11', 'delta_c0', 'delta_c1'),
        (initial_V_c0, delta_c0_group_0, axes[1, 0], 'Group 0', 'initial_V_c0', 'delta_c0'),
        (initial_V_c0, delta_c0_group_1, axes[1, 1], 'Group 1', 'initial_V_c0', 'delta_c0'),
        (initial_V_c0, delta_c0_group_2, axes[1, 2], 'Group 2', 'initial_V_c0', 'delta_c0'),
        (initial_V_c0, delta_c0_group_3, axes[1, 3], 'Group 3', 'initial_V_c0', 'delta_c0'),
        (initial_V_c0, delta_c0_group_4, axes[1, 4], 'Group 4', 'initial_V_c0', 'delta_c0'),
        (initial_V_c0, delta_c0_group_5, axes[1, 5], 'Group 5', 'initial_V_c0', 'delta_c0'),
        (initial_V_c0, delta_c0_group_6, axes[1, 6], 'Group 6', 'initial_V_c0', 'delta_c0'),
        (initial_V_c0, delta_c0_group_7, axes[1, 7], 'Group 7', 'initial_V_c0', 'delta_c0'),
        (initial_V_c0, delta_c0_group_8, axes[1, 8], 'Group 8', 'initial_V_c0', 'delta_c0'),
        (initial_V_c0, delta_c0_group_9, axes[1, 9], 'Group 9', 'initial_V_c0', 'delta_c0'),
        (initial_V_c0, delta_c0_group_10, axes[1, 10], 'Group 10', 'initial_V_c0', 'delta_c0'),
        (initial_V_c0, delta_c0_group_11, axes[1, 11], 'Group 11', 'initial_V_c0', 'delta_c0'),
        (initial_V_c0, delta_c1_group_0, axes[2, 0], 'Group 0', 'initial_V_c0', 'delta_c1'),
        (initial_V_c0, delta_c1_group_1, axes[2, 1], 'Group 1', 'initial_V_c0', 'delta_c1'),
        (initial_V_c0, delta_c1_group_2, axes[2, 2], 'Group 2', 'initial_V_c0', 'delta_c1'),
        (initial_V_c0, delta_c1_group_3, axes[2, 3], 'Group 3', 'initial_V_c0', 'delta_c1'),
        (initial_V_c0, delta_c1_group_4, axes[2, 4], 'Group 4', 'initial_V_c0', 'delta_c1'),
        (initial_V_c0, delta_c1_group_5, axes[2, 5], 'Group 5', 'initial_V_c0', 'delta_c1'),
        (initial_V_c0, delta_c1_group_6, axes[2, 6], 'Group 6', 'initial_V_c0', 'delta_c1'),
        (initial_V_c0, delta_c1_group_7, axes[2, 7], 'Group 7', 'initial_V_c0', 'delta_c1'),
        (initial_V_c0, delta_c1_group_8, axes[2, 8], 'Group 8', 'initial_V_c0', 'delta_c1'),
        (initial_V_c0, delta_c1_group_9, axes[2, 9], 'Group 9', 'initial_V_c0', 'delta_c1'),
        (initial_V_c0, delta_c1_group_10, axes[2, 10], 'Group 10', 'initial_V_c0', 'delta_c1'),
        (initial_V_c0, delta_c1_group_11, axes[2, 11], 'Group 11', 'initial_V_c0', 'delta_c1')
    ]

    for group_x, group_y, ax, group_label, x_label, y_label in plots:
        for i, (col_x, col_y) in enumerate(zip(group_x.columns, group_y.columns)):
            x_valid = pd.to_numeric(group_x[col_x], errors='coerce').dropna()
            y_valid = pd.to_numeric(group_y[col_y], errors='coerce').dropna()
            print(f"Column {col_x} (x) length: {len(x_valid)}")
            print(f"Column {col_y} (y) length: {len(y_valid)}")
            ax.scatter(group_x[col_x], group_y[col_y], color=colors(i), label=f'Cell {i + 1}')
        if group_label == 'Group 0':
            title = '25 min'
        elif group_label == 'Group 1':
            title = '45 min'
        elif group_label == 'Group 2':
            title = '65 min'
        elif group_label == 'Group 3':
            title = '85 min'
        elif group_label == 'Group 4':
            title = '105 min'
        elif group_label == 'Group 5':
            title = '125 min'
        elif group_label == 'Group 6':
            title = '145 min'
        elif group_label == 'Group 7':
            title = '165 min'
        elif group_label == 'Group 8':
            title = '185 min'
        elif group_label == 'Group 9':
            title = '205 min'
        elif group_label == 'Group 10':
            title = '225 min'
        elif group_label == 'Group 11':
            title = '245 min'

        ax.set_ylabel(y_label)
        #ax.legend()
        ax.set_title(f'{x_label} vs {y_label} ({title})')
        # ax.grid(True)
        ax.set_xlabel(x_label)
        for row_idx, row_axes in enumerate(axes): #各行のサブプロットを取得
            for col_idx, ax in enumerate (row_axes): #各列のサブプロットを取得
                if row_idx == 0:
                    ax.set_xlim(-100, 300)
                    ax.set_ylim(-100, 600)
                elif row_idx == 1:
                    ax.set_xlim(0, 3.2)
                    ax.set_ylim(-100, 300)
                elif row_idx == 2:
                    ax.set_xlim(0, 3.2)
                    ax.set_ylim(-100, 600)




    # 全体をPDFに保存
    pdf_output_file = os.path.join(dir, "_scatter_plots_by_12group.pdf")
    plt.tight_layout()
    plt.savefig(pdf_output_file)

#def initial_V_collection(dir, channel_list=["c0", "c1"], sp_cond_list=["stim"]):


# ディレクトリを選択して処理を開始
root = tkinter.Tk()
root.withdraw()
dir = tkinter.filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate")
df = data_collection(dir)


