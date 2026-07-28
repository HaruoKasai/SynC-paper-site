import pandas as pd
import numpy as np
import os
import tifffile as tiff
from scipy import stats
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams["font.family"] = "Arial"
import matplotlib.pyplot as plt
from read_roi import read_roi_zip, read_roi_file
import glob
import tkinter.filedialog
import tkinter.messagebox
import sys
import pathlib
import re
from collections import Counter


def delta_data_collection(dir, channel_list=["c0", "c1", "c2"]):
    cond_list = sorted(glob.glob(os.path.join(dir, "[!_]*")))  # 条件リストをソート
    print("number of conditions: %s" % len(cond_list))

    # Figure for delta(%).csv data
    fig_delta, axes_delta = plt.subplots(nrows=len(channel_list), ncols=1, figsize=(3, 3 * len(channel_list)))

    # Figure for AU.csv data (for c1 only)
    fig_au, axes_au = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))  # Only for c1

    delta_dataframes = {}  # To store delta data
    au_dataframes = {}  # To store AU data

    # 条件とsp_condの組み合わせごとの色割り当て
    color_map = {
        ("iAS_APV", "stim"): "dodgerblue",
        ("iAS_APV", "neighbor"): "orangered",
        ("iAS_APV+anisomycin", "stim"): "forestgreen",
        ("iAS_APV+anisomycin", "neighbor"): "dimgray",
        ("iAS_APV+lactacystin", "stim"): "black",
        ("iAS_APV+lactacystin", "neighbor"): "gray",
        ("iAS_spine", "stim"): "blue",
        ("iAS_spine", "neighbor"): "dimbrown"
    }

    for cond in cond_list:
        print("cond!!!!   " + cond)
        base_cond_name = os.path.basename(cond)  # 条件名

        # `cond_list` に "iAS_APV" が含まれている場合とその他の条件で `sp_cond_list` を設定
        if base_cond_name == "iAS_APV" or "iAS_spine":
            sp_cond_list = ["stim", "neighbor"]
        else:
            sp_cond_list = ["stim"]

        for sp_cond in sp_cond_list:
            date_list = sorted(glob.glob(os.path.join(cond, "[!_]*")))

            for c in range(len(channel_list)):
                df_list_delta = []  # List for delta(%).csv data
                df_list_au = []  # List for AU.csv data (only for c1)

                for date in date_list:
                    day = os.path.basename(date)[2:6]
                    csv_list_delta = sorted(glob.glob(os.path.join(date, "timeseries_ind", sp_cond,
                                                                   "*" + channel_list[c] + "*delta(%).csv")))
                    print(f"number of dendrites (delta): {len(csv_list_delta)}")

                    if channel_list[c] == 'c1':  # Only look for AU.csv files for c1
                        csv_list_au = sorted(glob.glob(os.path.join(date, "timeseries_ind", sp_cond,
                                                                    "*" + channel_list[c] + "*AU.csv")))
                        print(f"number of dendrites (AU): {len(csv_list_au)}")

                    # Process delta(%).csv files
                    for i, csv in enumerate(csv_list_delta):
                        tp = pd.read_csv(csv).loc[:, ["min"]]
                        if sp_cond == "neighbor":
                            df_ind = pd.read_csv(csv).loc[:, ["average"]]
                            df_ind.rename(columns={'average': day}, inplace=True)
                        else:
                            df_ind = pd.read_csv(csv).loc[:, ["1"]]
                            df_ind.rename(columns={'1': day}, inplace=True)

                        # 中間結果の出力: 各 dendrite のデータを出力
                        intermediate_csv = os.path.join(dir, "_summary",
                                                        f"{base_cond_name}_{sp_cond}_{channel_list[c]}_dendrite_{i}_intermediate_delta.csv")
                        df_ind.to_csv(intermediate_csv, index=False)

                        df_list_delta.append(df_ind)

                    # Process AU.csv files (only for c1)
                    if channel_list[c] == 'c1':
                        for i, csv in enumerate(csv_list_au):
                            tp_au = pd.read_csv(csv).loc[:, ["min"]]
                            if sp_cond =='neighbor':
                                df_ind_au = pd.read_csv(csv).loc[:, ["average"]]
                                df_ind_au.rename(columns={'average': day}, inplace=True)
                            else:
                                df_ind_au = pd.read_csv(csv).loc[:, ["1"]]
                                df_ind_au.rename(columns={'1': day}, inplace=True)

                            # 中間結果の出力: 各 dendrite の AU データを出力
                            intermediate_csv_au = os.path.join(dir, "_summary",
                                                               f"{base_cond_name}_{sp_cond}_{channel_list[c]}_dendrite_{i}_intermediate_au.csv")
                            df_ind_au.to_csv(intermediate_csv_au, index=False)

                            df_list_au.append(df_ind_au)


                # Process and plot delta(%).csv data
                if df_list_delta:
                    df_delta = pd.concat(df_list_delta, axis=1)

                    # Save concatenated delta data
                    concatenated_csv_delta = os.path.join(dir, "_summary",
                                                          f"{base_cond_name}_{sp_cond}_{channel_list[c]}_concatenated_delta.csv")
                    df_delta.to_csv(concatenated_csv_delta, index=False)

                    # Calculate average and standard error
                    df_delta["average"] = df_delta.mean(axis="columns").values
                    df_delta["sem"] = df_delta.sem(axis="columns").values

                    # Add time points
                    df_delta = pd.concat([df_delta, tp], axis=1)

                    # Save averaged delta data
                    averaged_csv_delta = os.path.join(dir, "_summary",
                                                      f"{base_cond_name}_{sp_cond}_{channel_list[c]}_averaged_delta.csv")
                    df_delta.to_csv(averaged_csv_delta, index=False)

                    # Store delta dataframe
                    delta_dataframes[f"{base_cond_name}_{sp_cond}_{channel_list[c]}"] = df_delta

                    # Plot delta data
                    color_key = (base_cond_name, sp_cond)
                    line_color = color_map.get(color_key, "black")
                    ax_delta = df_delta.plot(x="min", y="average", yerr="sem", elinewidth=0.6, ax=axes_delta[c],
                                             legend=False,
                                             linewidth=1.2, color=line_color)
                    ax_delta.axvspan(0, 15, color='gray', alpha=0.3, linewidth=0)
                    ax_delta.spines['right'].set_visible(False)
                    ax_delta.spines['top'].set_visible(False)
                    ax_delta.axhline(y=0, color="gray", ls="dotted", lw=1)

                    if channel_list[c] == 'c0':
                        ax_delta.set_ylabel('ΔV(%)')
                        #ax_delta.set_ylim(-50, 500)
                    elif channel_list[c] == 'c1':
                        ax_delta.set_ylabel('ΔiAS(%)')
                        #ax_delta.set_ylim(-100, 3500)
                        # inset_ax = inset_axes(ax_delta, width='40%', height='40%', loc="lower right")
                        # # インセットにデータをプロット
                        # inset_ax.plot(df_delta["min"], df_delta["average"], color=line_color, linewidth=1.0)
                        #
                        # # インセット内の範囲設定
                        # inset_ax.set_xlim(-60, 360)
                        # inset_ax.set_ylim(-20, 100)
                        #
                        # # インセットの美観調整
                        # inset_ax.axvspan(0, 15, color='gray', alpha=0.3, linewidth=0)
                        # inset_ax.axhline(y=0, color="gray", ls="dotted", lw=1)
                        # inset_ax.tick_params(axis='both', which='both', length=2, labelsize=8)
                        # inset_ax.spines['right'].set_visible(False)
                        # inset_ax.spines['top'].set_visible(False)

                    elif channel_list[c] == 'c2':
                        ax_delta.set_ylabel('ΔiAS(%)/ΔV(%)')

                    ax_delta.set_xlim(-60, 360)
                    ax_delta.set_xticks(np.arange(-60, 360, 60))

                # Process and plot AU.csv data (for c1 only)
                if channel_list[c] == 'c1' and df_list_au:
                    df_au = pd.concat(df_list_au, axis=1)

                    # Save concatenated AU data
                    concatenated_csv_au = os.path.join(dir, "_summary",
                                                       f"{base_cond_name}_{sp_cond}_{channel_list[c]}_concatenated_au.csv")
                    df_au.to_csv(concatenated_csv_au, index=False)

                    # Calculate average and standard error for AU data
                    df_au["average"] = df_au.mean(axis="columns").values
                    df_au["sem"] = df_au.sem(axis="columns").values

                    # Add time points for AU data
                    df_au = pd.concat([df_au, tp_au], axis=1)

                    # Save averaged AU data
                    averaged_csv_au = os.path.join(dir, "_summary",
                                                   f"{base_cond_name}_{sp_cond}_{channel_list[c]}_averaged_au.csv")
                    df_au.to_csv(averaged_csv_au, index=False)

                    # Store AU dataframe
                    au_dataframes[f"{base_cond_name}_{sp_cond}_{channel_list[c]}"] = df_au

                    # Plot AU data on a separate figure
                    line_color_au = color_map.get(color_key, "orange")  # Different color for AU
                    ax_au = df_au.plot(x="min", y="average", yerr="sem", elinewidth=0.6, ax=axes_au, legend=False,
                                       linewidth=1.2, color=line_color_au)
                    ax_au.axvspan(0, 15, color='gray', alpha=0.3, linewidth=0)
                    ax_au.spines['right'].set_visible(False)
                    ax_au.spines['top'].set_visible(False)
                    ax_au.axhline(y=0, color="gray", ls="dotted", lw=1)
                    ax_au.set_ylabel('iAS(a.u)')
                    ax_au.set_xlim(-60, 360)
                    ax_au.set_xticks(np.arange(-60, 360, 60))


    # Save delta figure
    fig_delta.tight_layout()
    fig_delta.savefig(os.path.join(dir, "_summary", "_timeseries_group_neighbor_APV_only_delta.pdf"), dpi=300,
                      transparent=True)


    # Save AU figure
    fig_au.tight_layout()
    fig_au.savefig(os.path.join(dir, "_summary", "_timeseries_group_neighbor_APV_only_au.pdf"), dpi=300,
                   transparent=True)

    # Return both delta and AU dataframes as a dictionary
    return {
        "delta": delta_dataframes,
        "au": au_dataframes
    }



def delta_data_shaft(dir, channel_list=["c0", "c1", "c2"]):
    cond_list = sorted(glob.glob(os.path.join(dir, "[!_]*")))  # 条件リストをソート
    print("number of conditions: %s" % len(cond_list))
    fig, axes = plt.subplots(nrows=len(channel_list), ncols=1, figsize=(3, 3 * len(channel_list)))

    # 条件とsp_condの組み合わせごとの色割り当て
    color_map = {
        ("iAS_APV", "shaft"): "dodgerblue",
        ("iAS_APV+anisomycin", "shaft"): "forestgreen",
        ("iAS_APV+lactacystin", "shaft"): "black"
    }

    for cond in cond_list:
        print("cond!!!!   " + cond)
        base_cond_name = os.path.basename(cond)  # 条件名

        sp_cond_list = ["shaft"]
        for sp_cond in sp_cond_list:
            date_list = sorted(glob.glob(os.path.join(cond, "[!_]*")))

            for c in range(len(channel_list)):
                df_list = []
                for date in date_list:
                    day = os.path.basename(date)[2:6]
                    csv_list = sorted(glob.glob(os.path.join(date, "timeseries_ind", sp_cond,
                                                             "*" + channel_list[c] + "*delta(%).csv")))
                    print("number of dendrites: %s" % len(csv_list))

                    # 各csvファイルのデータを読み込み、結合
                    for i, csv in enumerate(csv_list):
                        tp = pd.read_csv(csv).loc[:, ["min"]]
                        if sp_cond == "neighbor":
                            df_ind = pd.read_csv(csv).loc[:, ["average"]]
                            df_ind.rename(columns={'average': day}, inplace=True)
                        else:
                            df_ind = pd.read_csv(csv).loc[:,["1"]]
                            df_ind.rename(columns={'1': day}, inplace=True)


                        # 中間結果の出力: 各 dendrite のデータを出力
                        intermediate_csv = os.path.join(dir, "_summary", f"{base_cond_name}_{sp_cond}_{channel_list[c]}_dendrite_{i}_intermediate.csv")
                        df_ind.to_csv(intermediate_csv, index=False)

                        df_list.append(df_ind)

                # 各チャンネルの全日付データを結合
                df_shaft = pd.concat(df_list, axis=1)

                # 中間結果の出力: 結合されたデータを保存
                concatenated_csv = os.path.join(dir, "_summary", f"{base_cond_name}_{sp_cond}_{channel_list[c]}_concatenated.csv")
                df_shaft.to_csv(concatenated_csv, index=False)

                # 平均と標準誤差を計算
                df_shaft["average"] = df_shaft.mean(axis="columns").values
                df_shaft["sem"] = df_shaft.sem(axis="columns").values

                # タイムポイントを追加
                df_shaft = pd.concat([df_shaft, tp], axis=1)

                # 中間結果の出力: 平均と標準誤差が計算されたデータを保存
                averaged_csv = os.path.join(dir, "_summary", f"{base_cond_name}_{sp_cond}_{channel_list[c]}_averaged.csv")
                df_shaft.to_csv(averaged_csv, index=False)

                # グラフ描画
                color_key = (base_cond_name, sp_cond)  # cond_list と sp_cond_list の組み合わせ
                color = color_map.get(color_key, "black")  # 一致していれば指定色、一致していなければ黒を使う
                line_color = color_map.get(color_key, "black")  # 同じ条件が揃った時のみ揃える

                ax = df_shaft.plot(x="min", y="average", yerr="sem", elinewidth=0.6, ax=axes[c], legend=False,
                             linewidth=1.2, color=line_color)  # 線の色
                ax.axvspan(0, 15, color='gray', alpha=0.3, linewidth=0)  # 塗りつぶしの色
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.axhline(y=0, color="gray", ls="dotted", lw=1)
                if channel_list[c] == 'c0':
                    ax.set_ylabel('ΔV (%)')
                elif channel_list[c] == 'c1':
                    ax.set_ylabel('ΔiAS (%)')
                elif channel_list[c] == 'c2':
                    ax.set_ylabel('ΔiAS (%)/ΔV (%)')

                ax.set_title("shaft_signal")
                ax.set_xlim(-60, 360)
                ax.set_xticks(np.arange(-60, 360, 60))

    fig.tight_layout()
    # グループ全体の折れ線グラフを保存
    fig.savefig(os.path.join(dir, "_summary", "_timeseries_group_shaft.pdf"), dpi=300, transparent=True)

    return df_shaft
# #def run_one_way_anova_and_save_combined_csv(dir, channel_list=["c0", "c1", "c2"]):
# #    for channel in channel_list:
#         # 各チャンネルの `concatenated.csv` ファイルを取得
# #        csv_files = sorted(glob.glob(os.path.join(dir, "_summary", f"*_{channel}_concatenated.csv")))
#
#         # 全データを結合するためのリスト
# #        combined_data = []
#
# #        for csv_file in csv_files:
#             # CSVファイルの読み込み
#             df = pd.read_csv(csv_file)
#
#             # 日付のヘッダーをグループラベルとして使用
#             date_columns = df.columns  # すべての日付列を取得
#
#             # グループ化されたデータを準備
#             group_data = [df[date].dropna().values for date in date_columns]  # 各日付のデータをリストに格納
#
#             # 各データフレームにラベルを付けて結合用にリストに追加
#             df["channel"] = channel  # チャンネル名を新しい列に追加
#             df["source_file"] = os.path.basename(csv_file)  # 元のファイル名を記録
#             combined_data.append(df)
#
#             # One-way ANOVA を実施
#             f_value, p_value = stats.f_oneway(*group_data)
#
#             # 結果を表示
#             print(f"ANOVA results for channel {channel} ({os.path.basename(csv_file)}):")
#             print(f"F-value: {f_value}")
#             print(f"P-value: {p_value}")
#
#             # P値の解釈
#             if p_value < 0.05:
#                 print("有意な差があります。")
#             else:
#                 print("有意な差はありません。")
#
#         # すべてのデータフレームを結合
#         combined_df = pd.concat(combined_data)
#
#         # 結合されたデータをCSVに書き出す
#         output_csv_path = os.path.join(dir, f"combined_data_{channel}.csv")
#         combined_df.to_csv(output_csv_path, index=False)
#         print(f"Combined CSV saved: {output_csv_path}")
#




root = tkinter.Tk()
root.withdraw()
dir = tkinter.filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate")
df = delta_data_collection(dir)
dV = df["delta"]["iAS_APV_stim_c0"].iloc[3:7, :df["delta"]["iAS_APV_stim_c0"].columns.get_loc('average')]  #iAS_APVとiAS_spineの際書き換え
diAS = df["delta"]["iAS_APV_stim_c1"].iloc[3:7, :df["delta"]["iAS_APV_stim_c1"].columns.get_loc('average')]
au_iAS = df["au"]["iAS_APV_stim_c1"].iloc[3:7, :df["au"]["iAS_APV_stim_c1"].columns.get_loc('average')]

print(dV)
print(diAS)
print(au_iAS)

fig, axes = plt.subplots(2, 4, figsize=(12,6))

for row_index in range(dV.shape[0]):
    ax = axes[0, row_index % 4]
    dV_row = dV.iloc[row_index, :]
    diAS_row = diAS.iloc[row_index, :]
    ax.scatter(dV_row, diAS_row, marker='.', s=200, edgecolor='none', label="dV vs diAS", alpha=0.7)
    ax.set_xlabel("ΔV")
    ax.set_ylabel("ΔiAS(%)")

for row_index in range(dV.shape[0]):
    ax = axes[1, row_index % 4]
    dV_row = dV.iloc[row_index, :]
    au_iAS_row = au_iAS.iloc[row_index, :]
    ax.scatter(dV_row, au_iAS_row, marker='.', s=200, edgecolor='none', label="dV vs au_iAS", alpha=0.7)

    ax.set_xlabel("ΔV")
    ax.set_ylabel("iAS(a.u)")
    #ax.set_ylim(-500, 6000)
    if row_index % 4 == 0:
        ax.set_xlim(0,900)
    elif row_index % 4 == 1 or row_index % 4 ==2:
        ax.set_xlim(0,700)
    else:
        ax.set_xlim(0,500)


fig.tight_layout()
fig.savefig(os.path.join(dir, "_summary", "dV_vs_iAS.pdf"), dpi=300, transparent=True)

df_shaft = delta_data_shaft(dir)
# フォルダを指定してOne-way ANOVAを実行し、データを結合して保存
#run_one_way_anova_and_save_combined_csv(dir)

