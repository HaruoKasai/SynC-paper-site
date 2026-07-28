import pandas as pd
import numpy as np
import os
from sklearn import linear_model
import glob
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.ticker as ticker

#TODO 全体関数化して整理しないと

def calculate_as_score(spine_csv, back_csv, cell_count, cond):
    dir = os.path.dirname(os.path.dirname(spine_csv))
    fname = os.path.basename(spine_csv)
    spine_df = pd.read_csv(spine_csv)
    back_df = pd.read_csv(back_csv)

    # Drop NaN values
    spine_df = spine_df.dropna()
    back_df = back_df.dropna()

    # 'area' 列を優先し、存在しない場合は 'area_in_pixel' を使用
    area_column = 'area' if 'area' in spine_df.columns else 'area_in_pixel'

    # 選択された列を使用
    area = spine_df[area_column].values

    label = spine_df['label']
    sp_minus_back_x_area = (spine_df['mean_intensity-0'].values - back_df['mean_intensity-0'].values) * area
    sp_minus_back_y_area = (spine_df['mean_intensity-1'].values - back_df['mean_intensity-1'].values) * area

    # 最終行の値を保存（dendrite）
    #last_x_value = sp_minus_back_x_area[-1]
    #last_y_value = sp_minus_back_y_area[-1]

    # 最終行を除外 (dendrite)
    sp_minus_back_x_area = sp_minus_back_x_area[:-1]
    sp_minus_back_y_area = sp_minus_back_y_area[:-1]

    #scoreの値を計算
    AS_filler_ratio = sp_minus_back_y_area / sp_minus_back_x_area

    # 0.01刻みのbinを作成
    bin_edges = np.arange(0, AS_filler_ratio.max() + 0.01, 0.01)
    bin = pd.cut(AS_filler_ratio, bins=bin_edges)
    #最頻値のbinを取得
    mode_bin = pd.Series(bin).mode()[0]
    #最頻値binの下限値を取得
    mode_bin_lower_bound = float(str(mode_bin).split(",")[0][1:])

    AS_filler_ratio_mode_subtracted = AS_filler_ratio - mode_bin_lower_bound

    # 体積は平均で正規化
    X_normalized = sp_minus_back_x_area / np.mean(sp_minus_back_x_area)
    AS_normV_ratio = sp_minus_back_y_area / X_normalized

    # データフレームの作成
    df = pd.DataFrame({
        'label': label[:-1].values.flatten(),  # labelもflattenしておく
        'AS/filler': AS_filler_ratio,
        'AS/filler (mode-subtracted)': AS_filler_ratio_mode_subtracted.flatten(),  # 1次元に変換
        'filler sum': sp_minus_back_x_area,
        'mVenus sum': sp_minus_back_y_area,
        'bin': bin,
        'mode_bin': mode_bin_lower_bound,
        'condition': cond,
        'cell': "cell"+str(cell_count)
    })

    # データフレームをCSVファイルに保存
    as_csv_path = os.path.join(dir, f'subtract_as_scores_{fname[:-9]}.csv')
    df.to_csv(as_csv_path, index=False)

    return AS_filler_ratio_mode_subtracted, AS_filler_ratio, AS_normV_ratio, X_normalized, sp_minus_back_y_area, df




################################################################

root = tk.Tk()
root.withdraw()
day_list = glob.glob(os.path.join(r"U:\Probe_paper_2023\dissociate\c-fos_iAS_correlation", "[!_]*"))

fig = plt.figure(figsize=(50, 40))
gs = gridspec.GridSpec(3, 6, width_ratios=[1,1, 1, 1, 1,1])
plt.rc('font', size=20)

df_all = df = pd.DataFrame()
for p, day in enumerate(day_list):
    dir = os.path.join(day, "AS_filler_ratio")
    spine_csv_files = glob.glob(os.path.join(dir, "*spine.csv"))
    back_csv_files = [spine_csv[:-9] + "back.csv" for spine_csv in spine_csv_files]
    print(os.path.basename(day))
    print("Number of files: %s" % len(spine_csv_files))

    control_as_scores = []

    control_as_normV_ratio = []


    control_subtract_as_scores = []


    control_X_normalized = []
    as_Y = []



    ctrl_count = 0

    for spine_csv, back_csv in zip(spine_csv_files, back_csv_files):
        ctrl_count += 1
        AS_filler_ratio_mode_subtracted, AS_filler_ratio, AS_normV_ratio, X_norm, scatter_y, df= calculate_as_score(spine_csv, back_csv, cell_count = ctrl_count, cond = "Ctrl")
        control_subtract_as_scores.extend(AS_filler_ratio_mode_subtracted)
        control_as_scores.extend(AS_filler_ratio)
        control_as_normV_ratio.extend(AS_normV_ratio)
        control_X_normalized.extend(X_norm)
        as_Y.extend(scatter_y)
        df_all = pd.concat([df_all, df], ignore_index=True)

    print("ctrl_count="+str(ctrl_count))
    # prepare DataFrame for plotting and CSV writing
    control_subtract_as_scores_df = pd.DataFrame(np.array(control_subtract_as_scores).flatten(), columns=['AS/filler (mode-subtracted)'])
    control_subtract_as_scores_df['Group'] = 'control'


    control_subtract_as_scores_df.to_csv(os.path.join(dir, 'control_subtract_as_scores.csv'), index=False)


    # Combine DataFrames for plot and reset index to ensure unique indices
    #combined_as_df = pd.concat([apv_subtract_as_scores_df, control_subtract_as_scores_df])


    s=5 #scatter point size


    ax = fig.add_subplot(gs[(p, 1)])
    ax.scatter(control_X_normalized, as_Y, s=s,alpha=0.5, label='Control')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x / 1000:.1f}k'))
    ax.set_xlabel('V (a.u)')
    ax.set_ylabel('AS sum ' + os.path.basename(day))
    ax.set_ylim(-1000, 100000)  # Adjust the y-axis limit as needed
    ax.set_title('NoAPV AS_sum')
    ax.set_xlim(0, 6)



    ax = fig.add_subplot(gs[(p, 2)])
    ax.scatter(control_X_normalized, control_subtract_as_scores, s=s,alpha=0.5, label='Control')
    ax.set_xlabel('V (a.u)')
    ax.set_ylabel('AS/filler (mode-subtracted)')
    ax.set_ylim(-0.1,1.2)
    ax.set_title('NoAPV AS/filler (mode-subtracted)')
    ax.set_xlim(0, 6)



    ax = fig.add_subplot(gs[(p, 3)])
    ax.scatter(control_X_normalized, control_as_scores, s=s,alpha=0.5, label='Control')
    ax.set_xlabel('V (a.u)')
    ax.set_ylabel('AS/filler')
    ax.set_ylim(-0.02, 1.2)
    ax.set_title('NoAPV AS/filler')
    ax.set_xlim(0, 6)




    ax = fig.add_subplot(gs[(p, 4)])
    ax.scatter(control_X_normalized, control_as_normV_ratio, s=s, alpha=0.5, label='Control')
    ax.set_xlabel('V (a.u)')
    ax.set_ylabel('AS/norm V')
    # ax.set_ylim(-0.1, 1.2)
    ax.set_title('NoAPV AS/norm V')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 12000)

    # Plot AS box plot and strip plot
    ax = fig.add_subplot(gs[(p, 5)])
    sns.boxplot(x='Group', y='AS/filler (mode-subtracted)', data=control_subtract_as_scores_df, whis=np.inf, linewidth=1.5, ax=ax)
    sns.stripplot(x='Group', y='AS/filler (mode-subtracted)', data=control_subtract_as_scores_df, jitter=True, color='black',
                  alpha=0.5, ax=ax)
    ax.set_ylabel(os.path.basename(day) + " AS/filler (mode-subtracted)")
    ax.set_ylim(-0.2, 1.2)













plt.tight_layout()
fig.savefig(os.path.join(r"U:\Probe_paper_2023\dissociate\c-fos_iAS_correlation","_summary.pdf"), dpi=300, transparent=True)
