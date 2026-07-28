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
from scipy.stats import skew, kurtosis
import re
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

#TODO 全体関数化して整理しないと

hotspot_index_avg ={}

def calculate_as_score(spine_csv, back_csv, cell_count, cond, probe_name):
    dir = os.path.dirname(os.path.dirname(spine_csv))
    fname = os.path.basename(spine_csv)
    spine_df = pd.read_csv(spine_csv)
    spine_df =spine_df[spine_df['dendrite'].notna()].reset_index(drop=True)
    back_df = pd.read_csv(back_csv)[:len(spine_df)]

    # 'area' 列を優先し、存在しない場合は 'area_in_pixel' を使用
    area_column = 'area' if 'area' in spine_df.columns else 'area_in_pixel'
    area = spine_df[area_column].values

    label = spine_df['label']
    sp_minus_back_x_area = (spine_df['mean_intensity-0'].values - back_df['mean_intensity-0']) * area
    sp_minus_back_y_area = (spine_df['mean_intensity-1'].values - back_df['mean_intensity-1']) * area
    dendrite = spine_df['dendrite'] #TODO

    #scoreの値を計算
    AS_filler_ratio = sp_minus_back_y_area / sp_minus_back_x_area

    # 体積は平均で正規化

    X_normalized = sp_minus_back_x_area / np.mean(sp_minus_back_x_area)
    # データフレームの作成

    df = pd.DataFrame({
        'spine_label': label.values.flatten(),  # labelもflattenしておく
        'AS/filler': AS_filler_ratio,
        'filler sum': sp_minus_back_x_area,
        'mVenus sum': sp_minus_back_y_area,
        'probe': probe_name,
        'condition': cond,
        'cell': ["cell"+str(cell_count)]*len(AS_filler_ratio),
        'dendrite': dendrite
    })

    # データフレームをCSVファイルに保存
    as_csv_path = os.path.join(dir, f'AS_filler_ratio_{fname[:-9]}.csv')
    df.to_csv(as_csv_path, index=False)

    return AS_filler_ratio, sp_minus_back_y_area, X_normalized, df

def gini_coefficient(x):
    sorted_x = sorted(x)
    n = len(x)
    gini = (2 * sum([(i+1) * sorted_x[i] for i in range(n)]) - (n + 1) * sum(sorted_x)) / (n * sum(sorted_x))
    return gini

def calculate_index (input_df, col_name, probe_name, index_df, index_type="hot_spot"):
    global hotspot_index_avg
    cell_list = input_df['cell'].unique().tolist()
    for cell in cell_list:
        df = input_df[input_df['cell'] == cell]
        dend_list = df['dendrite'].unique().tolist()
        for dend in dend_list:
            df_dend = df[df['dendrite'] == dend]
            if index_type=="hot_spot":
                differences = df_dend[col_name].diff()
                differences.iloc[0] = df_dend[col_name].iloc[-1] - df_dend[col_name].iloc[0]
                hotspot_index = differences.abs().mean()
                print(hotspot_index)
                if probe_name not in hotspot_index_avg:
                    hotspot_index_avg[probe_name] = []
                hotspot_index_avg[probe_name].append(hotspot_index)
                index_df.loc[len(index_df)] = [hotspot_index, probe_name, cell, dend]
            if index_type=="skewness":
                skewness = skew(df_dend[col_name])
                index_df.loc[len(index_df)] = [skewness, probe_name, cell, dend]
            if index_type=="kurtosis":
                kurto = kurtosis(df_dend[col_name])
                index_df.loc[len(index_df)] = [kurto, probe_name, cell, dend]
            if index_type=="gini":
                gini = gini_coefficient(df_dend[col_name])
                index_df.loc[len(index_df)] = [gini, probe_name, cell, dend]
            if re.match(r"percentage_above_\d+SD", index_type):
                s = float(index_type.split("_")[2].split("SD")[0])
                mean = df_dend[col_name].mean()
                std = df_dend[col_name].std()
                count = ((df_dend[col_name]) > (mean + s * std)).sum()
                percentage = count / len(df_dend[col_name]) * 100
                index_df.loc[len(index_df)] = [percentage, probe_name, cell, dend]
            if index_type == "above_2SD_mean":
                mean = df_dend[col_name].mean()
                std = df_dend[col_name].std()
                above_2SD_values = df_dend[col_name][df_dend[col_name] > (mean + 2 * std)]
                mean_above_2SD = above_2SD_values.mean() if not above_2SD_values.empty else np.nan
                index_df.loc[len(index_df)] = [mean_above_2SD, probe_name, cell, dend]
            if index_type=="SD":
                std = df_dend[col_name].std()
                index_df.loc[len(index_df)] = [std, probe_name, cell, dend]






def plot_graph (df, position, title):
    ax = fig.add_subplot(gs[(position)])
    sns.boxplot(x='probe', y='index_val', data=df, ax=ax, showfliers=False)
    sns.stripplot(x='probe', y='index_val', data=df, ax=ax, color='black', alpha=0.6, size=2, jitter=True)

    ax.set_title(title)
    #ax.set_ylim(-2.5, 12)
    ax.set_xlabel('Probe')

    if position == (0,6):
        ax.set_ylim(-1, 25)
        ax.set_ylabel("z-score")
    elif position == (4,6):
        ax.set_ylim(-5, 30)
    elif position == (6,6):
        ax.set_ylabel("z-score")
        ax.set_ylim(-5, 100)
    else:
        ax.set_ylabel('Index')

################################################################


root = tk.Tk()
root.withdraw()
probe_list = glob.glob(os.path.join(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\distribution", "[!_]*"))

fig = plt.figure(figsize=(50, 40))
gs = gridspec.GridSpec(7, 7, width_ratios=[1,1, 1, 1, 1, 1,1])
plt.rc('font', size=20)

df_all = df = pd.DataFrame()
for p, probe in enumerate(probe_list):
    dir = os.path.join(probe, "boxplot")
    spine_csv_files = glob.glob(os.path.join(dir, "*spine.csv"))
    back_csv_files = [spine_csv[:-9] + "back.csv" for spine_csv in spine_csv_files]
    print(os.path.basename(probe))
    print("Number of files: %s" % len(spine_csv_files))

    control_as_scores = []
    apv_as_scores = []
    as_Y = []
    apv_as_Y = []
    control_X_normalized = []
    apv_X_normalized = []

    ctrl_count, apv_count =0,0

    for spine_csv, back_csv in zip(spine_csv_files, back_csv_files):
        # controlとapvを分ける
        if "control" in spine_csv.lower():
            ctrl_count += 1
            AS_filler_ratio, scatter_y, X_norm, df= calculate_as_score(spine_csv, back_csv, cell_count = ctrl_count, cond = "Ctrl", probe_name = os.path.basename(probe))
            control_as_scores.extend(AS_filler_ratio)
            as_Y.extend(scatter_y)
            control_X_normalized.extend(X_norm)
            df_all = pd.concat([df_all, df], ignore_index=True)


        elif "apv" in spine_csv.lower():
            apv_count += 1
            AS_filler_ratio, scatter_y, X_norm, df= calculate_as_score(spine_csv, back_csv, cell_count = apv_count, cond = "APV", probe_name = os.path.basename(probe))
            apv_as_scores.extend(AS_filler_ratio)
            apv_as_Y.extend(scatter_y)
            apv_X_normalized.extend(X_norm)
            df_all = pd.concat([df_all, df], ignore_index=True)


    print("ctrl_count="+str(ctrl_count))
    print("apv_count="+str(apv_count))
    # prepare DataFrame for plotting and CSV writing
    control_as_scores_df = pd.DataFrame(np.array(control_as_scores).flatten(), columns=['AS/filler'])
    control_as_scores_df['Group'] = 'control'
    apv_as_scores_df = pd.DataFrame(np.array(apv_as_scores).flatten(), columns=['AS/filler'])
    apv_as_scores_df['Group'] = 'APV'

    control_as_scores_df.to_csv(os.path.join(dir, 'control_as_scores.csv'), index=False)
    apv_as_scores_df.to_csv(os.path.join(dir, 'APV_as_scores.csv'), index=False)

    combined_as_df = pd.concat([apv_as_scores_df, control_as_scores_df])

    s=5 #scatter point size

    ax = fig.add_subplot(gs[(p, 0)])
    ax.scatter(apv_X_normalized, apv_as_Y, s=s, alpha=0.5, label='APV')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x / 1000:.1f}k'))
    ax.set_xlabel('V (a.u)')
    ax.set_ylabel('AS sum '+ os.path.basename(probe))
    ax.set_ylim(-1000, 100000)  # Adjust the y-axis limit as needed
    ax.set_title('APV AS_sum')
    ax.set_xlim(0, 6)

    ax = fig.add_subplot(gs[(p, 1)])
    ax.scatter(control_X_normalized, as_Y, s=s,alpha=0.5, label='Control')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x / 1000:.1f}k'))
    ax.set_xlabel('V (a.u)')
    ax.set_ylabel('AS sum ' + os.path.basename(probe))
    ax.set_ylim(-1000, 100000)  # Adjust the y-axis limit as needed
    ax.set_title('NoAPV AS_sum')
    ax.set_xlim(0, 6)


    ax = fig.add_subplot(gs[(p, 2)])
    ax.scatter(apv_X_normalized, apv_as_scores, s=s,alpha=0.5, label='APV')
    ax.set_xlabel('V (a.u)')
    ax.set_ylabel('AS/filler')
    ax.set_ylim(-0.01, 0.05)
    ax.set_title('APV AS/filler')
    ax.set_xlim(0, 6)

    ax = fig.add_subplot(gs[(p, 3)])
    ax.scatter(control_X_normalized, control_as_scores, s=s,alpha=0.5, label='Control')
    ax.set_xlabel('V (a.u)')
    ax.set_ylabel('AS/filler')
    ax.set_ylim(-0.01, 0.05)
    ax.set_title('NoAPV AS/filler')
    ax.set_xlim(0, 6)


    # Plot AS box plot and strip plot
    ax = fig.add_subplot(gs[(p, 4)])
    sns.boxplot(x='Group', y='AS/filler', data=combined_as_df, whis=np.inf, linewidth=1.5, ax=ax)
    sns.stripplot(x='Group', y='AS/filler', data=combined_as_df, jitter=True, color='black',
                  alpha=0.5, ax=ax)
    ax.set_ylabel(os.path.basename(probe) + " AS/filler")
    ax.set_ylim(-0.02, 0.1)

    # Z-score defined by APV distribution
    APVmean = apv_as_scores_df['AS/filler'].mean()
    APVstd = apv_as_scores_df['AS/filler'].std()
    control_as_scores_df["z-score"] = (control_as_scores_df['AS/filler']) / APVstd
    ax = fig.add_subplot(gs[(p, 5)])
    sns.boxplot(x='Group', y='z-score', data=control_as_scores_df, whis=np.inf, linewidth=1.5, ax =ax)
    sns.stripplot(x='Group', y='z-score', data=control_as_scores_df, jitter=True, color='black', alpha=0.5, ax=ax)
    ax.set_ylabel("Z Score (defined by APV group distribution)")
    ax.set_ylim(-10, 200)


df_all.to_csv (os.path.join(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\distribution","_all_data.csv"))
probe_list = df_all['probe'].unique().tolist()
scoring = "AS/filler"
hotspot_index_df = pd.DataFrame(columns=["index_val", "probe", "cell", "dendrite"])
skewness_df = pd.DataFrame(columns=["index_val", "probe", "cell", "dendrite"])
kurtosis_df = pd.DataFrame(columns=["index_val", "probe", "cell", "dendrite"])
gini_df = pd.DataFrame(columns=["index_val", "probe", "cell", "dendrite"])
SD_df =pd.DataFrame(columns=["index_val", "probe", "cell", "dendrite"])
percentage_2sd_df =pd.DataFrame(columns=["index_val", "probe", "cell", "dendrite"])
above_2SD_mean_df =pd.DataFrame(columns=["index_val", "probe", "cell", "dendrite"])
zscore_combined_df = pd.DataFrame()

for probe in probe_list:
    df = df_all[df_all['probe'] == probe]
    APV_df = df[df['condition'] == "APV"]
    Ctrl_df = df[df['condition'] == "Ctrl"]
    APVmean = APV_df[scoring].mean()
    APVstd = APV_df[scoring].std()
    Ctrl_df["z_score"] = (Ctrl_df[scoring]) / APVstd #TODO　上と2回連続で計算しているので直す
    zscore_combined_df = pd.concat([zscore_combined_df, Ctrl_df[['z_score', 'probe', 'cell', 'dendrite']]], ignore_index=True)
    calculate_index (Ctrl_df, col_name = "z_score", probe_name=probe, index_df=hotspot_index_df, index_type="hot_spot")
    calculate_index(Ctrl_df, col_name="z_score", probe_name=probe, index_df=skewness_df, index_type="skewness")
    calculate_index(Ctrl_df, col_name="z_score", probe_name=probe, index_df=kurtosis_df, index_type="kurtosis")
    calculate_index(Ctrl_df, col_name="z_score", probe_name=probe, index_df=gini_df, index_type="gini")
    calculate_index(Ctrl_df, col_name="z_score", probe_name=probe, index_df=SD_df,index_type="SD")
    calculate_index(Ctrl_df, col_name="z_score", probe_name=probe, index_df=percentage_2sd_df, index_type="percentage_above_2SD")
    calculate_index(Ctrl_df, col_name="z_score", probe_name=probe, index_df=above_2SD_mean_df,index_type="above_2SD_mean")

for probe in hotspot_index_avg:
    avg_hotspot_index = np.mean(hotspot_index_avg[probe])
    print(f"{probe}: {avg_hotspot_index}")

plot_graph(hotspot_index_df, position=(0,6), title="Hot spot index")
plot_graph(skewness_df, position=(1,6), title="Skewness")
plot_graph(kurtosis_df, position=(2,6), title="Kurtosis")
plot_graph(gini_df, position=(3,6), title="Gini coefficient")
plot_graph(SD_df, position=(4,6), title="SD")
plot_graph(percentage_2sd_df, position=(5,6), title="Percentage of spines above 2SD")
plot_graph(above_2SD_mean_df, position=(6,6), title="above_2SD_mean")
plot_graph(zscore_combined_df.rename(columns={'distributionratio': 'index_val'}), position=(6,5), title="Z-score")


hotspot_index_df.to_csv(os.path.join(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\distribution", "_hotspot_index.csv"), index=False)
zscore_combined_df.to_csv(os.path.join(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\distribution", "_z_score.csv"), index=False)
SD_df.to_csv(os.path.join(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\distribution", "_standard_deviation.csv"), index=False)





plt.tight_layout()
fig.savefig(os.path.join(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\distribution","_summary.pdf"), dpi=300, transparent=True)
