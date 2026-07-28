import pandas as pd
import numpy as np
import os
import re
from sklearn import linear_model
import glob
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.ticker as ticker
import h5py

def extract_cell_name(cell_path):
    basename = os.path.basename(cell_path)
    match = re.search(r"N\d+", basename)  # "N"に続く数字を検索
    if match:
        return match.group(0)  # "N" + 数字 の部分を返す
    else:
        return None  # マッチしない場合はNoneを返す

def calculate_AS_filler_ratio(back_filler, back_AS, prop_path, date, cell_name, df_all):
    df = pd.read_csv(prop_path)
    df = df[df['analysis'] == True]
    df = df[['label', 'area', 'mean_intensity-0', 'mean_intensity-1']][:-1]


    area = df["area"].values
    label = df['label']
    filler = (df['mean_intensity-0'].values - back_filler) * area  # 最終行を除外 (dendrite)
    AS = (df['mean_intensity-1'].values - back_AS) * area

    #scoreの値を計算
    AS_filler_ratio = AS / filler

    # 体積は平均で正規化
    filler_norm = filler / np.mean(filler)

    part_name = prop_path.split('_region_')[1].split('.tif')[0]

    df_to_add = pd.DataFrame({
        'date': date,
        'cell': cell_name,
        'part': part_name,
        'label': label.values.flatten(),  # labelもflattenしておく
        'area' : area,
        'filler': filler,
        'filler_mean': np.mean(filler),
        'filler_norm': filler_norm,
        'AS': AS,
        'AS_filler_ratio': AS_filler_ratio,
    })
    df_all = pd.concat([df_all, df_to_add], ignore_index=True)
    return df_all

def get_background(h5_path, image):
    with h5py.File(h5_path, 'r') as hdf:
        bg_mean = hdf["images/"+os.path.basename(image)+"/bg_mean"] #240708_3channel_c-fos_iAS_correlation_N22_al_unmix_region_right_upper.tif/bg_mean"][:])
        filler_back = bg_mean[:][0]
        AS_back = bg_mean[:][1]
        return filler_back, AS_back

def plot_AS_filler_ratio(df, gs, ax, threshold):
    ax = fig.add_subplot(gs[ax])
    cell_name = df["cell"].unique()[0]
    X = df["filler_norm"].values
    Y = df["AS_filler_ratio"].values
    s=8
    ax.axhline(y=threshold, color='k', linestyle='--')
    ax.scatter(X,Y,s=s,alpha=0.8)
    ax.set_xlabel('V (a.u)')
    ax.set_ylabel('AS/filler')
    ax.set_ylim(-0.005,0.08)
    ax.set_title(cell_name)
    ax.set_xlim(0, 6)

def analyse_AS_positive (date, cell, threshold, df_cell, df_soma, df_neurons):
    as_positive_df = df_cell[df_cell['AS_filler_ratio'] > threshold]
    ASpositive_intensity_sum = as_positive_df['AS_filler_ratio'].sum()
    ASpositive_percentage = len(as_positive_df) / len(df_cell) *100
    df_soma['date'] = df_soma['date'].astype(str)
    df_temp = df_soma[(df_soma['date'] == date) & (df_soma['cell'] == cell)]
    df_to_add = pd.DataFrame({ #'date', 'cell', 'AS(+)_intensity_sum', 'AS(+)_percentage', 'AS_soma', 'c-fos_nucleus', 'c_fos_nucleus_norm'
        'date': date,
        'cell': cell,
        'AS(+)_intensity_sum': ASpositive_intensity_sum,
        'AS(+)_percentage': ASpositive_percentage,
        'AS_soma': df_temp['mVenus'].values,
        'filler_soma':df_temp['filler'].values,
        'AS_filler_ratio': df_temp['mvenus/filler'].values,
        'c-fos_nucleus': df_temp['c-fos_nucleus'].values,
        'c_fos_nucleus_norm_max': df_temp['c-fos_nucleus_norm_max'].values
    })
    df_neurons = pd.concat([df_neurons, df_to_add], ignore_index=True)
    return df_neurons
################################################################



# fig = plt.figure(figsize=(50, 40))
# gs = gridspec.GridSpec(3, 6, width_ratios=[1,1, 1, 1, 1,1])
# plt.rc('font', size=20)

h5_path = r"\\Synology\arima\hdf5\as_proj.h5"
soma_data_path = r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_c-fos_iAS_correlatoin_correct_1slice_ROI.csv"
df_soma = pd.read_csv(soma_data_path)
day_list = glob.glob(os.path.join(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation", "[!_]*"))
df_all = pd.DataFrame(columns=['date', 'cell', 'part', 'label', 'area', 'filler', 'filler_norm', 'AS', 'AS_filler_ratio'])

for d, day in enumerate(day_list):
    dir = os.path.join(day, "_unmixing_mVenus_c-fos", "okazaki_analysis_crop")
    prop_list = glob.glob(os.path.join(dir, "*tif_props.csv"))
    cell_list = [p.split('_al_unmix')[0] for p in prop_list]
    cell_list = list(set(cell_list))
    for c, cell in enumerate(cell_list):
        date = os.path.basename(cell)[:6]
        cell_name= extract_cell_name(cell)
        cell_prop_list = glob.glob(cell+"_al_unmix_*.tif_props.csv")
        for prop in cell_prop_list:
            image = prop.split('_props')[0]
            back_filler, back_AS = get_background(h5_path, image)
            df_all = calculate_AS_filler_ratio(back_filler, back_AS, prop, date, cell_name, df_all)

df_all.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_all_spine.csv")


#AS(+)の基準を決める
mean = df_all['AS_filler_ratio'].mean()
std = df_all['AS_filler_ratio'].std()
threshold = mean + 1*std
print(threshold)

df_neurons = pd.DataFrame(columns=['date', 'cell', 'AS(+)_intensity_sum', 'AS(+)_percentage', 'AS_soma', 'filler_soma'
        ,'AS_filler_ratio','c-fos_nucleus', 'c_fos_nucleus_norm_max'])
#全細胞　AS/fillerのplot
fig = plt.figure(figsize=(130, 50))
gs = gridspec.GridSpec(10, 25)
plt.rc('font', size=15)
date_list  = df_all['date'].unique()
for d, date in enumerate(date_list):
    df_date = df_all[df_all['date'] == date]
    cell_list = df_date['cell'].unique()
    for c, cell in enumerate(cell_list):
        df_cell = df_date[df_date['cell'] == cell]
        plot_AS_filler_ratio(df_cell, gs=gs, ax=(d,c), threshold=threshold)
        df_neurons = analyse_AS_positive (date, cell, threshold, df_cell, df_soma, df_neurons)

plt.tight_layout()
fig.savefig(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_AS_filler_ratio.pdf", dpi=300, transparent=True)



df_neurons.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_cell_base.csv")
fig = plt.figure(figsize=(10, 3))
gs = gridspec.GridSpec(1, 3)
plt.rc('font', size=5)
colors = np.random.rand(len(df_neurons))

ax = fig.add_subplot(gs[(0, 0)])
ax.scatter(df_neurons['AS(+)_intensity_sum'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors, cmap='viridis')
ax.set_xlabel('Positive spine sum intensity')
ax.set_ylabel('Somatic AS intensity')
# ax.set_xlim(0, 6)
# ax.set_ylim(0, 12000)

ax = fig.add_subplot(gs[(0, 1)])
ax.scatter(df_neurons['AS(+)_percentage'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors, cmap='viridis')
ax.set_xlabel('Positive spine (%)')
ax.set_ylabel('Somatic AS intensity')
# ax.set_xlim(0, 6)
# ax.set_ylim(0, 12000)


ax = fig.add_subplot(gs[(0, 2)])
ax.scatter(df_neurons['c_fos_nucleus_norm_max'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors, cmap='viridis')
ax.set_xlabel('c-fos (a.u.)')
ax.set_ylabel('Somatic AS intensity')
# ax.set_xlim(0, 6)
# ax.set_ylim(0, 12000)

plt.tight_layout()
fig.savefig(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_summary.pdf", dpi=300, transparent=True)


#AS(+)個数/sum - AS somaのplot





# df_all = df = pd.DataFrame()
# for p, day in enumerate(day_list):
#     dir = os.path.join(day, "AS_filler_ratio")
#     spine_csv_files = glob.glob(os.path.join(dir, "*spine.csv"))
#     back_csv_files = [spine_csv[:-9] + "back.csv" for spine_csv in spine_csv_files]
#     print(os.path.basename(day))
#     print("Number of files: %s" % len(spine_csv_files))
#
#     control_as_scores = []
#
#     control_as_normV_ratio = []
#
#
#     control_subtract_as_scores = []
#
#
#     control_X_normalized = []
#     as_Y = []
#
#
#
#     ctrl_count = 0
#
#     for spine_csv, back_csv in zip(spine_csv_files, back_csv_files):
#         ctrl_count += 1
#         AS_filler_ratio_mode_subtracted, AS_filler_ratio, AS_normV_ratio, X_norm, scatter_y, df= calculate_as_score(spine_csv, back_csv, cell_count = ctrl_count, cond = "Ctrl")
#         control_subtract_as_scores.extend(AS_filler_ratio_mode_subtracted)
#         control_as_scores.extend(AS_filler_ratio)
#         control_as_normV_ratio.extend(AS_normV_ratio)
#         control_X_normalized.extend(X_norm)
#         as_Y.extend(scatter_y)
#         df_all = pd.concat([df_all, df], ignore_index=True)
#
#     print("ctrl_count="+str(ctrl_count))
#     # prepare DataFrame for plotting and CSV writing
#     control_subtract_as_scores_df = pd.DataFrame(np.array(control_subtract_as_scores).flatten(), columns=['AS/filler (mode-subtracted)'])
#     control_subtract_as_scores_df['Group'] = 'control'
#
#
#     control_subtract_as_scores_df.to_csv(os.path.join(dir, 'control_subtract_as_scores.csv'), index=False)
#
#
#     # Combine DataFrames for plot and reset index to ensure unique indices
#     #combined_as_df = pd.concat([apv_subtract_as_scores_df, control_subtract_as_scores_df])
#
#
#     s=5 #scatter point size
#
#
#     ax = fig.add_subplot(gs[(p, 1)])
#     ax.scatter(control_X_normalized, as_Y, s=s,alpha=0.5, label='Control')
#     ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x / 1000:.1f}k'))
#     ax.set_xlabel('V (a.u)')
#     ax.set_ylabel('AS sum ' + os.path.basename(day))
#     ax.set_ylim(-1000, 100000)  # Adjust the y-axis limit as needed
#     ax.set_title('NoAPV AS_sum')
#     ax.set_xlim(0, 6)
#
#
#
#     ax = fig.add_subplot(gs[(p, 2)])
#     ax.scatter(control_X_normalized, control_subtract_as_scores, s=s,alpha=0.5, label='Control')
#     ax.set_xlabel('V (a.u)')
#     ax.set_ylabel('AS/filler (mode-subtracted)')
#     ax.set_ylim(-0.1,1.2)
#     ax.set_title('NoAPV AS/filler (mode-subtracted)')
#     ax.set_xlim(0, 6)
#
#
#
#     ax = fig.add_subplot(gs[(p, 3)])
#     ax.scatter(control_X_normalized, control_as_scores, s=s,alpha=0.5, label='Control')
#     ax.set_xlabel('V (a.u)')
#     ax.set_ylabel('AS/filler')
#     ax.set_ylim(-0.02, 1.2)
#     ax.set_title('NoAPV AS/filler')
#     ax.set_xlim(0, 6)
#
#
#
#
#     ax = fig.add_subplot(gs[(p, 4)])
#     ax.scatter(control_X_normalized, control_as_normV_ratio, s=s, alpha=0.5, label='Control')
#     ax.set_xlabel('V (a.u)')
#     ax.set_ylabel('AS/norm V')
#     # ax.set_ylim(-0.1, 1.2)
#     ax.set_title('NoAPV AS/norm V')
#     ax.set_xlim(0, 6)
#     ax.set_ylim(0, 12000)
#
#     # Plot AS box plot and strip plot
#     ax = fig.add_subplot(gs[(p, 5)])
#     sns.boxplot(x='Group', y='AS/filler (mode-subtracted)', data=control_subtract_as_scores_df, whis=np.inf, linewidth=1.5, ax=ax)
#     sns.stripplot(x='Group', y='AS/filler (mode-subtracted)', data=control_subtract_as_scores_df, jitter=True, color='black',
#                   alpha=0.5, ax=ax)
#     ax.set_ylabel(os.path.basename(day) + " AS/filler (mode-subtracted)")
#     ax.set_ylim(-0.2, 1.2)
