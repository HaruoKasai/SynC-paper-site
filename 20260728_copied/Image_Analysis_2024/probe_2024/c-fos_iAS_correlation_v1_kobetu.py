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
import h5py


def calculate_AS_filler_ratio(back_filler, back_AS, prop_path, date, cell_name, df_all):
    df = pd.read_csv(prop_path)
    df = df[df['analysis'] == True]
    df = df[['label', 'area', 'mean_intensity-0', 'mean_intensity-1']][:-1]

    area = df["area"].values
    label = df['label']
    filler = (df['mean_intensity-0'].values - back_filler) * area  # 最終行を除外 (dendrite)
    AS = (df['mean_intensity-1'].values - back_AS) * area

    # AS/filler ratioの計算
    AS_filler_ratio = AS / filler

    # 体積は平均で正規化
    filler_norm = filler / np.mean(filler)

    part_name = prop_path.split('_region_')[1].split('.tif')[0]

    df_to_add = pd.DataFrame({
        'date': date,
        'cell': cell_name,
        'part': part_name,
        'label': label.values.flatten(),
        'area': area,
        'filler': filler,
        'filler_norm': filler_norm,
        'AS': AS,
        'AS_filler_ratio': AS_filler_ratio,
    })
    df_all = pd.concat([df_all, df_to_add], ignore_index=True)
    return df_all


def get_background(h5_path, image):
    with h5py.File(h5_path, 'r') as hdf:
        bg_mean = hdf["images/" + os.path.basename(image) + "/bg_mean"]
        filler_back = bg_mean[:][0]
        AS_back = bg_mean[:][1]
        return filler_back, AS_back


def plot_AS_filler_ratio(df, gs, ax, threshold):
    ax = fig.add_subplot(gs[ax])
    cell_name = df["cell"].unique()[0]
    X = df["filler_norm"].values
    Y = df["AS_filler_ratio"].values
    s = 8
    ax.axhline(y=threshold, color='k', linestyle='--')
    ax.scatter(X, Y, s=s, alpha=0.8)
    ax.set_xlabel('V (a.u)')
    ax.set_ylabel('AS/filler')
    ax.set_ylim(-0.005, 0.10)
    ax.set_title(cell_name)
    ax.set_xlim(0, 6)


def analyse_AS_positive(date, cell, threshold, df_cell, df_soma, df_neurons):
    as_positive_df = df_cell[df_cell['AS_filler_ratio'] > threshold]
    ASpositive_intensity_sum = as_positive_df['AS_filler_ratio'].sum()
    ASpositive_percentage = len(as_positive_df) / len(df_cell) * 100
    df_soma['date'] = df_soma['date'].astype(str)
    df_temp = df_soma[(df_soma['date'] == date) & (df_soma['cell'] == cell)]
    df_to_add = pd.DataFrame({
        'date': date,
        'cell': cell,
        'AS(+)_intensity_sum': ASpositive_intensity_sum,
        'AS(+)_percentage': ASpositive_percentage,
        'AS_soma': df_temp['mVenus'].values,
        'filler_soma': df_temp['filler'].values,
        'AS_filler_ratio': df_temp['mvenus/filler'].values,
        'c-fos_nucleus': df_temp['c-fos_nucleus'].values,
        'c_fos_nucleus_norm_max': df_temp['c-fos_nucleus_norm_max'].values
    })
    df_neurons = pd.concat([df_neurons, df_to_add], ignore_index=True)
    return df_neurons


h5_path = r"\\DESKTOP-WS2\data\arima\hdf5\as_proj.h5"
soma_data_path = r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_c-fos_iAS_correlatoin_correct_1slice_ROI.csv"
df_soma = pd.read_csv(soma_data_path)
day_list = glob.glob(
    os.path.join(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation", "[!_]*"))
df_all = pd.DataFrame(
    columns=['date', 'cell', 'part', 'label', 'area', 'filler', 'filler_norm', 'AS', 'AS_filler_ratio'])

for d, day in enumerate(day_list):
    dir = os.path.join(day, "_unmixing_mVenus_c-fos", "okazaki_analysis_crop")
    prop_list = glob.glob(os.path.join(dir, "*tif_props.csv"))
    cell_list = [p.split('_al_unmix')[0] for p in prop_list]
    cell_list = list(set(cell_list))
    for c, cell in enumerate(cell_list):
        date = os.path.basename(cell)[:6]
        cell_name = "N" + os.path.basename(cell).split("_N")[1]
        cell_prop_list = glob.glob(cell + "_al_unmix_*.tif_props.csv")
        for prop in cell_prop_list:
            image = prop.split('_props')[0]
            back_filler, back_AS = get_background(h5_path, image)
            df_all = calculate_AS_filler_ratio(back_filler, back_AS, prop, date, cell_name, df_all)

df_all.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_all_spine.csv")

df_neurons = pd.DataFrame(
    columns=['date', 'cell', 'AS(+)_intensity_sum', 'AS(+)_percentage', 'AS_soma', 'filler_soma', 'AS_filler_ratio',
             'c-fos_nucleus', 'c_fos_nucleus_norm_max'])

# 全細胞 AS/fillerのplot
fig = plt.figure(figsize=(130, 50))
gs = gridspec.GridSpec(10, 25)
plt.rc('font', size=15)
date_list = df_all['date'].unique()

for d, date in enumerate(date_list):
    df_date = df_all[df_all['date'] == date]
    cell_list = df_date['cell'].unique()
    for c, cell in enumerate(cell_list):
        df_cell = df_date[df_date['cell'] == cell]

        # 細胞ごとのthresholdを計算
        mean = df_cell['AS_filler_ratio'].mean()
        std = df_cell['AS_filler_ratio'].std()
        threshold = mean + 3 * std  # 細胞ごとのthresholdを設定

        # 各細胞に対して解析とプロットを実行
        plot_AS_filler_ratio(df_cell, gs=gs, ax=(d, c), threshold=threshold)
        df_neurons = analyse_AS_positive(date, cell, threshold, df_cell, df_soma, df_neurons)

plt.tight_layout()
fig.savefig(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_AS_filler_ratio.pdf",
            dpi=300, transparent=True)

df_neurons.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_cell_base.csv")
fig = plt.figure(figsize=(10, 3))
gs = gridspec.GridSpec(1, 3)
plt.rc('font', size=5)
colors = np.random.rand(len(df_neurons))

# Plot 1
ax = fig.add_subplot(gs[(0, 0)])
ax.scatter(df_neurons['AS(+)_intensity_sum'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors,
           cmap='viridis')
ax.set_xlabel('Positive spine sum intensity')
ax.set_ylabel('Somatic AS intensity')

# Plot 2
ax = fig.add_subplot(gs[(0, 1)])
ax.scatter(df_neurons['AS(+)_percentage'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors, cmap='viridis')
ax.set_xlabel('Positive spine (%)')
ax.set_ylabel('Somatic AS intensity')

# Plot 3
ax = fig.add_subplot(gs[(0, 2)])
ax.scatter(df_neurons['c_fos_nucleus_norm_max'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors, cmap='viridis')
ax.set_xlabel('c-fos (a.u.)')
ax.set_ylabel('Somatic AS intensity')

plt.tight_layout()
fig.savefig(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_summary.pdf", dpi=300, transparent=True)



