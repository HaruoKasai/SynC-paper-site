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
from scipy import stats
import h5py

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
    #ax.axhline(y=threshold, color='k', linestyle='--')
    ax.scatter(X,Y,s=s,alpha=0.8)
    ax.set_xlabel('V (a.u)')
    ax.set_ylabel('AS/filler')
    ax.set_ylim(-0.005,0.08)
    ax.set_title(cell_name)
    ax.set_xlim(0, 6)

def subtraction_AS_filler_ratio(date, cell, df_cell, df_soma, df_neurons):
    df_soma['date'] = df_soma['date'].astype(str)
    df_temp = df_soma[(df_soma['date'] == date) & (df_soma['cell'] == cell)]
    if df_temp.empty:
        print(f"No matching data found for date: {date}, cell: {cell}")
        return df_neurons
    spine_intensity_subtract_median = (df_cell['AS'].sum() / len(df_cell['AS'])) - df_cell['AS'].median()
    rounded_data = df_cell['AS'].round(2)
    mode_value = rounded_data.mode()[0]
    spine_intensity_mode_subtracted = df_cell['AS'].sum() / len(df_cell['AS']) - mode_value
    spine_intensity_subtract_soma = df_cell['AS'].sum() / len(df_cell['AS']) - df_temp['mVenus'].values[0]
    spine_AS_filler_ratio = df_cell['AS'].sum() / df_cell['filler'].sum()

    df_to_add = pd.DataFrame({
        'date': [date],
        'cell': [cell],
        'AS_soma': df_temp['mVenus'].values,
        'filler_soma': df_temp['filler'].values,
        'spine_intensity_subtract_median': [spine_intensity_subtract_median],
        'mode_value': [mode_value],
        'spine_intensity_mode_subtracted': [spine_intensity_mode_subtracted],
        'spine_intensity_subtract_soma': [spine_intensity_subtract_soma],
        'spine_AS_filler_ratio': [spine_AS_filler_ratio],
        'spine_number': len(df_cell['AS']),
        'c-fos_nucleus': df_temp['c-fos_nucleus'].values,
        'c-fos_nucleus_norm_max': df_temp['c-fos_nucleus_norm_max'].values,
        'c-fos_nucleus_norm_median': df_temp['c-fos_nucleus_norm_median'].values,
        'c-fos_nucleus_norm_mean': df_temp['c-fos_nucleus_norm_mean'].values
    })

    df_neurons = pd.concat([df_neurons, df_to_add], ignore_index=True)


    return df_neurons


# 読み込んだファイルの処理
h5_path = r"\\DESKTOP-WS2\data\arima\hdf5\as_proj.h5"
soma_data_path = r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_c-fos_iAS_correlatoin_correct_1slice_ROI.csv"
df_soma = pd.read_csv(soma_data_path)
day_list = glob.glob(os.path.join(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation", "[!_]*"))

# 全てのデータを保存するデータフレーム
df_all = pd.DataFrame(
    columns=['date', 'cell', 'part', 'label', 'area', 'filler', 'filler_norm', 'AS', 'AS_filler_ratio'])



# 各細胞ごとにループして計算を実行
for d, day in enumerate(day_list):
    dir = os.path.join(day, "_unmixing_mVenus_c-fos", "okazaki_analysis_crop")
    prop_list = glob.glob(os.path.join(dir, "*tif_props.csv"))
    cell_list = [p.split('_al_unmix')[0] for p in prop_list]
    cell_list = list(set(cell_list))

    for c, cell in enumerate(cell_list):
        date = os.path.basename(cell)[:6]
        cell_name = "N" + os.path.basename(cell).split("_N")[1]
        cell_prop_list = glob.glob(cell + "_al_unmix_*.tif_props.csv")

        # 各セルのプロパティを処理
        for prop in cell_prop_list:
            image = prop.split('_props')[0]
            back_filler, back_AS = get_background(h5_path, image)
            df_all = calculate_AS_filler_ratio(back_filler, back_AS, prop, date, cell_name, df_all)

df_all.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_all3_spine.csv")

df_neurons = pd.DataFrame(columns=['date', 'cell', 'AS_soma', 'filler_soma', 'spine_intensity_subtract_median'
    , 'mode_value','spine_intensity_mode_subtracted', 'spine_intensity_subtract_soma', 'spine_AS_filler_ratio', 'spine_number', 'c-fos_nucleus', 'c-fos_nucleus_norm_max', 'c-fos_nucleus_norm_median', 'c-fos_nucleus_norm_mean'])



date_list  = df_all['date'].unique()
for d, date in enumerate(date_list):
    df_date = df_all[df_all['date'] == date]
    cell_list = df_date['cell'].unique()
    for c, cell in enumerate(cell_list):
        df_cell = df_date[df_date['cell'] == cell]

        df_neurons = subtraction_AS_filler_ratio(date, cell, df_cell, df_soma, df_neurons)


# 結果を CSV として保存
df_neurons.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_cell_base_subtraction.csv")



# 結果をプロット
fig = plt.figure(figsize=(6, 4))
gs = gridspec.GridSpec(3, 4)
plt.rc('font', size=5)
colors = np.random.rand(len(df_neurons))

c_fos_threshold = 1
iAS_threshold = 0.4

ax = fig.add_subplot(gs[(0, 0)])
ax.scatter(df_neurons['spine_intensity_subtract_median'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors,
           cmap='viridis')
ax.set_xlabel('spine_intensity_subtract_median')
ax.set_ylabel('Somatic AS intensity')

ax = fig.add_subplot(gs[(0, 1)])
ax.scatter(df_neurons['spine_intensity_mode_subtracted'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors,
           cmap='viridis')
ax.set_xlabel('spine_intensity_mode_subtracted')
ax.set_ylabel('Somatic AS intensity')

ax = fig.add_subplot(gs[(0, 2)])
ax.scatter(df_neurons['spine_intensity_subtract_soma'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors, cmap='viridis')
ax.set_xlabel('spine_intensity_subtract_soma')
ax.set_ylabel('Somatic AS intensity')

ax = fig.add_subplot(gs[(1, 0)])
ax.scatter(df_neurons['c-fos_nucleus_norm_max'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors,
           cmap='viridis')
ax.set_xlabel('c-fos (a.u.) max_norm')
ax.set_ylabel('Somatic AS intensity')

ax = fig.add_subplot(gs[(1, 1)])
ax.scatter(df_neurons['c-fos_nucleus_norm_median'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors,
           cmap='viridis')
ax.set_xlabel('c-fos (a.u.) median_norm')
ax.set_ylabel('Somatic AS intensity')

ax = fig.add_subplot(gs[(1, 2)])
ax.scatter(df_neurons['c-fos_nucleus_norm_mean'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors,
           cmap='viridis')
ax.axhline(y=iAS_threshold, linestyle='--')
ax.axvline(x=c_fos_threshold, linestyle='--')
ax.set_xlabel('c-fos (a.u.) mean_norm')
ax.set_ylabel('Somatic AS intensity')



df_neurons['c-fos_category'] = np.where(df_neurons['c-fos_nucleus_norm_mean'] < c_fos_threshold, 'c-fos(-)', 'c-fos(+)')
df_neurons['iAS_category'] = np.where(df_neurons['AS_soma'] < iAS_threshold, 'iAS(-)', 'iAS(+)')
category_counts = df_neurons.groupby(['c-fos_category', 'iAS_category']).size().unstack(fill_value=0)
category_percentages = category_counts.div(category_counts.sum(axis=1), axis=0) * 100

ax = fig.add_subplot(gs[(2, 0)])
data_points = np.arange(len(category_percentages.index))
bottom_data = pd.Series(np.zeros(len(category_percentages.index)), index=category_percentages.index.tolist())
colors = {'iAS(-)': 'lightgray', 'iAS(+)': 'blue'}

for i, column in enumerate(category_percentages.columns):
    bar_list = ax.bar(data_points, category_percentages[column], bottom=bottom_data, label=column)

    for bar in bar_list:
        bar.set_color(colors[column])

    bottom_data += category_percentages[column]

ax.set_xticks(data_points)
ax.set_xticklabels(category_percentages.index)

ax.set_ylabel('Percentage (%)')
ax.legend(title='iAS Category')

# グラフ表示
plt.tight_layout()


# 結果をファイルに保存
plt.savefig(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_stacked_bar_chart_100percent.pdf", dpi=300, transparent=True)

df_neurons.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\combined_c-fos_iAS_data.csv", index=False)


plt.tight_layout()
fig.savefig(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_subtraction_mode_summary.pdf", dpi=300,
            transparent=True)



