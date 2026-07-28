import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# グラフ設定
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# グローバル変数
scoring = "AS/filler_normalized"
hotspot_index_avg = {}

# 関数定義
def calculate_as_score(spine_csv, back_csv, cell_count, cond, probe_name):
    dir = os.path.dirname(os.path.dirname(spine_csv))
    fname = os.path.basename(spine_csv)
    spine_df = pd.read_csv(spine_csv)
    spine_df = spine_df[spine_df['dendrite'].notna()].reset_index(drop=True)
    back_df = pd.read_csv(back_csv)[:len(spine_df)]

    area_column = 'area' if 'area' in spine_df.columns else 'area_in_pixel'
    area = spine_df[area_column].values

    label = spine_df['label']
    sp_minus_back_x_area = (spine_df['mean_intensity-0'].values - back_df['mean_intensity-0']) * area
    sp_minus_back_y_area = (spine_df['mean_intensity-1'].values - back_df['mean_intensity-1']) * area
    dendrite = spine_df['dendrite']

    valid_indices = (sp_minus_back_y_area > 0) & (sp_minus_back_x_area > 0)
    sp_minus_back_x_area = sp_minus_back_x_area[valid_indices]
    sp_minus_back_y_area = sp_minus_back_y_area[valid_indices]
    label = label[valid_indices]
    dendrite = dendrite[valid_indices]

    AS_filler_ratio = sp_minus_back_y_area / sp_minus_back_x_area

    kde = gaussian_kde(AS_filler_ratio)
    x_grid = np.linspace(AS_filler_ratio.min(), AS_filler_ratio.max(), 1000)
    kde_values = kde(x_grid)

    peaks, _ = find_peaks(kde_values)

    if len(peaks) > 0:
        mode_as_filler_ratio = x_grid[peaks].min()
    else:
        mode_as_filler_ratio = x_grid[np.argmax(kde_values)]

    AS_filler_ratio_normalized = AS_filler_ratio / mode_as_filler_ratio if mode_as_filler_ratio else AS_filler_ratio
    X_normalized = sp_minus_back_x_area / np.mean(sp_minus_back_x_area)
    SD = (mode_as_filler_ratio - AS_filler_ratio.min()) / mode_as_filler_ratio
    z_score = AS_filler_ratio_normalized / SD

    df = pd.DataFrame({
        'spine_label': label.values.flatten(),
        'AS/filler': AS_filler_ratio,
        'AS/filler_normalized': AS_filler_ratio_normalized,
        'filler sum': sp_minus_back_x_area,
        'V_normalized': X_normalized,
        'mVenus sum': sp_minus_back_y_area,
        'probe': probe_name,
        'condition': cond,
        'cell': [f"cell{cell_count}"] * len(AS_filler_ratio),
        'mode_AS_filler_ratio': mode_as_filler_ratio,
        'SD': SD,
        'z-score': z_score,
        'dendrite': dendrite
    })

    as_csv_path = os.path.join(dir, f'AS_filler_ratio_{fname[:-9]}.csv')
    df.to_csv(as_csv_path, index=False)

    return AS_filler_ratio, AS_filler_ratio_normalized, sp_minus_back_y_area, X_normalized, df


def calculate_index(input_df, col_name, probe_name, index_df, index_type="hot_spot"):
    global hotspot_index_avg
    cell_list = input_df['cell'].unique().tolist()
    for cell in cell_list:
        df = input_df[input_df['cell'] == cell]
        dend_list = df['dendrite'].unique().tolist()
        for dend in dend_list:
            df_dend = df[df['dendrite'] == dend]
            if index_type == "hot_spot":
                differences = df_dend[col_name].diff()
                differences.iloc[0] = df_dend[col_name].iloc[-1] - df_dend[col_name].iloc[0]
                hotspot_index = differences.abs().mean()
                if probe_name not in hotspot_index_avg:
                    hotspot_index_avg[probe_name] = []
                hotspot_index_avg[probe_name].append(hotspot_index)
                index_df.loc[len(index_df)] = [hotspot_index, probe_name, cell, dend]


def plot_graph(df, position, title):
    ax = fig.add_subplot(gs[(position)])
    sns.barplot(x='probe', y='index_val', data=df, ax=ax)
    sns.stripplot(x='probe', y='index_val', data=df, ax=ax, color='black', alpha=0.6, size=2, jitter=True)

    ax.set_title(title)
    ax.set_xlabel('Probe')
    ax.set_ylabel('Index')


# メイン処理
probe_list = glob.glob(os.path.join(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\UTR_screening_data", "[!_]*"))

fig = plt.figure(figsize=(50, 40))
gs = gridspec.GridSpec(7, 7, width_ratios=[1, 1, 1, 1, 1, 1, 1])
plt.rc('font', size=20)

df_all = pd.DataFrame()
for probe in probe_list:
    dir = os.path.join(probe, "violin_plot_box_plot")
    spine_csv_files = glob.glob(os.path.join(dir, "*spine.csv"))
    back_csv_files = [spine_csv[:-9] + "back.csv" for spine_csv in spine_csv_files]

    ctrl_count = 0
    for spine_csv, back_csv in zip(spine_csv_files, back_csv_files):
        if "control" in spine_csv.lower():
            ctrl_count += 1
            _, _, _, _, df = calculate_as_score(
                spine_csv, back_csv, cell_count=ctrl_count, cond="Ctrl", probe_name=os.path.basename(probe))
            df_all = pd.concat([df_all, df], ignore_index=True)

# プローブ単位と樹状突起単位のスコア計算
df_all['AS_score_probe'] = np.nan
df_all['AS_score_dendrite'] = np.nan

for probe in df_all['probe'].unique():
    df_probe = df_all[df_all['probe'] == probe]

    # プローブ単位のスコア
    SD_probe = 1 - df_probe[scoring].min()
    df_all.loc[df_all['probe'] == probe, 'AS_score_probe'] = (df_probe[scoring] / SD_probe) - 1

    # 樹状突起単位のスコア
    for dendrite in df_probe['dendrite'].unique():
        df_dendrite = df_probe[df_probe['dendrite'] == dendrite]
        SD_dendrite = 1 - df_dendrite[scoring].min()
        df_all.loc[(df_all['probe'] == probe) & (df_all['dendrite'] == dendrite), 'AS_score_dendrite'] = (
            df_dendrite[scoring] / SD_dendrite
        ) - 1

# 結果の保存
output_dir = r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\UTR_screening_data"
df_all.to_csv(os.path.join(output_dir, "_all_data_with_AS_scores.csv"), index=False)

# プロットをPDFに保存
plt.tight_layout()
output_pdf_path = os.path.join(output_dir, "_summary_NoAPV_as_score.pdf")
fig.savefig(output_pdf_path, dpi=300, transparent=True)

print(f"Summary figure saved to {output_pdf_path}")
