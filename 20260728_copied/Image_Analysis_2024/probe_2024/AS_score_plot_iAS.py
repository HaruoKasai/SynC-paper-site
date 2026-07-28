import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from scipy import stats
from scipy.stats import skew, kurtosis
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.ticker as mticker
import re
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

hotspot_index_avg = {}


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

    # valid_indices = (sp_minus_back_y_area > 0) & (sp_minus_back_x_area > 0)
    # sp_minus_back_x_area = sp_minus_back_x_area[valid_indices]
    # sp_minus_back_y_area = sp_minus_back_y_area[valid_indices]
    # label = label[valid_indices]
    # dendrite = dendrite[valid_indices]

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
        'mode_AS_filler_ratio' : mode_as_filler_ratio,
        'SD': SD,
        'z-score': z_score,
        'dendrite': dendrite
    })

    as_csv_path = os.path.join(dir, f'AS_filler_ratio_{fname[:-9]}.csv')
    df.to_csv(as_csv_path, index=False)

    return AS_filler_ratio, AS_filler_ratio_normalized, sp_minus_back_y_area, X_normalized, df

def calculate_index(input_df, col_name, probe_name, index_df, index_type="hot_spot", c_fit=None, ave=None):
    global hotspot_index_avg
    c_fit = c_fit_dict.get(os.path.basename(probe_name))
    if c_fit is None:
        raise KeyError(f"c_fit for probe '{probe_name}' is not found in c_fit_dict.")
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
            elif index_type == "SD":
                std_dev = c_fit
                index_df.loc[len(index_df)] = [std_dev, probe_name, cell, dend]
            elif index_type == "percentage_above_3SD":
                mean = ave
                std = c_fit
                count = ((df_dend[col_name]) > (mean + 3 * std)).sum()
                percentage = count / len(df_dend[col_name]) * 100
                index_df.loc[len(index_df)] = [percentage, probe_name, cell, dend]
                print(f"count {probe_name}: {count}")
                print(f"all count {probe_name}: {len(df_dend[col_name])}")
            elif index_type == "above_2SD_mean":
                mean = ave
                std = c_fit
                above_2SD_values = df_dend[col_name][df_dend[col_name] > (mean + 2 * std)]
                mean_above_2SD = above_2SD_values.mean() if not above_2SD_values.empty else np.nan
                index_df.loc[len(index_df)] = [mean_above_2SD, probe_name, cell, dend]
                print(f"mean {probe_name}: {mean}")
                print(f"std {probe_name}: {std}")


def plot_graph (df, position, title):
    ax = fig.add_subplot(gs[(position)])
    sns.barplot(x='probe', y='index_val', data=df, ax=ax)
    sns.stripplot(x='probe', y='index_val', data=df, ax=ax, color='black', alpha=0.6, size=2, jitter=True)

    ax.set_title(title)
    #ax.set_ylim(-2.5, 12)
    ax.set_xlabel('Probe')

    if position == (0,6):
        ax.set_ylim(-1, 8)
        ax.set_ylabel("z-score")
    elif position == (6,6):
        ax.set_ylabel("AS_index")
        ax.set_ylim(0, 10)
    else:
        ax.set_ylabel('Index')

def gaussian(x, a, b, c):
    return a * np.exp(-((x-b)**2) / (2 * c **2))

# メイン処理
#probe_list = glob.glob(os.path.join(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\AAV_iAS", "[!_]*"))
probe_list = glob.glob(os.path.join(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\distribution", "[!_]*"))


df_all = pd.DataFrame()
for p, probe in enumerate(probe_list):
    dir = os.path.join(probe, "plot")
    spine_csv_files = glob.glob(os.path.join(dir, "*spine.csv"))
    back_csv_files = [spine_csv[:-9] + "back.csv" for spine_csv in spine_csv_files]
    print(os.path.basename(probe))
    print("Number of files: %s" % len(spine_csv_files))

    as_Y, control_as_scores, control_as_scores_normalized, control_X_normalized = [], [], [], []
    ctrl_count = 0

    for spine_csv, back_csv in zip(spine_csv_files, back_csv_files):
        if "control" in spine_csv.lower():
            ctrl_count += 1
            AS_filler_ratio, AS_filler_ratio_normalized, scatter_y, X_norm, df = calculate_as_score(
                spine_csv, back_csv, cell_count=ctrl_count, cond="Ctrl", probe_name=os.path.basename(probe))
            control_as_scores.extend(AS_filler_ratio)
            control_as_scores_normalized.extend(AS_filler_ratio_normalized)
            as_Y.extend(scatter_y)
            control_X_normalized.extend(X_norm)
            df_all = pd.concat([df_all, df], ignore_index=True)



unique_filler_mean = df_all.groupby(['probe', 'cell'])['filler sum'].mean().reset_index()
unique_mVenus_mean = df_all.groupby(['probe', 'cell'])['mVenus sum'].mean().reset_index()
all_filler_mean = unique_filler_mean['filler sum'].mean()
print(f"all_filler_mean: {all_filler_mean}")
unique_filler_mean['across_cell_filler_coefficient'] = unique_filler_mean['filler sum'] / all_filler_mean
df_all = df_all.merge(unique_filler_mean[['probe', 'cell', 'across_cell_filler_coefficient']], on=['probe', 'cell'],how='left')
df_all['across_spine_filler'] = df_all['filler sum'] / df_all['across_cell_filler_coefficient']
df_all['AS/filler_ratio_filler_normalized'] = df_all['mVenus sum'] / df_all['across_spine_filler']

df_cell = pd.DataFrame()
cell_list = []
for probe in probe_list:
    dir = os.path.join(probe, "plot")
    unique_filler_mean = df_all[df_all['probe'] == os.path.basename(probe)].groupby(['probe', 'cell'])[
        'filler sum'].mean().reset_index()
    unique_mVenus_mean = df_all[df_all['probe'] == os.path.basename(probe)].groupby(['probe', 'cell'])[
        'mVenus sum'].mean().reset_index()
    for cell in unique_filler_mean['cell'].unique():
        cell_data = {
            'probe': os.path.basename(probe),
            'cell': cell,
            'filler_intensity': unique_filler_mean[unique_filler_mean['cell'] == cell]['filler sum'].iloc[0],
            'iAS_intensity': unique_mVenus_mean[unique_mVenus_mean['cell'] == cell]['mVenus sum'].iloc[0]
        }
        cell_list.append(cell_data)

cell_list_df = pd.DataFrame(cell_list)
df_cell = cell_list_df

fig = plt.figure(figsize=(75, 60))
gs = gridspec.GridSpec(13, 13, width_ratios=[1,1,1,1,1,1,1,1,1,1,1,1,1])
plt.rc('font', size=20)


#df_probe, cell毎の計算を先に行ってしまいグラフ表示だけ後でやるようにしてみる
group_iAS = df_all.groupby(['probe', 'cell'])
mode_list = []
subtraction_mode_list = []


for (probe, cell), group in group_iAS:
    print(f"Processing probe: {probe}, cell: {cell}")
    print(group)
    data_as = group['AS/filler_ratio_filler_normalized'].values
    data_as = group['AS/filler_ratio_filler_normalized'].values
    print(f"data_as for probe {probe}, cell {cell}: {data_as}")
    kde = gaussian_kde(data_as)
    x_grid = np.linspace(min(data_as), max(data_as), 100)
    kde_values = kde(x_grid)
    peaks, _ = find_peaks(kde_values)

    if len(peaks) > 0:
        mode_as_filler_ratio = x_grid[peaks].min()
    else:
        mode_as_filler_ratio = x_grid[np.argmax(kde_values)]

    print(f'mode_as_filler_ratio: {mode_as_filler_ratio}')


    subtraction_mode_AS_intensity = group['mVenus sum'] - (group['across_spine_filler'] * mode_as_filler_ratio)
    subtraction_mode_AS_filler_ratio = subtraction_mode_AS_intensity / group['across_spine_filler']

    for label, intensity, ratio in zip(group['spine_label'], subtraction_mode_AS_intensity, subtraction_mode_AS_filler_ratio):
        subtraction_mode_list.append({
            'probe': os.path.basename(probe),
            'cell': cell,
            'spine_label': label,
            'subtraction_mode_AS_intensity': intensity,
            'subtraction_mode_AS_filler_ratio': ratio
        })
    print("subtraction_mode_AS_intensity:")
    print(subtraction_mode_AS_intensity.head())
    print("subtraction_mode_AS_filler_ratio:")
    print(subtraction_mode_AS_filler_ratio.head())
    print(subtraction_mode_list[:5])

    mode_list.append({
        'probe': probe,
        'cell': cell,
        'mode_AS/filler ratio': mode_as_filler_ratio
    })

subtraction_mode_df = pd.DataFrame(subtraction_mode_list)
print(subtraction_mode_df.columns)
print(subtraction_mode_df.head())
df_all = df_all.merge(subtraction_mode_df, on=['spine_label', 'probe', 'cell'], how='left')
print(df_all.columns)
print(df_all.head())
print(df_all[['spine_label', 'probe', 'cell']].head())
print(subtraction_mode_df[['spine_label', 'probe', 'cell']].head())
mode_cell_df = pd.DataFrame(mode_list)
df_cell = df_cell.merge(mode_cell_df, on=['probe','cell'], how='left')

c_fit_dict = {}
mean_subtraction_ratio_dict = {}


for p, probe in enumerate(probe_list):
    df_probe = df_all[df_all['probe'] == os.path.basename(probe)]
    unique_cells = df_probe['cell'].unique()
    colors = cm.get_cmap('tab10', len(unique_cells))

    # Scatter plot
    ax = fig.add_subplot(gs[(p, 0)])
    for i, cell in enumerate(unique_cells):
        cell_data = df_probe[df_probe['cell'] == cell]
        ax.scatter(cell_data['V_normalized'], cell_data['mVenus sum'],  s=50, edgecolor='none', alpha=0.5, label='Control')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x / 1000:.1f}k'))
        ax.set_xlabel('V (a.u.)')
        ax.set_ylabel(f'AS sum {os.path.basename(probe)}')
        #ax.set_ylim(-1000, 100000)
        ax.set_title('Control AS_sum')
        ax.set_xlim(0, 4)

    ax = fig.add_subplot(gs[(p, 1)])
    for i, cell in enumerate(unique_cells):
        cell_data = df_probe[df_probe['cell'] == cell]
        ax.scatter(cell_data['V_normalized'], cell_data['AS/filler'], s=50, edgecolor='none', alpha=0.5,
                   label='Control')
        #ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x / 1000:.1f}k'))
        ax.set_xlabel('V (a.u.)')
        ax.set_ylabel(f'AS ratio {os.path.basename(probe)}')
        # ax.set_ylim(-1000, 100000)
        ax.set_title('Control AS_sum')
        #ax.set_xlim(0, 6)

    ax = fig.add_subplot(gs[(p, 2)])
    for i, cell in enumerate(unique_cells):
        cell_data = df_probe[df_probe['cell'] == cell]
        ax.scatter(cell_data['V_normalized'], cell_data['AS/filler_normalized'], s=50, edgecolor='none', alpha=0.5,
                   label='Control')
        # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x / 1000:.1f}k'))
        ax.set_xlabel('V (a.u.)')
        ax.set_ylabel(f'AS ratio (normalized) {os.path.basename(probe)}')
        # ax.set_ylim(-1000, 100000)
        ax.set_title('Control AS ratio')
        # ax.set_xlim(0, 6)

    ax = fig.add_subplot(gs[(p, 3)])
    sns.boxplot(x='condition', y='AS/filler_normalized', data=df_probe, whis=np.inf, linewidth=1.5, ax=ax)
    sns.stripplot(x='condition', y='AS/filler_normalized', data=df_probe, jitter=True, color='black', alpha=0.5, ax=ax)
    ax.set_ylabel(f'{os.path.basename(probe)} AS ratio (Normalized)')
    #ax.set_ylim(-0.2, 1.2)


    ax = fig.add_subplot(gs[p,4])
    for i, cell in enumerate(unique_cells):
        cell_data = df_probe[df_probe['cell'] == cell]
        ax.scatter(cell_data['V_normalized'], cell_data['AS/filler_ratio_filler_normalized'], s=50, edgecolor='none', alpha=0.5,
                   label='Control')
        # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x / 1000:.1f}k'))
        ax.set_xlabel('V (a.u.)')
        ax.set_ylabel(f'AS ratio (filler normalized) {os.path.basename(probe)}')
        ax.set_ylim(-0.002, 0.03)  #targetの方
        #ax.set_ylim(-0.002, 0.10)  # iAS
        ax.set_title('Control AS ratio')
        #ax.set_xlim(0, 6)

    ax = fig.add_subplot(gs[p,5])
    hist, bin_edges = np.histogram(df_probe['AS/filler_ratio_filler_normalized'], bins=100, density=True)
    hist_max = hist.max()  # ヒストグラムの最大値
    hist_normalized = hist / hist_max  # 最大値で正規化
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    weights = np.ones_like(df_probe['AS/filler_ratio_filler_normalized']) / len(
        df_probe['AS/filler_ratio_filler_normalized'])

    ax.bar(bin_centers, hist_normalized, width=bin_edges[1] - bin_edges[0], alpha=0.7)
    ax.set_title(f"histogram for {os.path.basename(probe)}")
    ax.set_xlabel("AS_filler_ratio (filler_norm)")
    ax.set_ylabel("frequency")
    ax.set_xlim(-0.0005, 0.030) #target
    #ax.set_xlim(-0.002, 0.10)   #iAS

    ax = fig.add_subplot(gs[p,6])
    for i, cell in enumerate(unique_cells):
        cell_data = df_probe[df_probe['cell'] == cell]
        ax.scatter(cell_data['V_normalized'], cell_data['subtraction_mode_AS_filler_ratio'], s=50, edgecolor='none', alpha=0.5,
                   label='Control')
        # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x / 1000:.1f}k'))
        ax.set_xlabel('V (a.u.)')
        ax.set_ylabel(f'subtraction_ratio {os.path.basename(probe)}')
        # ax.set_ylim(-1000, 100000)
        ax.set_title('subtraction mode AS/filler ratio')
        ax.set_ylim(-0.002, 0.028)  #target
        #ax.set_ylim(-0.02, 0.10)  # iAS

    ax = fig.add_subplot(gs[(p, 7)])
    kde = stats.gaussian_kde(df_all['subtraction_mode_AS_filler_ratio'])
    x_vals = np.linspace(min(df_all['subtraction_mode_AS_filler_ratio']),
                         max(df_all['subtraction_mode_AS_filler_ratio']), 100)
    kde_vals = kde(x_vals)
    mode_value_kde = x_vals[np.argmax(kde_vals)]

    hist_all, bin_edges_all = np.histogram(df_probe['subtraction_mode_AS_filler_ratio'], bins=100, density=True)
    hist_max_all = hist_all.max()  # ヒストグラムの最大値
    hist_normalized_all = hist_all / hist_max_all  # 最大値で正規化
    bin_centers_all = 0.5 * (bin_edges_all[1:] + bin_edges_all[:-1])

    weights = np.ones_like(df_probe['subtraction_mode_AS_filler_ratio']) / len(
        df_probe['subtraction_mode_AS_filler_ratio'])

    ax.bar(bin_centers_all, hist_normalized_all, width=bin_edges_all[1] - bin_edges_all[0], alpha=0.7)
    ax.set_title(f"histogram subtraction for {os.path.basename(probe)}")
    # ax.set_xlabel("subtraction_mode_AS_filler_ratio")
    ax.set_ylabel("frequency")
    ax.set_xlim(-0.002, 0.01)   #target
    #ax.set_xlim(-0.02, 0.08)   #iAS
    # inset_ax = inset_axes(ax, width='40%', height='40%', loc="upper right")
    # inset_ax.set_xlim(0.005, 0.012)
    # inset_ax.set_ylim(0, 0.2)
    # inset_ax.bar(bin_centers_all, hist_normalized_all, width=bin_edges_all[1] - bin_edges_all[0], alpha=0.7)
    # # inset_counts, inset_bins, inset_patches = inset_ax.bar(df_probe['subtraction_mode_AS_filler_ratio'], bins=100, alpha=0.7)


    ax = fig.add_subplot(gs[(p, 8)])
    kde = stats.gaussian_kde(df_probe['subtraction_mode_AS_filler_ratio'])
    x_vals = np.linspace(min(df_probe['subtraction_mode_AS_filler_ratio']), max(df_probe['subtraction_mode_AS_filler_ratio']), 100)
    kde_vals = kde(x_vals)
    mode_value_kde = x_vals[np.argmax(kde_vals)]

    bin_width = 1.5e-04 # target
    #bin_width =
    bin_edges_all = np.arange(min(df_probe['subtraction_mode_AS_filler_ratio']), max(df_probe['subtraction_mode_AS_filler_ratio']) + bin_width, bin_width)

    hist_all, bin_edges_all = np.histogram(df_probe['subtraction_mode_AS_filler_ratio'], bins=bin_edges_all)
    hist_max_all = hist_all.max()  # ヒストグラムの最大値
    hist_normalized_all = hist_all / hist_max_all  # 最大値で正規化
    bin_centers_all = 0.5 * (bin_edges_all[1:] + bin_edges_all[:-1])

    # 左側データを抽出
    filtered_data = df_probe['subtraction_mode_AS_filler_ratio'][
        df_probe['subtraction_mode_AS_filler_ratio'] <= mode_value_kde]
    fold_data = mode_value_kde + np.abs(mode_value_kde - filtered_data)
    symmetric_data = np.concatenate([filtered_data, fold_data])
    print(os.path.basename(probe))
    print(len(filtered_data))
    print(min(filtered_data))
    print(len(fold_data))
    print(max(fold_data))
    print(len(symmetric_data))
    print(mode_value_kde)


    hist, bin_edges = np.histogram(symmetric_data, bins=bin_edges_all)
    hist_max = hist.max()
    hist_normalized = hist / hist_max
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    print(max(hist))


    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[max(hist), mode_value_kde, np.std(symmetric_data)],maxfev=10000)
    a_fit, b_fit, c_fit = popt
    c_fit_dict[os.path.basename(probe)] = c_fit
    mean_subtraction_ratio_dict[os.path.basename(probe)] = df_probe['subtraction_mode_AS_filler_ratio'].mean()

    gaussian_peak = gaussian(b_fit, *popt)
    scaling_factor = 1 / gaussian_peak
    a_fit_scaled = a_fit * scaling_factor

    x_fit = np.linspace(min(symmetric_data)-0.1, max(df_probe['subtraction_mode_AS_filler_ratio']), 10000)
    y_fit = gaussian(x_fit, a_fit_scaled, b_fit, c_fit)

    weights = np.ones_like(df_probe['subtraction_mode_AS_filler_ratio']) / len(df_probe['subtraction_mode_AS_filler_ratio'])

    ax.bar(bin_centers_all, hist_normalized_all, width=bin_edges_all[1] - bin_edges_all[0], alpha=0.7)
    ax.plot(x_fit, y_fit / max(y_fit), color='blue', label="Gaussian Fit (Left)")
    ax.set_title(f"histogram and gaussian fit for {os.path.basename(probe)}")
    ax.set_xlabel("subtraction_mode_AS_filler_ratio")
    ax.set_ylabel("frequency")
    #ax.set_xlim(-0.002, 0.012)  #target
    ax.set_xlim(-0.02, 0.08)  # iAS
    # inset_ax = inset_axes(ax, width='40%', height='40%', loc="upper right")
    # inset_ax.set_xlim(0.005, 0.012)  #target
    # inset_ax.set_ylim(0, 0.2)
    # inset_ax.bar(bin_centers_all, hist_normalized_all, width=bin_edges_all[1] - bin_edges_all[0], alpha=0.7)


    print(f"a_fit : {a_fit}, a_fit_scaled: {a_fit_scaled}, b_fit:{b_fit}, c_fit:{c_fit}")
    print(f"{os.path.basename(probe)} gaussian SD: {c_fit_dict[os.path.basename(probe)]}")

    ax = fig.add_subplot(gs[(p, 9)])
    # 横軸をz_scoreのhistogramにするために変換;zscoreの定義よりmodeの値が0なのでそこまでのデータでまたgaussian fitを行う
    z_scores = (df_probe['subtraction_mode_AS_filler_ratio'] - mode_value_kde) / c_fit
    filtered_zscore_data = z_scores[z_scores <= 0]
    folded_zscore_data = np.abs(filtered_zscore_data)
    symmetric_zscore_data = np.concatenate([filtered_zscore_data, folded_zscore_data])

    bin_width_z0 = 0.25
    bin_edges_z0 = np.arange(min(symmetric_zscore_data), max(symmetric_zscore_data) + bin_width_z0, bin_width_z0)
    # gussian fit scale after zscore normalization
    hist_z0, bin_edges_z0 = np.histogram(symmetric_zscore_data, bins=bin_edges_z0)
    hist_z0_max = hist_z0.max()
    hist_z0_normalized = hist_z0 / hist_z0_max
    bin_z0_centers = 0.5 * (bin_edges_z0[1:] + bin_edges_z0[:-1])

    popt, _ = curve_fit(gaussian, bin_z0_centers, hist_z0,
                        p0=[max(hist_z0), np.median(symmetric_zscore_data), 2 * np.std(symmetric_zscore_data)],
                        maxfev=10000)
    a_fit_z0, b_fit_z0, c_fit_z0 = popt


    gaussian_peak = gaussian(b_fit_z0, *popt)
    scaling_factor = 1 / gaussian_peak
    a_fit_z0_scaled = a_fit_z0 * scaling_factor


    x_fit_z0 = np.linspace(min(z_scores) - 5, max(z_scores), 1000)
    y_fit_z0 = gaussian(x_fit_z0, a_fit_z0_scaled, b_fit_z0, c_fit_z0)

    bin_edges_zscores = np.arange(min(z_scores), max(z_scores) + bin_width_z0, bin_width_z0)
    hist_zscores, bin_zscores = np.histogram(z_scores, bins=bin_edges_zscores)#, density=True)
    hist_zscores_max = hist_zscores.max()
    hist_zscores_normalized = hist_zscores / hist_zscores_max
    bin_zscores_centers = 0.5 * (bin_zscores[1:] + bin_zscores[:-1])
    print(f"max_hist_zscores:{hist_zscores.max()}")
    print(f"a_fit_z0 : {a_fit_z0}, a_fit_z0_scaled: {a_fit_z0_scaled}, b_fit_z0:{b_fit_z0}, c_fit_z0:{c_fit_z0}")

    #weights = np.ones_like(z_scores) / len(z_scores)
    ax.bar(bin_zscores_centers, hist_zscores_normalized, width=bin_width_z0, alpha=0.7)
    ax.plot(x_fit_z0, y_fit_z0 / max(y_fit_z0), color='blue', label="Gaussian Fit (Left)")
    ax.set_title(f"zscore_histogram and gaussian fit for {os.path.basename(probe)}")
    ax.set_xlabel("zscore")
    ax.set_ylabel("frequency")
    #ax.set_xlim(-10, 330)
    # # 縦軸を対数表記に設定
    # ax.set_yscale('log')
    #
    # # 必要に応じて軸の範囲を設定（例）
    # ax.set_ylim(1e-3, 1e0)  # 対数表記の範囲指定（適宜調整）

    ax = fig.add_subplot(gs[(p, 10)])
    threshold = b_fit_z0 + c_fit_z0 * 3
    zscores_above_threshold = z_scores[z_scores >= threshold]
    bin_width_zscore_above = bin_width_z0 * 10
    bin_edges_zscore_above = np.arange(threshold, max(z_scores) + bin_width_zscore_above, bin_width_zscore_above)
    hist_zscores_above, bin_zscores_above = np.histogram(zscores_above_threshold, bins=bin_edges_zscore_above)

    max_frequency = max(hist_zscores.max(), hist_zscores_above.max()/5)
    print(f"max_hist_zscores:{hist_zscores.max()}")
    print(f"max_hist_zscore_above:{hist_zscores_above.max()}")
    print(max_frequency)
    hist_zscore_normalized = hist_zscores  / max_frequency
    hist_zscores_above_normalized = (hist_zscores_above / (bin_width_zscore_above/bin_width_z0)) / max_frequency
    bin_zscores_above_centers = 0.5 * (bin_zscores_above[1:] + bin_zscores_above[:-1])
    print(f"hist_zscore_normalized:{hist_zscore_normalized}")
    print(f"hist_zscores_above_normalized:{hist_zscores_above_normalized}")
    print(f"bin_width_z0: {bin_width_z0}")
    print(f"bin_width_zscore_above:{bin_width_zscore_above}")
    for i in range (len(hist_zscores)):
        print(f"hist_zscores_Bin {i+1} ({bin_zscores[i]:.2f} to {bin_zscores[i+1]:.2f}): {hist_zscores[i]} elements")
    for j in range(len(hist_zscores_above)):
        print(f"hist_zscores_above_Bin {j+1} ({bin_zscores_above[j]:.2f} to {bin_zscores_above[j+1]:.2f}): {hist_zscores_above[j]} elements")

    ax.bar(bin_zscores_centers, hist_zscore_normalized, width=bin_width_z0, alpha=0.7)
    ax.bar(bin_zscores_above_centers, hist_zscores_above_normalized, width=bin_width_zscore_above, alpha=0.7)
    ax.plot(x_fit_z0, y_fit_z0 / max(y_fit_z0), color='blue', label="Gaussian Fit (Left)")
    ax.set_title(f"hist and gaussian fit for {os.path.basename(probe)}")
    ax.set_xlabel("zscore")
    ax.set_ylabel("frequency")
    ax.set_xlim(-10, 100) #target
    #ax.set_xlim(-10, 50)

    # 縦軸を対数表記に設定
    ax.set_yscale('log')

    # 必要に応じて軸の範囲を設定（例）
    ax.set_ylim(1e-3, 1e0)

# 全データの保存
# df_all.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\AAV_iAS\_all_data.csv", index=False)
# df_cell.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\AAV_iAS\_cell_data.csv", index=False)
df_all.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\distribution\_all_data.csv", index=False)
df_cell.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\distribution\_cell_data.csv", index=False)

# 各種指数の計算
probe_list = df_all['probe'].unique().tolist()
scoring = "subtraction_mode_AS_filler_ratio"
hotspot_index_df = pd.DataFrame(columns=["index_val", "probe", "cell", "dendrite"])
SD_df = pd.DataFrame(columns=["index_val", "probe", "cell", "dendrite"])
percentage_3sd_df = pd.DataFrame(columns=["index_val", "probe", "cell", "dendrite"])
above_2SD_mean_df = pd.DataFrame(columns=["index_val", "probe", "cell", "dendrite"])
AS_score_spine_df = pd.DataFrame(columns=["index_val", "probe", "cell", "dendrite"])
zscore_combined_df = pd.DataFrame()

for probe in df_all['probe'].unique():
    df = df_all[df_all['probe'] == probe]
    SD = c_fit_dict[os.path.basename(probe)]
    df["z_score"] = (df[scoring]) / SD

    print(df.columns)
    print(df.head())
    zscore_combined_df = pd.concat([zscore_combined_df, df[['z_score', 'probe', 'cell', 'dendrite']]], ignore_index=True)
    calculate_index(df, col_name="z_score", probe_name=probe, index_df=hotspot_index_df, index_type="hot_spot", c_fit=c_fit_dict[os.path.basename(probe)], ave=mean_subtraction_ratio_dict[os.path.basename(probe)])
    calculate_index(df, col_name="z_score", probe_name=probe, index_df=SD_df, index_type="SD", c_fit=c_fit_dict[os.path.basename(probe)], ave=mean_subtraction_ratio_dict[os.path.basename(probe)])
    calculate_index(df, col_name=scoring, probe_name=probe, index_df=percentage_3sd_df, index_type="percentage_above_3SD", c_fit=c_fit_dict[os.path.basename(probe)], ave=mean_subtraction_ratio_dict[os.path.basename(probe)])
    calculate_index(df, col_name="z_score", probe_name=probe, index_df=above_2SD_mean_df, index_type="above_2SD_mean", c_fit=c_fit_dict[os.path.basename(probe)], ave=mean_subtraction_ratio_dict[os.path.basename(probe)])

for probe in hotspot_index_avg:
    avg_hotspot_index = np.mean(hotspot_index_avg[probe])
    print(f"{probe}: {avg_hotspot_index}")

plot_graph(hotspot_index_df, position=(0,12), title="Hot spot index")
plot_graph(SD_df, position=(4,12), title="SD")
plot_graph(percentage_3sd_df, position=(5,12), title="Percentage of spines above 3SD")
plot_graph(above_2SD_mean_df, position=(6,12), title="above_2SD_mean")
plot_graph(zscore_combined_df.rename(columns={'z_score': 'index_val'}), position=(3, 12), title='z-score')



# 結果の保存
# hotspot_index_df.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\AAV_iAS\_hotspot_index.csv", index=False)
# SD_df.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\AAV_iAS\_standard_deviation.csv", index=False)
# percentage_3sd_df.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\AAV_iAS\_percentage_above_3SD.csv", index=False)
# above_2SD_mean_df.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\AAV_iAS\_above_2SD_mean.csv", index=False)
# zscore_combined_df.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\AAV_iAS\_z_score.csv", index=False)
hotspot_index_df.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\distribution\_hotspot_index.csv", index=False)
SD_df.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\distribution\_standard_deviation.csv", index=False)
percentage_3sd_df.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\distribution\_percentage_above_3SD.csv", index=False)
above_2SD_mean_df.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\distribution\_above_2SD_mean.csv", index=False)
zscore_combined_df.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\distribution\_z_score.csv", index=False)

# プロットをPDFに保存
plt.tight_layout()
#fig.savefig(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\AAV_iAS\_summary_NoAPV.pdf", dpi=100, transparent=True)
fig.savefig(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\distribution\_summary_NoAPV.pdf", dpi=100, transparent=True) 