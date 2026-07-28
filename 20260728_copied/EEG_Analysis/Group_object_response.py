import numpy as np
import pandas as pd
from _archive.EEG_Analysis import plot_timeseries
import os
import matplotlib.pyplot as plt
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})
plt.rcParams['pdf.fonttype'] = 42
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import json
import lib.DLCAnalysis as DA
import glob
import seaborn as sns


def extract_event_windows(timeseries, event_frame_array, real_frame_time, t_pre, t_post, boundary):
    extracted_windows = []
    bout_duration = []
    for frame in event_frame_array:
        start = frame - int(t_pre / real_frame_time)
        end = frame + int(t_post / real_frame_time)
        window_size = end - start  # 期待する長さ
        window = np.full((window_size,), np.nan)

        ts_start = max(0, start)
        ts_end = min(len(timeseries), end)
        insert_start = ts_start - start  # NaN部分を考慮したデータの開始位置
        insert_end = insert_start + (ts_end - ts_start)

        window[insert_start:insert_end] = timeseries[ts_start:ts_end]
        extracted_windows.append (window)

        #bout duration
        subarray = window[int(t_pre / real_frame_time):]
        mask = subarray < boundary
        # 最初から連続するTrueの数を数える
        count = np.argmax(~mask) if np.any(~mask) else len(mask)
        bout_duration.append(count*real_frame_time)
    return extracted_windows, bout_duration


def process_object_responses(dlc_dir, exp_dir, object_time, contime, t_pre, t_post):
    arena_mm_per_pix = 0.6
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    dlc_output_dir = os.path.join(exp_dir, "_Object_responses")
    results = [[],[]]
    approaches_per_min = []
    bout_durations = []
    for i, index in enumerate(object_time): #基本はpre, post
        ax = fig.add_subplot(gs[0, i])
        dlc_exp_dir = glob.glob(os.path.join(dlc_dir, "day*"))[index]
        dlc_h5_path = os.path.join(dlc_exp_dir, "dlc_raw.h5")
        param_ind = os.path.join(dlc_exp_dir, "param_individual.json")
        df = pd.read_hdf(dlc_h5_path, key='dlc_data')
        if not os.path.exists(dlc_output_dir):
            os.makedirs(dlc_output_dir)
        # real_frame_time = (df["time"].iloc[-1] - df["time"].iloc[0]) / (len(df) - 1)  # real frame time
        # real_frame_time = round(pd.to_timedelta(real_frame_time).total_seconds(), 5)
        real_frame_time = 0.05

        object_coordinate = DA.get_roi_coordinate("Object", param_ind=param_ind)
        distance_to_object, frame_approaching, _ = DA.time_series_distance_to_object(df, object_coordinate, real_frame_time, arena_mm_per_pix, body_part = "centroid", distance_to_boundary_mm = 100, plot=True)
        peth_array, bout_duration= extract_event_windows(distance_to_object, frame_approaching, real_frame_time, t_pre, t_post, boundary=150)

        obj_exp_time = contime[index][1]-contime[index][0]
        print("obj_exp_time:" +str(obj_exp_time))
        approaches_per_min.append(len(peth_array)/obj_exp_time)
        bout_duration_mean = np.mean(bout_duration)
        bout_durations.append(bout_duration_mean)
        if peth_array:
            average =np.nanmean(peth_array, axis=0)
            tp = np.arange( -t_pre, t_post+real_frame_time, real_frame_time)
            for p, peth in enumerate(peth_array):
                plot_timeseries(tp, peth, 1, ax, plt.get_cmap("tab10")(p), 0.5, "Approach to Novel Object ", "Distance to object (mm/s)", (60,400), label=int(p))
            plot_timeseries(tp, average, 1, ax, "red", 2, "Approach to Novel Object ", "Distance to object (mm)",
                            (60, 500), label=None)
            if i>1: #仮で含める
                i=1
            results[i].append(average)

    plt.tight_layout()
    pdf_path = os.path.join(dlc_output_dir, "Object_response_PETH.pdf")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)
    return results, approaches_per_min, bout_durations


def process_folder(data_folder, t_pre, t_post):
    json_path = os.path.join(data_folder, "_analysis_param.json")
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    dlc_dir = data["DLC"].get("dir", None)
    contime = data["Time"]["Continuous"]
    object_time = data.get("Time", {}).get("Object", None)
    mouse_time = data.get("Time", {}).get("Mouse", None)

    # if object_time:
    # print("yes")
    results, approaches_per_min, bout_durations= process_object_responses(dlc_dir, data_folder, object_time, contime, t_pre, t_post)

    return results, approaches_per_min, bout_durations


def group_analysis_object(json_path, group_dict):
    dir = os.path.dirname(os.path.dirname(json_path))
    group_num = len(group_dict)
    fig = plt.figure(figsize=(10,5*group_num))
    gs = gridspec.GridSpec(group_num+1, 2) #height_ratios=[1, 1]
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    t_pre = 30
    t_post = 30
    df = pd.DataFrame(columns=["Group", "approaches_per_min", "bout_duration", "Before/After"])
    for g, (group, exp_list) in enumerate(group_dict.items()):
        print(group)
        result_array = [[] for _ in range(2)]
        approaches_per_min_array = [[] for _ in range(2)]
        bout_duration_array = [[] for _ in range(2)]
        for folder in exp_list:
            print(folder)
            path = os.path.join(dir, folder)
            results, approaches_per_min, bout_durations= process_folder(path, t_pre, t_post)
            results = [[x - 75 for x in sublist] for sublist in results] #70mm程度の余白
            for t in range(2):
                result_array[t].extend(results[t])
                approaches_per_min_array[t].append(approaches_per_min[t])
                bout_duration_array[t].append(bout_durations[t])
        # print("###########################")
        #
        # print(approaches_per_min_array)
        # print(bout_duration_array)

        a_values = np.array(approaches_per_min_array).flatten()
        b_values = np.array(bout_duration_array).flatten()
        group_array = np.repeat(group, len(a_values))
        before_after = np.repeat(["Before", "After"], np.array(approaches_per_min_array).shape[1])
        new_data = pd.DataFrame({
            "Group": group_array,
            "approaches_per_min": a_values,
            "bout_duration": b_values,
            "Before/After": before_after
        })
        df = pd.concat([df, new_data], ignore_index=True)


        title_list = ["Before", "After"]
        for t in range(2):
            ax = fig.add_subplot(gs[g,t])
            array = result_array[t]
            average = np.nanmean(array, axis=0)
            real_frame_time = 0.05
            tp = np.arange(-t_pre, t_post + real_frame_time, real_frame_time)
            for m, mouse_data in enumerate(array):
                plot_timeseries(tp, mouse_data, 1, ax, plt.get_cmap("tab10")(m), 0.25, None,
                                None, (0, 400), label=None)
            ylabel = group if t==0 else "Distance to object (mm/s)"
            if np.isscalar(average):
                print("average")
                print(average)
            else:
                plot_timeseries(tp, average, 1, ax, "red", 2.5, title_list[t], ylabel,
                            (0, 400), label=None)


    print(df)
    df.to_csv(os.path.join(dir, "_Group_Analysis_Behavior", "_Object_response_PETH_summary.csv"))


    #Approach per min ##
    ax = fig.add_subplot(gs[group_num,0])
    group_order = sorted(df["Group"].unique())
    condition_order = ["Before", "After"]

    # ---------- Barplot ----------
    sns.barplot(
        data=df,
        x="Group",
        y="approaches_per_min",
        hue="Before/After",
        hue_order=condition_order,
        order=group_order,
        estimator=np.mean,
        errorbar='se',
        capsize=0.2,
        palette="pastel",
        ax=ax,
        errwidth=1.5,
    )
    # ---------- 個体ごとの線プロット ----------
    # 前提: Before/After の順で並んでいてペア数が等しい（or調整する）
    for group_idx, group in enumerate(group_order):
        group_df = df[df["Group"] == group]
        before = group_df[group_df["Before/After"] == "Before"]["approaches_per_min"].reset_index(drop=True)
        after = group_df[group_df["Before/After"] == "After"]["approaches_per_min"].reset_index(drop=True)
        n = min(len(before), len(after))
        # seabornは hue で2カテゴリ分横にずれる → center ± offset
        x_before = group_idx - 0.2
        x_after = group_idx + 0.2
        for i in range(n):
            ax.plot([x_before, x_after], [before[i], after[i]], color="gray", alpha=0.5)
    ax.set_ylabel("Approaches per min")


    #####
    # bout_duration ##
    ax = fig.add_subplot(gs[group_num, 1])

    # ---------- Barplot ----------
    sns.barplot(
        data=df,
        x="Group",
        y="bout_duration",
        hue="Before/After",
        hue_order=condition_order,
        order=group_order,
        estimator=np.mean,
        errorbar='se',
        capsize=0.2,
        palette="pastel",
        ax=ax,
        errwidth=1.5,
    )
    # ---------- 個体ごとの線プロット ----------
    # 前提: Before/After の順で並んでいてペア数が等しい（or調整する）
    for group_idx, group in enumerate(group_order):
        group_df = df[df["Group"] == group]
        before = group_df[group_df["Before/After"] == "Before"]["bout_duration"].reset_index(drop=True)
        after = group_df[group_df["Before/After"] == "After"]["bout_duration"].reset_index(drop=True)
        n = min(len(before), len(after))
        # seabornは hue で2カテゴリ分横にずれる → center ± offset
        x_before = group_idx - 0.2
        x_after = group_idx + 0.2
        for i in range(n):
            ax.plot([x_before, x_after], [before[i], after[i]], color="gray", alpha=0.5)
    ax.set_ylabel("Bout duration (sec)")

    plt.tight_layout()
    pdf_path = os.path.join(dir, "_Group_Analysis_Behavior", "_Object_response_PETH_summary.pdf")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)


def main():
    # json_path = select_json_path()
    json_path =r"X:\Behavior\Openfield_EEG\_Group_Analysis_Behavior\_group_analysis_object_0422.json"
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    group_dict = data["Group"]
    group_analysis_object(json_path, group_dict)


if __name__ == "__main__":
    main()