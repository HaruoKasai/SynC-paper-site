import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})
plt.rcParams['pdf.fonttype'] = 42
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import lib.DLCAnalysis as DA
from Group_Analysis_Centertime import calculate_center_time


def Openfield_for_M1(index, folder, event_df, start_time, velocity_boundary):
    arena_mm_per_pix = 0.4
    dlc_h5_path = os.path.join(folder, "dlc_raw.h5")
    param_ind = os.path.join(folder, "param_individual.json")
    df = pd.read_hdf(dlc_h5_path, key='dlc_data')
    dlc_output_dir = os.path.join(folder, "_DLC_analysis")
    if not os.path.exists(dlc_output_dir):
        os.makedirs(dlc_output_dir)
    df.to_csv(os.path.join(dlc_output_dir, "dlc_data.csv"))

    real_frame_time = (df["time"].iloc[-1] - df["time"].iloc[0]) / (len(df) - 1)  # real frame time
    real_frame_time = round(pd.to_timedelta(real_frame_time).total_seconds(), 5)
    velocity, cumulative_distance, frames_extracted_by_velocity, likelihood = DA.time_series_velocity(df, real_frame_time,arena_mm_per_pix,"centroid", velocity_boundary)

    for v in range(len(frames_extracted_by_velocity)):
        event_name = "Centroid~" + str(velocity_boundary[v]) + "mm_per_s"
        event_df = DA.frame_to_sec(frames_extracted_by_velocity[v], real_frame_time, event_df, event_name,start_time, tolerable_frame_drop=2, min_duration=10) #min_duration:sec

    arena_coordinate = DA.get_roi_coordinate("arena_box", param_ind=param_ind)
    distance_to_center, frame_approaching, frame_leaving = DA.time_series_distance_to_object(df, arena_coordinate,real_frame_time,arena_mm_per_pix,body_part="centroid",distance_to_boundary_mm=100)

    event_df.to_csv(os.path.join(dlc_output_dir, "event.csv"))

    return real_frame_time, velocity, cumulative_distance, distance_to_center, event_df, likelihood

def process_dir(dir):
    exp_list = glob.glob(os.path.join(dir, "z*"))
    time_list = [[0,30], [-15,0]]
    data_list = [[] for _ in range(2)]  # before vs after
    vdata_list = [[] for _ in range(2)]
    time_blocks = [[-3, 0], [3, 90]] #to be analyzed
    for exp in exp_list:
        print(exp)
        event_combined = pd.DataFrame(columns=["start_time", "end_time", "event_name"])
        for t, time in enumerate(["after", "before"]):
            folder =glob.glob(os.path.join(exp, "*"+time+"*"))[0]
            event_df = pd.DataFrame(columns=["start_time", "end_time", "event_name"])
            velocity_boundary =[10] #mm/s
            video_frame_time, velocity, cumulative_distance, distance_to_center,event_df, likelihood= Openfield_for_M1(t, folder, event_df, time_list[1][0]*60, velocity_boundary)
            # velocity = velocity[:int(exp_dur / video_frame_time)]
            # likelihood = likelihood[:int(exp_dur / video_frame_time)]
            # distance_to_center = distance_to_center[:int(exp_dur / video_frame_time)]
            OF_tp = time_list[t][0] * 60 + np.arange(len(velocity)) * video_frame_time
            # print(video_frame_time)
            event_combined = pd.concat([event_combined,event_df], ignore_index=True )
            if t==1:
                event_combined.to_csv(os.path.join(exp, "event_combined.csv"))
            # print(time)
            # print(velocity)
            # print(velocity.shape)
            if event_df is None:
                active_velocity = velocity
            else:
                # active_velocity = event_mask(OF_tp, velocity, event_df, 5)
                active_velocity = velocity
            # print(active_velocity)
            # print(OF_tp)
            # print(time_blocks[1-t][0] * 60)
            mask = (OF_tp >= time_blocks[1-t][0] * 60) & (OF_tp <= time_blocks[1-t][1] * 60)
            masked_velocity = active_velocity[mask]
            # np.set_printoptions(threshold=np.inf)
            binned = active_velocity.reshape(20, -1).mean(axis=1)
            print(binned)


            # print(masked_velocity)
            mean_velocity = np.nanmean(masked_velocity)
            vdata_list[1-t].append(mean_velocity)


        data_list = calculate_center_time(exp, time_list, data_list, event_df, time_blocks)


    df = pd.DataFrame({"Before": data_list[0], "After": data_list[1]})
    df.to_csv(os.path.join(r"X:\Behavior\Openfield_EEG\_Group_Analysis_Behavior", "Center_time_M1local.csv"))
    df_melted = df.melt(var_name="Group", value_name="Time in center (%)")

    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    sns.barplot(x="Group", y="Time in center (%)", data=df_melted, estimator=np.mean, errorbar=('ci', 68),
                edgecolor='black', alpha=1, facecolor='none')
    for i in range(len(data_list[0])):
        plt.plot(["Before", "After"], [data_list[0][i], data_list[1][i]], color="gray",  # linestyle="--",
                 alpha=0.5)
    plt.ylim(0, 60)

    plt.tight_layout()
    pdf_path = os.path.join(r"X:\Behavior\Openfield_EEG\_Group_Analysis_Behavior", "Center_time_M1local.pdf")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)






    df = pd.DataFrame({"Before": vdata_list[0], "After": vdata_list[1]})
    df.to_csv(os.path.join(r"X:\Behavior\Openfield_EEG\_Group_Analysis_Behavior", "Velocity_M1local.csv"))
    df_melted = df.melt(var_name="Group", value_name="Velocity")

    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    sns.barplot(x="Group", y="Velocity", data=df_melted, estimator=np.mean, errorbar=('ci', 68),
                edgecolor='black', alpha=1, facecolor='none')
    for i in range(len(vdata_list[0])):
        plt.plot(["Before", "After"], [vdata_list[0][i], vdata_list[1][i]], color="gray",  # linestyle="--",
                 alpha=0.5)
    plt.ylim(0, 60)

    plt.tight_layout()
    pdf_path = os.path.join(r"X:\Behavior\Openfield_EEG\_Group_Analysis_Behavior", "Velocity_M1local.pdf")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)


def main():
    # json_path = select_json_path()
    dir = r"Z:\ProbeG\cond_M1"
    process_dir(dir)


if __name__ == "__main__":
    main()