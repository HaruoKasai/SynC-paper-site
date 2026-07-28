import os
import tkinter as tk
from tkinter import filedialog
import glob
import PETH
import EEG_Analysis


def main():
    root = tk.Tk()
    root.withdraw()
    parent_folder_path = filedialog.askdirectory(
        title="Select the parent directory",
        # initialdir=r"X:\Behavior"
        initialdir=r"X:\Behavior"
    )
    mouse_list = glob.glob(os.path.join(parent_folder_path, "[!_]*"))
    for mouse in mouse_list:
        if os.path.exists(os.path.join(mouse, "_analysis_param.json")):
            EEG_Analysis.process_folder(mouse)
            # if os.path.exists(os.path.join(mouse, "_Combined", "manual_event.csv")):
            #     PETH.process_folder(mouse)
            # Object_response.process_folder(mouse)
            # Light_timer.extract_light_timing(mouse, "top", 60, 60, "bottom-left")
            # Light_timer.extract_light_timing(mouse, "side", 200, 400, "bottom-right")
        # if os.path.exists(os.path.join(mouse, "_Combined", "manual_event.csv")):
        #     PETH.process_folder(mouse)
            # Movie_Behavior_EEG.process_folder(mouse)
        # if os.path.exists(os.path.join(mouse, "_Combined", "food_approach.csv")):
        #     Food_movie.process_folder(mouse)



if __name__ == "__main__":
    main()


