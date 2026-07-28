import json
import os
import tkinter.filedialog
import tkinter.messagebox

data = {
    "tp_list":[
        0,
        60,
        120,
        180,
        240,
        300
    ],
    "tp_base": [
        0,
        1
    ],
    "shade_min": 0,
    "ms": "LeicaSP8",
    "spine_dict":{
        "stim":[
            0,
            15
        ]
    }

}

root = tkinter.Tk()
root.withdraw()
dir = tkinter.filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate")

file_path = os.path.join(dir, "_param.json")
with open(file_path, "w") as json_file:
    json.dump(data, json_file, indent=4)


