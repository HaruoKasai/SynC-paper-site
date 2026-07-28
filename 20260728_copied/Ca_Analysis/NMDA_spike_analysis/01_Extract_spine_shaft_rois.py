"""
Interactive spine / shaft ROI selection and low-memory trace extraction
from suite2p-registered movies (data.bin).

Why this exists
----------------
suite2p's automated ROI detection + neuropil correction is built for soma-like
signals and is not appropriate for isolating individual dendritic spines from
their parent shaft. This script skips suite2p's detection/classifier entirely
and lets you hand-draw polygon ROIs (spine heads + paired local shaft) on the
mean/max projection image, then extracts raw fluorescence traces directly
from data.bin using a memory-mapped, chunked read -- so RAM usage stays small
even if data.bin is 90+ GB.

Workflow
--------
1. A file dialog asks you to select the suite2p 'planeX' folder that contains
   ops.npy and data.bin.
2. A menu lets you repeatedly:
     - "画新的 ROI"          draw a new polygon (scroll to zoom, 'r' to reset
                              view; pick spine/shaft via the on-plot radio
                              buttons; name is auto-generated: spine1, shaft1,
                              spine2, ...)
     - "撤销上一步"          undo the last draw / delete / pairing action
     - "查看/删除已有 ROI"   list current ROIs and remove one (cascades to
                              remove any pairs that referenced it)
     - "完成"                finish and extract traces
   Whenever you finish drawing a SPINE, you'll be asked how to link it to a
   shaft: click on an already-drawn shaft (no need to redraw it), type an
   existing shaft's name manually, or skip pairing for now. Each such link
   gets its own fresh, unique pair_id -- reusing the same shaft for several
   different spines will NOT make their pair_ids collide.
3. Outputs (written next to the input folder):
     - <session>_roi_masks.npz    boolean masks + ROI metadata (name, type)
     - <session>_roi_overview.pdf projection image with ROI outlines + pair
                                   labels, for QC
     - <session>_roi_traces.csv   long-format RAW fluorescence, one row per
                                   (roi_name, frame) -- each physical ROI is
                                   stored only once, even if reused in several
                                   pairs
     - <session>_pairs.csv        pair_id, spine_name, shaft_name -- join
                                   this against roi_traces.csv downstream to
                                   get matched spine/shaft traces per pair

This script only extracts RAW fluorescence. Event detection / peak_dff /
Wilcoxon comparisons stay in your existing pandas pipeline.
"""

import os
import sys
import copy
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Compatibility shim: ops.npy may have been pickled under NumPy 2.x, which
# renamed the internal 'numpy.core' package to 'numpy._core'. If you're
# running this under NumPy 1.x, that module name doesn't exist and np.load
# fails with "ModuleNotFoundError: No module named 'numpy._core'". This maps
# the new name back onto the old package so unpickling works without an
# upgrade. Safe no-op if you're already on NumPy 2.x.
# --------------------------------------------------------------------------
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core
    for _sub in ("multiarray", "numeric", "umath", "fromnumeric",
                 "_multiarray_umath", "records", "arrayprint"):
        _mod = getattr(np.core, _sub, None)
        if _mod is not None:
            sys.modules[f"numpy._core.{_sub}"] = _mod

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector, RadioButtons
from matplotlib.path import Path

import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox


# --------------------------------------------------------------------------
# File selection
# --------------------------------------------------------------------------
def select_suite2p_folder(root):
    folder = filedialog.askdirectory(
        title="Select suite2p plane folder (must contain ops.npy and data.bin)",
        parent=root
    )
    if not folder:
        print("No folder selected, exiting.")
        sys.exit(0)
    ops_path = os.path.join(folder, "ops.npy")
    bin_path = os.path.join(folder, "data.bin")
    if not os.path.exists(ops_path) or not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"ops.npy or data.bin not found in {folder}. "
            "Point this at the suite2p 'planeX' folder, not a parent folder."
        )
    return folder, ops_path, bin_path


# --------------------------------------------------------------------------
# Projection image for drawing ROIs on
# --------------------------------------------------------------------------
def get_projection_image(ops):
    if "max_proj" in ops and ops["max_proj"] is not None:
        img = np.asarray(ops["max_proj"], dtype=np.float32)
    else:
        img = np.asarray(ops["meanImg"], dtype=np.float32)
    lo, hi = np.percentile(img, [1, 99.5])
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return img


# --------------------------------------------------------------------------
# Shared zoom / reset-view behavior for any figure showing the projection image
# --------------------------------------------------------------------------
def attach_zoom(fig, ax, image):
    def on_scroll(event):
        if event.inaxes != ax or event.xdata is None:
            return
        base_scale = 1.2
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        scale = 1 / base_scale if event.button == "up" else base_scale
        xdata, ydata = event.xdata, event.ydata
        new_w = (cur_xlim[1] - cur_xlim[0]) * scale
        new_h = (cur_ylim[1] - cur_ylim[0]) * scale
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        ax.set_xlim([xdata - new_w * (1 - relx), xdata + new_w * relx])
        ax.set_ylim([ydata - new_h * (1 - rely), ydata + new_h * rely])
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "r":
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(image.shape[0], 0)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)


def draw_existing_rois(ax, masks, only_type=None):
    """masks: name -> (mask, roi_type)"""
    for name, (mask, roi_type) in masks.items():
        if only_type is not None and roi_type != only_type:
            continue
        ys, xs = np.where(mask)
        if len(ys):
            color = "tab:red" if roi_type == "spine" else "tab:cyan"
            ax.scatter(xs, ys, s=1, alpha=0.35, color=color)
            ax.annotate(name, (xs.mean(), ys.mean()), color="yellow", fontsize=8)


# --------------------------------------------------------------------------
# Generic multi-button chooser (replaces the old linear yes/no-only flow)
# --------------------------------------------------------------------------
def ask_action(root, title, options):
    result = {"value": None}
    top = tk.Toplevel(root)
    top.title(title)
    tk.Label(top, text=title, padx=20, pady=10).pack()

    def make_cb(opt):
        def cb():
            result["value"] = opt
            top.destroy()
        return cb

    for opt in options:
        tk.Button(top, text=opt, width=32, command=make_cb(opt)).pack(padx=20, pady=4)

    def on_close():
        result["value"] = None
        top.destroy()

    top.protocol("WM_DELETE_WINDOW", on_close)
    top.grab_set()
    root.wait_window(top)
    return result["value"]


# --------------------------------------------------------------------------
# Interactive polygon drawing with in-figure type selector + zoom
# --------------------------------------------------------------------------
def draw_polygon_roi(image, existing_masks, default_type="spine", title=""):
    verts_holder = []
    type_holder = [default_type]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray")
    draw_existing_rois(ax, existing_masks)

    ax.set_title(
        (title or "Draw one ROI")
        + "\nScroll = zoom, 'r' = reset view. Click vertices, close near the "
          "start point, pick spine/shaft on the left, then close this window."
    )

    def onselect(vertices):
        verts_holder.clear()
        verts_holder.extend(vertices)

    selector = PolygonSelector(ax, onselect)  # noqa: F841 keep alive

    radio_ax = fig.add_axes([0.01, 0.45, 0.11, 0.12])
    radio = RadioButtons(radio_ax, ("spine", "shaft"),
                          active=0 if default_type == "spine" else 1)

    def on_type_change(label):
        type_holder[0] = label

    radio.on_clicked(on_type_change)  # noqa: F841

    attach_zoom(fig, ax, image)
    plt.show()  # blocks until window closed
    return verts_holder, type_holder[0]


def polygon_to_mask(verts, Ly, Lx):
    if len(verts) < 3:
        return np.zeros((Ly, Lx), dtype=bool)
    y_grid, x_grid = np.mgrid[0:Ly, 0:Lx]
    points = np.vstack((x_grid.ravel(), y_grid.ravel())).T
    path = Path(verts)
    mask = path.contains_points(points).reshape(Ly, Lx)
    return mask


# --------------------------------------------------------------------------
# Click-to-pick an existing shaft (instead of retyping / redrawing)
# --------------------------------------------------------------------------
def pick_existing_shaft(image, existing_shafts):
    """existing_shafts: name -> (mask, roi_type), all roi_type == 'shaft'"""
    picked_holder = [None]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray")
    draw_existing_rois(ax, existing_shafts)
    ax.set_title(
        "Click on (or near) the shaft this spine belongs to, "
        "then close the window.\nScroll = zoom, 'r' = reset view."
    )

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        for name, (mask, _t) in existing_shafts.items():
            Ly, Lx = mask.shape
            if 0 <= y < Ly and 0 <= x < Lx and mask[y, x]:
                picked_holder[0] = name
                plt.close(fig)
                return
        best_name, best_dist = None, None
        for name, (mask, _t) in existing_shafts.items():
            ys, xs = np.where(mask)
            if len(ys):
                d = np.min((ys - y) ** 2 + (xs - x) ** 2)
                if best_dist is None or d < best_dist:
                    best_dist, best_name = d, name
        picked_holder[0] = best_name
        plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    attach_zoom(fig, ax, image)
    plt.show()
    return picked_holder[0]


def manual_shaft_name_or_none(root, masks):
    name = simpledialog.askstring(
        "Shaft 名称",
        "输入要关联的 shaft ROI 名称（留空则暂不配对）:",
        parent=root
    )
    if name:
        name = name.strip()
        if name in masks and masks[name][1] == "shaft":
            return name
        print(f"  未找到名为 '{name}' 的 shaft ROI，暂不配对。")
    return None


def ask_shaft_link(root, image, masks):
    """Returns the shaft name to pair a just-drawn spine with, or None."""
    shaft_names = [n for n, (_m, t) in masks.items() if t == "shaft"]
    if not shaft_names:
        messagebox.showinfo(
            "还没有 shaft",
            "目前还没有画过任何 shaft ROI，可以先手动输入名称占位，"
            "或者暂不配对，之后画好 shaft 再回来配对。",
            parent=root
        )
        return manual_shaft_name_or_none(root, masks)

    action = ask_action(root, "这个 spine 要配对到哪个 shaft？",
                         ["点击已有 shaft", "手动输入 shaft 名称", "暂不配对"])
    if action == "点击已有 shaft":
        shaft_subset = {n: masks[n] for n in shaft_names}
        return pick_existing_shaft(image, shaft_subset)
    elif action == "手动输入 shaft 名称":
        return manual_shaft_name_or_none(root, masks)
    else:
        return None


# --------------------------------------------------------------------------
# View / delete existing ROIs
# --------------------------------------------------------------------------
def manage_roi_list(root, masks, pairs):
    top = tk.Toplevel(root)
    top.title("已有 ROI 列表")
    tk.Label(top, text="选中一个 ROI，点“删除选中的 ROI”\n"
                        "（删除 shaft 会一并移除引用它的配对）").pack(padx=10, pady=6)

    lb = tk.Listbox(top, width=55, height=15, selectmode=tk.SINGLE)
    names = list(masks.keys())
    for n in names:
        mask, t = masks[n]
        n_pairs = sum(1 for p in pairs if p["spine"] == n or p["shaft"] == n)
        lb.insert(tk.END, f"{n}  ({t}, {int(mask.sum())} px, {n_pairs} 个配对引用)")
    lb.pack(padx=10, pady=6)

    selected = {"name": None}

    def on_delete():
        sel = lb.curselection()
        if sel:
            selected["name"] = names[sel[0]]
            top.destroy()

    def on_close():
        top.destroy()

    tk.Button(top, text="删除选中的 ROI", command=on_delete).pack(pady=4)
    tk.Button(top, text="关闭（不删除）", command=on_close).pack(pady=4)
    top.grab_set()
    root.wait_window(top)
    return selected["name"]


# --------------------------------------------------------------------------
# Main interactive collection loop, with undo
# --------------------------------------------------------------------------
def collect_rois(root, image):
    masks = {}          # name -> (mask, roi_type)
    pairs = []           # list of {"pair_id": int, "spine": name, "shaft": name}
    counters = {"spine": 0, "shaft": 0}
    next_pair_id = [1]
    undo_stack = []
    default_type = "spine"

    def snapshot():
        undo_stack.append((
            copy.deepcopy(masks), copy.deepcopy(pairs),
            copy.deepcopy(counters), next_pair_id[0]
        ))

    while True:
        action = ask_action(
            root, "下一步操作",
            ["画新的 ROI", "撤销上一步", "查看/删除已有 ROI", "完成"]
        )
        if action is None or action == "完成":
            break

        elif action == "画新的 ROI":
            verts, roi_type = draw_polygon_roi(
                image, masks, default_type=default_type,
                title=f"下一个 ROI（当前类型：{default_type}）"
            )
            if len(verts) < 3:
                print("Polygon too small / not closed, skipping this ROI.")
                continue

            snapshot()
            counters[roi_type] += 1
            name = f"{roi_type}{counters[roi_type]}"
            mask = polygon_to_mask(verts, image.shape[0], image.shape[1])
            masks[name] = (mask, roi_type)
            print(f"  saved ROI '{name}' ({roi_type}), {int(mask.sum())} pixels")
            default_type = roi_type

            if roi_type == "spine":
                shaft_name = ask_shaft_link(root, image, masks)
                if shaft_name is not None:
                    pid = next_pair_id[0]
                    next_pair_id[0] += 1
                    pairs.append({"pair_id": pid, "spine": name, "shaft": shaft_name})
                    print(f"  linked '{name}' <-> '{shaft_name}' as pair {pid}")
                else:
                    print(f"  '{name}' not paired yet (delete/redraw or pair it "
                          f"later by editing pairs.csv if needed).")

        elif action == "撤销上一步":
            if undo_stack:
                masks_prev, pairs_prev, counters_prev, pid_prev = undo_stack.pop()
                masks.clear(); masks.update(masks_prev)
                pairs.clear(); pairs.extend(pairs_prev)
                counters.clear(); counters.update(counters_prev)
                next_pair_id[0] = pid_prev
                print("  已撤销上一步操作。")
            else:
                print("  没有可撤销的操作了。")

        elif action == "查看/删除已有 ROI":
            to_delete = manage_roi_list(root, masks, pairs)
            if to_delete is not None:
                snapshot()
                del masks[to_delete]
                removed = [p for p in pairs if p["spine"] == to_delete or p["shaft"] == to_delete]
                for p in removed:
                    pairs.remove(p)
                print(f"  已删除 ROI '{to_delete}'"
                      + (f"，并移除了 {len(removed)} 条相关配对。" if removed else "。"))

    return masks, pairs


# --------------------------------------------------------------------------
# Low-memory trace extraction
# --------------------------------------------------------------------------
def extract_traces(bin_path, ops, masks, chunk_size=2000):
    Ly, Lx = ops["Ly"], ops["Lx"]
    nframes = ops["nframes"]
    dtype = ops.get("dtype", "int16")

    mov = np.memmap(bin_path, dtype=dtype, mode="r", shape=(nframes, Ly, Lx))

    pixel_idx = {name: np.where(mask) for name, (mask, _t) in masks.items()}
    traces = {name: np.zeros(nframes, dtype=np.float32) for name in masks}

    print(f"Extracting traces for {len(masks)} ROIs across {nframes} frames "
          f"(chunk size {chunk_size})...")
    for start in range(0, nframes, chunk_size):
        end = min(start + chunk_size, nframes)
        block = np.asarray(mov[start:end])  # only this chunk is read from disk
        for name, (ys, xs) in pixel_idx.items():
            traces[name][start:end] = block[:, ys, xs].mean(axis=1)
        print(f"  frames {start}-{end}/{nframes}", end="\r")
    print("\nDone extracting traces.")
    return traces


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    root = tk.Tk()
    root.withdraw()

    folder, ops_path, bin_path = select_suite2p_folder(root)
    ops = np.load(ops_path, allow_pickle=True).item()

    mouse_id = simpledialog.askstring("Mouse ID", "Mouse identifier for this session:", parent=root)
    mouse_id = mouse_id or os.path.basename(os.path.normpath(folder))

    proj_img = get_projection_image(ops)

    print("Draw ROIs on the projection image (spine heads + paired local shaft).")
    masks, pairs = collect_rois(root, proj_img)
    if not masks:
        print("No ROIs drawn, exiting.")
        root.destroy()
        return

    session_name = os.path.basename(os.path.normpath(folder))
    out_dir = os.path.dirname(os.path.normpath(folder)) or "."

    # --- save masks + metadata for reproducibility ---
    masks_out = os.path.join(out_dir, f"{session_name}_roi_masks.npz")
    np.savez_compressed(
        masks_out,
        **{f"{name}__mask": m for name, (m, _t) in masks.items()},
        roi_meta=np.array(
            [(name, t) for name, (_m, t) in masks.items()],
            dtype=[("name", "U64"), ("roi_type", "U16")]
        ),
        pair_meta=np.array(
            [(p["pair_id"], p["spine"], p["shaft"]) for p in pairs],
            dtype=[("pair_id", "i4"), ("spine", "U64"), ("shaft", "U64")]
        ) if pairs else np.array([], dtype=[("pair_id", "i4"), ("spine", "U64"), ("shaft", "U64")]),
    )
    print(f"Saved ROI masks: {masks_out}")

    # --- QC overview figure ---
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(proj_img, cmap="gray")
    pair_lookup = {}
    for p in pairs:
        pair_lookup.setdefault(p["spine"], []).append(p["pair_id"])
        pair_lookup.setdefault(p["shaft"], []).append(p["pair_id"])
    for name, (mask, roi_type) in masks.items():
        ys, xs = np.where(mask)
        color = "tab:red" if roi_type == "spine" else "tab:cyan"
        ax.scatter(xs, ys, s=1, alpha=0.4, color=color)
        pid_str = ",".join(str(x) for x in pair_lookup.get(name, [])) or "-"
        ax.annotate(f"{name}\n(pair {pid_str})", (xs.mean(), ys.mean()),
                    color="yellow", fontsize=7)
    ax.set_title(f"{session_name} ROI overview  (red=spine, cyan=shaft)")
    ax.axis("off")
    overview_out = os.path.join(out_dir, f"{session_name}_roi_overview.pdf")
    fig.savefig(overview_out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROI overview figure: {overview_out}")

    # --- extract traces (each physical ROI extracted exactly once) ---
    traces = extract_traces(bin_path, ops, masks)

    rows = []
    for name, (_mask, roi_type) in masks.items():
        trace = traces[name]
        for frame_i, val in enumerate(trace):
            rows.append((mouse_id, name, roi_type, frame_i, val))
    df = pd.DataFrame(rows, columns=["mouse", "roi_name", "roi_type", "frame", "raw_f"])
    traces_out = os.path.join(out_dir, f"{session_name}_roi_traces.csv")
    df.to_csv(traces_out, index=False)
    print(f"Saved traces: {traces_out}")

    # --- pairs table (unique pair_id per spine-shaft link, even if a shaft is reused) ---
    pairs_df = pd.DataFrame(pairs, columns=["pair_id", "spine", "shaft"])
    pairs_df.insert(0, "mouse", mouse_id)
    pairs_out = os.path.join(out_dir, f"{session_name}_pairs.csv")
    pairs_df.to_csv(pairs_out, index=False)
    print(f"Saved pairs table: {pairs_out}")

    root.destroy()


if __name__ == "__main__":
    main()