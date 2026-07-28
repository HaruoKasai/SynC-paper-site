import pandas as pd
import numpy as np
from scipy import stats
import itertools
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
from datetime import datetime
import os

# ===== GUI準備 =====
root = tk.Tk()
root.withdraw()


def process_file(file_path):
    """1つのCSVファイルに対してMann-Whitney U検定を実行し、結果を保存する"""
    df = pd.read_csv(file_path)
    print(f"\n=== {os.path.basename(file_path)} ===")
    print(df.head())

    # ===== 数値列のみを対象に =====
    all_cols = list(df.columns)
    num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        print(f"スキップ（数値列が2列未満）: {file_path}")
        return None

    target_cols = [c for c in all_cols if c in num_cols]

    # ===== Mann–Whitney U（全ペア） =====
    pairs, u_stats, p_vals, n1_list, n2_list = [], [], [], [], []

    for a, b in itertools.combinations(target_cols, 2):
        x = df[a].to_numpy()
        y = df[b].to_numpy()
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]

        if len(x) == 0 or len(y) == 0:
            U, p = np.nan, np.nan
        else:
            U, p = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")

        pairs.append(f"{a} vs {b}")
        u_stats.append(U)
        p_vals.append(p)
        n1_list.append(len(x))
        n2_list.append(len(y))

    mwu_df = pd.DataFrame({
        "test": "Mann-Whitney U (two-sided)",
        "comparison": pairs,
        "n_group1": n1_list,
        "n_group2": n2_list,
        "U_statistic": u_stats,
        "p_value": p_vals,
    })

    # ===== 保存先 =====
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(file_path)),
        os.path.basename(file_path)[:-8] + ".csv"
    )

    with open(save_path, "w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
        mwu_df.to_csv(f, index=False)

    print("Saved:", save_path)
    return save_path


# ===== メインループ：ファイル選択→処理→再度選択、を繰り返す =====
while True:
    file_paths = tkinter.filedialog.askopenfilenames(
        initialdir=r"\\Synology\zhou\SynC_Fig\_stats\_raw",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        title="解析するCSVファイルを選択（複数選択可・Ctrl/Shiftで複数選択）"
    )

    if not file_paths:
        # キャンセル or ファイル未選択 → ループ終了
        break

    saved_paths = []
    errors = []
    for fp in file_paths:
        try:
            sp = process_file(fp)
            if sp:
                saved_paths.append(sp)
        except Exception as e:
            errors.append(f"{os.path.basename(fp)}: {e}")

    msg = f"{len(saved_paths)} 件を保存しました。"
    if errors:
        msg += "\n\nエラー:\n" + "\n".join(errors)
    cont = tkinter.messagebox.askyesno(
        "完了",
        msg + "\n\n続けて別のファイルを選択しますか？"
    )
    if not cont:
        break

print("すべて終了しました。")