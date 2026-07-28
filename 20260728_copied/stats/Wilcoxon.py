import pandas as pd
import tkinter as tk
from tkinter import filedialog
from scipy.stats import wilcoxon

# Choose *.csv file
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Choose the csv file",
    filetypes=[("CSV files", "*.csv")]
)

if not file_path:
    print("No selection, finished.")
else:
    print(f"Choosing: {file_path}")

    # Read CSV
    df = pd.read_csv(file_path)

    # Detect columns
    if df.shape[1] < 2:
        raise ValueError("At least 2 groups")

    # Read names（Row1 as group names）
    group_names = list(df.columns)

    print("Group names:", group_names)

    # Read the datas
    group1 = df[group_names[0]].dropna().values
    group2 = df[group_names[1]].dropna().values

    if len(group1) != len(group2):
        raise ValueError("Unbalanced data, Please check.")

    # Calculate the difference
    differences = group1 - group2


    # Wilcoxon signed rank test
    stat, p_value = wilcoxon(group1, group2)
    method = "Wilcoxon signed rank test"

    # === Final reuslt ===
    print(f"Method: {method}")
    print(f"Nums = {stat:.0f}, p = {p_value:.8f}")

    # === Significant ===
    if p_value < 0.05:
        print("Significant: True (p < 0.05)")
    else:
        print("Significant: False (p ≥ 0.05)")

