import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog

def filter_and_save_files(folder_path):
    # 指定されたフォルダ内の全てのファイルを処理
    for file_name in os.listdir(folder_path):
        if file_name.endswith('props.csv'):
            file_path = os.path.join(folder_path, file_name)
            # CSVファイルを読み込む
            data = pd.read_csv(file_path)
            # 'analysis'がTrueの行をフィルタリング
            filtered_data = data[data['analysis'] == True]
            # 必要な列を選択
            selected_columns = filtered_data[['label', 'area', 'mean_intensity-0', 'mean_intensity-1']]
            # 最後の1行を除外
            selected_columns = selected_columns[:-1]
            # 新しいファイル名を生成
            new_file_name = f'filtered_{file_name.replace("props.csv", "spine.csv")}'
            output_file_path = os.path.join(folder_path, new_file_name)
            # フィルタリングされたデータを新しいCSVファイルに保存
            selected_columns.to_csv(output_file_path, index=False)
            print(f'Filtered data saved to: {output_file_path}')

def main():
    root = tk.Tk()
    root.withdraw()  # Tkinterウィンドウを隠す

    # フォルダ選択ダイアログを表示
    folder_path = filedialog.askdirectory(title="Select Folder", initialdir=r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate")
    if folder_path:
        filter_and_save_files(folder_path)
    else:
        print("No folder selected.")

if __name__ == "__main__":
    main()


