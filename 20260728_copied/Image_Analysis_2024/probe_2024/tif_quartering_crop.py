import imagej
import os
from pathlib import Path
from tkinter import Tk, filedialog

# フォルダ選択ダイアログを表示
root = Tk()
root.withdraw()  # Tkinterのルートウィンドウを非表示にする
input_dir = filedialog.askdirectory(title="入力フォルダを選択してください")
output_dir = filedialog.askdirectory(title="出力フォルダを選択してください")

if not input_dir or not output_dir:
    print("フォルダが選択されていません。処理を中止します。")
    exit()

# TIFFファイルを数える
tiff_files = [f for f in os.listdir(input_dir) if f.endswith(".tif")]
num_files = len(tiff_files)
print(f"tiff file number: {num_files}")

if num_files == 0:
    print("処理対象のTIFFファイルが見つかりません。処理を終了します。")
    exit()

# ImageJのインスタンスを作成
ij = imagej.init('sc.fiji:fiji', mode='headless')

# 出力フォルダが存在しない場合は作成
Path(output_dir).mkdir(parents=True, exist_ok=True)

# フォルダ内のすべてのTIFFファイルに対して処理を行う
for file_name in tiff_files:
    image_path = os.path.join(input_dir, file_name).replace('\\', '/')
    output_path = os.path.join(output_dir, file_name.replace('.tif', '')).replace('\\', '/')
    print(f"Processing {file_name}...")

    # マクロコード (ROIマネージャーを使用しない)
    macro_code = f"""
        open("{image_path}");

        // 画像のサイズを取得
        width = getWidth();
        height = getHeight();

        // 4つのROIを定義して処理
        makeRectangle(0, 0, width/2, height/2); // 左上
        run("Crop");
        saveAs("Tiff", "{output_path}_region_left_upper.tif");
        close();
        open("{image_path}");

        makeRectangle(width/2, 0, width/2, height/2); // 右上
        run("Crop");
        saveAs("Tiff", "{output_path}_region_right_upper.tif");
        close();
        open("{image_path}");

        makeRectangle(0, height/2, width/2, height/2); // 左下
        run("Crop");
        saveAs("Tiff", "{output_path}_region_left_below.tif");
        close();
        open("{image_path}");

        makeRectangle(width/2, height/2, width/2, height/2); // 右下
        run("Crop");
        saveAs("Tiff", "{output_path}_region_right_below.tif");
        close();
        """

    # マクロコードをFijiで実行
    ij.py.run_macro(macro_code)

    print(f"Finished processing {file_name}")

print("すべてのファイルの処理が完了しました。")
