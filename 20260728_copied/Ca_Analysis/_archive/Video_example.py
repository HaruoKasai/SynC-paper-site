# pip install tifffile opencv-python numpy tqdm
from tifffile import TiffFile
import numpy as np
import cv2
from tqdm import tqdm
from math import ceil
from typing import Optional

def _estimate_contrast(tif, pages, sample_frames=200, p_low=1.0, p_high=99.5):
    """巨大TIFFのごく一部をサンプリングして全体の輝度範囲を推定"""
    idxs = np.linspace(0, len(pages)-1, min(sample_frames, len(pages))).astype(int)
    vals = []
    for i in idxs:
        arr = pages[i].asarray()
        if arr.ndim == 3:  # HWC
            # 輝度推定のためにY風にまとめる（単純平均でもOK）
            arr_mono = arr.mean(axis=-1)
        else:
            arr_mono = arr
        vals.append(arr_mono.ravel()[::64])  # 間引きで軽量化
    vals = np.concatenate(vals)
    lo = np.percentile(vals, p_low)
    hi = np.percentile(vals, p_high)
    if hi <= lo:
        hi = lo + 1
    return float(lo), float(hi)

def _to_uint8(frame, lo, hi):
    frame = np.clip((frame - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return frame

def tiff_to_mp4(
    tiff_path: str,
    out_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,   # ← 修正
    speed: float = 1.0,
    mode: str = "fps",
    resize: float = 1.0,
    input_fps: float = 31.0,
    fourcc: str = "mp4v",
):
    assert speed > 0, "speed は正の値にしてください"
    with TiffFile(tiff_path) as tf:
        pages = tf.pages
        n_pages = len(pages)
        if end_frame is None:
            end_frame = n_pages
        start = max(0, start_frame)
        end = min(end_frame, n_pages)
        if end <= start:
            raise ValueError("範囲指定が不正です")

        # サイズ・カラー判定
        sample = pages[start].asarray()
        if sample.ndim == 2:
            h, w = sample.shape
            is_gray = True
        elif sample.ndim == 3:
            h, w, c = sample.shape
            is_gray = False
        else:
            raise ValueError("想定外の配列形状です: ndim={}".format(sample.ndim))

        # コントラスト推定（パーセンタイル）
        lo, hi = _estimate_contrast(tf, pages)

        # 出力FPSとスキップステップ
        if mode == "fps":
            out_fps = input_fps * speed
            step = 1
        elif mode == "skip":
            out_fps = input_fps
            step = max(1, int(round(speed)))
        else:
            raise ValueError('mode は "fps" か "skip" を指定してください')

        # リサイズ後のサイズ
        out_w = int(round(w * resize))
        out_h = int(round(h * resize))

        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*fourcc),
            out_fps,
            (out_w, out_h),
            True,  # True=カラー動画として書く（グレーでも3ch化して扱う）
        )
        if not writer.isOpened():
            raise RuntimeError("VideoWriterを開けません。コーデックや出力パスを確認してください。")

        try:
            # 進捗バー
            total = ceil((end - start) / step)
            for i in tqdm(range(start, end, step), total=total, desc="Writing"):
                arr = pages[i].asarray()

                # 8bit化
                if arr.dtype != np.uint8:
                    if arr.ndim == 3 and arr.shape[-1] in (3, 4) and arr.dtype == np.uint16:
                        # 16bitカラー → 一旦各chをスケール
                        arr8 = _to_uint8(arr, lo, hi)
                    else:
                        arr8 = _to_uint8(arr, lo, hi)
                else:
                    arr8 = arr

                # チャンネル整形（OpenCVはBGR 8bit 3chを想定）
                if is_gray:
                    frame = cv2.cvtColor(arr8, cv2.COLOR_GRAY2BGR)
                else:
                    # tifffileはHWCのはず。RGB→BGRへ
                    if arr8.shape[-1] == 4:
                        arr8 = arr8[..., :3]  # アルファ捨てる
                    frame = cv2.cvtColor(arr8, cv2.COLOR_RGB2BGR)

                # リサイズ
                if resize != 1.0:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

                writer.write(frame)
        finally:
            writer.release()

if __name__ == "__main__":
    # 例:
    # 0〜100000フレームを4倍速で出力（出力FPSを上げる方式）
    tiff_to_mp4(
        tiff_path=r"X:\Behavior\Ca_imaging\20250724_z253-4_IRES-2x_GCaMP-3e12_soma_imaging_EEG\_GCaMP\02.tif",
        out_path=r"X:\Behavior\Ca_imaging\20250724_z253-4_IRES-2x_GCaMP-3e12_soma_imaging_EEG\_GCaMP\_02_movie.mp4",
        start_frame=1000,
        end_frame=1900,
        speed=4.0,
        mode="fps",        # ← 出力FPS=31*4=124fps（再生環境次第で重いなら "skip" へ）
        resize=1.0,
        input_fps=31.0,
        fourcc="mp4v",
    )

    # # 0〜300000フレームを8倍速で出力（フレーム間引き方式・出力は31fps）
    # tiff_to_mp4(
    #     tiff_path="input.tif",
    #     out_path="clip_0_300k_8x_skip.mp4",
    #     start_frame=0,
    #     end_frame=300000,
    #     speed=8.0,
    #     mode="skip",
    #     resize=1.0,
    #     input_fps=31.0,
    #     fourcc="mp4v",
    # )
