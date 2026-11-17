"""Kết nối mã nguồn với các file ảnh trong Data_Trafic (Train/, Test/, Meta/ + 3 file CSV).



Chạy kiểm tra:  python load_data.py --data_dir "C:\\Users\\<ten>\\Documents\\Data_Trafic"
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

IMG_SIZE = 32


def read_image(path, roi=None, size=IMG_SIZE, gray=True):
    """Đọc 1 ảnh (an toàn với đường dẫn có dấu tiếng Việt trên Windows), cắt ROI, resize."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    if roi is not None:
        x1, y1, x2, y2 = roi
        crop = img[y1:y2 + 1, x1:x2 + 1]
        if crop.size:
            img = crop
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def check_folder(data_dir):
    """Kiểm tra cấu trúc thư mục và các đường dẫn trong CSV có tồn tại không."""
    data_dir = Path(data_dir)
    for name in ("Train.csv", "Test.csv", "Meta.csv"):
        if not (data_dir / name).exists():
            raise FileNotFoundError(f"Thiếu {data_dir / name}")
    for name in ("Train", "Test", "Meta"):
        df = pd.read_csv(data_dir / f"{name}.csv")
        missing = [p for p in df["Path"] if not (data_dir / p).exists()]
        print(f"{name}: {len(df)} dòng CSV, thiếu {len(missing)} ảnh trên đĩa")
        if missing:
            print("  ví dụ file thiếu:", missing[:3])


def load_split(data_dir, name, limit=0, size=IMG_SIZE, use_roi=True, gray=True):
    """name = 'Train' hoặc 'Test'. Trả về (X: N x size x size, y: N, df)."""
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / f"{name}.csv")
    if limit:
        df = df.sample(min(limit, len(df)), random_state=42).reset_index(drop=True)
    rois = df[["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"]].values if use_roi else [None] * len(df)
    X = np.stack([read_image(data_dir / p, r, size, gray) for p, r in zip(df["Path"], rois)])
    return X, df["ClassId"].values, df


def load_dataset(data_dir, limit=0, **kw):
    Xtr, ytr, _ = load_split(data_dir, "Train", limit, **kw)
    Xte, yte, _ = load_split(data_dir, "Test", limit, **kw)
    return Xtr, ytr, Xte, yte


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--limit", type=int, default=500, help="số ảnh thử đọc mỗi tập")
    a = ap.parse_args()
    check_folder(a.data_dir)
    Xtr, ytr, Xte, yte = load_dataset(a.data_dir, limit=a.limit)
    print("X_train", Xtr.shape, Xtr.dtype, "| y_train", ytr.shape, "| số lớp", len(set(ytr)))
    print("X_test ", Xte.shape, "| y_test ", yte.shape)
    print("Đọc ảnh thành công.")
