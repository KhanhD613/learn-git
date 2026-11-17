"""Hàm dùng chung: đọc ảnh, tiền xử lý, trích đặc trưng, tên lớp."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from joblib import Parallel, delayed
from skimage.feature import hog

SEED = 42
IMG_SIZE = 32

CLASS_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End speed limit 80", "Speed limit 100",
    "Speed limit 120", "No passing", "No passing (>3.5t)", "Right-of-way at intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Vehicles >3.5t prohibited",
    "No entry", "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware ice/snow", "Wild animals crossing",
    "End of all speed/passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing", "End no passing (>3.5t)",
]


def get_args(extra=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="Data_Trafic",
                   help="Thư mục chứa Train/, Test/, Meta/, Train.csv, Test.csv, Meta.csv")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--limit", type=int, default=0,
                   help="Chỉ dùng N ảnh mỗi tập (để chạy thử nhanh); 0 = dùng hết")
    if extra:
        extra(p)
    return p.parse_args()


def track_id(path):
    """Train/20/00020_00000_00007.png -> '00020_00000' (cùng 1 biển vật lý, 30 khung hình)."""
    stem = Path(path).stem.split("_")
    return stem[0] + "_" + stem[1]


def read_gray_roi(path, roi, size, clahe):
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if roi is not None:
        x1, y1, x2, y2 = roi
        crop = img[y1:y2 + 1, x1:x2 + 1]
        if crop.size > 0:
            img = crop
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if clahe:
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(img)
    return img


def load_images(df, data_dir, size=IMG_SIZE, use_roi=True, clahe=False):
    data_dir = Path(data_dir)
    rois = df[["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"]].values if use_roi else [None] * len(df)
    out = Parallel(n_jobs=-1)(
        delayed(read_gray_roi)(data_dir / p, r, size, clahe) for p, r in zip(df["Path"], rois))
    return np.stack(out).astype(np.uint8)


def load_csv(data_dir, name, limit=0):
    df = pd.read_csv(Path(data_dir) / f"{name}.csv")
    if limit and limit < len(df):
        df = df.sample(limit, random_state=SEED).reset_index(drop=True)
    return df


def pixel_features(imgs):
    return imgs.reshape(len(imgs), -1).astype(np.float32) / 255.0


def hog_features(imgs, ppc=4, orient=9):
    f = Parallel(n_jobs=-1)(
        delayed(hog)(im, orientations=orient, pixels_per_cell=(ppc, ppc),
                     cells_per_block=(2, 2), block_norm="L2-Hys") for im in imgs)
    return np.asarray(f, dtype=np.float32)
