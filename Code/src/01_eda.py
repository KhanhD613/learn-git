"""EDA: kích thước ảnh, phân phối lớp, ảnh mẫu, ảnh Meta, kiểm tra trùng/thiếu."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

args = get_args()
out = Path(args.out_dir); out.mkdir(exist_ok=True, parents=True)
tr = load_csv(args.data_dir, "Train"); te = load_csv(args.data_dir, "Test")
meta = pd.read_csv(Path(args.data_dir) / "Meta.csv")

lines = []
lines.append(f"Train: {tr.shape}, Test: {te.shape}, Meta: {meta.shape}")
lines.append(f"Số lớp train={tr.ClassId.nunique()}, test={te.ClassId.nunique()}")
lines.append(f"Thiếu dữ liệu train: {int(tr.isna().sum().sum())}, test: {int(te.isna().sum().sum())}")
lines.append(f"Trùng đường dẫn train: {int(tr.Path.duplicated().sum())}")
cnt = tr.ClassId.value_counts().sort_index()
lines.append(f"Lớp ít nhất: {cnt.idxmin()} ({cnt.min()}), nhiều nhất: {cnt.idxmax()} ({cnt.max()}), "
             f"tỉ lệ imbalance = {cnt.max()/cnt.min():.1f}")
tr["track"] = tr.Path.map(track_id)
lines.append(f"Số track (biển vật lý) trong train: {tr.track.nunique()} -> ~{len(tr)/tr.track.nunique():.0f} ảnh/track")
lines.append(tr[["Width", "Height"]].describe().to_string())
(out / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

# phân phối lớp train vs test
fig, ax = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
cnt.plot.bar(ax=ax[0], color="steelblue"); ax[0].set_title("Phân phối lớp - Train")
te.ClassId.value_counts().sort_index().plot.bar(ax=ax[1], color="darkorange"); ax[1].set_title("Phân phối lớp - Test")
plt.tight_layout(); plt.savefig(out / "eda_class_distribution.png", dpi=150); plt.close()

# kích thước ảnh
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(tr.Width, bins=40); ax[0].set_title("Width (train)")
ax[1].scatter(tr.Width, tr.Height, s=2, alpha=.3); ax[1].set_title("Width vs Height")
plt.tight_layout(); plt.savefig(out / "eda_image_sizes.png", dpi=150); plt.close()

# lưới ảnh mẫu: mỗi lớp 1 ảnh gốc (Meta) + 5 ảnh train
fig, axes = plt.subplots(43, 6, figsize=(8, 55))
rng = np.random.RandomState(SEED)
for c in range(43):
    sub = tr[tr.ClassId == c]
    rows = [Path(args.data_dir) / meta[meta.ClassId == c].Path.iloc[0]] + \
           [Path(args.data_dir) / p for p in sub.Path.sample(5, random_state=SEED)]
    for j, p in enumerate(rows):
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        axes[c, j].imshow(im); axes[c, j].axis("off")
    axes[c, 0].set_title(f"{c}: {CLASS_NAMES[c]}", fontsize=7, loc="left")
plt.tight_layout(); plt.savefig(out / "eda_samples.png", dpi=80); plt.close()

# độ sáng trung bình -> ảnh tối/sáng
sub = tr.sample(min(3000, len(tr)), random_state=SEED)
imgs = load_images(sub, args.data_dir)
plt.figure(figsize=(5, 3.5)); plt.hist(imgs.reshape(len(imgs), -1).mean(1), bins=40)
plt.title("Độ sáng trung bình (3000 ảnh train)"); plt.tight_layout()
plt.savefig(out / "eda_brightness.png", dpi=150); plt.close()
print("EDA xong ->", out)
