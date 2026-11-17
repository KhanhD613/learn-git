"""Thí nghiệm chính: baseline, so sánh tiền xử lý (CLAHE), biểu diễn (pixel/HOG/PCA),
thuật toán (K-NN vs Cây quyết định), quét siêu tham số trên validation (chia theo track),
đánh giá cuối trên Test.csv, phân tích lỗi."""
import time
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from common import *

args = get_args()
out = Path(args.out_dir); out.mkdir(exist_ok=True, parents=True)
cache = out / "cache"; cache.mkdir(exist_ok=True)

# ------------------------------------------------------------------ 1. Dữ liệu
tr = load_csv(args.data_dir, "Train", args.limit)
te = load_csv(args.data_dir, "Test", args.limit)
ytr, yte = tr.ClassId.values, te.ClassId.values
groups = tr.Path.map(track_id).values


def get_imgs(df, name, clahe):
    f = cache / f"{name}_clahe{int(clahe)}_{IMG_SIZE}_{len(df)}.npy"
    if f.exists():
        return np.load(f)
    a = load_images(df, args.data_dir, clahe=clahe); np.save(f, a); return a


IM = {c: (get_imgs(tr, "train", c), get_imgs(te, "test", c)) for c in (False, True)}

# Chia fit/val theo TRACK để tránh data leakage: 30 ảnh của cùng một biển vật lý
# gần như giống hệt nhau, chia ngẫu nhiên theo ảnh sẽ làm validation quá lạc quan.
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
fit_idx, val_idx = next(sgkf.split(np.zeros(len(tr)), ytr, groups))
print(f"fit={len(fit_idx)} val={len(val_idx)} test={len(te)} | "
      f"track chung fit/val = {len(set(groups[fit_idx]) & set(groups[val_idx]))} (phải = 0)")

# ------------------------------------------------------------------ 2. Biểu diễn đặc trưng
REPS = {}  # name -> (Xtr_full, Xte)  [pixel/hog không học tham số; PCA fit riêng bên dưới]


def build_rep(rep, clahe):
    itr, ite = IM[clahe]
    key = (rep, clahe)
    if key not in REPS:
        if rep == "pixel":
            REPS[key] = (pixel_features(itr), pixel_features(ite))
        elif rep == "hog":
            REPS[key] = (hog_features(itr), hog_features(ite))
    return REPS[key]


def make_features(rep, clahe, fit_rows, other_sets):
    """Trả về (X_fit, [X_other...]). PCA chỉ fit trên fit_rows (chống leakage)."""
    base, extra = rep.replace("pca_", ""), None
    Xtr_all, Xte_all = build_rep(base, clahe)
    Xfit = Xtr_all[fit_rows]
    others = [Xtr_all[o] if isinstance(o, np.ndarray) else Xte_all for o in other_sets]
    if rep.startswith("pca_"):
        p = PCA(n_components=100, random_state=SEED).fit(Xfit)
        Xfit, others = p.transform(Xfit), [p.transform(o) for o in others]
    return Xfit, others


def make_model(algo, **kw):
    if algo == "knn":
        return KNeighborsClassifier(n_jobs=-1, **kw)
    return DecisionTreeClassifier(random_state=SEED, **kw)


def run(algo, rep, clahe, params, fit_rows, y_fit, eval_rows_or_test, y_eval):
    Xf, (Xe,) = make_features(rep, clahe, fit_rows, [eval_rows_or_test])
    m = make_model(algo, **params)
    t0 = time.time(); m.fit(Xf, y_fit); t_fit = time.time() - t0
    t0 = time.time(); yp = m.predict(Xe); t_pred = time.time() - t0
    p, r, f, _ = precision_recall_fscore_support(y_eval, yp, average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(y_eval, yp), macro_precision=p, macro_recall=r,
                macro_f1=f, fit_s=round(t_fit, 2), pred_s=round(t_pred, 2)), yp


# ------------------------------------------------------------------ 3. Baseline & so sánh tiền xử lý (validation)
rows = []
def log(stage, algo, rep, clahe, params, res):
    rows.append(dict(stage=stage, algo=algo, rep=rep, clahe=clahe, params=str(params), **res))
    print(f"[{stage}] {algo:3s} {rep:9s} clahe={int(clahe)} {params} "
          f"acc={res['accuracy']:.4f} f1={res['macro_f1']:.4f}", flush=True)

BASE = dict(algo="knn", rep="pixel", clahe=False, params=dict(n_neighbors=5))
res, _ = run(**BASE, fit_rows=fit_idx, y_fit=ytr[fit_idx], eval_rows_or_test=val_idx, y_eval=ytr[val_idx])
log("baseline_val", **BASE, res=res)

for algo, params in [("knn", dict(n_neighbors=5)), ("dt", dict())]:
    for rep in ["pixel", "hog"]:
        for clahe in (False, True):
            res, _ = run(algo, rep, clahe, params, fit_idx, ytr[fit_idx], val_idx, ytr[val_idx])
            log("preproc_val", algo, rep, clahe, params, res)

# ------------------------------------------------------------------ 4. Quét siêu tham số (validation)
GRID = {
    "knn": [dict(n_neighbors=k, metric=m) for k in (1, 3, 5, 7, 9) for m in ("euclidean", "manhattan")],
    "dt": [dict(max_depth=d, criterion=c) for d in (10, 20, None) for c in ("gini", "entropy")],
}
# CLAHE chọn theo kết quả bước 3 (trung bình macro-F1 theo rep)
pre = pd.DataFrame(rows); pre = pre[pre.stage == "preproc_val"]
best_clahe = {r: bool(pre[pre.rep == r].groupby("clahe").macro_f1.mean().idxmax()) for r in ("pixel", "hog")}
best_clahe["pca_pixel"], best_clahe["pca_hog"] = best_clahe["pixel"], best_clahe["hog"]
print("CLAHE được chọn:", best_clahe)

def metrics(y, yp):
    p, r, f, _ = precision_recall_fscore_support(y, yp, average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(y, yp), macro_precision=p, macro_recall=r, macro_f1=f)


def knn_sweep(rep, clahe):
    """Tìm 9 láng giềng gần nhất MỘT lần cho mỗi (rep, metric), rồi bỏ phiếu cho k=1,3,5,7,9.
    Kết quả giống K-NN thường (bỏ phiếu đa số) nhưng nhanh hơn ~5 lần. pred_s = thời gian tìm láng giềng."""
    Xf, (Xv,) = make_features(rep, clahe, fit_idx, [val_idx])
    ks = sorted({p["n_neighbors"] for p in GRID["knn"]})
    yf, yv = ytr[fit_idx], ytr[val_idx]
    for metric in sorted({p["metric"] for p in GRID["knn"]}):
        nn = NearestNeighbors(n_neighbors=max(ks), metric=metric, n_jobs=-1).fit(Xf)
        t0 = time.time(); idx = nn.kneighbors(Xv, return_distance=False); t = time.time() - t0
        nb = yf[idx]
        for k in ks:
            yp = (nb[:, :k, None] == np.arange(43)).sum(1).argmax(1)
            log("grid_val", "knn", rep, clahe, dict(n_neighbors=k, metric=metric),
                dict(**metrics(yv, yp), fit_s=0.0, pred_s=round(t, 2)))


for rep in ("pixel", "hog", "pca_pixel", "pca_hog"):
    knn_sweep(rep, best_clahe[rep])
    for params in GRID["dt"]:
        res, _ = run("dt", rep, best_clahe[rep], params, fit_idx, ytr[fit_idx], val_idx, ytr[val_idx])
        log("grid_val", "dt", rep, best_clahe[rep], params, res)

df = pd.DataFrame(rows); df.to_csv(out / "validation_results.csv", index=False)
grid = df[df.stage == "grid_val"]
best = grid.loc[grid.groupby(["algo", "rep"]).macro_f1.idxmax()].reset_index(drop=True)
best.to_csv(out / "best_per_config_val.csv", index=False)

# biểu đồ ảnh hưởng của k
g = grid[grid.algo == "knn"].copy()
g["k"] = g.params.str.extract(r"'n_neighbors': (\d+)").astype(int)
g["metric"] = g.params.str.extract(r"'metric': '(\w+)'")
plt.figure(figsize=(7, 4))
for (rep, met), s in g.groupby(["rep", "metric"]):
    plt.plot(s.sort_values("k").k, s.sort_values("k").macro_f1, marker="o", label=f"{rep}/{met}")
plt.xlabel("k"); plt.ylabel("Macro-F1 (val)"); plt.legend(fontsize=7); plt.grid(alpha=.3)
plt.title("Ảnh hưởng của k và metric (K-NN)"); plt.tight_layout()
plt.savefig(out / "fig_knn_k.png", dpi=150); plt.close()

# ------------------------------------------------------------------ 5. Đánh giá cuối trên Test
all_idx = np.arange(len(tr))
final, preds = [], {}
def eval_test(name, algo, rep, clahe, params):
    res, yp = run(algo, rep, clahe, params, all_idx, ytr, "test", yte)
    final.append(dict(name=name, algo=algo, rep=rep, clahe=clahe, params=str(params), **res))
    preds[name] = yp
    print(f"[TEST] {name}: acc={res['accuracy']:.4f} macroF1={res['macro_f1']:.4f}", flush=True)

eval_test("BASELINE pixel+KNN(k=5)", **BASE)
for _, r in best.iterrows():
    eval_test(f"{r.algo}+{r.rep} (tuned)", r.algo, r.rep, bool(r.clahe), eval(r.params))
fin = pd.DataFrame(final); fin.to_csv(out / "test_results.csv", index=False)
print(fin.drop(columns=["params"]).to_string())

# ------------------------------------------------------------------ 6. Phân tích lỗi cho mô hình tốt nhất
bn = fin.sort_values("macro_f1", ascending=False).iloc[0]["name"]; yp = preds[bn]
print("Mô hình tốt nhất:", bn)
rep_txt = classification_report(yte, yp, labels=list(range(43)), target_names=[f"{i}-{n}" for i, n in enumerate(CLASS_NAMES)],
                                digits=4, zero_division=0)
(out / "classification_report_best.txt").write_text(f"{bn}\n{rep_txt}", encoding="utf-8")

cm = confusion_matrix(yte, yp, labels=range(43))
cmn = cm / cm.sum(1, keepdims=True).clip(1)
plt.figure(figsize=(14, 12)); sns.heatmap(cmn, cmap="Blues", xticklabels=range(43), yticklabels=range(43))
plt.xlabel("Dự đoán"); plt.ylabel("Thực tế"); plt.title(f"Confusion matrix (chuẩn hoá theo hàng) - {bn}")
plt.tight_layout(); plt.savefig(out / "fig_confusion_best.png", dpi=130); plt.close()

off = cm.copy(); np.fill_diagonal(off, 0)
pairs = sorted([(off[i, j], i, j) for i in range(43) for j in range(43) if off[i, j]], reverse=True)[:10]
pd.DataFrame([dict(count=c, true=f"{i}-{CLASS_NAMES[i]}", pred=f"{j}-{CLASS_NAMES[j]}") for c, i, j in pairs]
             ).to_csv(out / "top_confusions.csv", index=False)
f1c = precision_recall_fscore_support(yte, yp, labels=range(43), zero_division=0)[2]
pd.DataFrame(dict(ClassId=range(43), name=CLASS_NAMES, f1=f1c, support=cm.sum(1))
             ).sort_values("f1").to_csv(out / "per_class_f1.csv", index=False)

# lỗi theo kích thước ảnh (biển nhỏ/xa khó hơn?)
te2 = te.copy(); te2["ok"] = (yp == yte)
te2["size_bin"] = pd.cut(te2[["Width", "Height"]].min(axis=1), [0, 30, 40, 60, 100, 1000])
te2.groupby("size_bin", observed=True).ok.agg(["mean", "count"]).rename(columns={"mean": "accuracy"}
    ).to_csv(out / "accuracy_by_size.csv")

# lưới ảnh dự đoán sai
wrong = np.where(yp != yte)[0]
sel = np.random.RandomState(SEED).choice(wrong, min(24, len(wrong)), replace=False) if len(wrong) else []
fig, ax = plt.subplots(4, 6, figsize=(12, 8.5)); ax = ax.ravel()
for a in ax: a.axis("off")
for a, i in zip(ax, sel):
    a.imshow(IM[best_clahe.get(bn.split("+")[1].split(" ")[0], False)][1][i], cmap="gray")
    a.set_title(f"T:{yte[i]} P:{yp[i]}", fontsize=8, color="red")
plt.suptitle(f"Ảnh dự đoán sai (T=thực tế, P=dự đoán) - {bn}"); plt.tight_layout()
plt.savefig(out / "fig_errors.png", dpi=120); plt.close()
print("Hoàn tất ->", out)
