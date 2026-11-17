# Đề tài 6: Nhận dạng biển báo giao thông (FIT4013)

**Sinh viên thực hiện:** Dương Duy Khánh (MSV 1871020326)  
**Học phần:** Nhập môn Học máy (FIT4013)  
**Giảng viên:** ThS. Lê Thị Thùy Trang

## 1. Nguồn dữ liệu

- **Bộ dữ liệu:** German Traffic Sign Recognition Benchmark (GTSRB) — 39.209 ảnh huấn luyện (`Train/`), 12.630 ảnh kiểm thử (`Test/`), 43 lớp biển báo, kèm 43 ảnh biển chuẩn (`Meta/`).
- **Liên kết tải:** [GTSRB - German Traffic Sign (Kaggle)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Cách chuẩn bị:** Tải và giải nén, đặt các thư mục `Train/`, `Test/`, `Meta/` và các tệp `Train.csv`, `Test.csv`, `Meta.csv` vào thư mục `Code_NhapMonHocMay/`. Dữ liệu không được đưa lên repository vì dung lượng lớn (khoảng 500 MB).

## 2. Cấu trúc thư mục

- `Code_NhapMonHocMay/common.py`: Hàm dùng chung — đọc ảnh (cắt theo khung biển ROI, đổi kích thước 32×32, tùy chọn CLAHE), trích đặc trưng pixel và HOG, tên 43 lớp.
- `Code_NhapMonHocMay/load_data.py`: Kiểm tra dữ liệu (số ảnh thiếu trên đĩa) và đọc thử một phần ảnh.
- `Code_NhapMonHocMay/01_eda.py`: Khám phá dữ liệu (kích thước, dữ liệu thiếu, trùng lặp, mất cân bằng lớp, số track) và xuất các biểu đồ EDA vào `results/`.
- `Code_NhapMonHocMay/02_experiments.py`: Baseline, so sánh CLAHE, quét siêu tham số **K-NN** và **Cây quyết định** trên 3 biểu diễn (**Pixel**, **HOG**, **PCA**), đánh giá cuối trên `Test.csv`, phân tích lỗi.
- `Code_NhapMonHocMay/results/`: Bảng kết quả (`validation_results.csv`, `test_results.csv`, ...), biểu đồ (`.png`), phân tích lỗi (`top_confusions.csv`, `per_class_f1.csv`, `accuracy_by_size.csv`) và nhật ký chạy (`run_log_eda.txt`, `run_log_experiments.txt`).
- `Code_NhapMonHocMay/AI_USAGE.md`: Khai báo sử dụng công cụ AI.
- `BaoCao_BTL_NhapMonHocMay/`: Mã nguồn báo cáo LaTeX (`main.tex`) và bản PDF (`main.pdf`).

## 3. Kết quả thực nghiệm nổi bật

- Chia dữ liệu huấn luyện/kiểm định **theo track** (`StratifiedGroupKFold`) để tránh Data Leakage: 31.349 ảnh huấn luyện, 7.860 ảnh kiểm định, số track chung giữa hai tập bằng 0. PCA chỉ fit trên tập huấn luyện.
- Baseline (Pixel + K-NN, k=5): Accuracy = 0.5253, Macro-F1 = 0.4682.
- Mô hình tốt nhất: **K-NN + HOG** (k=9, euclidean) đạt **Accuracy = 0.9082**, **Macro-F1 = 0.8840** trên tập Test. Kết hợp PCA (100 chiều) cho kết quả gần tương đương (Accuracy = 0.9051, Macro-F1 = 0.8817) và dự đoán nhanh hơn hơn hai lần.
- Cây quyết định tốt nhất (HOG): Accuracy = 0.7595, Macro-F1 = 0.6887.
- Lỗi tập trung ở các biển giới hạn tốc độ (ví dụ biển 30 bị dự đoán thành biển 80) và các biển cảnh báo hình tam giác có hình minh họa nhỏ.
- Hai lần chạy đầy đủ độc lập cho cùng Accuracy và Macro-F1 ở cả chín phương án (chỉ cột thời gian khác nhau).

## 4. Hướng dẫn cài đặt và chạy lại từ đầu

```
cd Code_NhapMonHocMay
python -m venv venv
.\venv\Scripts\activate
python -m pip install numpy pandas opencv-python scikit-learn scikit-image joblib matplotlib seaborn

# Bước 0: Kiểm tra dữ liệu
python load_data.py --data_dir .

# Bước 1: Khám phá dữ liệu và xuất biểu đồ EDA
python 01_eda.py --data_dir .

# Bước 2: Chạy thử nhanh với 1200 ảnh (vài phút, ghi vào thư mục smoke)
python 02_experiments.py --data_dir . --out_dir smoke --limit 1200

# Bước 3: Thí nghiệm đầy đủ (mất vài giờ)
python 02_experiments.py --data_dir .
```

Biên dịch báo cáo LaTeX (dùng XeLaTeX, chạy hai lần để cập nhật mục lục):

```
cd ..\BaoCao_BTL_NhapMonHocMay
xelatex main.tex
xelatex main.tex
```
