# Giai đoạn 1 (Tuần 1–2): Dữ liệu, bảng nhãn theo QCVN và baseline

> Mục tiêu cuối giai đoạn: có **bộ dữ liệu sạch + bảng nhãn chuẩn QCVN 41:2024/BGTVT**, **model baseline chạy được**, **API và giao diện khung (mock)** kết nối được với nhau.

## 0. Nguồn dữ liệu và căn cứ pháp lý

| Nguồn | Dùng để | Lưu ý |
|---|---|---|
| Kaggle: `maitam/vietnamese-traffic-signs` | Ảnh huấn luyện | Tôi chưa xem được nội dung chi tiết của trang (chỉ thấy thẻ: image, law, computer vision, image classification, pytorch). **Tuần 1 phải tự kiểm tra**: số lớp, số ảnh/lớp, ảnh đã cắt hay ảnh cả cảnh, có file bounding box không, license, cách đặt tên lớp |
| QCVN 41:2024/BGTVT (thuvienphapluat.vn) | Danh mục biển, mã biển, nhóm, ý nghĩa | Ban hành 15/11/2024, thay QCVN 41:2019. Trang chỉ hiển thị đầy đủ Điều 22 (P/DP), 28 (W), 32 (R), 36 (I), 41 (S) và Điều 47 (IE). Phụ lục B–F (ý nghĩa chi tiết) bị ẩn sau đăng nhập, cần tải văn bản gốc/PDF từ nguồn chính thống. **Cần đối chiếu ngày có hiệu lực từ văn bản gốc** (trang không hiển thị rõ) |

Nhóm mã theo QCVN: **P/DP** (cấm/hết cấm), **W** (nguy hiểm, cảnh báo), **R/R.E** (hiệu lệnh), **I** (chỉ dẫn), **S/S.G/S.H** (biển phụ), **IE** (chỉ dẫn đường cao tốc).

## 1. Tuần 1: Khảo sát dữ liệu và chuẩn hóa nhãn

| Ngày | Công việc | Đầu ra |
|---|---|---|
| 1 | Tạo repo monorepo (`web/`, `ml-service/`), quy ước nhánh Git, lint/format, `README` | Repo + CI lint cơ bản |
| 1 | Tải dataset Kaggle (Kaggle API), ghi phiên bản/ngày tải/license | `data/raw/`, `DATA_SOURCES.md` |
| 2 | **Audit dataset**: đếm lớp, ảnh/lớp, kích thước ảnh, ảnh trùng, ảnh hỏng, có bbox hay không | `notebooks/01_audit.ipynb` |
| 2 | Chốt **quyết định rẽ nhánh** (mục 3) | Ghi vào `DECISIONS.md` |
| 3 | Lập **bảng ánh xạ nhãn**: tên lớp trong dataset → mã biển QCVN 2024 (vd. `P.102`, `P.124a`, `W.224`) | `ml-service/data/label_map.csv` |
| 3 | Phát hiện biển thuộc QCVN 2019 mà đã đổi/bỏ ở bản 2024, đánh dấu `status` | Cột `status` trong `label_map.csv` |
| 4 | Xây `signs.json` (id, code, group, name_vi, name_en, meaning, action, source) từ các Điều của QCVN; ý nghĩa **tự diễn đạt lại**, ghi nguồn điều/phụ lục | `signs.json` v0 |
| 4 | Thu thập ảnh minh họa chuẩn (SVG/PNG tự vẽ hoặc nguồn có license) cho thư viện biển | `assets/signs/` |
| 5 | EDA: biểu đồ phân bố lớp, mẫu ảnh mỗi lớp, thống kê độ sáng/kích thước | `eda.ipynb`, hình cho báo cáo |
| 5 | Họp nhóm: chốt **danh sách lớp MVP** (20–30 lớp phổ biến, đủ ảnh) | `classes_mvp.json` |

**Quy tắc gán nhãn:** dùng mã QCVN làm khóa chính (`code`), biển có biến thể (a, b, c…) coi là lớp riêng nếu hình dạng khác nhau rõ (vd. P.123a cấm rẽ trái, P.123b cấm rẽ phải).

## 2. Tuần 2: Pipeline dữ liệu, baseline và khung hệ thống

| Ngày | Công việc | Đầu ra |
|---|---|---|
| 6 | `data_loader.py`: đọc ảnh, map nhãn, làm sạch (xóa trùng bằng hash, bỏ ảnh hỏng) | Script + test |
| 6 | Chia dữ liệu **stratified** 70/15/15, cố định seed, lưu file split (không chia lại mỗi lần chạy) | `splits/train.csv, val.csv, test.csv` |
| 7 | `preprocess.py` + `augment.py` (Albumentations): xoay ±10°, sáng/tương phản, blur, nhiễu, mưa/bóng đổ. **Không lật ngang** biển có hướng (rẽ trái/phải, P.123, P.136–P.139, R.301) | Pipeline augmentation |
| 7 | Xử lý mất cân bằng lớp: class weights hoặc oversampling | Cấu hình `imbalance` |
| 8 | `model.py`: CNN baseline (Conv-Pool×3 + Dense), 224 hoặc 64×64 tùy kích thước ảnh gốc | Baseline model |
| 9 | `train.py`: Adam, CrossEntropy, EarlyStopping, ReduceLROnPlateau, lưu checkpoint tốt nhất; log bằng TensorBoard hoặc CSV | Baseline đã train |
| 9 | `evaluate.py`: accuracy, precision/recall/F1 macro + per-class, confusion matrix | `reports/baseline_metrics.json` |
| 10 | **FastAPI skeleton**: `/health`, `/signs`, `/predict` (trả kết quả giả từ baseline), schema Pydantic, xử lý lỗi chuẩn | API chạy local |
| 10 | **Frontend skeleton** (React + TS + Vite + Tailwind): theme sáng/tối bằng CSS variables, trang chủ có Uploader + gọi `/predict`, hiển thị top-1 | Web chạy local |

## 3. Quyết định rẽ nhánh (chốt cuối ngày 2)

| Tình huống dataset | Hướng đi cho giai đoạn 2–3 |
|---|---|
| Chỉ có **ảnh biển đã cắt** (classification) | Classifier là lõi. Với ảnh cảnh đường: tìm thêm dataset có bounding box (Roboflow Universe…) hoặc tự gán nhãn ~300–500 ảnh để train detector 1 lớp "biển báo" (pipeline 2 tầng). Nếu không kịp: MVP chỉ nhận ảnh biển đã cắt, ghi rõ hạn chế |
| Có **ảnh cảnh + bbox** | Train YOLOv8 trực tiếp (1 tầng), hoặc 2 tầng để tăng độ chính xác |
| Ít ảnh/lớp (< 100) | Giảm số lớp MVP, augmentation mạnh, transfer learning ngay từ giai đoạn 2 |

## 4. Phân công gợi ý (điều chỉnh theo số thành viên)

| Vai trò | Nhiệm vụ chính trong 2 tuần |
|---|---|
| Data lead | Audit, làm sạch, label map, EDA |
| ML engineer | Pipeline, baseline, đánh giá |
| Backend (ml-service) | FastAPI skeleton, schema, `signs.json` |
| Frontend | Design tokens, theme, Uploader, gọi API |

## 5. Tiêu chí hoàn thành (Definition of Done)

- [ ] `label_map.csv` phủ 100% lớp MVP, mỗi lớp có mã QCVN 2024 và nguồn điều luật
- [ ] Split cố định, không rò rỉ (không có ảnh trùng giữa train/test)
- [ ] Baseline đạt accuracy đo được trên test và có confusion matrix
- [ ] `POST /predict` trả JSON đúng schema; web upload ảnh và hiện kết quả
- [ ] Theme sáng/tối hoạt động, không nháy màu khi tải trang
- [ ] `DECISIONS.md` ghi rõ hướng dataset (mục 3) và danh sách lớp MVP

## 6. Rủi ro giai đoạn 1

| Rủi ro | Giảm thiểu |
|---|---|
| Tên lớp trong dataset không khớp mã QCVN 2024 | Dành trọn ngày 3, kiểm chéo 2 người |
| Dataset có ảnh trùng/rò rỉ giữa các tập | Hash ảnh trước khi chia |
| Không xem được Phụ lục QCVN | Tải PDF văn bản gốc; tạm chỉ dùng tên biển ở các Điều |
| Ảnh quá nhỏ (vd. 32×32) | Cân nhắc input 64×64 cho baseline; dùng 224 khi transfer learning |