# Giai đoạn 2 (Tuần 3–4): Mô hình chính, detection và tích hợp web

> Điều kiện đầu vào: hoàn thành Giai đoạn 1 (dataset đã chia, `label_map.csv`, baseline, skeleton API/web).
> Mục tiêu cuối giai đoạn: **model tốt nhất đã export ONNX**, **API thật** (detect + classify + Grad-CAM), **giao diện có đủ chức năng lõi** (kết quả, bounding box, chi tiết biển, lịch sử, thư viện).

## 1. Tuần 3: Huấn luyện, so sánh model, detection

| Ngày | Công việc | Đầu ra |
|---|---|---|
| 11 | Transfer learning **MobileNetV2** (ImageNet): đóng băng backbone, train head, sau đó fine-tune mở dần các block cuối; label smoothing 0.1 | Model TL v1 |
| 11 | Thử thêm EfficientNet-B0 (hoặc ResNet18/50) để so sánh | Model TL v2 |
| 12 | Tinh chỉnh siêu tham số (learning rate, batch size, dropout, augmentation mạnh/nhẹ); ghi mọi lần chạy vào bảng thí nghiệm | `experiments.csv` |
| 13 | Đánh giá đầy đủ trên tập test: accuracy, top-3 accuracy, F1 macro/per-class, confusion matrix | `reports/*.json`, hình |
| 13 | **Phân tích lỗi**: liệt kê cặp lớp nhầm nhiều (biển tốc độ P.127, các biển cấm rẽ P.123/P.124/P.137–P.139), xem ảnh sai | Mục "Error analysis" cho báo cáo |
| 14 | Kiểm tra **robustness**: ảnh tối, mờ, nghiêng, che một phần; đo giảm accuracy | Bảng robustness |
| 14 | Detection: theo hướng đã chốt ở Giai đoạn 1 — YOLOv8n/s (1 tầng) hoặc detector 1 lớp "biển báo" + classifier (2 tầng) | Detector v1, mAP@0.5 |
| 15 | Thêm lớp `not_sign` (ảnh nền) và **ngưỡng tin cậy**; chọn ngưỡng bằng tập val (cân bằng precision/recall) | `thresholds.json` |

**Quy tắc so sánh công bằng:** cùng split, cùng seed, cùng augmentation cơ sở; báo cáo trung bình ± độ lệch chuẩn nếu chạy ≥ 3 seed.

## 2. Tuần 4: Xuất model, API thật, giao diện lõi

| Ngày | Công việc | Đầu ra |
|---|---|---|
| 16 | Chọn model cuối (cân nhắc accuracy vs ms/ảnh vs kích thước); **export ONNX**, kiểm tra kết quả ONNX khớp PyTorch (sai số nhỏ) | `models/*.onnx`, `manifest.json` |
| 16 | **Calibration**: temperature scaling để độ tin cậy phản ánh đúng xác suất | `calibration.json` |
| 17 | ml-service: `predictor.py` (load model 1 lần lúc khởi động, warm-up), tiền xử lý ảnh (EXIF, RGB, giới hạn cạnh), NMS, top-K, cờ `uncertain` | `/predict` thật |
| 17 | Endpoint `/models`, `/metrics/{model}`, `/signs`, `/signs/{id}`; validate file bằng magic bytes, giới hạn 10 MB, rate-limit | API v1 |
| 18 | **Grad-CAM** (`/predict?explain=true`) cho classifier, trả heatmap dạng base64 | Grad-CAM chạy được |
| 18 | Test API: pytest cho preprocess, postprocess, endpoint, ảnh lỗi/quá lớn/sai định dạng | Bộ test ml-service |
| 19 | Frontend: `ImageCanvas` vẽ bounding box (SVG overlay), `ResultList`, `ConfidenceBar`, `SignDetail` (ảnh chuẩn, ý nghĩa, nguồn QCVN) | Trang chính hoàn chỉnh |
| 19 | Frontend: trạng thái rỗng/đang tải/lỗi/không tìm thấy, hủy request (AbortController), thanh chỉnh ngưỡng | UX đầy đủ trạng thái |
| 20 | Frontend: trang **Lịch sử** (localStorage), trang **Thư viện biển báo** (tìm không dấu, lọc theo nhóm P/W/R/I/S) | 2 trang mới |
| 20 | Sinh type TypeScript từ OpenAPI của FastAPI (`openapi-typescript`) để FE/BE không lệch schema | `api/types.ts` |

## 3. Đặc tả kỹ thuật cần chốt trong giai đoạn này

**Schema kết quả (rút gọn):**
```json
{
  "request_id": "…",
  "model": {"name": "two-stage", "version": "1.0.0"},
  "inference_ms": 84.3,
  "detections": [{
    "bbox": {"x": 412, "y": 190, "w": 96, "h": 96},
    "code": "P.102",
    "name_vi": "Cấm đi ngược chiều",
    "group": "prohibitory",
    "confidence": 0.93,
    "uncertain": false,
    "top_k": [{"code": "P.102", "confidence": 0.93}, {"code": "P.101", "confidence": 0.04}]
  }]
}
```

**Ngưỡng gợi ý (điều chỉnh theo tập val):** `conf` mặc định 0.5; dưới ngưỡng → `uncertain=true` và giao diện hiện "Chưa chắc chắn", kèm top-3.

## 4. Phân công gợi ý

| Vai trò | Tuần 3 | Tuần 4 |
|---|---|---|
| ML engineer 1 | Transfer learning, tinh chỉnh | Export ONNX, calibration |
| ML engineer 2 | Detection, `not_sign` | Grad-CAM, robustness |
| Backend | Chuẩn bị predictor/schema | API thật, test, bảo mật |
| Frontend | Component kết quả, canvas | Lịch sử, thư viện, type OpenAPI |
| Data lead | Error analysis, bổ sung ảnh lớp yếu | Bổ sung `signs.json` đủ ý nghĩa/nguồn |

## 5. Tiêu chí hoàn thành

- [ ] Bảng so sánh ≥ 3 model (baseline, MobileNetV2, model thứ 3 hoặc YOLO) với accuracy, F1 macro, ms/ảnh, kích thước
- [ ] Error analysis có ví dụ ảnh sai và giải thích
- [ ] Model ONNX chạy được trên CPU, inference ≤ ~300 ms/ảnh (ảnh ≤ 1280 px)
- [ ] `/predict` trả đúng schema; ảnh lỗi trả mã lỗi chuẩn (400/413/415/422)
- [ ] Web hiển thị box + tên + mã + độ tin cậy + ý nghĩa; bật/tắt Grad-CAM
- [ ] Test ml-service chạy xanh; contract FE/BE sinh từ OpenAPI

## 6. Rủi ro giai đoạn 2

| Rủi ro | Giảm thiểu |
|---|---|
| Overfit vì ít ảnh/lớp | Augmentation mạnh, dropout, early stopping, giảm số lớp MVP |
| Detection không đủ dữ liệu | Ưu tiên pipeline 2 tầng; nếu không kịp, giữ classification cho ảnh biển đã cắt và ghi rõ hạn chế |
| Nhầm các biển giống nhau | Thu thêm ảnh cặp dễ nhầm, xem confusion matrix, tăng độ phân giải input |
| Model tốt nhưng chậm | Dùng ONNX Runtime, model nhỏ (MobileNet, YOLOv8n), giảm kích thước ảnh |
| Lệch schema FE/BE | Sinh type từ OpenAPI, không viết tay |