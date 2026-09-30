# Đặc tả hệ thống: Nhận diện Biển báo Giao thông Việt Nam (VN-SignVision)

> Môn Học máy · Frontend: TypeScript (React + Vite) · ML service: Python (FastAPI)

## 0. Lưu ý quan trọng trước khi bắt đầu

1. **Dùng dữ liệu biển báo Việt Nam từ Kaggle trực tiếp** (bộ dữ liệu maitam/vietnamese-traffic-signs). Đây là dữ liệu chính, không cần pretrain GTSRB. GTSRB (biển báo Đức) chỉ dùng để so sánh architecture/baseline nếu muốn, nhưng không nên dùng làm dữ liệu huấn luyện vì hình dạng/màu/ký hiệu hoàn toàn khác biệt.
2. **Quy chuẩn hiện hành:** QCVN 41:2024/BGTVT (thay QCVN 41:2019, có hiệu lực từ 01/01/2025). Nhóm nên đối chiếu tên/mã biển theo quy chuẩn này (nhóm P cấm, R hiệu lệnh, W nguy hiểm/cảnh báo, S phụ, I chỉ dẫn). Cần kiểm tra lại mã biển cụ thể trong văn bản gốc, vì một số mã đã đổi so với bản 2019.
3. **Ảnh người dùng upload thường là ảnh cả cảnh đường**, không phải ảnh đã cắt. Vì vậy hệ thống cần **detection** (tìm biển) chứ không chỉ classification. Đề xuất: MVP làm classification + detection cho một tập lớp nhỏ trước, mở rộng dần.
4. **Dữ liệu:** sử dụng bộ dữ liệu Kaggle `maitam/vietnamese-traffic-signs` làm nền tảng chính. Nếu cần bổ sung (lớp thiếu, ảnh chất lượng thấp), có thể thu thêm từ Roboflow Universe hoặc tự chụp (đảm bảo ghi nguồn, kiểm tra license). Gán nhãn thêm bằng Roboflow/CVAT/LabelImg nếu cần bounding box cho detection.

---

## 1. Phạm vi và mục tiêu

| Mức | Mục tiêu |
| --- | --- |
| **MVP (bắt buộc)** | Upload ảnh → hiển thị tên biển báo, mã biển, độ tin cậy, ý nghĩa |
| **Nên có** | Detection nhiều biển trong 1 ảnh (bounding box), kéo-thả/dán/chụp camera, lịch sử, dark/light mode |
| **Nâng cao** | Grad-CAM giải thích, so sánh model, webcam realtime, batch upload, tra cứu thư viện biển báo, quiz học luật, PWA/offline |

Người dùng mục tiêu: người học lái xe, sinh viên, giảng viên chấm đồ án (cần thấy rõ metrics + demo ổn định).

## 2. Chức năng chi tiết

### 2.1 Nhận diện (core)

- F1. Upload 1 ảnh: kéo-thả, chọn file, **dán từ clipboard (Ctrl+V)**, nhập URL ảnh (tùy chọn), chụp bằng camera (mobile).
- F2. Định dạng JPG/PNG/WEBP, tối đa 10 MB, validate ở cả client và server.
- F3. Hiển thị ảnh gốc + **bounding box** cho từng biển; nhãn + % tin cậy trên box.
- F4. Danh sách kết quả bên cạnh: tên biển, mã biển (vd. P.102), nhóm, độ tin cậy, thanh confidence.
- F5. Bấm vào 1 kết quả → panel chi tiết: ảnh mẫu chuẩn, ý nghĩa, hành vi cần làm, mức phạt tham khảo (ghi rõ "tham khảo", có nguồn/nghị định).
- F6. **Top-K dự đoán** (top-3) khi độ tin cậy thấp: "Có thể là: A (62%), B (21%)…".
- F7. **Ngưỡng tin cậy & "không chắc chắn"**: nếu conf \< ngưỡng → hiển thị "Chưa chắc chắn / có thể không phải biển báo", không ép ra nhãn sai.
- F8. Slider chỉnh ngưỡng confidence (mặc định 0.5) và IoU (NMS).
- F9. Trạng thái: rỗng, đang tải, đang suy luận (skeleton/progress), lỗi, không tìm thấy biển.

### 2.2 Giải thích & phân tích (điểm cộng cho môn Học máy)

- F10. **Grad-CAM heatmap** (bật/tắt overlay) cho classifier.
- F11. Hiển thị thời gian suy luận (ms), tên model, phiên bản model.
- F12. Trang **Model & Metrics**: accuracy, precision/recall/F1 per-class, confusion matrix, [mAP@0.5](mailto:mAP@0.5), biểu đồ loss/accuracy, so sánh baseline CNN vs MobileNetV2 vs YOLOv8.
- F13. Trang **Dataset**: số ảnh/lớp (biểu đồ cột), ví dụ ảnh, phân chia train/val/test, mô tả augmentation.
- F14. **Chọn model** trong UI (baseline / transfer / YOLO) để so sánh trực tiếp trên cùng ảnh.
- F15. Nút **"Báo sai / góp ý nhãn đúng"**: lưu ảnh + nhãn đề xuất để làm dữ liệu cải thiện model (human-in-the-loop, cần đồng ý của người dùng).

### 2.3 Tiện ích

- F16. **Lịch sử** nhận diện (lưu localStorage ở MVP; DB ở giai đoạn sau), xóa từng mục/tất cả.
- F17. Tải kết quả: ảnh đã vẽ box (PNG), kết quả JSON.
- F18. **Thư viện biển báo**: lưới toàn bộ lớp model hỗ trợ, lọc theo nhóm, tìm kiếm không dấu ("cam re trai").
- F19. **Quiz** "Đây là biển gì?" từ thư viện (vui + học luật).
- F20. Webcam/video realtime (nâng cao): gửi frame \~3–5 fps hoặc chạy ONNX Runtime Web trong trình duyệt.
- F21. Batch upload nhiều ảnh, bảng kết quả.
- F22. Đa ngôn ngữ vi/en (i18n), mặc định tiếng Việt.
- F23. PWA: cài như app, cache tài nguyên tĩnh.
- F24. Ảnh mẫu có sẵn ("Thử ảnh mẫu") để demo không cần ảnh riêng.

### 2.4 Quản trị (tùy chọn, giai đoạn 2)

- Đăng nhập admin, duyệt góp ý (F15), xem thống kê sử dụng, tải dữ liệu góp ý về để retrain.

## 3. Kiến trúc hệ thống

```
┌──────────────────────┐   HTTPS/JSON     ┌────────────────────────┐
│ web (React + TS)     │ ───────────────▶ │ ml-service (FastAPI)   │
│ Vite, Tailwind,      │ ◀─────────────── │ - /predict (detect+cls)│
│ TanStack Query       │                  │ - /explain (Grad-CAM)  │
└─────────┬────────────┘                  │ - /models, /health     │
          │ (giai đoạn 2)                 │ ONNX Runtime / PyTorch │
          ▼                               └───────────┬────────────┘
┌──────────────────────┐                              │
│ api-gateway (tùy chọn)│                  ┌──────────▼───────────┐
│ Node/Express+TypeORM │ ───────────────▶ │ models/ (best.onnx…) │
│ PostgreSQL: history, │                  │ data/signs.json      │
│ feedback, users      │                  └──────────────────────┘
└──────────────────────┘
```

- **MVP:** `web` gọi thẳng `ml-service` (bật CORS). Lịch sử lưu localStorage.
- **Giai đoạn 2:** thêm `api-gateway` (Node/Express + PostgreSQL) cho lịch sử, feedback, auth, rate-limit; ml-service chỉ làm suy luận.
- **Pipeline 2 tầng (khuyến nghị nếu ít dữ liệu box):** YOLOv8 phát hiện "biển báo" (1 hoặc vài nhóm lớp) → cắt vùng → classifier (MobileNetV2/EfficientNet) phân loại chi tiết. Cho phép thêm lớp mới mà không cần gán box lại toàn bộ.
- **Pipeline 1 tầng:** YOLOv8 detect + classify trực tiếp N lớp VN (đơn giản hơn, cần nhiều ảnh có box).
- Xuất model sang **ONNX** để suy luận nhanh, nhẹ, dễ deploy CPU.

### 3.1 Cấu trúc repo (monorepo)

```
vn-signvision/
├── web/                          # TypeScript
│   ├── src/
│   │   ├── app/ (router, providers, theme)
│   │   ├── pages/ (Home, History, Library, Quiz, Metrics, About)
│   │   ├── components/ (Uploader, ImageCanvas, ResultList, SignDetail,
│   │   │                ConfidenceBar, ThemeToggle, ModelSelect, ...)
│   │   ├── features/ (predict, history, library, feedback)
│   │   ├── api/ (client.ts, types.ts)
│   │   ├── store/ (Zustand: settings, history)
│   │   ├── i18n/ (vi.json, en.json)
│   │   ├── styles/ (tokens.css, globals.css)
│   │   └── utils/ (image.ts, format.ts, normalize.ts)
│   ├── tests/ (Vitest, Playwright)
│   └── vite.config.ts, tailwind.config.ts, tsconfig.json
├── ml-service/                   # Python
│   ├── app/ (main.py, routers/, schemas.py, config.py, deps.py)
│   ├── ml/ (detector.py, classifier.py, preprocess.py, postprocess.py,
│   │        gradcam.py, registry.py)
│   ├── training/ (data_loader.py, augment.py, train_cls.py,
│   │              train_yolo.py, evaluate.py, export_onnx.py)
│   ├── models/ (best.onnx, classifier.onnx, manifest.json)
│   ├── data/ (signs.json, raw/, processed/, labels.csv)
│   ├── reports/ (metrics.json, confusion_matrix.png)
│   ├── notebooks/ (eda.ipynb)
│   ├── tests/ (pytest)
│   └── requirements.txt
├── docker-compose.yml
└── README.md
```

## 4. Đặc tả ML

### 4.1 Tập lớp

- Bắt đầu **20–30 biển phổ biến** (cấm đi ngược chiều, cấm dừng/đỗ, cấm rẽ trái/phải, hạn chế tốc độ 40/50/60…, dừng lại, nhường đường, đường ưu tiên, giao nhau, người đi bộ cắt ngang, trường học, công trường, đường trơn…). Mở rộng sau khi pipeline ổn.
- Mỗi lớp có: `id`, `code` (P.102…), `name_vi`, `name_en`, `group`, `meaning`, `action`, `reference_image`.
- Thêm lớp **`background/not_sign`** (ảnh không phải biển) để giảm dự đoán sai.

### 4.2 Dữ liệu và tiền xử lý

- Chia **theo ảnh gốc / theo cảnh** (không chia ngẫu nhiên từng crop) để tránh rò rỉ dữ liệu giữa train/test. Tỷ lệ 70/15/15, stratified theo lớp.
- Cân bằng lớp: class weights hoặc oversampling; báo cáo số ảnh/lớp.
- Augmentation (Albumentations): xoay nhẹ ±10°, đổi sáng/tương phản, blur/chuyển động, nhiễu, mưa/sương/bóng đổ, cắt lệch, thay đổi màu nhẹ. **Không lật ngang** với biển có hướng (rẽ trái/phải).
- Chuẩn hóa: resize 224×224 (classifier) hoặc 640 letterbox (YOLO), theo mean/std ImageNet.

### 4.3 Model

| Model | Vai trò |
| --- | --- |
| CNN baseline (Conv-Pool×3 + Dense) | Mốc so sánh |
| MobileNetV2 / EfficientNet-B0 (fine-tune) | Classifier chính |
| YOLOv8n/s | Detection |
| (tùy chọn) ResNet50 | So sánh độ chính xác vs tốc độ |

Huấn luyện: Adam, CrossEntropy (+ label smoothing 0.1), EarlyStopping, ReduceLROnPlateau, lưu best checkpoint; fine-tune 2 giai đoạn (đóng băng backbone → mở dần). Đặt seed cố định để tái lập.

### 4.4 Đánh giá

- Classification: accuracy, precision/recall/F1 (macro + per-class), confusion matrix, top-3 accuracy.
- Detection: [mAP@0.5](mailto:mAP@0.5), [mAP@0.5](mailto:mAP@0.5):0.95, precision/recall theo lớp, IoU.
- Robustness: test riêng ảnh tối/mưa/mờ/nghiêng/che một phần.
- Tốc độ: ms/ảnh trên CPU, kích thước model (MB).
- Phân tích lỗi: liệt kê cặp lớp hay nhầm (vd. các biển hạn chế tốc độ), đưa ví dụ ảnh sai vào báo cáo.
- Calibration: kiểm tra độ tin cậy có phản ánh đúng xác suất (reliability diagram / temperature scaling).

## 5. Đặc tả API (ml-service)

Base URL: `/api/v1`. Tài liệu tự sinh tại `/docs`.

### `POST /predict` (multipart/form-data)

| Trường | Kiểu | Mô tả |
| --- | --- | --- |
| `file` | file | Ảnh JPG/PNG/WEBP ≤ 10 MB |
| `model` | string | `yolo`, `two-stage`, `cnn`… (mặc định từ manifest) |
| `conf` | float | 0.05–0.95, mặc định 0.5 |
| `iou` | float | mặc định 0.45 |
| `top_k` | int | mặc định 3 |
| `explain` | bool | trả Grad-CAM (base64), mặc định false |

Response 200:

```json
{
  "request_id": "b1f0…",
  "model": {"name": "two-stage", "version": "1.2.0"},
  "image": {"width": 1280, "height": 720},
  "inference_ms": 84.3,
  "detections": [
    {
      "id": 0,
      "bbox": {"x": 412, "y": 190, "w": 96, "h": 96},
      "class_id": 7,
      "code": "P.124",
      "name_vi": "Cấm quay đầu xe",
      "confidence": 0.93,
      "top_k": [
        {"class_id": 7, "name_vi": "Cấm quay đầu xe", "confidence": 0.93},
        {"class_id": 8, "name_vi": "Cấm rẽ trái", "confidence": 0.04}
      ],
      "uncertain": false,
      "gradcam": null
    }
  ]
}
```

`bbox` là **pixel theo ảnh gốc** (frontend tự scale khi vẽ).

### Các endpoint khác

| Method | Path | Mục đích |
| --- | --- | --- |
| GET | `/health` | Trạng thái service, model đã load |
| GET | `/models` | Danh sách model + metrics tóm tắt |
| GET | `/signs` , `/signs/{id}` | Thư viện biển báo (từ `signs.json`) |
| GET | `/metrics/{model}` | Metrics chi tiết, confusion matrix |
| POST | `/feedback` | `{request_id, detection_id, correct_class_id, consent}` |
| POST | `/predict/batch` | Nhiều ảnh (giới hạn 10) |
| WS | `/ws/stream` | Frame webcam (nâng cao) |

### Lỗi (chuẩn hóa)

```json
{"error": {"code": "UNSUPPORTED_MEDIA_TYPE", "message": "Chỉ hỗ trợ JPG, PNG, WEBP", "request_id": "…"}}
```

Mã: 400 `INVALID_IMAGE`, 413 `FILE_TOO_LARGE`, 415 `UNSUPPORTED_MEDIA_TYPE`, 422 `INVALID_PARAMS`, 429 `RATE_LIMITED`, 503 `MODEL_NOT_READY`.

### Yêu cầu triển khai server

- Load model **một lần** lúc khởi động (lifespan), warm-up 1 lần.
- Đọc ảnh bằng Pillow, sửa EXIF orientation, chuyển RGB, giới hạn cạnh dài (vd. 1920px) trước khi suy luận.
- Chạy suy luận nặng trong threadpool để không chặn event loop.
- Kiểm tra magic bytes (không tin đuôi file/`Content-Type`), giới hạn kích thước, rate-limit theo IP.
- Log có `request_id`, thời gian, model; **không lưu ảnh** mặc định.

## 6. Đặc tả Frontend (TypeScript)

**Stack:** React 18 + Vite + TypeScript (strict), Tailwind CSS, Zustand, TanStack Query, React Router, Recharts (biểu đồ), react-i18next, Zod (validate response), Vitest + Playwright.

### 6.1 Trang

| Route | Nội dung |
| --- | --- |
| `/` | Uploader + kết quả (trang chính) |
| `/history` | Lịch sử, lọc, xóa |
| `/library` | Thư viện biển báo, tìm kiếm, lọc nhóm |
| `/quiz` | Trò chơi đoán biển |
| `/models` | Metrics, confusion matrix, so sánh model |
| `/dataset` | Thống kê dữ liệu |
| `/about` | Nhóm, công nghệ, hạn chế, nguồn dữ liệu, quy chuẩn |

### 6.2 Bố cục trang chính

- **Desktop:** 2 cột — trái: vùng upload/ảnh + canvas box (60%); phải: thanh cài đặt (model, ngưỡng) + danh sách kết quả + chi tiết (40%).
- **Mobile:** 1 cột, nút "Chụp ảnh" nổi bật, kết quả dạng bottom sheet.
- Header: logo, điều hướng, chọn ngôn ngữ, **icon đổi theme sáng/tối** ở góc phải.

### 6.3 Luồng người dùng

1. Chọn/kéo/dán ảnh → preview ngay, validate client (loại, kích thước).
2. Bấm "Nhận diện" (hoặc tự chạy sau khi chọn, có thể cấu hình).
3. Hiển thị tiến trình; có thể **hủy** (AbortController).
4. Vẽ box lên canvas (SVG overlay), hover box ↔ hover hàng kết quả (highlight 2 chiều).
5. Chọn kết quả → chi tiết; bật Grad-CAM; lưu vào lịch sử; tải xuống; báo sai.

### 6.4 Kiểu dữ liệu chính

```ts
export interface Detection {
  id: number;
  bbox: { x: number; y: number; w: number; h: number };
  classId: number;
  code: string;
  nameVi: string;
  confidence: number;
  topK: { classId: number; nameVi: string; confidence: number }[];
  uncertain: boolean;
  gradcam?: string | null;
}
export interface PredictResponse {
  requestId: string;
  model: { name: string; version: string };
  image: { width: number; height: number };
  inferenceMs: number;
  detections: Detection[];
}
```

Validate response bằng Zod; chuyển snake_case → camelCase trong `api/client.ts`.

### 6.5 Yêu cầu chất lượng

- Nén/resize ảnh phía client nếu > 2500px trước khi gửi (giữ tỷ lệ) để nhanh hơn.
- Debounce slider ngưỡng; cache theo (ảnh hash, tham số) bằng TanStack Query.
- Không chặn UI: dùng Web Worker nếu xử lý ảnh nặng.
- Hỗ trợ bàn phím: Tab/Enter/Esc, phím tắt `U` upload, `D` đổi theme.

## 7. Thiết kế giao diện sáng/tối

### 7.1 Nguyên tắc

- Chỉ cần **1 icon toggle** ở góc phải header để chuyển Sáng ↔ Tối (không cần lựa chọn "Hệ thống" riêng). Lần đầu vào trang, đọc `prefers-color-scheme` để chọn theme mặc định; sau đó người dùng bấm icon là đổi và ghi đè lựa chọn đó vào localStorage. Áp `data-theme` lên `<html>` **trước khi render** (inline script) để không bị nháy sai màu.
- Dùng **design tokens (CSS variables)**, Tailwind tham chiếu token; không hard-code màu trong component.
- Chuyển theme mượt (`transition` 150–200ms cho background/color), tắt khi `prefers-reduced-motion`.
- Màu box/nhãn phải đủ tương phản trên **ảnh bất kỳ**: nhãn có nền đặc + chữ trắng/đen tự chọn, viền 2px.

### 7.2 Token màu đề xuất

| Token | Sáng | Tối |
| --- | --- | --- |
| `--bg` | #F7F8FA | #0E1116 |
| `--surface` | #FFFFFF | #161B22 |
| `--surface-2` | #EEF1F5 | #1F2630 |
| `--border` | #D8DEE6 | #2D3542 |
| `--text` | #111827 | #E6EAF0 |
| `--text-muted` | #5B6472 | #9AA4B2 |
| `--primary` | #1D4ED8 | #5B8DEF |
| `--accent` (cảnh báo/biển) | #D97706 | #F59E0B |
| `--success` | #15803D | #3FB871 |
| `--danger` | #B91C1C | #F0605D |

Màu theo nhóm biển (dùng cho chip/box, khớp màu biển thật): **Cấm** đỏ, **Hiệu lệnh** xanh dương, **Nguy hiểm** vàng, **Chỉ dẫn** xanh lam nhạt, **Biển phụ** xám. Không chỉ dùng màu để phân biệt: kèm icon/chữ (a11y).

### 7.3 Typography và bố cục

- Font: Inter (hoặc Be Vietnam Pro — hỗ trợ tiếng Việt tốt); cỡ cơ sở 16px; heading 24/20/18.
- Khoảng cách theo thang 4px; bo góc 12px (card), 8px (nút/input); bóng nhẹ ở sáng, viền + độ sáng surface ở tối (thay vì bóng).
- Grid responsive: breakpoint 640 / 1024 / 1280.

### 7.4 Thành phần UI chính

- **Uploader:** vùng viền đứt, đổi trạng thái khi dragover, icon + gợi ý "Kéo thả, dán (Ctrl+V) hoặc chọn ảnh".
- **ImageCanvas:** ảnh + SVG overlay box, zoom/pan (wheel/pinch), nút reset.
- **ResultCard:** thumbnail crop, tên, mã, chip nhóm, `ConfidenceBar` (màu theo ngưỡng: \<50% cảnh báo).
- **SignDetail:** ảnh chuẩn, ý nghĩa, hành động, nguồn quy chuẩn.
- **Toast** cho lỗi/thành công, **Skeleton** khi tải.

### 7.5 Khả năng tiếp cận (WCAG 2.1 AA)

- Tương phản chữ ≥ 4.5:1 ở cả hai theme; focus ring rõ.
- `alt` cho ảnh, `aria-live="polite"` thông báo kết quả, nhãn cho mọi nút icon.
- Kết quả có thể đọc bằng screen reader: "Phát hiện 2 biển: Cấm quay đầu xe, độ tin cậy 93%…".

## 8. Dữ liệu thư viện biển (`signs.json`)

```json
{
  "id": 7,
  "code": "P.124",
  "group": "prohibitory",
  "name_vi": "Cấm quay đầu xe",
  "name_en": "No U-turn",
  "meaning": "Cấm các phương tiện quay đầu (theo kiểu chữ U).",
  "action": "Không quay đầu xe tại vị trí đặt biển.",
  "reference_image": "signs/p124.svg",
  "source": "QCVN 41:2024/BGTVT"
}
```

Mã và mô tả **phải đối chiếu với văn bản quy chuẩn**; chú thích rõ đây là ví dụ định dạng. Mức phạt (nếu hiển thị) ghi kèm nghị định hiện hành và ngày cập nhật.

## 9. Phi chức năng

| Hạng mục | Mục tiêu |
| --- | --- |
| Hiệu năng | Inference CPU ≤ 300 ms/ảnh (ảnh ≤ 1280px); TTFB trang \< 1s |
| Độ chính xác (mục tiêu tham khảo) | Classifier ≥ 95% acc trên tập test VN; detection [mAP@0.5](mailto:mAP@0.5) ≥ 0.85 (điều chỉnh theo dữ liệu thực tế) |
| Bảo mật | Validate file, giới hạn kích thước, CORS whitelist, rate-limit, không lưu ảnh mặc định, header bảo mật |
| Quyền riêng tư | Thông báo rõ ảnh có được lưu hay không; chỉ lưu khi người dùng đồng ý (F15); nên làm mờ mặt/biển số nếu lưu |
| Tin cậy | Health check, retry có backoff ở client, thông báo lỗi thân thiện |
| Khả bảo trì | Type-safe end-to-end, lint (ESLint, Ruff), format (Prettier, Black), CI |
| Tái lập | Seed, `requirements.txt` pin phiên bản, `manifest.json` lưu hash model + dataset version |

## 10. Kiểm thử và triển khai

- **ml-service:** pytest (preprocess, postprocess/NMS, endpoint với ảnh mẫu, ảnh lỗi/quá lớn).
- **web:** Vitest cho component + hook; Playwright E2E: upload → thấy kết quả → đổi theme → xem lịch sử.
- **Contract test:** sinh type TS từ OpenAPI của FastAPI (`openapi-typescript`) để FE/BE không lệch.
- **Docker Compose:** `web` (nginx build tĩnh) + `ml-service` (uvicorn, CPU) (+ `postgres`, `api-gateway` giai đoạn 2).
- **Deploy demo:** Hugging Face Spaces / Render / VPS nhỏ cho ml-service; Vercel/Netlify cho web. Chuẩn bị **phương án offline** (chạy local bằng docker) cho buổi bảo vệ.
- CI (GitHub Actions): lint + test + build.

## 11. Kế hoạch 6 tuần (đề xuất)

| Tuần | Data/ML | Web/API |
| --- | --- | --- |
| 1 | Chốt danh sách lớp, thu thập + gán nhãn, EDA | Khởi tạo repo, thiết kế UI (Figma/wireframe), token theme |
| 2 | Pipeline tiền xử lý + augmentation, CNN baseline | Uploader, ImageCanvas, theme toggle, mock API |
| 3 | Train baseline + tinh chỉnh, bắt đầu YOLO | FastAPI `/predict` (model tạm), kết nối FE |
| 4 | Transfer learning, so sánh model, đánh giá | Kết quả + chi tiết biển, lịch sử, thư viện |
| 5 | Export ONNX, Grad-CAM, robustness test | Trang Metrics, Grad-CAM UI, i18n, a11y |
| 6 | Chốt model, báo cáo metrics | Đóng gói Docker, test E2E, slide + demo |

**Phân công gợi ý:** 1 người data & gán nhãn; 1–2 người train/đánh giá; 1 người ml-service; 1 người frontend. (Điều chỉnh theo số thành viên.)

## 12. Rủi ro và cách giảm thiểu

| Rủi ro | Giảm thiểu |
| --- | --- |
| Thiếu dữ liệu biển VN | Bắt đầu ít lớp; augmentation mạnh; pretrain trên GTSRB; ghép dữ liệu tự chụp |
| Nhãn sai/không đồng nhất | Quy ước gán nhãn rõ, kiểm chéo 10% mẫu |
| Model nhầm biển giống nhau (tốc độ) | Thu thêm dữ liệu cặp dễ nhầm, phân tích confusion matrix |
| Ảnh demo ngoài phân phối | Ngưỡng tin cậy + lớp `not_sign`; báo "không chắc chắn" |
| Deploy chậm/mất mạng khi bảo vệ | Bản chạy local Docker + video demo dự phòng |
| Vấn đề bản quyền ảnh | Ghi nguồn, dùng dữ liệu có license cho phép, dùng cho mục đích học tập |

## 13. Checklist demo và báo cáo

- [ ] Demo 5–8 ảnh: rõ, tối, mưa, nghiêng, nhiều biển, không có biển, biển lạ
- [ ] Hiển thị Grad-CAM cho 1 ca đúng và 1 ca sai
- [ ] Bảng so sánh baseline vs transfer vs YOLO (acc/F1/mAP/ms)
- [ ] Confusion matrix + phân tích lỗi
- [ ] Nêu hạn chế: số lớp hỗ trợ, điều kiện ánh sáng, biển bị che
- [ ] Đối chiếu quy chuẩn QCVN 41:2024/BGTVT, ghi nguồn dữ liệu
- [ ] Hướng phát triển: thêm lớp, ứng dụng mobile, chạy realtime trên thiết bị (ONNX/TFLite)