# Giai đoạn 3 (Tuần 5–6): Hoàn thiện, giải thích model, đóng gói và bảo vệ

> Điều kiện đầu vào: hoàn thành Giai đoạn 2 (model ONNX, `/predict` thật + Grad-CAM, trang chính, Lịch sử, Thư viện, type sinh từ OpenAPI, bộ test ml-service xanh).
> Mục tiêu cuối giai đoạn: **sản phẩm demo ổn định** (chạy được cả online lẫn offline bằng Docker), **trang Model & Metrics / Dataset / About đầy đủ**, **báo cáo + slide + kịch bản demo** sẵn sàng cho buổi bảo vệ.

## 0. Nguyên tắc của giai đoạn cuối

1. **Không thêm tính năng mới sau ngày 28.** Ngày 28 là mốc *code freeze*; ngày 29–30 chỉ sửa lỗi, luyện demo, hoàn thiện báo cáo.
2. **Không huấn luyện lại model sau ngày 26** (model freeze). Nếu số liệu thay đổi, mọi bảng/hình trong báo cáo phải sinh lại từ cùng một nguồn (`reports/*.json`) để không lệch.
3. **Ưu tiên theo thứ tự:** (1) demo không lỗi → (2) số liệu đúng và giải thích được → (3) tính năng phụ. Tính năng phụ (quiz, batch, PWA, webcam) làm sau cùng, cắt được.
4. **Mọi con số trong slide/báo cáo phải truy được về một file** trong `reports/` kèm commit hash và `manifest.json`.

## 1. Tuần 5: Giải thích model, trang phân tích, chất lượng UX

| Ngày | Công việc | Đầu ra |
|---|---|---|
| 21 | Hoàn thiện đánh giá cuối: chạy lại toàn bộ metrics trên tập test bằng **model ONNX** (không dùng checkpoint PyTorch), so sánh với PyTorch; vẽ **reliability diagram** trước/sau temperature scaling, báo cáo ECE | `reports/final_metrics.json`, `reliability.png` |
| 21 | Endpoint `GET /metrics/{model}` trả đủ: accuracy, top-3, F1 macro/per-class, confusion matrix, mAP@0.5 (và @0.5:0.95), ms/ảnh, kích thước, đường cong loss/accuracy | API metrics hoàn chỉnh |
| 21 | Endpoint `GET /dataset/stats`: số ảnh/lớp theo train/val/test, mô tả augmentation, nguồn dữ liệu, license | `dataset_stats.json` |
| 22 | Frontend trang **Model & Metrics** (`/models`): bảng so sánh model, confusion matrix (heatmap, hover xem cặp nhầm), biểu đồ loss/accuracy (Recharts), bảng F1 per-class sắp xếp được | Trang `/models` |
| 22 | Frontend trang **Dataset** (`/dataset`): biểu đồ cột số ảnh/lớp, lưới ảnh mẫu, sơ đồ chia train/val/test | Trang `/dataset` |
| 23 | **ModelSelect** trên trang chính: chọn baseline / transfer / two-stage (/YOLO) và **chế độ so sánh** (chạy cùng một ảnh trên 2 model, hiển thị cạnh nhau: box, nhãn, ms) | So sánh model trực tiếp |
| 23 | **Grad-CAM UI hoàn chỉnh**: bật/tắt overlay, thanh độ trong suốt, hiển thị heatmap theo từng detection; chuẩn bị 2 ca minh họa (1 đúng, 1 sai) lưu thành **ảnh mẫu** cho demo | Ảnh mẫu + Grad-CAM |
| 24 | **Phản hồi (F15)**: `POST /feedback` `{request_id, detection_id, correct_class_id, consent}`; UI nút "Báo sai" + hộp thoại xin đồng ý; chỉ lưu ảnh khi `consent=true` | Feedback có consent |
| 24 | Tính năng phụ 1: **Ảnh mẫu có sẵn (F24)** + **xuất kết quả** (PNG có box, JSON) (F17) | Demo không cần ảnh riêng |
| 25 | **i18n vi/en** (`react-i18next`), mặc định tiếng Việt; rà soát mọi chuỗi cứng | `vi.json`, `en.json` |
| 25 | **Khả năng tiếp cận (WCAG 2.1 AA)**: tương phản ≥ 4.5:1 cả 2 theme, focus ring, `aria-live` cho kết quả, nhãn cho nút icon, thao tác bằng bàn phím; chạy Lighthouse/axe | Báo cáo a11y, danh sách lỗi đã sửa |
| 25 | Rà soát theme sáng/tối trên mọi trang mới (không nháy màu, box/nhãn đủ tương phản trên ảnh bất kỳ) | Theme nhất quán |

## 2. Tuần 6: Chốt model, đóng gói, kiểm thử, báo cáo và bảo vệ

| Ngày | Công việc | Đầu ra |
|---|---|---|
| 26 | **Model freeze**: chốt model cuối, ghi `manifest.json` (hash model, phiên bản dataset, seed, ngưỡng, calibration); tag Git `v1.0-model` | Model + manifest cố định |
| 26 | Kiểm thử hiệu năng: đo ms/ảnh trên CPU (ảnh ≤ 1280 px, 3 lần đo, bỏ lần warm-up), đo RAM, thời gian khởi động container | `reports/perf.json` |
| 26 | Chạy lại **robustness** (tối, mưa, mờ, nghiêng, che một phần) trên model cuối; điền bảng giảm accuracy | Bảng robustness cuối |
| 27 | **Docker Compose**: `web` (build tĩnh + nginx) + `ml-service` (uvicorn, CPU); healthcheck, biến môi trường, CORS whitelist; `docker compose up` chạy được trên máy sạch | `docker-compose.yml`, `.env.example` |
| 27 | Tài liệu vận hành trong `README`: cài đặt, chạy local, chạy Docker, cấu trúc repo, cách tái lập kết quả (seed, lệnh train/evaluate) | `README` hoàn chỉnh |
| 28 | **Test E2E (Playwright)**: upload → thấy box + kết quả → bật Grad-CAM → đổi theme → xem lịch sử → mở thư viện → đổi ngôn ngữ; test ảnh lỗi/quá lớn/sai định dạng | Bộ E2E xanh |
| 28 | **CI (GitHub Actions)**: lint (ESLint, Ruff, Prettier, Black) + pytest + Vitest + build; type từ OpenAPI không lệch | CI xanh |
| 28 | **Code freeze** cuối ngày 28; tag `v1.0` | Bản phát hành demo |
| 29 | **Deploy demo** (Hugging Face Spaces / Render / VPS cho ml-service; Vercel/Netlify cho web) và kiểm tra bằng điện thoại; chuẩn bị **bản offline** (Docker) và **video demo dự phòng** | URL demo + video dự phòng |
| 29 | Hoàn thiện trang **About**: nhóm, công nghệ, hạn chế, nguồn dữ liệu, căn cứ QCVN 41:2024/BGTVT, lưu ý về quyền riêng tư | Trang `/about` |
| 29 | Viết **báo cáo** (mục 4) và chốt hình/bảng từ `reports/` | Báo cáo bản đầy đủ |
| 30 | Làm **slide** (mục 5), tập demo theo kịch bản (mục 6) ít nhất 2 lần có bấm giờ; kiểm tra mạng, thiết bị, tài khoản, bản offline | Slide + kịch bản demo |
| 30 | Tổng duyệt: một người ngoài nhóm dùng thử; sửa lỗi nghiêm trọng còn lại; đóng băng repo | Bản nộp cuối |

## 3. Đặc tả kỹ thuật cần chốt trong giai đoạn này

**Endpoint bổ sung**

| Method | Path | Ghi chú |
|---|---|---|
| GET | `/dataset/stats` | Thống kê dữ liệu cho trang `/dataset` |
| POST | `/feedback` | `{request_id, detection_id, correct_class_id, consent}`; không lưu ảnh nếu `consent=false`; rate-limit; trả 204 |
| POST | `/predict/batch` | Tùy chọn, ≤ 10 ảnh/lần, xử lý tuần tự trong threadpool, trả mảng kết quả kèm lỗi riêng từng ảnh |

Mọi endpoint mới phải xuất hiện trong OpenAPI và được sinh lại `api/types.ts`; lỗi dùng đúng định dạng chuẩn đã chốt (400/413/415/422/429/503).

**Quyền riêng tư cho `/feedback`:** thông báo rõ trước khi gửi; chỉ lưu khi đồng ý; lưu kèm `request_id`, nhãn đề xuất, thời điểm; không lưu IP; nên làm mờ mặt/biển số xe trước khi lưu (hoặc ghi rõ đây là hạn chế nếu không kịp).

**Cấu hình triển khai (Docker):**
- `web`: nginx phục vụ bản build tĩnh, header bảo mật, proxy `/api` sang `ml-service` (tránh CORS khi chạy local).
- `ml-service`: uvicorn, 1 worker (model load một lần), giới hạn CPU/RAM, healthcheck gọi `/health`, model mount qua volume hoặc copy vào image.
- Phiên bản pin trong `requirements.txt` và `package-lock.json`; image không chứa dữ liệu thô.

**Chiến lược cắt giảm (làm theo thứ tự, dừng khi hết thời gian):**

| Ưu tiên | Hạng mục | Ghi chú |
|---|---|---|
| Bắt buộc | Metrics, Dataset, About, Grad-CAM UI, Docker, E2E, báo cáo, slide | Không được cắt |
| Nên có | So sánh model cạnh nhau, i18n vi/en, a11y, feedback, ảnh mẫu, xuất kết quả | Cắt i18n `en` trước nếu thiếu thời gian (giữ `vi`) |
| Cắt được | Quiz (F19), batch upload (F21), PWA (F23) | Chỉ làm khi ngày 25 đã xong phần trên |
| Chỉ ghi vào "Hướng phát triển" | Webcam realtime (F20), api-gateway + PostgreSQL + đăng nhập admin (mục 2.4) | Không triển khai trong 6 tuần |

## 4. Cấu trúc báo cáo (gợi ý)

1. Đặt vấn đề, mục tiêu, phạm vi (số lớp MVP, điều kiện ảnh).
2. Cơ sở lý thuyết ngắn: CNN, transfer learning, detection, calibration, Grad-CAM.
3. Dữ liệu: nguồn, license, audit, ánh xạ nhãn theo QCVN 41:2024/BGTVT, chia tập, augmentation, EDA.
4. Phương pháp: kiến trúc hệ thống (1 tầng hay 2 tầng và lý do), huấn luyện, siêu tham số (`experiments.csv`).
5. Kết quả: bảng so sánh baseline / MobileNetV2 / model thứ 3 hoặc YOLO (accuracy, F1 macro, mAP, ms/ảnh, MB); confusion matrix; reliability diagram.
6. **Phân tích lỗi** và **robustness** có ví dụ ảnh; Grad-CAM cho 1 ca đúng, 1 ca sai.
7. Hệ thống: API, giao diện, bảo mật, quyền riêng tư, triển khai.
8. Hạn chế (số lớp hỗ trợ, ánh sáng, biển bị che, phụ thuộc chất lượng dữ liệu) và hướng phát triển.
9. Phụ lục: hướng dẫn chạy lại, phân công, tài liệu tham khảo, nguồn dữ liệu.

## 5. Cấu trúc slide (gợi ý ~12–15 slide, 10–15 phút)

Vấn đề & mục tiêu → Dữ liệu & QCVN → Kiến trúc → Mô hình & huấn luyện → **Bảng so sánh model** → Phân tích lỗi → Robustness & Grad-CAM → **Demo trực tiếp** → Triển khai & bảo mật → Hạn chế & hướng phát triển → Phân công & Q&A.

## 6. Kịch bản demo (5–8 ảnh, có bấm giờ)

| # | Ảnh | Điều cần chứng minh |
|---|---|---|
| 1 | Một biển rõ, ban ngày | Luồng chính, box + mã + ý nghĩa |
| 2 | Cảnh có nhiều biển | Detection nhiều biển, hover 2 chiều |
| 3 | Ảnh tối hoặc mưa | Robustness |
| 4 | Ảnh nghiêng / bị che một phần | Giới hạn, top-3 |
| 5 | Ảnh không có biển | Lớp `not_sign`, "không tìm thấy biển" |
| 6 | Biển lạ ngoài tập lớp | `uncertain`, không ép nhãn sai |
| 7 | Ca đúng + ca sai với Grad-CAM | Khả năng giải thích |
| 8 | Chạy so sánh 2 model, đổi theme, xem trang Metrics | Chiều sâu học thuật |

Chuẩn bị sẵn: ảnh demo trong `web/public/samples/`, bản offline Docker, video dự phòng, thẻ ghi lệnh khởi động nhanh. Phân vai: một người thuyết trình, một người thao tác, một người trực Q&A và xử lý sự cố.

## 7. Phân công gợi ý

| Vai trò | Tuần 5 | Tuần 6 |
|---|---|---|
| ML engineer 1 | Đánh giá cuối bằng ONNX, reliability diagram, `/metrics` | Model freeze, `manifest.json`, perf, robustness cuối |
| ML engineer 2 | Ca minh họa Grad-CAM, chuẩn bị ảnh mẫu | Phần kết quả + phân tích lỗi trong báo cáo |
| Backend | `/dataset/stats`, `/feedback`, (batch) | Docker, CI, deploy, hardening |
| Frontend | Metrics, Dataset, ModelSelect, i18n, a11y | E2E, About, sửa lỗi UI, slide demo |
| Data lead | `dataset_stats.json`, rà `signs.json` cuối | Phần dữ liệu + QCVN trong báo cáo, nguồn/license |

## 8. Tiêu chí hoàn thành (Definition of Done)

- [ ] Trang `/models` hiển thị bảng so sánh ≥ 3 model, confusion matrix, biểu đồ huấn luyện, F1 per-class; số liệu khớp `reports/final_metrics.json`
- [ ] Trang `/dataset` và `/about` đầy đủ, có nguồn dữ liệu, license, căn cứ QCVN 41:2024/BGTVT và hạn chế
- [ ] Grad-CAM bật/tắt được, có 2 ca minh họa (đúng/sai); chế độ so sánh 2 model chạy được
- [ ] `/feedback` chỉ lưu khi có đồng ý; không lưu ảnh mặc định
- [ ] Lighthouse Accessibility ≥ 90; không có lỗi axe mức nghiêm trọng; dùng được bằng bàn phím
- [ ] `docker compose up` chạy được trên máy sạch; inference CPU ≤ ~300 ms/ảnh (ảnh ≤ 1280 px)
- [ ] CI xanh (lint + pytest + Vitest + build); E2E Playwright xanh
- [ ] Có URL demo, bản offline Docker và video demo dự phòng
- [ ] Báo cáo và slide hoàn chỉnh; tập demo đủ 2 lần đúng thời gian
- [ ] Repo đóng băng với tag `v1.0`, `README` cho phép người khác chạy lại và tái lập kết quả

## 9. Rủi ro giai đoạn 3

| Rủi ro | Giảm thiểu |
|---|---|
| Số liệu trong báo cáo lệch với model demo | Model freeze ngày 26; sinh mọi bảng/hình từ `reports/`, ghi hash model trong báo cáo |
| Deploy chậm hoặc mất mạng khi bảo vệ | Bản offline Docker + video dự phòng; kiểm tra trên mạng khác trước ngày bảo vệ |
| Docker chạy khác máy phát triển | Test trên máy sạch/CI, pin phiên bản, healthcheck |
| Thêm tính năng phút chót gây lỗi | Code freeze ngày 28; chỉ sửa lỗi sau đó |
| Demo ra kết quả sai trước hội đồng | Chọn sẵn ảnh đã kiểm tra; giải thích ca sai bằng Grad-CAM và phân tích lỗi thay vì né tránh |
| Hội đồng hỏi về dữ liệu/bản quyền/quyền riêng tư | Chuẩn bị `DATA_SOURCES.md`, trang About, chính sách `/feedback`; nói rõ ảnh không được lưu mặc định |
| Thiếu thời gian cho báo cáo | Viết các mục dữ liệu/phương pháp song song từ tuần 5; dùng khung ở mục 4 |
| Thành viên vắng hoặc quá tải | Mỗi hạng mục có người dự phòng; các mục "Cắt được" (mục 3) được bỏ trước |