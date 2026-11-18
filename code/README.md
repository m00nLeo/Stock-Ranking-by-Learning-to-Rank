# code/: mã nguồn luận văn (Giai đoạn 3-5, luồng S)

Đây là phần duy nhất của dự án `E:/Thạc sĩ/Internship2/` được đẩy lên GitHub. Mọi tài liệu luận văn
(đề cương, giao thức, quyết định, PDF gốc...) nằm ngoài thư mục này và bị `.gitignore` chặn ở gốc dự án,
xem `../README.md` và `../00_Quan_ly_du_an/` (chỉ trên máy cục bộ, không lên GitHub).

## Nguyên tắc

- Mọi hằng số đã chốt ở Giai đoạn 2 (mốc chia dữ liệu, h, K, M, chi phí, lưới mô hình...) nằm ở
  `config/config.yaml`. Code không được lặp lại các con số này; luôn đọc từ file cấu hình.
- Không chọn tham số trên test. Cấu hình chỉ chọn một lần trên validation (charter mục 5, điều 3 và 10).
- Mỗi module code có test đi kèm trong `tests/`; sau mỗi module mới, chạy thêm một lượt kiểm chứng
  riêng (look-ahead, rò rỉ giữa các tầng, nhất quán as-of) theo quy ước ở charter mục 8.
- Artefact mô hình (trong `artifacts/`) phải có tên phiên bản và log số lần thử.

## Cấu trúc thư mục

```
code/
├── config/           # config.yaml, mọi hằng số đã chốt (D54-D65)
├── vnrank/           # package Python lõi, chạy được cả cục bộ lẫn trên Colab
│   ├── data/         # tải, làm sạch dữ liệu: entrade_client.py (3.2, D71), làm sạch (3.3)
│   ├── universe/     # universe theo thời điểm (3.4)
│   ├── features/     # feature_registry (3.5)
│   ├── labels/       # nhãn lợi suất + relevance (3.6), nhãn xu hướng triple-barrier (3.10),
│   │                 # trạng thái thị trường (3.11)
│   ├── splits/       # chia train/val/test, purge, embargo (3.7)
│   ├── asof/         # pipeline chế độ as-of (3.12)
│   ├── costs/        # mô hình chi phí VN (2.6, dùng ở 3.6 và nhãn xu hướng)
│   ├── models/       # baseline B0-B4, hồi quy-rồi-topK R1-R3, ranking L1, phân loại xu hướng T2 (4.x)
│   ├── evaluation/   # harness đánh giá: Newey-West, bootstrap khối, Holm, DSR/PBO (2.4, 4.1)
│   └── system/       # ba hàm đọc `xep_hang`, `xu_huong`, `ket_qua`; chatbot (S1-S12);
│                     # rag.py (D91, tra cứu code/docs/), sector_analysis.py (D91/D92, S13)
├── notebooks/        # notebook chạy trên Google Colab: pipeline dữ liệu, huấn luyện, đánh giá, demo
├── tests/            # unit test cho vnrank/, test khung của bước 3.1
├── docs/             # báo cáo kiểm tra theo bước (ví dụ 3_2_kiem_tra_du_lieu.md)
├── data/             # dữ liệu cục bộ (raw/interim/processed), KHÔNG lên GitHub, chỉ giữ cấu trúc
├── artifacts/        # model artefact có tên phiên bản, KHÔNG lên GitHub
├── outputs/          # bảng, hình, log chạy ra, KHÔNG lên GitHub
├── requirements.txt
└── README.md         # file này
```

## Chạy ở đâu

Theo quyết định ở `00_Quan_ly_du_an/decision_log.md` (bước 3.1, sau Run 31): thực nghiệm và hệ thống
minh họa đều chạy trên **Google Colab**, để đồng nhất môi trường. `vnrank/` là package Python thuần,
import được cả từ notebook Colab lẫn khi chạy thử cục bộ (ví dụ chạy `pytest` trước khi đẩy lên Colab).

Quy trình dự kiến trên Colab:
1. Clone repo GitHub này vào Colab (`!git clone ...`), hoặc mount Google Drive nếu cần dữ liệu lớn.
2. `pip install -r code/requirements.txt`.
3. Chạy notebook tương ứng trong `notebooks/`, notebook gọi vào các hàm trong `vnrank/`.
4. Model artefact và log ghi ra Google Drive (D64), không ghi vào `artifacts/` của repo.

## Cài đặt cục bộ (tùy chọn, để chạy test nhanh trước khi đẩy lên Colab)

```bash
pip install -r requirements.txt
pytest
```

## Trạng thái

**[SỬA, 11/11/2026]** Mục này trước chỉ cập nhật tới 3.9 (69 test); đã bị bỏ quên không làm mới
qua nhiều bước sau đó — cùng lỗi quy trình đã phát hiện và sửa ở `BRIEF_phien_moi.md` (D90). Nội
dung 3.1-3.9 cũ giữ nguyên bên dưới, nối thêm phần còn lại.

Khung dự án (3.1) xong. Bước 3.2 (kiểm tra nguồn dữ liệu) xong, có giới hạn: `vnrank/data/entrade_client.py`
gọi API công khai Entrade/DNSE thay cho `vnstock` (bị PyPI cách ly, xem `docs/3_2_kiem_tra_du_lieu.md` và
`decision_log.md` D71); KHÔNG có dữ liệu mã đã hủy niêm yết, chấp nhận làm giới hạn của luận văn. Bước 3.3
(làm sạch dữ liệu) xong: `vnrank/data/cleaning.py` xác nhận giá đã điều chỉnh, phát hiện khóa trần/sàn qua
OHLC suy biến (D73). Bước 3.4 (universe theo thời điểm) xong: `vnrank/universe/build_universe.py` dựng
$U_t$ hằng ngày (150 mã, 01/2023-11/2026), tra VN30 theo kỳ, ước lượng số mã thiếu do bỏ mã hủy niêm yết
(D74, xem `docs/3_4_universe.md`). Bước 3.5 (sổ đăng ký feature) xong: `vnrank/features/feature_registry.py`
sinh 11 đặc trưng và chuẩn hóa chéo theo ngày (D76); phát hiện và sửa lỗi nguồn Entrade trả trùng ngày ở
3 ngày cụ thể, ảnh hưởng 521 mã (D77, xem `docs/3_5_feature_registry.md`). Bước 3.6 (nhãn 5/10/21 ngày,
gộp cả nội dung 3.10) xong: `vnrank/labels/labeling.py` tính nhãn hồi quy, mức liên quan, nhãn xu hướng
triple-barrier, quy tắc khóa trần/sàn (D78, xem `docs/3_6_nhan.md`). Bước 3.7 (chia train/validation/test,
purge, embargo) xong: `vnrank/splits/split_protocol.py` xác định cụ thể `test_end` (02/10/2026) và cài đặt
purge theo $t^1$ (D79, xem `docs/3_7_chia_du_lieu.md`). Bước 3.8 (kiểm tra leakage) xong:
`vnrank/evaluation/leakage_checks.py` kiểm tra xáo trộn nhãn và nhãn ngẫu nhiên (Rank IC gần 0 trên
validation, mô hình Ridge chỉ để kiểm tra, không phải mô hình chính thức, D80, xem
`docs/3_8_kiem_tra_leakage.md`). Bước 3.9 (tài liệu dataset) xong: data card đầy đủ cho bốn bảng
universe/feature/label/split, `data/processed/README.md` và `MANIFEST.md` (phiên bản `v20261107-ce5d2e5c`,
D81, xem `docs/3_9_dataset_card.md`).

**Giai đoạn 3 (3.1-3.13) nay đã XONG HOÀN TOÀN** (D85): thêm nhãn xu hướng/kiểm tra rò rỉ cho nhãn
xu hướng (3.13, D83), trạng thái thị trường theo ngày (3.11, D84, lần đầu Cline làm), pipeline chế
độ as-of mô phỏng hệ thống chạy thật (3.12, D85).

**Giai đoạn 4 đang làm:** harness đánh giá — Rank IC/ICIR, Newey-West, bootstrap khối, Holm (4.1,
D86, `vnrank/evaluation/harness.py` và `multiple_testing.py`); baseline không dùng ML B0-B4, không
baseline tín hiệu nào có Rank IC dương có ý nghĩa trên test (4.2, D88, `vnrank/models/baselines.py`).
Tiếp theo: 4.3 (baseline hồi quy rồi top-K).

**Mở rộng hệ thống minh họa (luồng S, không ảnh hưởng phần benchmark trên):** RAG tra cứu báo cáo
kỹ thuật cho chatbot (D91, `vnrank/system/rag.py`, TF-IDF tự cài) và phân tích/lọc cổ phiếu theo
nhóm ngành GICS trên toàn thị trường, 707/707 mã (D91/D92, `vnrank/system/sector_analysis.py`,
`data/processed/sector_mapping.csv`). Theo yêu cầu của giảng viên hướng dẫn.

134 test PASS cục bộ. Chi tiết đầy đủ từng quyết định: `00_Quan_ly_du_an/decision_log.md`.
