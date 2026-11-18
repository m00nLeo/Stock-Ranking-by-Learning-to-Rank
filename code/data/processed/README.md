# code/data/processed/ — dữ liệu đã qua xử lý

Bốn bảng chính của pipeline Giai đoạn 3, theo thứ tự tạo ra:

1. `universe_daily.csv` — universe $U_t$ theo ngày (3.4).
2. `features_daily.csv` — 11 đặc trưng, đã chuẩn hóa chéo theo ngày (3.5).
3. `labels_daily.csv` — nhãn hồi quy, mức liên quan, nhãn xu hướng triple-barrier (3.6/3.10). Bản trung
   gian trước khi giới hạn vào $U_t$: `labels_raw.csv`.
4. `splits_daily.csv` — gán train/validation/test theo ngày, CHƯA purge (3.7); purge khi dùng thật qua
   `vnrank.splits.split_protocol.purge_by_realization`.

Mô tả đầy đủ từng cột, nguồn, giới hạn đã biết và cách tái lập: xem `code/docs/3_9_dataset_card.md`.
Mã băm xác nhận phiên bản: `MANIFEST.md` trong thư mục này.

**Lưu ý:** các file CSV thật (không phải hai file `.md` này) KHÔNG lên GitHub (`.gitignore`), vì dữ liệu
tự sinh được từ `code/data/raw/` (cũng không lên GitHub) qua các module trong `code/vnrank/`. Chạy lại
theo đúng thứ tự ở mục 4 của `3_9_dataset_card.md` để tái tạo.

**[SỬA, D92, 11/11/2026]** Thư mục này còn có `sector_mapping.csv` (707 mã, ánh xạ ngành GICS) —
đây KHÔNG thuộc bốn bảng chính ở trên, không tính vào mã băm `MANIFEST.md`, chỉ phục vụ phần mở
rộng hệ thống minh họa (S13, `vnrank/system/sector_analysis.py`), không dùng cho benchmark xếp
hạng/đánh giá ở Giai đoạn 4. Sinh lại bằng `python -m vnrank.system.sector_analysis` (đọc
`code/data/raw/sector_inputs/`).
