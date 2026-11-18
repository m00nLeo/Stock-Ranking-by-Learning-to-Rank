# 3.11 — Trạng thái xu hướng thị trường theo ngày (RQ3)

Thực hiện bởi Trường hợp 3 (Cline), rà soát và tích hợp bởi Trường hợp 1 (Claude Code), 07/11/2026.

## 1. Mục đích và luật (D63 mục 6, dinh_nghia_xu_huong.md)

Trạng thái thị trường chỉ dùng để chia kết quả khi báo cáo RQ3 (số phiên trong từng trạng thái ở
train, validation, test). Không làm đặc trưng, không dùng chọn mô hình (sau D43 bỏ cổng tin cậy).

Luật (đặt trước, chỉ dùng dữ liệu <= t), trên VN-Index:
- Tăng: giá đóng cửa > SMA200 và SMA50 > SMA200.
- Giảm: giá đóng cửa < SMA200 và SMA50 < SMA200.
- Đi ngang: các trường hợp còn lại.

Cần 200 phiên lịch sử trước khi có trạng thái; lịch sử từ 2022 (D62) đáp ứng được.

## 2. Mã nguồn

- `code/vnrank/labels/market_state.py`: hàm `compute_market_state(vnindex_df, sma_short=50,
  sma_long=200)`. Dùng `rolling(window=..., min_periods=window)` nên SMA tại ngày t chỉ dùng giá
  đóng cửa <= t (point-in-time đúng), và trả về `None`/NaN cho các ngày chưa đủ 200 phiên.
- `code/tests/test_market_state.py`: 4 test (tăng, giảm, đi ngang trên dữ liệu tổng hợp nhỏ với
  cửa sổ rút gọn để test nhanh; và một test xác nhận 199 ngày đầu chưa có trạng thái khi dùng cửa
  sổ mặc định 50/200). 50/200 là hằng số lấy thẳng từ D63, không chọn theo kết quả test.
- `__main__` trong `market_state.py`: đọc `data/raw/prices/VNINDEX.csv`, ghi
  `data/processed/market_state_daily.csv` (ngày, trạng thái) — theo đúng khuôn mẫu đã dùng ở
  `build_universe.py`, `feature_registry.py`.

## 3. Kết quả trên dữ liệu thật

`VNINDEX.csv`: 1201 phiên (2022-01-04 đến 2026-11-03). 199 phiên đầu chưa đủ 200 phiên lịch sử nên
`state` là NaN; từ phiên thứ 200 trở đi:

| Trạng thái | Số phiên (toàn bộ VN-Index) |
| :--- | ---: |
| Tăng | 607 |
| Giảm | 211 |
| Đi ngang | 184 |

## 4. Số phiên theo trạng thái ở train/validation/test (yêu cầu mục 6)

Ghép `market_state_daily.csv` với `splits_daily.csv` (3.7) theo `date`:

| Tập | Tăng | Giảm | Đi ngang | Không khớp VN-Index |
| :--- | ---: | ---: | ---: | ---: |
| Train | 183 | 114 | 71 | 1 |
| Validation | 115 | 17 | 55 | 1 |
| Test | 309 | 8 | 58 | 0 |

Ghi chú:
- `splits_daily.csv` có 954 dòng, trong đó 22 dòng không thuộc train/validation/test (khoảng purge
  giữa các tập, theo `split_protocol.py`) — không tính trong bảng trên vì mục 6 chỉ yêu cầu ba tập.
- 2 ngày (2023-04-07 thuộc train, 2024-08-12 thuộc validation) có trong `splits_daily.csv` nhưng
  không có dòng tương ứng trong `VNINDEX.csv` (chênh lệch lịch giao dịch giữa chỉ số và từng mã) →
  trạng thái NaN cho hai ngày này, không suy diễn thay. Tỷ lệ 2/954 (0,2%), không ảnh hưởng kết luận.
- Tập validation và test lệch mạnh về "Tăng" (115/188 và 309/375) trong khi train có nhiều phiên
  "Giảm" hơn (114/369) — phản ánh giai đoạn thị trường biến động trong lịch sử gần đây so với giai
  đoạn phục hồi gần cuối mẫu. Khi báo cáo RQ3 theo trạng thái, tập "Giảm" ở validation/test có rất ít
  phiên (17, 8) nên kết quả trong trạng thái đó chỉ mang tính minh họa (đúng lưu ý ở mục 6).

## 5. Kiểm tra không dùng dữ liệu sau t

`rolling(..., min_periods=window)` của pandas chỉ dùng các giá trị tại vị trí <= vị trí hiện tại
trong chuỗi đã sort theo `date`; không có phép toán nào nhìn về phía trước. Đã xác nhận bằng test
`test_state_nan_before_enough_history_for_long_sma` (199 ngày đầu phải là NaN, không phải suy diễn
ngược từ tương lai).

## 6. Phạm vi không làm trong bước này

- Không dùng trạng thái này làm đặc trưng hay để chọn mô hình (giữ đúng D63/D43).
- Không tính lại theo chỉ số đồng trọng số của universe (phương án bị loại ở D63, ghi giới hạn
  trong luận văn).
- Không merge cột trạng thái vào `labels_daily.csv`/`labels_raw.csv` vì đây là bảng theo ngày
  (không theo mã, không theo h), tách riêng ở `market_state_daily.csv` cho rõ ràng; khi viết báo
  cáo RQ3 sẽ join theo `date` khi cần.

## 7. Test

Toàn bộ suite: 80 PASS (76 trước đó + 4 test mới của `test_market_state.py`).
