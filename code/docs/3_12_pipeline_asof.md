# 3.12 — Đóng gói pipeline chế độ as-of

07/11/2026.

## 1. Mục đích

Bước cuối cùng của Giai đoạn 3: dựng hai hàm `get_universe(as_of)` và `get_features(as_of)` mô
phỏng đúng những gì một hệ thống chạy thật sẽ thấy tại một ngày bất kỳ, chỉ dùng dữ liệu giá đến hết
ngày đó. Đây là hạ tầng cho S2 (khung giao diện minh họa) và cho ba hàm đọc `xep_hang`, `xu_huong`,
`ket_qua` (D64) sau này.

## 2. Thiết kế

`code/vnrank/asof/pipeline.py`. Nguyên tắc: KHÔNG tính lại thuật toán 3.4/3.5, chỉ gọi thẳng các hàm
đã có (`build_daily_universe`, `compute_liquidity_features` của `build_universe.py`;
`compute_features`, `cross_sectional_normalize`, `fill_market_calendar_gaps` của
`feature_registry.py`), sau khi cắt dữ liệu đầu vào còn `<= as_of`.

- `_load_panel_asof(as_of, price_cache_dir, listing_path)`: đọc panel giá cổ phiếu và VN-Index đã
  cache, lọc còn `date <= as_of` TRƯỚC khi đưa vào bất kỳ hàm tính nào. Đây là nơi duy nhất thực
  hiện việc cắt; mọi hàm công khai đều đi qua đây.
- `get_universe(as_of)`: đọc ngưỡng lọc từ `config.yaml` mục `universe` (`n_size`,
  `liquidity_window_sessions`, `liquidity_lag_sessions`, `price_min_thousand_vnd`,
  `avg_value_min_thousand_vnd`, `min_active_sessions` — đúng các giá trị đã chốt ở D74, không chép
  lại thành hằng số trong code), gọi `compute_liquidity_features` rồi `build_daily_universe` trên
  panel đã cắt, lọc lấy đúng dòng ngày `as_of`.
- `get_features(as_of)`: gọi `get_universe(as_of)` trước để biết $U_{as\_of}$, tính đặc trưng thô
  bằng `compute_features` trên panel/market đã cắt, chuẩn hóa chéo bằng `cross_sectional_normalize`
  giới hạn trong $U_{as\_of}$, lọc lấy đúng dòng ngày `as_of`.
- Cả hai hàm nhận thêm `price_cache_dir`, `listing_path` (mặc định là đường dẫn thật của dự án) để
  test có thể tiêm dữ liệu tổng hợp mà không đụng vào `data/raw/` thật.
- Không cache giữa các lần gọi: mỗi lần gọi tính lại toàn bộ lịch sử đến `as_of` cho tất cả mã đã
  niêm yết. Ưu tiên đúng và rõ ràng hơn tốc độ (theo đúng yêu cầu của bước); thời gian thực đo trên
  707 mã khoảng 25-30 giây một lần gọi `get_features`.

## 3. Tại sao cắt dữ liệu không đổi kết quả (nhưng vẫn cần cắt)

Mọi công thức ở 3.4/3.5 vốn đã chỉ nhìn về quá khứ: `rolling().shift()`, `ewm(adjust=False)`, và
chuẩn hóa chéo tính riêng theo từng ngày (`groupby("date")`). Vì vậy, với các hàm này, giá trị tính
cho một ngày `t` không phụ thuộc vào việc có hay không có các dòng dữ liệu sau `t` trong bảng đầu
vào — đã xác nhận thực nghiệm (gỡ tạm dòng lọc `<= as_of`, chạy lại test, kết quả không đổi).

Dù vậy, việc cắt dữ liệu trong `_load_panel_asof` vẫn giữ lại, vì đây là một đảm bảo ở mức hợp đồng
(contract), không phụ thuộc vào tính chất toán học hiện tại của 3.4/3.5 — nếu sau này có ai thêm một
đặc trưng không thuần nhìn-về-quá-khứ (ví dụ một phép chuẩn hóa theo toàn bộ lịch sử thay vì theo
ngày), pipeline as-of vẫn an toàn nhờ lớp cắt dữ liệu này, không phải sửa lại từng hàm đặc trưng.

## 4. Test

`code/tests/test_asof_pipeline.py`, dữ liệu tổng hợp nhỏ (5 mã, 190 phiên, không đọc
`data/raw/universe_inputs/` hay `data/raw/prices/` thật, không gọi mạng — đúng quy ước các test khác
trong dự án):

1. `test_get_universe_and_features_match_offline_reconstruction` (3 ngày as_of: sớm/giữa/cuối lịch
   tổng hợp): `get_universe`/`get_features` phải khớp tuyệt đối với kết quả tính trực tiếp bằng
   `build_daily_universe`/`compute_features`/`cross_sectional_normalize` trên cùng panel, lọc lấy
   đúng ngày đó — mô phỏng đúng việc đối chiếu với `universe_daily.csv`/`features_daily.csv` đã dựng
   sẵn mà đặc tả yêu cầu.
2. `test_load_panel_asof_never_returns_rows_after_as_of`: kiểm trực tiếp hợp đồng cắt dữ liệu — gọi
   `_load_panel_asof` và xác nhận không có dòng nào có `date > as_of`, dù cache tổng hợp có đủ 190
   phiên. Đã xác nhận test này THẬT SỰ bắt được lỗi: gỡ tạm dòng lọc trong code, test báo FAIL ngay
   (`AssertionError: 2023-09-22 <= 2023-07-31` sai), rồi khôi phục lại đúng.
3. `test_get_universe_and_features_stable_when_cache_has_shocked_future_rows` (3 ngày as_of): cache
   có sẵn dữ liệu SAU as_of với giá/khối lượng bị chỉnh thành giá trị cực đoan (x1000); kết quả ngày
   as_of phải giữ nguyên, khớp với kết quả tính trên panel bị cắt vật lý.
4. `test_get_universe_empty_before_enough_liquidity_history`: ngày chưa đủ 21 phiên lịch sử thanh
   khoản thì universe rỗng, không suy đoán.

Toàn bộ suite: 88 PASS (từ 80, thêm 8 test mới).

## 5. Đối chiếu trên dữ liệu thật (chỉ đọc, không ghi)

Gọi `get_universe`/`get_features` trên dữ liệu thật của dự án (707 mã, panel giá đầy đủ) cho 3 ngày
mẫu (đầu train, giữa validation, giữa test), đối chiếu với `universe_daily.csv`/`features_daily.csv`
đã dựng sẵn ở 3.4/3.5:

| as_of | Số mã $U_t$ | Khớp universe | Khớp feature (rtol 1e-9) | Thời gian |
| :--- | ---: | :---: | :---: | ---: |
| 2023-10-02 | 150/150 | Đúng | Đúng | ~29s |
| 2024-11-12 | 150/150 | Đúng | Đúng | ~30s |
| 2025-12-26 | 150/150 | Đúng | Đúng | ~27s |

Khớp tuyệt đối ở cả ba ngày mẫu (đầu train, giữa validation, giữa test).

## 6. Phạm vi không làm trong bước này

- Không đụng tới S2 (khung giao diện minh họa) — đó là bước kế tiếp dùng module này làm hạ tầng.
- Không tối ưu tốc độ (không cache giữa các lần gọi, không giới hạn phạm vi panel đọc); để lại cho
  khi S2 hoặc hệ thống minh họa thật sự cần.
- Không gộp `get_universe`/`get_features` thành một hàm; giữ tách biệt theo đúng chữ ký đã chốt ở
  charter mục 7 dòng 3.12.
