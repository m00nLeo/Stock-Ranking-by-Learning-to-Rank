# Báo cáo bước 3.5: sổ đăng ký feature, chuẩn hóa chéo, sửa lỗi dữ liệu trùng ngày

**Ngày:** 06/11/2026. **Trạng thái:** Xong (D76, D77). **Căn cứ:** `bai_toan_hinh_thuc.md` mục 3 (D55);
`charter.md` mục 4; `giao_thuc_chia_du_lieu_bias.md` rủi ro #1; `config/config.yaml`; `3_4_universe.md`.

## 1. Sổ đăng ký feature (D76)

Spec gốc (`bai_toan_hinh_thuc.md` mục 3, `charter.md` mục 4) chỉ nêu tên nhóm đặc trưng, không có công
thức hay cửa sổ cụ thể: "nhóm giá và lợi suất quá khứ, SMA, RSI, MACD, khối lượng, thanh khoản, và một
số đặc trưng chéo mã đơn giản". Bước 3.5 chốt cụ thể 11 đặc trưng sau:

| Tên | Nhóm | Công thức | Cửa sổ | Độ trễ |
|---|---|---|---|---|
| `ret_21`, `ret_63`, `ret_126` | Lợi suất quá khứ | $\ln(C_t / C_{t-N})$ | 21, 63, 126 phiên | 0 |
| `sma_ratio_21`, `sma_ratio_63`, `sma_ratio_126` | SMA | $C_t / \text{SMA}_N(t) - 1$ | 21, 63, 126 phiên | 0 |
| `rsi_14` | RSI | RSI chuẩn Wilder | 14 phiên | 0 |
| `macd_hist` | MACD | $\text{EMA}_{12} - \text{EMA}_{26} - \text{EMA}_9(\text{EMA}_{12}-\text{EMA}_{26})$ | 12, 26, 9 | 0 |
| `volume_ratio_21` | Khối lượng | $v_t / \overline{v}_{t-21..t-1} - 1$ (không gồm phiên hiện tại) | 21 phiên | 0 |
| `avg_value_l` | Thanh khoản | Giá trị giao dịch TB 21 phiên (tái dùng 3.4) | 21 phiên | 0 |
| `beta_63` | Chéo mã đơn giản | Beta cuộn so với lợi suất log VN-Index | 63 phiên | 0 |

**Căn cứ chọn tham số:**
- **Độ trễ 0 cho mọi đặc trưng:** dữ liệu OHLCV của phiên $t$ đã là thông tin biết được khi đóng cửa
  phiên $t$ ($\mathcal{F}_t$, charter mục 5 điều 1), giống lý do chọn $\ell = 0$ ở 3.4 (D74).
- **Cửa sổ 21/63/126 cho lợi suất quá khứ và SMA:** tái dùng ĐÚNG ba cửa sổ đã chốt cho họ baseline
  momentum B2 (D59, `config.yaml` mục `model_grid.baseline.B2`), để không phát sinh một bộ cửa sổ mới
  không có căn cứ ("cần và đủ", nguyên tắc D63).
- **RSI(14), MACD(12,26,9):** quy ước kinh điển toàn cầu của hai chỉ báo này (Wilder 1978 cho RSI; Appel
  cho MACD), KHÔNG phải số liệu quy định thị trường VN nên không cần nguồn chính thức như charter mục 5
  điều 7 yêu cầu cho phí, thuế, biên độ (loại số liệu đó khác về bản chất với tham số định nghĩa toán học
  của một chỉ báo kỹ thuật phổ biến toàn cầu).
- **`volume_ratio_21` loại trừ phiên hiện tại khỏi trung bình:** đo mức bất thường của khối lượng hôm nay
  so với mức "bình thường" gần đây; nếu gồm cả phiên hiện tại, một phiên khối lượng đột biến sẽ tự làm
  loãng chính trung bình dùng để so sánh với nó.
- **`avg_value_l` tái dùng `vnrank.universe.build_universe.compute_liquidity_features`:** tránh tính lại
  logic đã có ở 3.4 (cùng $L=21$, $\ell=0$).
- **`beta_63` là đặc trưng chéo mã duy nhất:** dự án chưa có dữ liệu cơ bản (báo cáo tài chính, ngành) để
  làm đặc trưng chéo mã phức tạp hơn (ví dụ theo ngành); beta so với VN-Index là đặc trưng chéo mã đơn
  giản, chuẩn, không cần dữ liệu ngoài OHLCV đã có.
- **Không lấy log cho thanh khoản/khối lượng:** vì bước sau là chuẩn hóa theo THỨ HẠNG (không phải
  z-score), một phép biến đổi đơn điệu như lấy log không làm đổi thứ hạng, nên không cần thiết.

## 2. Phát hiện và sửa lỗi: nguồn Entrade trả trùng ngày (D77)

Khi chạy `compute_features` lần đầu trên dữ liệu thật, phát hiện một vài ngày trong `universe_daily.csv`
(3.4) có tới 162 mã thay vì đúng 150 — dấu hiệu có bản ghi (symbol, date) trùng trong bảng giá. Gọi trực
tiếp API thô (không qua `fetch_ohlcv`) cho mã AAA quanh 27/12/2022 để truy nguyên:

| Timestamp UTC | Giờ VN | Đóng cửa |
|---|---|---|
| 2022-12-27 00:00:00 | 07:00 | 6,19 |
| 2022-12-27 02:00:00 | 09:00 (giờ mở cửa, khớp quy ước mọi ngày khác trong chuỗi) | 6,33 |

Máy chủ trả HAI mốc timestamp cho cùng một ngày lịch VN, trong khi mọi ngày khác trong toàn bộ chuỗi chỉ
có đúng một mốc 02:00 UTC. Kiểm trên toàn bộ 707 mã: đúng **3 ngày** bị lỗi này (27/12/2022, 07/04/2023,
10/04/2023), ảnh hưởng **521/707 mã** (1372 dòng trùng trong bảng giá dài; 6 dòng trùng đã lọt vào
`universe_daily.csv` gốc của 3.4, không bị phát hiện khi đó vì 3.3 chỉ spot-check VNM/HPG/VIC, không rơi
đúng vào 3 ngày này).

**Cách sửa:** `entrade_client.fetch_ohlcv` nay giữ bản ghi có timestamp UTC gốc LỚN HƠN khi nhiều bản ghi
cùng đổi ra một ngày lịch (khớp quy ước 02:00 UTC dùng xuyên suốt phần còn lại của chuỗi). Không suy đoán
giá trị nào "đúng hơn" từ nội dung OHLCV (cả hai bản ghi đều có số liệu hợp lệ, không có nguồn độc lập để
xác minh mức giá thật của phiên đó); chỉ dùng tín hiệu khách quan là mốc thời gian có khớp quy ước nhất
quán của toàn chuỗi hay không.

**Việc đã làm lại:** tải lại toàn bộ cache giá (707 mã + VNINDEX) bằng client đã sửa; dựng lại
`universe_daily.csv` (không đổi tham số D74, chỉ loại dòng trùng — vẫn đúng 150 mã/ngày, 954 ngày, số
quan sát đạt điều kiện lọc đổi rất nhỏ, không đổi kết luận). Chi tiết: `3_4_universe.md` mục "Cập nhật
06/11/2026".

## 3. Chuẩn hóa chéo theo ngày

`cross_sectional_normalize` thực hiện đúng mục 3 của `bai_toan_hinh_thuc.md`: giới hạn vào $U_t$, thay
giá trị thiếu bằng trung vị chéo mã của ngày đó, rồi đổi mỗi đặc trưng thành thứ hạng trong $U_t$ và đưa
về $[-1, 1]$:
$$z_{i,t} = \frac{2(\text{rank}_{i,t} - 1)}{n_t - 1} - 1,$$
rank trung bình khi có giá trị trùng nhau. Trường hợp biên: nếu một ngày không có quan sát hợp lệ nào cho
một đặc trưng (ví dụ ngày VN-Index bị hụt dữ liệu chưa lấp được — xem mục 4), đặc trưng đó được gán 0
(điểm giữa thang) cho cả ngày.

## 4. Lấp khoảng trống lịch VN-Index cho beta

3.3 đã phát hiện VNINDEX qua Entrade thiếu vài phiên có thật (07/04/2023, 12/08/2024). `fill_market_calendar_gaps`
lấp các ngày này bằng giá đóng cửa gần nhất đã biết (forward-fill), không dùng thông tin tương lai, trước
khi tính lợi suất ngày của VN-Index cho đặc trưng `beta_63`.

## 5. Kết quả chạy trên dữ liệu thật

- Universe đầu vào: `code/data/processed/universe_daily.csv` (954 ngày, đúng 150 mã/ngày, đã sửa theo D77).
- Đầu ra: `code/data/processed/features_daily.csv`, **143.100 dòng, 290 mã khác nhau từng xuất hiện trong
  $U_t$** trong suốt mẫu (150 mã/ngày nhưng danh sách xoay vòng theo thanh khoản), **0 giá trị thiếu**
  (đã qua chuẩn hóa chéo). Mỗi đặc trưng có min = -1, max = 1, trung bình ≈ 0 (đúng như kỳ vọng của phép
  chuẩn hóa theo thứ hạng).

## 6. Module và test

`code/vnrank/features/feature_registry.py`: `FEATURE_REGISTRY`, `FEATURE_COLUMNS`, `past_return`,
`sma_ratio`, `rsi`, `macd_hist`, `volume_ratio`, `rolling_beta`, `fill_market_calendar_gaps`,
`compute_features`, `cross_sectional_normalize`. Test ở `code/tests/test_feature_registry.py` (12 test,
không gọi mạng thật), gồm một test bắt buộc theo rủi ro #1 của `giao_thuc_chia_du_lieu_bias.md`: dịch giá
tương lai (sau một mốc $t$) không được làm đổi đặc trưng tại các ngày $\le t$ — đã kiểm chứng PASS. Sửa
lỗi D77 thêm 1 test cho `code/tests/test_entrade_client.py`. Tổng 42 test PASS cục bộ cả dự án.

## 7. Việc không làm ở bước này (ngoài phạm vi 3.5)

- Không tạo nhãn (3.6), không chia train/validation/test (3.7).
- Không huấn luyện mô hình nào; `feature_registry` chỉ sinh đặc trưng đầu vào.
- Không kiểm tra rò rỉ đầy đủ theo lịch walk-forward thật (việc đó thuộc 3.13); test ở bước này chỉ kiểm
  tính chất "không nhìn trước" ở mức đơn vị (unit test), đúng phạm vi rủi ro #1 giao cho 3.5.
