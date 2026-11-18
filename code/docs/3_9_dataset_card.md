# Data card: bộ dữ liệu universe/feature/label/split (bước 3.9)

**Phiên bản dataset:** `v20261107-ce5d2e5c` (ngày tạo 07/11/2026; mã băm 8 ký tự = SHA256 của
`universe_daily.csv` + `features_daily.csv` + `labels_daily.csv` + `splits_daily.csv` nối tiếp nhau,
theo đúng thứ tự đó — xem `data/processed/MANIFEST.md` để đối chiếu và tái lập). **Trạng thái:** Xong
(D81). **Căn cứ:** `code/docs/3_4_universe.md`, `3_5_feature_registry.md`, `3_6_nhan.md`,
`3_7_chia_du_lieu.md`, `3_8_kiem_tra_leakage.md`.

## 1. Nguồn dữ liệu gốc

- **Giá OHLCV:** API công khai Entrade/DNSE, qua `vnrank/data/entrade_client.py` (3.2, D71). 707 mã
  HOSE+HNX (sau khi loại 24 mã quỹ ETF) + VNINDEX, 01/2022-03/11/2026. Cache cục bộ ở
  `code/data/raw/prices/` (không lên GitHub).
- **Danh sách mã niêm yết:** `finance.vietstock.vn/truy-xuat-du-lieu/tong-hop-doanh-nghiep.htm`, do
  người dùng tải 04/11/2026 (3.4, D74).
- **Thành viên VN30 theo kỳ:** 8 kỳ 01/2023-07/2026, do người dùng tìm và cung cấp 04/11/2026, kiểm
  chứng độc lập 4/8 kỳ qua WebSearch (3.4, D74). Không dùng trực tiếp trong 4 bảng dưới đây (chỉ dùng khi
  báo cáo kết quả theo lát cắt VN30 ở Giai đoạn 4).
- Cả giá thô lẫn danh sách mã/VN30 đều **không đi kèm theo dataset này**; dataset chỉ gồm bốn bảng đã qua
  xử lý ở mục 2.

## 2. Bốn bảng, cột và kiểu dữ liệu

### 2.1. `universe_daily.csv` (954 ngày × 150 mã = 143.100 dòng, từ 3.4, D74)

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `date` | ngày (YYYY-MM-DD) | Ngày quyết định $t$ |
| `symbol` | chuỗi | Mã cổ phiếu thuộc $U_t$ của ngày đó |

Đúng 150 mã mỗi ngày (cận trên của khoảng 50-150, D74), 01/2023-11/2026. Tập hợp 290 mã khác nhau từng
xuất hiện trong $U_t$ tại một thời điểm nào đó (danh sách xoay vòng theo thanh khoản).

### 2.2. `features_daily.csv` (143.100 dòng, 11 cột đặc trưng, từ 3.5, D76/D77)

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `date`, `symbol` | — | Khóa, trùng với `universe_daily.csv` |
| `ret_21`, `ret_63`, `ret_126` | số thực $\in [-1,1]$ | Lợi suất quá khứ, đã chuẩn hóa chéo theo ngày |
| `sma_ratio_21`, `sma_ratio_63`, `sma_ratio_126` | số thực $\in [-1,1]$ | Tỷ lệ giá so với SMA, đã chuẩn hóa chéo |
| `rsi_14` | số thực $\in [-1,1]$ | RSI Wilder 14 phiên, đã chuẩn hóa chéo |
| `macd_hist` | số thực $\in [-1,1]$ | Histogram MACD(12,26,9), đã chuẩn hóa chéo |
| `volume_ratio_21` | số thực $\in [-1,1]$ | Khối lượng so với 21 phiên liền trước, đã chuẩn hóa chéo |
| `avg_value_l` | số thực $\in [-1,1]$ | Thanh khoản (giá trị giao dịch TB 21 phiên), đã chuẩn hóa chéo |
| `beta_63` | số thực $\in [-1,1]$ | Beta cuộn 63 phiên so với VN-Index, đã chuẩn hóa chéo |

Mọi cột đặc trưng đã qua chuẩn hóa thứ hạng chéo ngày về $[-1,1]$ (3.5 mục 3); **không còn giá trị thiếu**
(trung vị chéo mã đã thay thế trước khi chuẩn hóa).

### 2.3. `labels_daily.csv` (423.934 dòng = 143.100 × 3 horizon trừ phần không tạo được nhãn, từ 3.6/3.10, D78)

**Lưu ý đối chiếu số dòng (D82):** 423.934 là tổng số dòng của bảng (mỗi $(i,t)$ có tối đa 3 dòng, một cho
mỗi $h$; 1.108 $(i,t)$ không vào lệnh được nên chỉ có 1 dòng toàn NaN, không phải 3). Trong số đó,
422.826 dòng có `exit_date_resolved` khác rỗng (nhãn hồi quy tính được) — con số này xuất hiện ở
`3_7_chia_du_lieu.md` khi tính tỷ lệ dời ngày thoát (423/422.826 = 0,10%). Hai số không mâu thuẫn, chỉ
khác mẫu số (tổng dòng so với dòng có `r` hợp lệ).

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `date`, `symbol` | — | Khóa, trùng `universe_daily.csv` |
| `h` | 5/10/21 | Horizon (ngày/phiên) |
| `r` | số thực hoặc NaN | Lợi suất thực hiện $C_{exit}/C_{t+1} - 1$ (đã áp quy tắc khóa trần/sàn, D66) |
| `exit_date_resolved` | ngày hoặc NaN | Ngày thoát lệnh thật (có thể trễ hơn $t+1+h$ nếu khóa sàn) |
| `exit_extended_khoa_san` | bool | True nếu ngày thoát bị dời do khóa sàn/tạm ngừng |
| `y` | −1/0/+1 hoặc NaN | Nhãn xu hướng triple-barrier (NaN nếu chưa đủ 20 phiên làm nóng EWMA) |
| `t1` | ngày hoặc NaN | Thời điểm chạm rào hoặc $t+1+h$ nếu không chạm (purge nhãn xu hướng) |
| `b` | số thực hoặc NaN | Giá trị rào đã dùng cho `y` |
| `san_chi_phi` | bool hoặc NaN | True nếu rào bị sàn chi phí 0,60% chặn |
| `khoa_tran_san` | bool hoặc NaN | True nếu có phiên khóa trần/sàn trong cửa sổ xét nhãn xu hướng |
| `tam_ngung` | bool hoặc NaN | True nếu phiên thoát dự kiến của nhãn xu hướng không có giao dịch thật |
| `exchange` | "HOSE"/"HNX" | Sàn giao dịch hiện hành của mã (KHÔNG theo dõi lịch sử chuyển sàn) |
| `r_excess` | số thực hoặc NaN | $r$ trừ trung bình chéo $U_t$ cùng ngày, cùng $h$ |
| `relevance` | 0-4 hoặc NaN | Mức liên quan cho ranker, phân vị $Q=5$ theo `r_excess` |

NaN ở các cột trên xảy ra khi: không vào lệnh được (khóa trần/tạm ngừng ở $t+1$, 0,77% dòng $(i,t)$,
không có dòng nào cho bất kỳ $h$), hoặc chưa đủ 20 phiên lịch sử để tính $\hat\sigma$ (chỉ ảnh hưởng `y`,
`t1`, `b`, `san_chi_phi`, không ảnh hưởng `r`).

`labels_raw.csv` (988.995 dòng) là bản TRUNG GIAN trước khi giới hạn vào $U_t$ và tính `r_excess`/
`relevance` — gồm nhãn cho mọi mã từng xuất hiện trong universe tại một thời điểm nào đó, kể cả ngày mã
đó không còn trong $U_t$. Không dùng trực tiếp để huấn luyện; chỉ giữ để truy vết/kiểm tra.

### 2.4. `splits_daily.csv` (954 dòng, từ 3.7, D79)

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `date` | ngày | Ngày quyết định |
| `split` | "train"/"validation"/"test"/rỗng | Nhãn chia dữ liệu |

Train: 369 ngày (01/2023-06/2024). Validation: 188 ngày (07/2024-03/2025). Test: 375 ngày
(04/2025-02/10/2026). 22 ngày cuối universe (03/10-03/11/2026) không có nhãn `split` vì thiếu dữ liệu
tương lai để tính nhãn $h=21$ đủ (`test_end` = 02/10/2026, xác định ở 3.7).

**Lưu ý purge:** bảng này CHƯA purge theo $t^1$/`exit_date_resolved`. Khi dùng cho huấn luyện, phải tự áp
`vnrank.splits.split_protocol.purge_by_realization` (xem `3_7_chia_du_lieu.md` mục 3-4) — ghép `split`
vào đây chỉ là gán nhãn theo mốc ngày, không tự động loại các dòng chưa hiện thực nhãn.

## 3. Giới hạn đã biết (ghi vào phần "Giới hạn" của luận văn)

- **Survivorship bias (D71):** universe chỉ gồm mã đang niêm yết tại thời điểm thu thập (11/2026), không
  giữ được mã đã hủy niêm yết trong giai đoạn mẫu. Ước lượng ở 3.4: khoảng 45-70 mã HOSE+HNX hủy niêm yết
  trong giai đoạn mẫu, trong đó khoảng 3-8 mã (~2-5% của $N=150$) có thể đã lọt vào $U_t$ nếu còn dữ liệu.
- **Không có giá thô:** chỉ có giá đã điều chỉnh (D71/3.3); khóa trần/sàn phát hiện qua tính chất OHLC suy
  biến (`detect_limit_lock`, D73), không suy ngược mức giá tuyệt đối.
- **`exchange` tĩnh:** lấy theo danh sách niêm yết hiện hành, không theo dõi lịch sử mã chuyển sàn (hiếm
  gặp, chưa xử lý, 3.6).
- **Đã sửa một lỗi dữ liệu nguồn (D77):** API Entrade/DNSE từng trả trùng hai phiên cho cùng một ngày lịch
  VN ở đúng 3 ngày (27/12/2022, 07/04/2023, 10/04/2023); đã khử trùng tại nguồn trước khi tạo bốn bảng
  này — dataset ở phiên bản này KHÔNG còn lỗi đó.
- **Thống kê mô tả trong các báo cáo 3.4-3.8 tính trên TOÀN mẫu** (gồm cả đoạn sẽ là test), chỉ để kiểm
  chất lượng dữ liệu (đúng rủi ro #12 của giao thức bias), KHÔNG dùng để chọn hay điều chỉnh tham số nào.
  Báo cáo mô tả riêng trên TRAIN (yêu cầu `dinh_nghia_xu_huong.md` mục 7) để ở 3.13.
- **Chưa kiểm tra rò rỉ theo đúng lịch walk-forward thật** (nhiều lần refit, cửa sổ mở rộng) — 3.8 (D80)
  mới kiểm ở mức purge tĩnh theo mốc train/validation, không mô phỏng toàn bộ 6 lần refit trong test.

## 4. Tái lập dataset này

1. Tải lại cache giá: `python -m vnrank.universe.build_universe` (707 mã) + tải riêng VNINDEX (xem
   `3_4_universe.md` mục 1.1). Cần hai file đầu vào ở `code/data/raw/universe_inputs/` (không lên GitHub,
   xem `3_4_universe.md` mục 0).
2. Dựng universe: xem lệnh cụ thể ở `3_4_universe.md` mục 1.3 (tham số D74: $N=150$, $L=21$, $\ell=0$,
   $p^{\min}=5,0$, $v^{\min}=100.000$, $n^{\min}=15$).
3. Tính feature: `python -m vnrank.features.feature_registry`.
4. Tính nhãn: gọi `vnrank.labels.labeling.compute_raw_labels` rồi `add_excess_and_relevance` (xem
   `3_6_nhan.md` mục 7, chưa có lệnh dòng lệnh sẵn, gọi trực tiếp hàm Python).
5. Gán split: `vnrank.splits.split_protocol.assign_split` cho từng ngày trong `universe_daily.csv`
   (xem `3_7_chia_du_lieu.md` mục 2).

**Lưu ý tái lập:** cache giá (`code/data/raw/prices/`) không lên GitHub; tái lập ở máy khác hoặc thời
điểm khác sẽ gọi lại API Entrade/DNSE, có thể cho dữ liệu khác đi nếu nguồn tự sửa số liệu lịch sử (đã
từng xảy ra, D77) hoặc nếu chạy sau ngày 03/11/2026 (thêm phiên mới, có thể đẩy `test_end` xa hơn). Hai
file đầu vào ở `universe_inputs/` (danh sách mã, VN30) cũng cần giữ nguyên bản đã dùng (04/11/2026) để
tái lập đúng, không dùng bản tải lại sau này.

## 5. Module nguồn (không lặp lại công thức, xem từng báo cáo bước)

`vnrank/universe/build_universe.py` (3.4) → `vnrank/features/feature_registry.py` (3.5) →
`vnrank/labels/labeling.py` (3.6/3.10) → `vnrank/splits/split_protocol.py` (3.7). Kiểm tra rò rỉ:
`vnrank/evaluation/leakage_checks.py` (3.8).
