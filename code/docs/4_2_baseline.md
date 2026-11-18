# 4.2 — Baseline không dùng ML

08/11/2026.

## 1. Mục đích

Tính điểm và đánh giá các baseline không học đã chốt ở 2.3 (B0-B4, `ma_tran_baseline_ablation.md`
mục 2, D59): B0 (đồng trọng số), B1 (VN30), B2 (momentum nhiều cửa sổ), B3 (IMOM), B4 (Amihud
illiquidity). Dùng `code/vnrank/evaluation/harness.py` (4.1) để đánh giá Rank IC/ICIR, Newey-West,
bootstrap khối tròn trên TEST, sau khi chọn cấu hình lưới trên VALIDATION.

## 2. Ba điểm công thức chưa đủ rõ trong hồ sơ — đã hỏi và chốt khi giao việc 4.2

Đặc tả gốc (`ma_tran_baseline_ablation.md`) không ghi công thức số học đầy đủ cho B2 (khi
`skip > 0`) và B3 (IMOM). Trước khi viết code, đã hỏi và người dùng chốt:

1. **B2, công thức momentum có `skip`:** $s(w, \text{skip}) = \ln(C_{t-\text{skip}}/C_{t-w})$,
   quy ước kinh điển bỏ qua `skip` phiên gần nhất (Jegadeesh-Titman). `skip=0` cho đúng kết quả
   như `ret_w` thô ở `feature_registry.py` (trước chuẩn hóa chéo).
2. **Ô suy biến của lưới B2:** lưới `window=[21,63,126] x skip=[0,21]` của `config.yaml` có ô
   `(21, 21)` suy biến (tử bằng mẫu, luôn bằng 0). Đã chốt: BỎ ô này, lưới thực chỉ có 5 ô:
   `(21,0), (63,0), (63,21), (126,0), (126,21)`.
3. **B3 (IMOM), beta dùng để tính phần dư:** `config.yaml` mục `model_grid.baseline.B3` ghi
   `estimation_window: 126`, khác với `beta_63` đã có sẵn ở `feature_registry.py` (window 63).
   Đã chốt: tính beta RIÊNG cho B3 với `estimation_window=126` (tái dùng hàm `rolling_beta` đã có,
   chỉ đổi `window`), không dùng chung `beta_63`.
4. **B3, cách cộng dồn phần dư:** tổng phần dư log-return ngày $(r_{i,d} - \beta_t \cdot r_{m,d})$
   trên $[t-\text{window}, t-\text{skip}]$, dùng MỘT $\beta_t$ (beta cuộn tại ngày $t$) cho toàn bộ
   tổng. Vì log-return cộng dồn theo đường thẳng (telescoping), tổng này bằng ĐÚNG (không xấp xỉ):
   $$\text{IMOM}_t = \ln(C_{t-\text{skip}}/C_{t-w}) - \beta_t \cdot \ln(\text{Mkt}_{t-\text{skip}}/\text{Mkt}_{t-w})$$
   nên code chỉ cần gọi lại `momentum_score` cho cả mã và thị trường, không cần vòng lặp cộng dồn
   từng ngày. Đã kiểm bằng test (`test_imom_score_near_zero_when_stock_is_pure_beta_times_market`):
   khi lợi suất mã đúng bằng $\beta \times$ lợi suất thị trường (không có phần riêng), IMOM → 0.

Amihud (B4) không hỏi: dùng công thức kinh điển toàn cầu Amihud (2002), cùng cách RSI/MACD ở
`feature_registry.py` được coi là quy ước kinh điển không cần quyết định riêng.

## 3. B1 (VN30): có dữ liệu thật, không cần proxy

Trước khi viết code đã thử gọi `fetch_ohlcv("VN30", ..., symbol_type="index")` — API Entrade/DNSE
ĐÃ CÓ dữ liệu chỉ số VN30 thật (không chỉ danh sách thành viên), cùng cách đã dùng cho VN-Index.
Vì vậy B1 dùng lợi suất chỉ số thật, cache ở `code/data/raw/prices/VN30.csv`, không cần tự dựng
proxy từ các mã thành viên.

## 4. Thiết kế

`code/vnrank/models/baselines.py`:

- `momentum_score(close, window, skip)` (B2): công thức mục 2 điểm 1.
- `imom_score(...)` (B3): công thức mục 2 điểm 4, tái dùng `rolling_beta` (D59, `estimation_window=126`).
- `amihud_illiquidity(close, volume, window)` (B4): $\text{ILLIQ} = \text{mean}(|r_d|/\text{giá trị giao dịch}_d)$
  trên `window` phiên; điểm CAO hơn = kém thanh khoản hơn (đúng chiều đặc tả).
- `equal_weight_portfolio_returns(labels, universe, h)` (B0): trung bình cột `r` (lợi suất thô,
  `exit_price/entry_price - 1`, đúng quy ước `labeling.py`) trên toàn $U_t$ mỗi ngày.
- `load_vn30_prices` / `vn30_index_returns(h)` (B1): tải/cache VN30, lợi suất cùng quy ước vào
  (đóng cửa $t+1$) / thoát (đóng cửa $t+1+h$) của hệ thống (`config.yaml` mục `execution`).
- `select_best_config`: chọn ô lưới có Rank IC trung bình CAO NHẤT trên VALIDATION — không bao giờ
  chọn trên test. Ghi lại Rank IC validation của MỌI ô (kể cả ô không chọn) vào
  `baseline_grid_search_log_4_2.csv`, đúng nguyên tắc 4 của `ma_tran_baseline_ablation.md` mục 1
  ("mọi biến thể, kể cả biến thể bị bỏ, ghi vào log thử nghiệm", dùng cho DSR/PBO sau này).
- `evaluate_on_test`: Rank IC/ICIR, Newey-West (`lag = h-1`), bootstrap khối tròn
  (`block_len = 2h`, `n_boot` đọc từ `config.yaml` mục `hypotheses.bootstrap_n`) — chỉ chạy MỘT
  LẦN trên test, sau khi đã chọn cấu hình trên validation.

**Bổ sung `config.yaml`:** thêm `hypotheses.bootstrap_n: 10000` và
`hypotheses.bootstrap_block_len_factor_h: 2` — đây là hằng số ĐÃ CHỐT ở D60
(`giao_thuc_danh_gia.md` mục 3: "độ dài khối $b=2h$, $B=10.000$ lần") nhưng trước đó chưa được
chép vào `config.yaml`; bổ sung để `evaluate_on_test` không hardcode mà đọc từ config, đúng quy
ước của dự án.

Rank IC dùng cột `r` (lợi suất thô theo $h$, KHÔNG phải `r_excess`), đúng định nghĩa ở
`giao_thuc_danh_gia.md` mục 2 ("Rank IC$_t$ = Spearman giữa điểm $s_{i,t}$ và $r^{(h)}_{i,t}$").

## 5. Test

`code/tests/test_baselines.py` (10 test, dữ liệu tổng hợp nhỏ): công thức momentum (`skip=0` khớp
`ret_w` thô, `skip>0` khớp tính tay); IMOM gần 0 khi mã thuần túy $\beta \times$ thị trường (không
phần riêng), dương khi có phần riêng dương ngoài vùng `skip`; Amihud khớp tính tay, cao hơn khi
giá trị giao dịch thấp hơn; B0 lấy đúng trung bình theo $U_t$ từng ngày, đúng $h$; B1 lấy đúng
entry $t+1$/exit $t+1+h$, NaN ở cuối chuỗi. Toàn bộ suite dự án: 122 PASS (từ 112, thêm 10 test
baseline).

## 6. Kết quả trên dữ liệu thật (test: 2025-04 đến 2026-10-02, 374-375 ngày quyết định)

Cấu hình chọn trên validation và kết quả test (`code/data/processed/baseline_results_4_2.csv`,
đầy đủ ở `baseline_grid_search_log_4_2.csv`):

| Baseline | $h$ | Chọn trên validation | Rank IC validation | Rank IC test (TB) | ICIR test | Newey-West $p$ | Cận dưới bootstrap 95% |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| B2 momentum | 5 | (21, 0) | 0,0305 | -0,0205 | -0,099 | 0,852 | -0,057 |
| B2 momentum | 10 | (21, 0) | 0,0434 | -0,0287 | -0,139 | 0,872 | -0,073 |
| B2 momentum | 21 | (21, 0) | 0,0218 | -0,0236 | -0,124 | 0,811 | -0,059 |
| B3 IMOM | 5 | (126, 21) | -0,0164 | 0,0083 | 0,047 | 0,318 | -0,026 |
| B3 IMOM | 10 | (126, 21) | -0,0265 | 0,0053 | 0,028 | 0,417 | -0,039 |
| B3 IMOM | 21 | (126, 21) | -0,0168 | 0,0004 | 0,002 | 0,495 | -0,047 |
| B4 Amihud | 5 | 21 | -0,0266 | -0,0393 | -0,270 | 0,997 | -0,066 |
| B4 Amihud | 10 | 21 | -0,0520 | -0,0488 | -0,302 | 0,987 | -0,090 |
| B4 Amihud | 21 | 21 | -0,0685 | -0,0534 | -0,339 | 0,965 | -0,111 |

Mô tả (không kiểm định, chỉ tham chiếu), trên test:

| Baseline | $h=5$ | $h=10$ | $h=21$ |
| :--- | ---: | ---: | ---: |
| B0 đồng trọng số, lợi suất TB/kỳ | 0,0015 | 0,0033 | 0,0077 |
| B0, Sharpe năm hóa ($\sqrt{250/h}$) | 0,37 | 0,46 | 0,48 |
| B1 VN30, lợi suất TB/kỳ | 0,0059 | 0,0110 | 0,0227 |
| B1 VN30, Sharpe năm hóa | 1,42 | 1,32 | 1,24 |

**Nhận xét (mô tả, không phải kết luận H1/H1b — đó là việc của 2.4/4.9):**
- Không baseline tín hiệu nào (B2, B3, B4) có Rank IC dương, có ý nghĩa thống kê trên test, dù
  được chọn tốt nhất trên validation; B2 (momentum ngắn hạn) còn ĐẢO CHIỀU từ dương (validation)
  sang âm (test) ở cả ba $h$ — một dấu hiệu bất ổn định cần lưu ý khi so sánh với L1/R2 ở 4.3-4.4.
  B4 (Amihud) có bằng chứng khá rõ theo chiều ngược (illiquidity có Rank IC âm, càng rõ ở $h$ dài
  hơn) nhưng vẫn không vượt ngưỡng ý nghĩa một phía ở $\alpha=0,05$ cho giả thuyết IC>0.
- B1 (VN30) có lợi suất và Sharpe mô tả cao hơn B0 (đồng trọng số toàn $U_t$) trong giai đoạn test
  — giai đoạn thị trường tăng mạnh ở nhóm vốn hóa lớn; đây là thông tin mô tả, KHÔNG dùng để kết
  luận gì về baseline tín hiệu B2-B4 (khác universe, D56).
- Các con số trên chỉ có giá trị tham chiếu cho bước này; kết luận chính thức về baseline mạnh
  nhất dùng cho H1b (so với L1 ở 4.4) để lại cho 4.6-4.9 sau khi có kết quả mô hình học.

## 7. Phạm vi không làm trong bước này

- Không chạy baseline trên các tập con (VN30, HOSE, HNX riêng) — để lại cho báo cáo D56 ở 4.x.
- Không tính Sharpe sau chi phí hay danh mục thực (turnover, MDD) cho B0-B4 — baseline ở đây chỉ
  đánh giá TÍN HIỆU (Rank IC) và lợi suất mô tả thô; danh mục thực chốt ở 4.7 (test một lần sau khi
  đóng băng cấu hình).
- Không so sánh B2-B4 với R1-R3/L1 (đó là 4.3, 4.4); bước này chỉ dựng xong các mốc baseline.
