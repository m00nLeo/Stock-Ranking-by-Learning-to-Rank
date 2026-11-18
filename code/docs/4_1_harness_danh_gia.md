# 4.1 — Harness đánh giá

08/11/2026.

## 1. Mục đích

Bước đầu tiên của Giai đoạn 4: dựng các hàm lõi dùng chung cho mọi phép đánh giá và kiểm định ở
`giao_thuc_danh_gia.md` (BẢN CHỐT, D60), rồi kiểm tra bằng tín hiệu tổng hợp đã biết trước đáp án
(tín hiệu ngẫu nhiên phải KHÔNG bị bác bỏ, tín hiệu momentum có tương quan thật PHẢI bị bác bỏ).

## 2. Thiết kế và phân công

`code/vnrank/evaluation/harness.py` (Trường hợp 1) và `code/vnrank/evaluation/multiple_testing.py`
(Trường hợp 3, Cline, model `deepseek-v4-pro-0813` — đặc tả đóng, không có quyết định thiết kế mở).
Chỉ `numpy`/`pandas` (harness.py) và thuần Python (`multiple_testing.py`), không phụ thuộc
`scipy`/`statsmodels` (cùng cách 3.8/D80, 3.13/D83).

- **Mục 1, 2 (Rank IC, ICIR, precision@K):** `rank_ic_series`, `icir`, `icir_nonoverlapping`,
  `precision_at_k_by_day`. Rank IC tính riêng theo từng ngày quyết định (tương quan hạng Spearman),
  bỏ qua ngày dưới 3 quan sát hoặc một cột không đổi giá trị. `icir_nonoverlapping` lấy mẫu cách `h`
  phiên để giảm tự tương quan do nhãn chồng lấn.
- **Mục 3 (Newey-West, bootstrap khối tròn):** `newey_west_se`, `newey_west_test`,
  `circular_block_bootstrap_mean_ci`. Giao thức ghi "quy tắc chọn độ trễ tự động cần chép từ bản gốc"
  (chưa có trong hồ sơ) nên `newey_west_test` KHÔNG tự chọn `lag`; người gọi bắt buộc truyền (sàn tối
  thiểu `h-1` theo mục 3). `circular_block_bootstrap_mean_ci` bắt buộc truyền `n_boot` và `block_len`,
  không hardcode số lần lặp trong hàm; `seed` mặc định đọc từ `config.yaml` nếu không truyền.
- **Mục 2 (macro-F1, balanced accuracy, ma trận nhầm lẫn):** `balanced_accuracy`, `confusion_matrix`.
- **Mục 4 (hiệu chỉnh Holm):** `holm_step_down` ở `multiple_testing.py`. Chuẩn Holm step-down: sắp
  p tăng dần, bác bỏ $H_{(j)}$ với mọi $j \le i$ nếu $p_{(j)} \le \alpha/(m-j+1)$, dừng ngay khi gặp
  giả thuyết không vượt ngưỡng.
- **Mục 6 (Brier đa lớp, BSS, ECE, chỉ mô tả):** `brier_multiclass`, `brier_skill_score`,
  `expected_calibration_error` (10 nhóm theo phân vị, không theo khoảng đều).
- **Mục 7 (DSR, PBO):** KHÔNG làm ở bước này. Giao thức ghi rõ công thức DSR và thủ tục CSCV của PBO
  "cần chép từ bản gốc", hồ sơ hiện chưa có — để lại cho khi có nguồn, không tự suy diễn công thức phổ
  biến thay thế (nguyên tắc đã áp dụng nhất quán từ 3.1).

## 3. Rà soát và sửa phần của Trường hợp 3 (Cline)

Cline giao `multiple_testing.py` và `test_multiple_testing.py` (4 test), kèm báo cáo đã dừng lại hỏi
khi phát hiện ví dụ trong đặc tả tính nhầm (công thức Holm chuẩn buộc dừng ở $p=0,03$ trong ví dụ
$[0,01;\ 0,04;\ 0,03;\ 0,20]$, trong khi đặc tả ghi $0,03$ "đạt"; người dùng xác nhận công thức chuẩn
đúng, ví dụ trong đặc tả sai, expected đúng là `[True, False, False, False]`) — đúng vai trò "không tự
đoán, dừng lại hỏi" quy định cho Trường hợp 3.

Rà soát theo ba quy tắc bắt buộc:
- **Point-in-time:** không áp dụng — hàm thuần túy thống kê trên danh sách p-giá trị, không có trục
  thời gian, không rủi ro rò rỉ.
- **Không chọn tham số trên test:** không vi phạm — $\alpha = 0,05$ là giá trị đã chốt theo giao thức
  (D54, D60; `config.yaml` mục `hypotheses.alpha`), không suy ra từ kết quả kiểm định.
- **Không hardcode hằng số — VI PHẠM, ĐÃ SỬA.** Bản Cline giao có `alpha: float = 0.05` làm giá trị
  mặc định trong chữ ký hàm, trùng với `hypotheses.alpha: 0.05` đã chốt trong `config.yaml`. Theo đúng
  quy ước đã áp dụng cho `newey_west_test` (bắt buộc `lag`) và `circular_block_bootstrap_mean_ci`
  (bắt buộc `n_boot`) ở `harness.py`: hằng số đã chốt theo giao thức không được mặc định trong hàm, để
  tránh lệch âm thầm nếu `config.yaml` đổi giá trị sau này mà hàm không được gọi lại với giá trị mới.
  **Sửa:** bỏ giá trị mặc định, `alpha` thành tham số bắt buộc; cập nhật hai lời gọi trong
  `test_multiple_testing.py` (trường hợp $m=1$) để truyền `alpha=0.05` tường minh. Thuật toán Holm
  step-down của Cline, sau khi sửa ví dụ, đúng chuẩn.

## 4. Test

`code/tests/test_harness.py` (20 test, dữ liệu tổng hợp tự dựng — lợi suất $= 0,5 \times$ chất lượng
$+$ nhiễu; tín hiệu momentum tương quan thật với chất lượng; tín hiệu ngẫu nhiên độc lập) và
`code/tests/test_multiple_testing.py` (4 test, đã sửa theo mục 3). Test tích hợp quan trọng nhất:
`test_holm_does_not_reject_random_signals_but_rejects_momentum` — mô phỏng 4 "giả thuyết" H1-H4 cùng
kiểm định Newey-West trên tín hiệu ngẫu nhiên và trên tín hiệu momentum (4 seed khác nhau), xác nhận
Holm không bác bỏ giả thuyết nào ở tín hiệu ngẫu nhiên (`[False, False, False, False]`) nhưng bác bỏ
hết ở tín hiệu momentum (`[True, True, True, True]`).

Toàn bộ suite dự án: 112 PASS (từ 88 ở 3.12, thêm 24 test mới: 20 ở `test_harness.py`, 4 ở
`test_multiple_testing.py`).

## 5. Phạm vi không làm trong bước này

- DSR, PBO (mục 7 giao thức) — để lại chờ nguồn công thức gốc, xem mục 2.
- Chạy harness trên dữ liệu thật của dự án (universe, nhãn, điểm mô hình) — bước này chỉ dựng và kiểm
  tra hạ tầng bằng dữ liệu tổng hợp đã biết trước đáp án; áp dụng trên dữ liệu thật và các mô hình cụ
  thể (baseline, ranker, T2) là việc của 4.2 trở đi.
- Không tự chọn độ trễ Newey-West tự động hay số lần lặp bootstrap mặc định — người gọi (bước 4.2 trở
  đi) phải truyền tường minh.
