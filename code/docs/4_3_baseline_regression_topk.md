# 4.3 — Baseline hồi quy rồi lấy Top-K

12/11/2026.

## 1. Mục đích

Tính điểm và đánh giá ba baseline hồi quy đã chốt ở 2.3 (R1-R3, `ma_tran_baseline_ablation.md` mục
2, D59): R1 (Ridge), R2 (LightGBM hồi quy, **ĐỐI CHỨNG CHÍNH của H1**), R3 (XGBoost hồi quy, kiểm
tra R2 không phải do riêng thư viện LightGBM). Cả ba hồi quy trên `r_excess` (lợi suất vượt trung
bình chéo, 3.6), dùng đúng `feature_registry` (11 đặc trưng, 3.5), cùng universe, cùng walk-forward
với B0-B4 (4.2) và L1 (4.4, chưa làm).

## 2. Môi trường — từ lệch phiên bản (D95) đến khớp đúng bản Colab (D96)

Trước khi viết code, đã thử `import sklearn, lightgbm, xgboost` cục bộ (quy ước D31: dự án chạy
thật trên Google Colab). Cả ba thư viện CÀI ĐƯỢC và CHẠY ĐƯỢC cục bộ (Windows, Python 3.14), nhưng
chỉ ở bản MỚI HƠN bản đã chốt trong `requirements.txt` (D67):

| Thư viện | Bản đã chốt cho Colab (`requirements.txt`, D67) | Bản cài được trên Python 3.14 |
| :--- | ---: | ---: |
| scikit-learn | 1.5.2 | 1.9.1 |
| lightgbm | 4.5.0 | 4.7.0 |
| xgboost | 2.1.1 | 3.4.1 |

**D95 (ban đầu):** đã hỏi người dùng và người dùng chọn chạy thật cục bộ bằng bản mới (ghi nhận
lệch), kèm khuyến nghị đối chiếu lại trên Colab khi có dịp.

**D96 (sửa lại ngay sau, cùng phiên):** người dùng hỏi thêm liệu Colab có hỗ trợ đầy đủ bản đã
chốt hay không, và việc thử cài cục bộ có lợi ích gì so với chạy thẳng trên Colab. Tra cứu
(WebSearch, tài liệu Colab Enterprise) cho thấy Colab chạy Python khoảng 3.12 (không phải 3.14),
tức bản Python mới nhất gây lỗi ở trên không phải là vấn đề của riêng Colab — ba thư viện pin ở
D67 (từ khoảng 2024) gần như chắc chắn có wheel cho Python 3.11/3.12. Quyết định: hạ xuống đúng
bản Colab thay vì chạy thật trên Colab thật — cài thêm Python 3.12.10 cục bộ (qua `winget install
Python.Python.3.12`, không đụng tới Python 3.14 mặc định của máy), dựng venv riêng `code/.venv`
(Python 3.12.10), cài ĐÚNG `requirements.txt` không sửa gì (scikit-learn 1.5.2, lightgbm 4.5.0,
xgboost 2.1.1 — cài sạch, không lỗi phụ thuộc), rồi CHẠY LẠI toàn bộ 4.3 trong venv này. Đây KHÔNG
phải chạy trên Colab thật (vẫn chưa làm), nhưng dùng đúng cặp (Python, thư viện) mà Colab nhiều khả
năng đang dùng, gần với Colab hơn nhiều so với D95, và giữ được tốc độ lặp cục bộ.

**Đối chiếu hai lần chạy** (D95 bản mới hơn so với D96 bản đã chốt đúng, cùng dữ liệu, cùng seed):

| Mô hình | $h$ | D95 (sklearn 1.9.1/lightgbm 4.7.0/xgboost 3.4.1) | D96 (sklearn 1.5.2/lightgbm 4.5.0/xgboost 2.1.1) |
| :--- | ---: | :--- | :--- |
| R1 Ridge | 5, 10, 21 | Rank IC test giống hệt D96 | Ridge không nhạy phiên bản |
| R2 LightGBM | 5, 10, 21 | Rank IC test giống hệt D96 | cùng cấu hình chọn, cùng số liệu |
| R3 XGBoost | 5 | max_depth=5 chọn, test -0,0135 | max_depth=7 chọn, test -0,0042 |
| R3 XGBoost | 10 | max_depth=7 chọn, test -0,0287 | max_depth=7 chọn, test -0,0178 |
| R3 XGBoost | 21 | max_depth=3 chọn, test -0,0408 | max_depth=7 chọn, test -0,0348 |

R1 (Ridge) và R2 (LightGBM) cho kết quả GIỐNG HỆT giữa hai bản thư viện. R3 (XGBoost) chọn
`max_depth` khác nhau trên validation và có Rank IC test khác nhau về độ lớn giữa xgboost 2.1.1 và
3.4.1 (khác biệt hành vi mặc định giữa major version, không điều tra sâu hơn, ngoài phạm vi bước
này), nhưng KẾT LUẬN ĐỊNH TÍNH KHÔNG ĐỔI: Rank IC test vẫn ÂM ở cả ba h, không cấu hình nào đạt ý
nghĩa thống kê một phía. Số liệu ở mục 5 dưới đây là của D96 (bản đã chốt D67) — số liệu CHÍNH
THỨC, thay cho bản D95. `code/requirements.lock.txt` ghi lại đúng bộ phụ thuộc đã cài (từ
`code/.venv`, Python 3.12.10, KHÔNG phải chạy trên Colab thật — vẫn khuyến nghị xác nhận lại trên
Colab thật khi có dịp).

## 3. Thiết kế

`code/vnrank/models/regression_topk.py` (mới), walk-forward hai giai đoạn, tái dùng tối đa hạ tầng
đã có (`split_protocol.py` 3.7, `baselines.py` 4.2, `harness.py` 4.1):

1. **Chọn cấu hình trên VALIDATION** (`select_config_on_validation`): fit MỘT LẦN trên cửa sổ TRAIN
   (purge theo `exit_date_resolved`, đúng quy ước nhãn hồi quy/mức liên quan của `split_protocol.py`
   mục purge — khác nhãn xu hướng dùng `t1`), dự báo điểm cho mọi ngày VALIDATION (không refit trong
   validation), chọn cấu hình Rank IC trung bình validation CAO NHẤT (`select_best_config`, tái dùng
   nguyên hàm của `baselines.py`, không viết lại). Không bao giờ chọn trên test.
2. **Test một lần duy nhất cho cấu hình đã chọn** (`walk_forward_test_scores`): refit mỗi quý (63
   phiên, D62, `build_refit_schedule`) trong TEST theo cửa sổ huấn luyện MỞ RỘNG
   (`refit_training_window`, purge tới đúng ngày refit), nối điểm dự báo ngoài mẫu của mọi đoạn, rồi
   `evaluate_on_test` (tái dùng nguyên hàm của `baselines.py`: Rank IC/ICIR, Newey-West `lag=h-1`,
   bootstrap khối tròn) MỘT LẦN — không chọn lại trên test.

GBDT (R2, R3) dừng sớm trên 20% CUỐI theo thời gian của mỗi cửa sổ huấn luyện
(`early_stopping_fraction_gbdt`, D62, `config.yaml` mục `data_split`) — không xáo trộn ngẫu nhiên,
để tập dừng sớm luôn nằm SAU tập fit theo thời gian, tránh rò rỉ thứ tự.

Lưới cấu hình (`config.yaml` mục `model_grid`, mỗi họ tối đa 3 cấu hình, đúng ngân sách 2.3 mục 6,
cùng điều kiện với L1):
- R1 Ridge: `alpha ∈ {0,1; 1; 10}`.
- R2 LightGBM: ba cặp `(num_leaves, min_data_in_leaf)` của `lightgbm_configs` (dùng chung với L1),
  cộng `gbdt_common` (`learning_rate=0,05`, `max_trees=2000`, `early_stopping_rounds=50`,
  `subsample_row=0,8`, `subsample_col=0,8`).
- R3 XGBoost: `max_depth ∈ {3; 5; 7}`, các tham số khác theo `gbdt_common` (giống R2).

Rank IC dùng cột `r` (lợi suất thô theo $h$), đúng định nghĩa `giao_thuc_danh_gia.md` mục 2, giống
4.2 — KHÔNG phải `r_excess` (dù `r_excess` là mục tiêu huấn luyện). Về mặt số học, hai cột cho đúng
CÙNG Rank IC vì $r_{excess} = r - \text{trung bình chéo ngày}(r)$: trừ một hằng số mỗi ngày không
đổi thứ hạng trong ngày, nên dùng cột nào của hai cột cũng ra cùng kết quả — `evaluate_on_test`/
`select_best_config` (tái dùng từ `baselines.py`) vẫn merge đúng cột `r` có sẵn trong
`labels_daily.csv`, không cần đổi.

## 4. Test

`code/tests/test_regression_topk.py` (10 test mới, dữ liệu tổng hợp nhỏ, không đọc dữ liệu thật):
ngân sách cấu hình ≤ 3/họ; ghép đặc trưng bỏ đúng mẫu chưa hiện thực nhãn; chia tập dừng sớm theo
thời gian (20% cuối); cả ba họ mô hình khôi phục được tín hiệu tuyến tính đơn giản
(`corr(dự báo, đặc trưng tín hiệu) > 0,5`); `select_config_on_validation` chọn đúng cấu hình có tín
hiệu thật (phân biệt với cấu hình suy biến LightGBM chỉ một lá do `min_data_in_leaf` quá lớn — Ridge
alpha lớn KHÔNG tạo được tình huống suy biến tương tự vì chỉ co đều mọi hệ số, không đổi thứ hạng dự
báo); `walk_forward_test_scores` phủ đúng mọi ngày test, và **không rò rỉ** mẫu chưa hiện thực nhãn
tại thời điểm refit (kiểm bằng cách so dự báo ngày đầu test giữa dữ liệu xóa hẳn dòng chưa hiện thực
và dữ liệu giữ dòng đó với giá trị cực đoan 999,0 nhưng `exit_date_resolved` rơi sau mốc refit — hai
kết quả phải khớp tuyệt đối nếu purge đúng). Toàn bộ suite dự án: 144 test PASS (từ 134, thêm 10 test
mới) — xác nhận PASS ở cả hai môi trường: Python 3.14 (bản thư viện mới hơn, D95) và `code/.venv`
Python 3.12.10 (bản thư viện đã chốt đúng `requirements.txt`, D96).

## 5. Kết quả trên dữ liệu thật (test: 2025-04 đến 2026-10-02, 375 ngày quyết định)

Cấu hình chọn trên validation và kết quả test
(`code/data/processed/regression_topk_results_4_3.csv`, đầy đủ 27 dòng ở
`regression_topk_grid_search_log_4_3.csv`):

| Mô hình | $h$ | Chọn trên validation | Rank IC validation | Rank IC test (TB) | ICIR test | Newey-West $p$ (IC>0) | Cận dưới bootstrap 95% |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| R1 Ridge | 5 | `alpha=10` | -0,0105 | -0,0136 | -0,075 | 0,791 | -0,0447 |
| R2 LightGBM | 5 | `num_leaves=15, min_data_in_leaf=200` | 0,0190 | -0,0106 | -0,071 | 0,781 | -0,0361 |
| R3 XGBoost | 5 | `max_depth=7` | -0,0010 | -0,0042 | -0,028 | 0,621 | -0,0289 |
| R1 Ridge | 10 | `alpha=10` | -0,0395 | -0,0322 | -0,175 | 0,914 | -0,0755 |
| R2 LightGBM | 10 | `num_leaves=15, min_data_in_leaf=200` | 0,0269 | -0,0130 | -0,084 | 0,756 | -0,0486 |
| R3 XGBoost | 10 | `max_depth=7` | -0,0142 | -0,0178 | -0,117 | 0,832 | -0,0509 |
| R1 Ridge | 21 | `alpha=10` | -0,0759 | -0,0368 | -0,191 | 0,858 | -0,1077 |
| R2 LightGBM | 21 | `num_leaves=31, min_data_in_leaf=100` | 0,0198 | -0,0442 | -0,314 | 0,974 | -0,0899 |
| R3 XGBoost | 21 | `max_depth=7` | -0,0119 | -0,0348 | -0,270 | 0,946 | -0,0771 |

**Nhận xét (mô tả, không phải kết luận H1 chính thức — đó là việc của 4.4, khi có L1 để so sánh):**

- **Không mô hình hồi quy nào (R1, R2, R3) có Rank IC dương, có ý nghĩa thống kê trên test, ở bất kỳ
  $h$ nào** — cùng kết luận với B2-B4 ở 4.2. Mọi $t$-stat Newey-West đều ÂM (IC test trung bình < 0),
  $p$-giá trị một phía cho $H_0: IC \le 0$ đều rất cao (0,62-0,97), tức hoàn toàn không có bằng chứng
  ủng hộ $IC > 0$.
- **R2 LightGBM (đối chứng chính của H1) đảo chiều dấu từ validation sang test ở cả ba $h$**: Rank
  IC validation dương (0,019 đến 0,027, mô hình học được một quan hệ nào đó trong mẫu) nhưng Rank IC
  test ÂM (-0,011 đến -0,044) — cùng kiểu bất ổn định đã thấy ở B2 (momentum) tại 4.2. Đây là tín
  hiệu quan trọng cho H1 (so R2 với L1 ở 4.4): nếu LambdaRank (L1) cũng không vượt qua được R2 theo
  hướng tương tự, giả thuyết H1 khó có cơ sở; nếu L1 ổn định hơn (không đảo chiều), đó là bằng chứng
  ủng hộ H1.
- R1 Ridge (mô hình tuyến tính) có Rank IC ÂM ở CẢ BA cấu hình alpha, ở validation lẫn test, tại mọi
  $h$ — `alpha=10` (co hệ số mạnh nhất trong lưới) luôn được chọn vì ít âm nhất, không phải vì tìm ra
  tín hiệu dương. Dấu hiệu 11 đặc trưng tuyến tính không mang quan hệ tuyến tính đơn giản, ổn định
  với `r_excess` trên dữ liệu này.
- R3 XGBoost cho kết quả tương tự R2 (Rank IC test âm cả ba $h$, không có cấu hình nào vượt ngưỡng ý
  nghĩa) — nhất quán với R2, ủng hộ cách đọc rằng vấn đề không nằm ở riêng LightGBM mà có thể ở chính
  cách đặt bài toán hồi quy / đặc trưng hiện có trên dữ liệu VN giai đoạn này (đúng vai trò R3 đề ra:
  "kiểm tra R2 không phải do thư viện").

## 6. Phạm vi không làm trong bước này

- Không so sánh chính thức R1-R3 với L1 (LambdaRank) — đó là trọng tâm của 4.4 và kiểm định H1
  (Newey-West giữa chuỗi Rank IC của L1 và R2, không phải so sánh Rank IC tuyệt đối từng mô hình như
  bảng trên).
- Không tính Sharpe sau chi phí hay danh mục thực (turnover, MDD) cho R1-R3 — chỉ đánh giá TÍN HIỆU
  (Rank IC), giống 4.2; danh mục thực chốt ở 4.7.
- Không chạy trên các tập con (VN30, HOSE, HNX riêng) — để lại cho báo cáo ở 4.x.
- Không đối chiếu lại kết quả trên Google Colab với bản thư viện đã chốt ở `requirements.txt`
  (D67) — khuyến nghị làm khi có dịp, xem mục 2.
