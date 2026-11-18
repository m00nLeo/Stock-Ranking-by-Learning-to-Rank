# Báo cáo bước 3.13: kiểm tra rò rỉ cho tầng xu hướng theo lịch walk-forward thật

**Ngày:** 07/11/2026. **Trạng thái:** Xong. **Căn cứ:** charter mục 7 dòng 3.13 (D66); `dinh_nghia_xu_huong.md`
mục 6-7; `split_protocol.py` (3.7, D79); `leakage_checks.py` (3.8, D80, D82).

## 0. Mô hình dùng để kiểm tra (không phải mô hình chính thức)

Dùng bộ phân loại 3 lớp ghép từ ba Ridge dạng đóng theo kiểu một-đấu-còn-lại (`ovr_ridge_classifier_fit`,
`leakage_checks.py`), tự cài bằng numpy thuần, cùng cách làm với 3.8 (không thêm phụ thuộc mới cho một
kiểm tra QA). Điểm số mỗi lớp lấy từ Ridge hồi quy trên biến chỉ báo {−1}/{0}/{+1}; dự đoán là lớp có điểm
cao nhất. Đo bằng AUC một-đấu-còn-lại trung bình ba lớp (`auc_one_vs_rest`, công thức Mann-Whitney U) và
macro-F1 (`macro_f1`). Đây chỉ để kiểm rò rỉ, không phải lựa chọn hay huấn luyện mô hình chính thức (việc
đó ở 4.10, có `model_grid` riêng).

## 1. Xác nhận purge theo $t^1$ cho nhãn xu hướng (không phải `exit_date_resolved`)

3.7 (D79) đã tách hai cột hiện thực hóa khác nhau: nhãn xu hướng dùng `t1` (thời điểm chạm rào hoặc
$t+1+h$ nếu không chạm), nhãn hồi quy/mức liên quan dùng `exit_date_resolved` (có thể trễ hơn do khóa
sàn, D79). Toàn bộ kiểm tra ở bước này gọi `split_protocol.purge_by_realization(df, tau, "t1")`, KHÔNG
dùng `exit_date_resolved`, đúng loại nhãn đang kiểm (xu hướng).

Purge mẫu huấn luyện ($\tau$ = cuối train 30/06/2024) và tập đánh giá đứng trước ($\tau$ = đầu test
01/04/2025), theo $t^1$, cho từng $h$ (khác 3.7 ở chỗ tách riêng theo $h$ thay vì gộp một con số, vì $t^1$
phụ thuộc $h$):

| $h$ | Train trước purge | Train sau purge (loại) | Validation trước purge | Validation sau purge (loại) |
|---|---|---|---|---|
| 5 | 55.300 | 54.465 (835, 1,5%) | 28.125 | 27.485 (640, 2,3%) |
| 10 | 55.300 | 53.960 (1.340, 2,4%) | 28.125 | 26.912 (1.213, 4,3%) |
| 21 | 55.300 | 52.695 (2.605, 4,7%) | 28.125 | 25.760 (2.365, 8,4%) |

Số dòng bị loại tăng theo $h$ (hợp lý: $h$ lớn hơn → $t^1$ xa hơn → nhiều mẫu cuối đoạn có hiện thực hóa
rơi sau mốc $\tau$ hơn). Khớp đúng cơ chế purge đã định ở `giao_thuc_chia_du_lieu_bias.md` mục 3.

## 2. Xáo trộn nhãn đường đi (y), đo AUC và macro-F1 trên validation với nhãn thật

**Quy trình:** với mỗi $h \in \{5, 10, 21\}$, ghép `features_daily.csv` + `labels_daily.csv` +
`splits_daily.csv` (11 cột `FEATURE_COLUMNS`, không chuẩn hóa lại — cùng cách 3.8). Huấn luyện bộ phân
loại trên TRAIN đã purge theo $t^1$; đo AUC một-đấu-còn-lại và macro-F1 trên VALIDATION đã purge theo
$t^1$, với nhãn $y$ **thật**. KHÔNG đụng tập test.

**Đối chứng** (`that`, không lặp lại): huấn luyện trên $y$ thật, đo trên validation với $y$ thật.

**Xáo trộn nhãn** (`xao_tron`): hoán vị ngẫu nhiên $y$ trong tập train (giữ nguyên phân bố lớp, phá vỡ
liên kết với đặc trưng), lặp lại độc lập 10 lần rồi 50 lần (theo đúng bài học D82: một lần lặp đơn lẻ có
thể lệch do trùng hợp, cần kiểm ổn định bằng nhiều số lần lặp/hạt giống khác nhau trước khi kết luận).

| $h$ | AUC thật (đối chứng) | macro-F1 thật | AUC xáo trộn, 10 lần (TB ± lệch chuẩn) | AUC xáo trộn, 50 lần | macro-F1 xáo trộn (10 và 50 lần) |
|---|---|---|---|---|---|
| 5 | 0,524 | 0,257 | 0,504 ± 0,011 | 0,500 ± 0,011 | 0,2572 (không đổi) |
| 10 | 0,513 | 0,241 | 0,491 ± 0,016 | 0,501 ± 0,015 | 0,2406 (không đổi) |
| 21 | 0,510 | 0,227 | 0,497 ± 0,014 | 0,502 ± 0,019 | 0,2272 (không đổi) |

**Nhận xét:**
- AUC xáo trộn nằm quanh 0,5 (ngẫu nhiên) ở cả 10 và 50 lần lặp cho cả ba $h$, KHÔNG có ô nào lệch rõ và
  đổi chỗ khi tăng số lần lặp — khác với tình huống ở 3.8 (D82), nơi một ô "có vẻ lệch" ở 10 lần biến mất
  rồi một ô khác lệch ở 50 lần. Ở đây cả hai mức lặp đều cho kết quả ổn định quanh 0,5 → không có dấu hiệu
  rò rỉ qua kênh AUC.
- macro-F1 xáo trộn gần như không đổi giữa 10 và 50 lần lặp (độ lệch chuẩn ~$10^{-17}$, tức bằng 0 về mặt
  số học), vì khi huấn luyện trên nhãn bị xáo trộn (gần như nhiễu thuần túy so với đặc trưng), mô hình
  Ridge một-đấu-còn-lại luôn hội tụ về dự đoán lớp đa số ("đi ngang", lớp 0) cho MỌI dòng validation.
  Đã xác nhận trực tiếp: macro-F1 của một bộ dự đoán "luôn chọn lớp đa số" trên chính tập validation thật
  là 0,2572 / 0,2406 / 0,2272 cho $h=5/10/21$ — khớp CHÍNH XÁC với macro-F1 xáo trộn ở trên. Đây đúng là
  "mức ngẫu nhiên" mà đặc tả yêu cầu ("macro-F1 ~ tỷ lệ lớp đa số" — chính xác hơn, bằng macro-F1 của một
  bộ phân loại hằng số dự đoán lớp đa số, không phải bằng tỷ lệ lớp đa số).
- AUC thật (đối chứng, không lặp) chỉ nhỉnh hơn 0,5 một chút (0,51-0,52), tức 11 đặc trưng hiện có mang
  tín hiệu phân loại xu hướng rất yếu — khớp với Rank IC gần 0 đã thấy ở 3.8 cho nhãn hồi quy cùng bộ đặc
  trưng này. Đây là nhận xét về CHẤT LƯỢNG đặc trưng (để 4.x cải thiện), không phải bằng chứng rò rỉ.

**Kết luận:** không có dấu hiệu rò rỉ nhãn → đặc trưng qua kênh phân loại xu hướng. Bằng chứng dựa trên
(a) AUC xáo trộn bám sát 0,5 và ỔN ĐỊNH ở cả hai mức lặp (không có ô nào đổi chỗ như D82 đã cảnh báo),
(b) macro-F1 xáo trộn khớp đúng hành vi lý thuyết của một mô hình học trên nhiễu thuần túy, (c) test cấu
trúc pipeline tĩnh ở 3.8 mục 1 (dịch giá sau một mốc không đổi đặc trưng/nhãn trước mốc đó) vẫn là bằng
chứng chính cho tính point-in-time của pipeline, áp dụng chung cho cả nhãn hồi quy và nhãn xu hướng vì
cùng một hàm `compute_symbol_labels`.

## 3. Kiểm tra "xếp tầng": không áp dụng

Đặc tả gốc (trước D66) có bước "kiểm tra xếp tầng: thay đầu ra ngoài mẫu của tầng trước bằng đầu ra trong
mẫu, nếu hiệu năng tăng bất thường là cảnh báo rò rỉ" — áp dụng cho kiến trúc có tầng lọc/cổng tin cậy nối
tiếp nhau (ví dụ: tầng xếp hạng → tầng cổng tin cậy dùng đầu ra của tầng xếp hạng làm đầu vào). D43
(26/10/2026) đã bỏ cổng tin cậy, gộp động, thẻ ý tưởng khỏi phạm vi Cấp 1 (giữ benchmark xếp hạng + dự báo
xu hướng 5/10/21 ngày + chatbot hỏi đáp đơn giản). Cấp 1 hiện hành KHÔNG còn kiến trúc nhiều tầng nối tiếp
nhau theo kiểu "đầu ra tầng này làm đầu vào tầng sau" — xếp hạng và dự báo xu hướng là hai đầu ra song
song từ cùng một tập đặc trưng, không phải tầng nối tiếp. Vì vậy bước kiểm tra xếp tầng không áp dụng ở
đây; không có mã hay kết quả nào tạo ra cho mục này. Nếu 4.x sau này thêm kiến trúc nhiều tầng (ví dụ
gộp xác suất xu hướng làm đặc trưng cho xếp hạng — ablation A4 đã ghi ở D80 mục "việc không làm"), kiểm
tra xếp tầng sẽ cần làm lại ở bước tương ứng.

## 4. Kiểm tra cột `b` (rào) không bao giờ thấp hơn sàn chi phí 0,60%

Kiểm trực tiếp trên `labels_daily.csv` (423.934 dòng, 416.503 dòng có `b` khác rỗng — các dòng rỗng là do
không vào lệnh được, không có nhãn nào ở $h$ đó):

```
so dong co b: 416503
min b: 0.006
so dong b < 0.0060: 0
```

0/416.503 dòng có $b < 0,60\%$ — đúng thiết kế $b = \max(k \cdot \sigma \cdot \sqrt{h},\ 0,60\%)$
(`labeling.py`, `C_RT_BASE = 0.0060`). Đây là một ràng buộc TOÁN HỌC của công thức (hàm `max`), không
phải một tính chất cần suy ra từ dữ liệu — nhưng vẫn kiểm trên dữ liệu thật để loại khả năng lỗi cài đặt
(ví dụ nhầm thứ tự tham số, áp sai `C_RT_BASE` cho một nhánh nào đó). Đã thêm test đơn vị
`test_barrier_b_never_below_cost_floor_even_with_high_volatility` (`test_labeling.py`) dựng một chuỗi giá
tổng hợp có cả đoạn đi ngang dài (sigma thấp) và đoạn biến động mạnh (sigma cao, hạt giống ngẫu nhiên cố
định), xác nhận bất biến này giữ qua cả ba $h$.

## 5. Thống kê mô tả nhãn xu hướng CHỈ TRÊN TẬP TRAIN

Khác với 3.6/3.8 (mô tả trên toàn mẫu 01/2023–11/2026, gồm cả đoạn sẽ là test), bảng dưới đây lọc theo
`splits_daily.csv` chỉ giữ các ngày quyết định thuộc TRAIN (2023-01 đến 2024-06, 369 ngày × 150 mã =
165.950 dòng $(i,t)$, mỗi dòng × 3 horizon = 55.300 dòng/$h$; KHÔNG purge theo $t^1$ ở bảng mô tả này, vì
đây là thống kê mô tả độ phủ/chất lượng dữ liệu, không phải tập dùng để huấn luyện/đánh giá mô hình — mục
2 ở trên mới là nơi áp dụng purge):

| $h$ | Giảm (−1) | Đi ngang (0) | Tăng (+1) | `san_chi_phi` (rào bị sàn chi phí chặn) | `khoa_tran_san` | `tam_ngung` |
|---|---|---|---|---|---|---|
| 5 | 16,3% | 62,5% | 21,2% | 0,13% (73/55.300) | 0,09% (51/55.300) | 0,06% (32/55.300) |
| 10 | 17,1% | 57,7% | 25,3% | 0,01% (6/55.300) | 0,21% (114/55.300) | 0,08% (46/55.300) |
| 21 | 18,5% | 52,6% | 28,9% | 0,00% (0/55.300) | 0,50% (277/55.300) | 0,10% (55/55.300) |

**So với toàn mẫu (3.6):** $h=5$ toàn mẫu là 18,1%/62,3%/19,6%, train riêng là 16,3%/62,5%/21,2% — lớp
"đi ngang" gần như không đổi, nhưng train có ÍT hơn "giảm" và NHIỀU hơn "tăng" so với toàn mẫu (tương tự
ở $h=10$: 19,8%/57,3%/22,9% toàn mẫu so với 17,1%/57,7%/25,3% train; $h=21$: 21,9%/52,4%/25,7% toàn mẫu so
với 18,5%/52,6%/28,9% train). Khớp hướng hợp lý: đoạn train (2023-01 đến 2024-06) là giai đoạn thị trường
Việt Nam phục hồi sau đáy 2022, nên tỷ lệ phiên "tăng" cao hơn trung bình toàn mẫu (gồm cả đoạn test sau
này biến động nhiều chiều hơn). Đây CHỈ là ghi nhận mô tả (không dùng để chọn hay điều chỉnh tham số),
nhưng đáng lưu ý cho 4.x: mô hình huấn luyện trên train sẽ thấy một phân bố lớp hơi lệch "tăng" hơn so với
toàn cục — cần theo dõi khi đánh giá trên validation/test (nơi phân bố có thể lệch khác).

Tỷ lệ `san_chi_phi` giảm dần theo $h$ (rào tỷ lệ $\sqrt{h}$ trong khi sàn cố định, khớp lý thuyết, giống
3.6). Tỷ lệ `khoa_tran_san` tăng theo $h$ (cửa sổ nhãn dài hơn → xác suất gặp khóa trần/sàn cao hơn).

## 6. Module và test

Mở rộng `code/vnrank/evaluation/leakage_checks.py` (D80 tạo ở 3.8): thêm `auc_one_vs_rest`, `macro_f1`,
`ovr_ridge_classifier_fit`, `ovr_ridge_classifier_predict`, `run_trend_leakage_check`,
`run_trend_leakage_check_repeated`; `run_label_leakage_check_repeated` trả thêm khóa `all_ics` (giá trị
thô từng lần lặp, phục vụ kiểm ổn định kiểu D82 cho các bước sau). Thêm 6 test mới ở
`test_leakage_checks.py` và 1 test ở `test_labeling.py` (mục 4 ở trên). Tổng 76 test PASS cục bộ cả dự án
(`PYTHONPATH=scratchpad/pylib pytest tests/ -q`).

## 7. Việc không làm ở bước này (ngoài phạm vi 3.13)

- Không huấn luyện hay chọn mô hình chính thức cho tầng xu hướng (4.10).
- Không làm ablation A4 (xác suất xu hướng làm đặc trưng cho xếp hạng) đã ghi ở D80 — chỉ ở 4.x khi có
  kiến trúc nhiều tầng thật, lúc đó kiểm tra xếp tầng ở mục 3 mới áp dụng lại.
- Không chạy lại 3.8 (nhãn hồi quy) — 3.8 đã xong, D82 đã sửa kết luận thống kê; 3.13 chỉ làm phần nhãn
  xu hướng (phân loại) mà 3.8 chưa làm.
