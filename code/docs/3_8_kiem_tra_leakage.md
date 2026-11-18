# Báo cáo bước 3.8: kiểm tra leakage

**Ngày:** 07/11/2026. **Trạng thái:** Xong (D80). **Căn cứ:** charter mục 7 dòng 3.8;
`giao_thuc_chia_du_lieu_bias.md` rủi ro #1 (nhìn trước).

## 0. Mô hình dùng để kiểm tra (không phải mô hình chính thức)

Dùng Ridge dạng đóng (`leakage_checks.ridge_fit`), tự cài bằng numpy thuần (không thêm phụ thuộc
`scikit-learn` dù đã khai trong `requirements.txt`, vì công thức Ridge chỉ là một phép giải tuyến tính,
không cần thư viện cho một kiểm tra QA). Đây CHỈ để kiểm tra rò rỉ, không phải lựa chọn hay huấn luyện mô
hình chính thức (việc đó ở 4.x, có `model_grid` riêng trong `config.yaml`).

## 1. Dịch feature trên pipeline ghép

`test_pipeline_leakage.py`: chạy NỐI TIẾP `feature_registry.compute_features` rồi
`labeling.compute_raw_labels` trên cùng một bộ dữ liệu tổng hợp, xáo trộn mạnh giá ở các phiên sau một
mốc, rồi so sánh đặc trưng VÀ nhãn ở các ngày trước mốc đó giữa hai lần chạy — phải giống hệt nhau. Khác
với test đơn vị ở 3.5/3.6 (chỉ kiểm từng hàm riêng lẻ), test này kiểm cả hai bước chạy nối tiếp như trong
pipeline thật. Thêm kiểm tra tĩnh: tập cột đặc trưng (`FEATURE_COLUMNS`) và tập cột nhãn (`r`, `y`, `t1`,
`relevance`, ...) phải rời nhau (không cột nhãn nào lọt vào đặc trưng). Cả hai PASS.

## 2. Xáo trộn nhãn và nhãn ngẫu nhiên: Rank IC trên validation

**Quy trình:** với mỗi $h \in \{5, 10, 21\}$, ghép `features_daily.csv` + `labels_daily.csv` +
`splits_daily.csv`. Huấn luyện Ridge trên TRAIN (purge theo `exit_date_resolved` $\le$ cuối train), đo
Rank IC (tương quan hạng Spearman trung bình theo ngày) trên VALIDATION (purge theo `exit_date_resolved`
$\le$ đầu test) với nhãn **thật** ($\tilde r$). KHÔNG đụng vào tập test (test chỉ chạy một lần).

Hai kiểm tra rò rỉ, mỗi kiểm tra lặp lại 10 lần với bản xáo trộn/ngẫu nhiên độc lập rồi lấy trung bình
(một lần đơn lẻ có thể tình cờ lệch khỏi 0 do số đặc trưng ít — xem `code/tests/test_leakage_checks.py`
để thấy hiệu ứng này trên dữ liệu tổng hợp):
- **Xáo trộn nhãn** (`shuffle`): hoán vị ngẫu nhiên các giá trị $\tilde r$ trong tập train.
- **Nhãn ngẫu nhiên thuần túy** (`random`): thay bằng nhiễu chuẩn, cùng độ lệch chuẩn với $\tilde r$ thật.

**Đối chứng** (`real`, không lặp lại, chỉ một lần): huấn luyện trên nhãn THẬT, đo IC trên validation —
không phải kết luận về chất lượng đặc trưng (Ridge không tinh chỉnh, chỉ để có điểm so sánh).

| $h$ | $n$ train / val | IC thật (đối chứng) | IC xáo trộn (TB ± lệch chuẩn, 10 lần) | IC ngẫu nhiên (TB ± lệch chuẩn, 10 lần) |
|---|---|---|---|---|
| 5 | 54.400 / 27.446 | −0,013 (lệch chuẩn theo ngày 0,186, 183 ngày) | 0,007 ± 0,017 | 0,001 ± 0,015 |
| 10 | 53.650 / 26.696 | −0,037 (lệch chuẩn theo ngày 0,201, 178 ngày) | −0,021 ± 0,030 | 0,003 ± 0,035 |
| 21 | 52.000 / 25.046 | −0,080 (lệch chuẩn theo ngày 0,210, 167 ngày) | −0,010 ± 0,047 | −0,006 ± 0,043 |

**[SỬA 3.8, D82] Rà soát lại kết luận thống kê (phát hiện khi Claude web viết mục "Dữ liệu" của chương
Phương pháp, 07/11/2026):** bản gốc viết "cả IC xáo trộn và IC ngẫu nhiên đều ... nằm trong khoảng sai số
chuẩn của chính chúng ... không có ý nghĩa thống kê", chỉ minh họa bằng ô $h=5$ ($|0,007|/(0,017/\sqrt{10})
\approx 1,3$) rồi suy rộng cho cả 6 ô. Kiểm lại từng ô cho thấy ô $h=10$ xáo trộn có $|{-0,021}|/(0,030/
\sqrt{10}) \approx 2,2$ lần sai số chuẩn — KHÔNG thuộc diện "trong sai số chuẩn" như câu kết luận cũ ngụ ý.
Đây là kết luận **phóng đại so với bằng chứng**, cần sửa.

**Chạy lại với 50 lần lặp (thay vì 10) để kiểm xem lệch này có ổn định không:**

| $h$ | IC xáo trộn (50 lần) | $\lvert t\rvert$ | IC ngẫu nhiên (50 lần) | $\lvert t\rvert$ |
|---|---|---|---|---|
| 5 | 0,0038 ± 0,0166 | 1,61 | 0,0033 ± 0,0171 | 1,36 |
| 10 | 0,0059 ± 0,0353 | 1,18 | 0,0087 ± 0,0246 | **2,50** |
| 21 | −0,0116 ± 0,0514 | 1,60 | −0,0049 ± 0,0546 | 0,64 |

Với 50 lần lặp (hạt giống ngẫu nhiên khác), ô $h=10$ xáo trộn KHÔNG còn lệch (1,18), nhưng ô $h=10$ ngẫu
nhiên lại lệch (2,50) — tức ô "có vẻ lệch" ĐỔI CHỖ khi đổi số lần lặp hoặc hạt giống. Đây chính là dấu hiệu
của nhiễu Monte Carlo thuần túy (kiểm 6 ô cùng lúc ở mức ý nghĩa ~5% thì xác suất thấy ít nhất một ô "có
vẻ lệch" chỉ do ngẫu nhiên đã xấp xỉ 1 − 0,95⁶ ≈ 26%, không nhỏ), KHÔNG phải một độ lệch ổn định, có thể
tái lập — nếu có rò rỉ thật, độ lệch phải xuất hiện lại ở cùng một ô khi đổi hạt giống.

**Kết luận viết lại (thay cho kết luận cũ):** không dùng tỷ số $t$ từng ô làm bằng chứng chính, vì số lần
lặp (10-50) còn quá ít để ước lượng sai số chuẩn ổn định cho một kiểm tra QA kiểu này (khác hẳn các phép
kiểm định chính thức H1-H4 của luận văn, dùng Newey-West trên chuỗi dài hơn nhiều, ở 2.4/4.x). Ba căn cứ
đáng tin hơn cho kết luận "không có dấu hiệu rò rỉ": (1) độ lớn IC xáo trộn/ngẫu nhiên luôn nhỏ (trong
khoảng [−0,02; 0,02] ở phần lớn trường hợp, hiếm khi tới 0,05) và cùng độ lớn với IC "thật" (đối chứng,
cũng gần 0 vì Ridge chưa tinh chỉnh); (2) không có ô nào lệch khỏi 0 một cách ỔN ĐỊNH, TÁI LẬP ĐƯỢC khi
đổi hạt giống/số lần lặp — ô "có vẻ lệch" thay đổi ngẫu nhiên giữa các lần chạy; (3) **quan trọng nhất**,
kiểm tra cấu trúc ở mục 1 (dịch giá trên pipeline ghép, cột đặc trưng và cột nhãn rời nhau) loại trừ trực
tiếp khả năng có đường rò rỉ cơ học trong code, không phụ thuộc vào ước lượng thống kê nhiều nhiễu. Kiểm
tra Rank IC ở mục này chỉ là bằng chứng BỔ SUNG, không phải bằng chứng duy nhất hay mạnh nhất.

## 3. Module và test

`code/vnrank/evaluation/leakage_checks.py`: `ridge_fit`, `ridge_predict`, `rank_ic_by_day`,
`run_label_leakage_check`, `run_label_leakage_check_repeated`. Test ở `code/tests/test_leakage_checks.py`
(6 test, dữ liệu tổng hợp có tín hiệu thật biết trước, xác nhận hàm có "sức mạnh" phát hiện tín hiệu khi
có, và cho IC gần 0 khi xáo trộn/ngẫu nhiên) và `code/tests/test_pipeline_leakage.py` (2 test, pipeline
ghép). Tổng 69 test PASS cục bộ cả dự án.

## 4. Việc không làm ở bước này (ngoài phạm vi 3.8)

- Không huấn luyện hay chọn mô hình chính thức (Ridge/LightGBM/XGBoost/LambdaRank của `model_grid`) — đó
  là 4.x.
- Không chạy kiểm tra trên tập test (giữ đúng nguyên tắc test chạy một lần).
- Không kiểm tra rò rỉ riêng cho xếp tầng (mô hình lọc dùng đầu ra ngoài mẫu của tầng trước) vì Cấp 1
  hiện hành không còn tầng lọc/cổng tin cậy (D43); kiểm tra rò rỉ cho tầng xu hướng cụ thể (nếu cần thêm,
  ví dụ khi dùng xác suất xu hướng làm đặc trưng ở ablation A4) để ở 3.13.
