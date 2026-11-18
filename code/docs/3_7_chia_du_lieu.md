# Báo cáo bước 3.7: chia train/validation/test, purge theo t1, embargo

**Ngày:** 07/11/2026. **Trạng thái:** Xong (D79). **Căn cứ:** `giao_thuc_chia_du_lieu_bias.md` mục 1-3
(D62); `config/config.yaml` (mục `data_split`).

## 1. Xác định `test_end` cụ thể từ dữ liệu thật

`config.yaml` để `test_end: null` ("đến phiên quyết định cuối có nhãn h=21 đủ; xác nhận ở 3.2"). Dữ liệu
giá thật có đến 03/11/2026 (3.2-3.5). Phiên quyết định cuối cùng còn đủ dữ liệu tương lai để tính nhãn
$h=21$ là **02/10/2026** (lấy trực tiếp từ ngày lớn nhất có nhãn $h=21$ trong `labels_daily.csv`, không
suy đoán — nhãn chỉ được tạo khi đủ dữ liệu tương lai, xem `labeling.compute_symbol_labels`). Khớp ước
tính ban đầu ở `giao_thuc_chia_du_lieu_bias.md` ("khoảng cuối tháng 9/2026" khi giả định dữ liệu chỉ đến
hết 10/2026; nay nhỉnh hơn một chút vì dữ liệu thật đến 03/11/2026).

`config/config.yaml` mục `data_split.test_end` cập nhật từ `null` thành `"2026-10-02"`.

## 2. Bảng chia (`splits_daily.csv`)

| Đoạn | Mốc | Số ngày quyết định trong universe |
|---|---|---|
| Train | 2023-01-01 .. 2024-06-30 | 369 |
| Validation | 2024-07-01 .. 2025-03-31 | 188 |
| Test | 2025-04-01 .. 2026-10-02 | 375 |
| (ngoài phạm vi, sau `test_end`) | 2026-10-03 .. 2026-11-03 | 22 (không gán nhãn split, vì thiếu dữ liệu tương lai cho $h=21$) |

Khớp ước tính ban đầu (train ~375, validation ~185, test ~350-375 phiên).

## 3. Purge theo $t^1$ (`giao_thuc_chia_du_lieu_bias.md` mục 3)

Hai cột hiện thực hóa khác nhau tùy loại nhãn (đã có sẵn trong `labels_daily.csv` từ 3.6):
- **Nhãn xu hướng:** cột `t1` (thời điểm chạm rào, hoặc $t+1+h$ nếu không chạm).
- **Nhãn hồi quy/mức liên quan:** [SỬA 3.7, D79] dùng cột `exit_date_resolved` thay vì $t+1+h$ thuần túy
  theo đặc tả gốc D62 — xem mục 4.

**Purge mẫu huấn luyện** (giữ mẫu có hiện thực $\le$ mốc $\tau$, ở đây $\tau$ = cuối train 30/06/2024):

| Loại nhãn | Số dòng train trước purge | Sau purge | Bị loại |
|---|---|---|---|
| Xu hướng (`t1`) | 165.950 | 161.120 | 4.830 (2,9%) |
| Hồi quy (`exit_date_resolved`) | 165.950 | 160.050 | 5.900 (3,6%) |

**Purge tập đánh giá đứng trước** (validation chỉ tính trên $t$ có hiện thực $\le$ đầu test 01/04/2025,
để chọn cấu hình không dùng thông tin từ giai đoạn test):

| Loại nhãn | Số dòng validation trước purge | Sau purge | Bị loại |
|---|---|---|---|
| Xu hướng (`t1`) | 84.592 | 80.157 | 4.435 (5,2%) |
| Hồi quy (`exit_date_resolved`) | 84.592 | 79.188 | 5.404 (6,4%) |

Tỷ lệ bị loại ở validation cao hơn train vì validation ngắn hơn (9 tháng) nên tỷ trọng các ngày quyết
định cuối cùng (có nhãn chưa kịp hiện thực trước đầu test) lớn hơn tương đối.

## 4. [SỬA 3.7, D79] Điều chỉnh mốc hiện thực cho nhãn hồi quy: dùng `exit_date_resolved` thay vì $t+1+h$

`giao_thuc_chia_du_lieu_bias.md` mục 3 (viết trước khi có quy tắc khóa sàn ở D66, 01/11/2026) ghi $t^1 =
t+1+h$ cho nhãn lợi suất/mức liên quan. Nhưng 3.6 (D78) đã thêm quy tắc "giữ đến phiên bán được" khi
khóa sàn ở phiên thoát dự kiến: thời điểm BIẾT giá thoát thật có thể trễ hơn $t+1+h$ (lưu ở cột
`exit_date_resolved`). Nếu purge vẫn dùng $t+1+h$ cho các mẫu này, một số mẫu sẽ được đưa vào huấn luyện
TRƯỚC khi giá thoát thật sự được biết — rò rỉ nhẹ, trái nguyên tắc point-in-time (charter mục 5 điều 1).

**Quyết định:** purge nhãn hồi quy dùng `exit_date_resolved` (đã có sẵn, không cần tính thêm). Ảnh hưởng
trên dữ liệu thật: chỉ 423/422.826 dòng (0,10%) có `exit_date_resolved` khác mốc $t+1+h$ danh nghĩa — khớp
đúng tỷ lệ mẫu bị dời ngày thoát đã báo ở 3.6 (`exit_extended_khoa_san`). Ảnh hưởng rất nhỏ nhưng là lựa
chọn nhất quán, không suy đoán.

## 5. Walk-forward: lịch refit trong test

Theo `giao_thuc_chia_du_lieu_bias.md` mục 2: refit mỗi quý (~63 phiên) trong test, cửa sổ huấn luyện MỞ
RỘNG (gồm train, validation, và phần test đã qua), cấu hình cố định sau khi chọn trên validation.
`build_refit_schedule` sinh đúng lịch này từ lịch giao dịch thật; `refit_training_window` trả về đúng tập
mẫu đã purge cho một ngày refit bất kỳ (ví dụ: cửa sổ cho ngày refit đầu tiên của test gồm toàn bộ
train + validation đã purge theo mốc đó). Không chạy thử các lần refit cụ thể ở bước này (chưa có mô hình,
để 4.x).

## 6. Embargo

Giữ nguyên quy định: embargo = 0 vì walk-forward chỉ đi tới (không có mẫu huấn luyện nằm sau tập đánh
giá). Embargo 22 phiên ($h_{\max}+1$) chỉ áp khi dùng thủ tục nhìn ngược (CSCV cho PBO), chưa cần ở bước
này — để 4.x khi tính PBO.

## 7. Module và test

`code/vnrank/splits/split_protocol.py`: `assign_split`, `determine_test_end`, `purge_by_realization`,
`build_refit_schedule`, `refit_training_window`. Test ở `code/tests/test_split_protocol.py` (7 test,
không gọi mạng thật). Tổng 60 test PASS cục bộ cả dự án.

## 8. Việc không làm ở bước này (ngoài phạm vi 3.7)

- Không kiểm tra rò rỉ đầy đủ (dịch nhãn, xáo trộn, mô hình trên nhãn ngẫu nhiên) — đó là 3.8.
- Không purge riêng cho mô hình gộp đa horizon F0 (theo $h$ lớn nhất của các thành phần) — việc này áp
  dụng khi huấn luyện F0 thật ở 4.x, không phải một bảng tĩnh cần dựng trước.
- Không chạy thử các lần refit cụ thể hay huấn luyện mô hình nào.
