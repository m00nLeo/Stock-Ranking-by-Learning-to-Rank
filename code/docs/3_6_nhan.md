# Báo cáo bước 3.6: nhãn 5/10/21 ngày (gộp cả nội dung 3.10)

**Ngày:** 06/11/2026. **Trạng thái:** Xong (D78). **Căn cứ:** `bai_toan_hinh_thuc.md` mục 4 (D55);
`dinh_nghia_xu_huong.md` (D63); `mo_hinh_chi_phi_vn.md` (D61); `config/config.yaml`.

## 0. Gộp 3.6 và 3.10 trong cùng phiên (D78)

Charter tách "Label 5/10/21 ngày" (3.6: lợi suất, mức liên quan, khóa trần/sàn) và "Nhãn triple-barrier"
(3.10) thành hai mã bước. Prompt mở đầu phiên này gộp cả hai nội dung dưới tên "Bước 3.6" (do phiên trước
soạn prompt đã gộp). Vì mọi công thức của cả hai đều đã chốt từ Giai đoạn 2 và dùng chung một cơ chế tính
đường giá, cửa sổ, quy tắc khóa trần/sàn, việc tách thành hai phiên riêng sẽ chỉ lặp lại cùng một hạ tầng
tính toán. Quyết định: hoàn thành cả hai mã bước (3.6 và 3.10) trong phiên này, ghi rõ ở `charter.md` và
`decision_log.md`.

## 1. Ba loại nhãn, đúng công thức đã chốt

### 1.1. Nhãn hồi quy và mức liên quan (`bai_toan_hinh_thuc.md` mục 4.1-4.3, D55, D59)

- $r^{(h)}_{i,t} = C_{i,t+1+h}/C_{i,t+1} - 1$, giá đóng cửa đã điều chỉnh.
- Lợi suất vượt trung bình: $\tilde r^{(h)}_{i,t} = r^{(h)}_{i,t} - \frac{1}{|U_t|}\sum_{j \in U_t} r^{(h)}_{j,t}$,
  trung bình CHỈ tính trên các mã trong $U_t$ ngày đó **và bán được** (xem mục 2).
- Mức liên quan $\ell^{(h)}_{i,t} \in \{0,\dots,4\}$: chia $U_t$ thành $Q=5$ nhóm theo thứ hạng của
  $\tilde r^{(h)}_{i,t}$.

### 1.2. Nhãn xu hướng ba lớp triple-barrier (`dinh_nghia_xu_huong.md`, D63)

- $\hat\sigma_{i,t}$: độ lệch chuẩn EWMA (span 20) của lợi suất log ngày, dữ liệu $\le t$.
- $b^{(h)}_{i,t} = \max(1 \cdot \hat\sigma_{i,t} \cdot \sqrt{h},\ 0{,}60\%)$.
- Chỉ xét chạm rào bằng giá đóng cửa, từ phiên $t+3$ (T+2) đến $t+1+h$.
- $y = +1$ nếu chạm rào trên trước; $-1$ nếu chạm rào dưới trước; $0$ nếu không chạm. $t^1$ = thời điểm
  chạm hoặc $t+1+h$ nếu không chạm.
- Khóa trần/sàn giữa cửa sổ: nhãn **không đổi**, chỉ gắn cờ `khoa_tran_san` (nhãn xu hướng mô tả đường đi
  giá, không mô phỏng giao dịch thật).

### 1.3. Quy tắc khóa trần/sàn cho nhãn hồi quy (charter mục 7 dòng 3.6, D66)

Khác với nhãn xu hướng, nhãn hồi quy (dùng để backtest danh mục thật) áp quy tắc thực thi:
- **Khóa trần ở phiên vào lệnh ($t+1$):** coi như không mua được → không tạo nhãn cho $(i,t)$ (mọi $h$).
- **Khóa sàn ở phiên thoát lệnh dự kiến ($t+1+h$):** "giữ đến phiên bán được" — dời ngày thoát tới phiên
  giao dịch thật gần nhất sau đó mà không bị khóa sàn (tối đa 60 phiên tìm kiếm; không tìm được thì bỏ mẫu
  đó, cực hiếm với mã đang thanh khoản tốt trong $U_t$).
- **Lý do phân biệt trần/sàn theo chiều mua/bán:** khóa trần (dư mua, hết người bán) cản trở MUA, không
  cản trở bán (ai cũng muốn mua ở giá trần); khóa sàn (dư bán, hết người mua) cản trở BÁN, không cản trở
  mua. Đây là suy luận từ cơ chế khớp lệnh, không phải số liệu cần nguồn chính thức.
- **Phiên vào/thoát không có giao dịch (tạm ngừng):** không tạo nhãn nếu phiên vào không giao dịch; nếu
  phiên thoát dự kiến tạm ngừng, áp cùng cơ chế "tìm phiên bán được tiếp theo" như khóa sàn (gộp xử lý, vì
  cả hai đều là "không bán được đúng lúc dự kiến").

## 2. Kết quả chạy trên dữ liệu thật

- Đầu vào: `code/data/processed/universe_daily.csv` (954 ngày, 150 mã/ngày, sau D77), giá từ cache
  `code/data/raw/prices/` (707 mã, sạch sau D77).
- `labels_raw.csv` (trung gian, không lọc theo $U_t$): nhãn cho mọi mã từng xuất hiện trong universe.
- `labels_daily.csv` (chính thức): giới hạn vào $U_t$, cộng $\tilde r$, $\ell$.
- Số liệu cụ thể (số dòng, tỷ lệ bỏ mẫu do khóa trần ở phiên vào, tỷ lệ dời ngày thoát do khóa sàn, phân bố
  ba lớp xu hướng): xem mục 3.

## 3. Thống kê mô tả (chỉ để kiểm chất lượng dữ liệu, không dùng chọn tham số)

Chạy trên toàn bộ 143.100 dòng $(i,t)$ của $U_t$ (954 ngày, 150 mã/ngày), mỗi dòng nhân 3 horizon:

- **Bỏ mẫu do không vào lệnh được (khóa trần hoặc tạm ngừng ở phiên $t+1$):** 1.108/143.100 dòng
  $(i,t)$ (0,77%) — không có nhãn ở bất kỳ $h$ nào cho các dòng này.
- **Dời ngày thoát do khóa sàn hoặc tạm ngừng ở phiên thoát dự kiến:** 0,10% số nhãn còn lại; không có
  trường hợp nào vượt quá giới hạn tìm kiếm 60 phiên (tham số kỹ thuật an toàn, không phải quyết định
  thiết kế).
- **Tỷ lệ rào bị sàn chi phí 0,60% chặn (`san_chi_phi`), theo $h$:** $h=5$: 0,16%; $h=10$: 0,06%;
  $h=21$: 0,01%. Giảm dần theo $h$ vì rào tỷ lệ $\sqrt{h}$ trong khi sàn cố định — khớp lý thuyết.
- **Tỷ lệ có cờ khóa trần/sàn trong cửa sổ nhãn xu hướng (`khoa_tran_san`):** 1,50%.
- **Phân bố ba lớp xu hướng, theo $h$ (trên toàn mẫu, không chỉ train — đúng cần báo lại trên train riêng
  ở 3.13):**

  | $h$ | Giảm (−1) | Đi ngang (0) | Tăng (+1) |
  |---|---|---|---|
  | 5 | 18,1% | 62,3% | 19,6% |
  | 10 | 19,8% | 57,3% | 22,9% |
  | 21 | 21,9% | 52,4% | 25,7% |

  Tỷ lệ "đi ngang" giảm dần và hai lớp còn lại tăng dần theo $h$ — khớp lý thuyết (horizon dài hơn, xác
  suất chạm một trong hai rào cao hơn).
- **Mức liên quan ($Q=5$):** cân bằng gần như tuyệt đối giữa 5 nhóm (~84.4-84.6 nghìn dòng/nhóm mỗi $h$),
  đúng kỳ vọng của phép chia theo phân vị.
- **$\tilde r$ (lợi suất vượt trung bình):** trung bình ≈ 0 (đúng theo định nghĩa, vì trừ trung bình
  chéo mỗi ngày), độ lệch chuẩn ≈ 6,9%.

Lưu ý: đây là mô tả trên TOÀN BỘ mẫu (01/2023-11/2026, gồm cả đoạn sẽ là test), chỉ để kiểm chất lượng dữ
liệu (độ phủ, tỷ lệ cờ) theo đúng rủi ro #12 của giao thức bias — không dùng số liệu này để chọn hay điều
chỉnh bất kỳ tham số nào. Báo cáo mô tả CHỈ TRÊN TRAIN theo đúng yêu cầu `dinh_nghia_xu_huong.md` mục 7 sẽ
làm ở 3.13, sau khi có bảng chia dữ liệu (3.7).

## 4. Module và test

`code/vnrank/labels/labeling.py`: `build_trading_calendar`, `ewma_sigma`, `compute_symbol_labels`,
`compute_raw_labels`, `add_excess_and_relevance`. Test ở `code/tests/test_labeling.py` (10 test, không gọi
mạng thật), gồm test kiểm không nhìn trước (giống rủi ro #1, áp dụng tương tự cho nhãn: thay đổi giá ở các
phiên ngoài cửa sổ cần dùng không được đổi nhãn).

## 5. Việc không làm ở bước này (ngoài phạm vi 3.6/3.10)

- Không chia train/validation/test, không purge/embargo (3.7).
- Không tính trạng thái thị trường theo ngày cho RQ3 (3.11, dù công thức đã có sẵn ở
  `dinh_nghia_xu_huong.md` mục 6 — để lại đúng mã bước của nó, không gộp thêm vì không dùng chung hạ tầng
  tính nhãn cổ phiếu).
- Không kiểm tra rò rỉ theo lịch walk-forward thật hay kiểm tra xếp tầng (3.13).
