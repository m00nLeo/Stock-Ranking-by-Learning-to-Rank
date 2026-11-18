# Báo cáo bước 3.4: dựng universe theo thời điểm, VN30 theo kỳ, ước lượng mã thiếu

**Ngày:** 04/11/2026. **Trạng thái:** Xong (D74). **Căn cứ:** `bai_toan_hinh_thuc.md` mục 2 (D55);
`giao_thuc_chia_du_lieu_bias.md` rủi ro #2, #3, #6, #12, #13 (D62); `config/config.yaml` (D54-D65, D71).

**Cập nhật 06/11/2026 (D77, phát hiện ở 3.5):** nguồn Entrade/DNSE trả trùng hai phiên cho cùng một
ngày lịch VN ở đúng 3 ngày (27/12/2022, 07/04/2023, 10/04/2023), ảnh hưởng 521/707 mã. Đã sửa tại
`entrade_client.fetch_ohlcv` (giữ bản ghi timestamp lớn hơn) và tải lại toàn bộ cache giá; dựng lại
`universe_daily.csv` từ dữ liệu sạch. Kết quả không đổi về bản chất: vẫn đúng 150 mã/ngày, 954 ngày;
số quan sát đạt điều kiện lọc trên train+validation đổi rất nhỏ (min 287 → 286 mã/ngày, không đổi kết
luận N=150 khả thi). Chi tiết lỗi: `code/docs/3_5_feature_registry.md` mục 2, `decision_log.md` D77.

## 0. Hai nguồn dữ liệu đầu vào do người dùng cung cấp

AI không tìm được nguồn tải trực tiếp, máy đọc được cho danh sách mã niêm yết (hsx.vn/hnx.vn không có
file tải công khai; API công khai kiểu TCBS bị chặn bởi Cloudflare, không nên cố vượt qua theo bài học
Run 38) và cho thông báo gốc của HOSE về VN30 theo kỳ. Người dùng tự tìm và cung cấp hai file (04/11/2026):

1. **`mã_niêm_yết_HOSE_HNX.xlsx`** — tải từ `finance.vietstock.vn/truy-xuat-du-lieu/tong-hop-doanh-nghiep.htm`,
   748 dòng. Sau khi loại 24 mã quỹ ETF/quỹ đầu tư (tên có "Quỹ", mã dài 8 ký tự, ví dụ `FUEVFVND`,
   `E1VFVN30`, không phải cổ phiếu phổ thông), còn **707 mã: 408 HOSE, 299 HNX**, không trùng lặp.
2. **`Thành_viên_VN30.csv`** — 8 kỳ xét duyệt (01/2023, 07/2023, 01/2024, 07/2024, 01/2025, 07/2025,
   01/2026, 07/2026), mỗi kỳ 30 mã, kèm ngày công bố, ngày hiệu lực, nguồn và liên kết (phần lớn là bản
   PDF công bố của HOSE hoặc công ty chứng khoán/báo tài chính dẫn lại nguyên văn thông báo của HOSE).

**Kiểm chứng độc lập của AI:** đối chiếu nội dung `Thành_viên_VN30.csv` với kết quả tìm kiếm độc lập
(WebSearch, không dùng file người dùng cung cấp) về các đợt đổi rổ VN30 đã biết công khai: kỳ 01/2023
(BCM vào), kỳ 07/2023 (SSB, SHB vào; NVL, PDR ra), kỳ 01/2025 (LPB vào; POW ra), kỳ 07/2025 (DGC vào;
BVH ra) — **khớp hoàn toàn** với dữ liệu trong 4 kỳ đó của file. Không phát hiện trùng lặp mã trong kỳ,
không thiếu mã (đúng 30/kỳ), không có giá trị rỗng. Cả hai file đều dùng được, không cần sửa.

Để `vnrank/` tự chứa trong `code/` (chạy được trên Colab chỉ với repo GitHub, không cần toàn bộ thư mục
dự án nằm ngoài `code/`), bản sao của hai file được đặt ở `code/data/raw/universe_inputs/` (thư mục này
không lên GitHub theo `.gitignore`, giống cách xử lý cache giá — khi chạy trên Colab cần tải lại hai file
này, ví dụ qua Google Drive).

## 1. Dựng $U_t$ theo từng ngày

### 1.1. Tải giá cho toàn bộ 707 mã

`vnrank/universe/build_universe.fetch_price_cache` gọi `entrade_client.fetch_ohlcv` cho từng mã,
01/2022-04/11/2026, cache CSV riêng từng mã ở `code/data/raw/prices/` (không lên GitHub). Chạy cục bộ:
**707/707 mã tải thành công, 0 lỗi** (199 giây, 6 luồng song song, không giả lập trình duyệt, không
vượt rào chống bot). Mã có ít dữ liệu nhất có 2 phiên (mới niêm yết gần đây).

### 1.2. Tham số lọc và N: chọn từ phân phối dữ liệu trên train + validation (rủi ro #12, #13)

Theo đúng ràng buộc (không chạm test): mọi thống kê dưới đây tính trên `data_split.train_start` đến
`validation_end` (01/2023-03/2025 theo `config.yaml`), 323.547 quan sát mã-ngày.

| Tham số | Giá trị chọn | Căn cứ từ dữ liệu train+validation |
|---|---|---|
| $L$ (cửa sổ thanh khoản) | 21 phiên (~1 tháng) | Cùng đơn vị "1 tháng" với cửa sổ ngắn nhất của momentum B2 (21 phiên, D59); không có lý do chọn cửa sổ khác cho universe |
| $\ell$ (độ trễ) | 0 phiên | Mọi đại lượng dùng dữ liệu $\le t$ đã là $\mathcal{F}_t$ theo định nghĩa (mục 1 của `bai_toan_hinh_thuc.md`); không cần trễ thêm, giữ "cần và đủ" (D63) |
| $p^{\min}$ (giá thấp/penny) | 5,0 (nghìn đồng) | Xấp xỉ bách phân vị thứ 10 của giá đóng cửa trên toàn bộ 323.547 quan sát (thực tế 4,93); làm tròn để dễ diễn giải. Loại 10,2% quan sát mã-ngày |
| $v^{\min}$ (giá trị giao dịch TB $L$ phiên) | 100.000 (nghìn đồng) = 100 triệu đồng/phiên | Giữa bách phân vị 25 (90,2 triệu) và 30 (139,9 triệu); loại 26,2% quan sát mã-ngày |
| $n^{\min}$ (số phiên có giao dịch trong $L$) | 15/21 phiên | Biện pháp phòng ngừa, **không ràng buộc trên dữ liệu hiện có**: 100% quan sát trong train+validation có đủ 21/21 phiên giao dịch (không có phiên khối lượng bằng 0 nào trong tập 707 mã hiện còn niêm yết, khớp phát hiện ở 3.3 trên mẫu nhỏ VNM/HPG/VIC) |
| $N$ (kích thước universe) | **150** (cận trên của khoảng 50-150, D55) | Sau khi lọc $p^{\min}$, $v^{\min}$, $n^{\min}$: số mã đạt điều kiện mỗi ngày trong suốt train+validation luôn $\ge 287$ (nhỏ nhất ngày 19/04/2023), tức luôn còn dư sức chọn đủ 150 mã thanh khoản cao nhất. Chọn $N$ ở cận trên để tăng số quan sát mã-ngày cho mẫu huấn luyện vốn đã ngắn (rủi ro đã nêu ở `giao_thuc_chia_du_lieu_bias.md` mục 7) |

Hai bộ lọc (penny, thanh khoản) loại tổng cộng 33,6% quan sát mã-ngày (phần giao nhau 8.935 quan sát bị
loại bởi cả hai), còn lại trung bình 385,5 mã/ngày đạt điều kiện trước khi cắt còn 150 mã có giá trị
giao dịch trung bình cao nhất.

### 1.3. Kết quả

`vnrank/universe/build_universe.build_daily_universe` áp cùng bốn tham số trên cho TOÀN BỘ giai đoạn có
nhãn (01/2023 đến phiên dữ liệu cuối, 03/11/2026 — chưa cắt theo `test_end`, việc đó để ở bước dùng
universe này, không thuộc phạm vi 3.4), không chọn lại tham số trên đoạn test. Kết quả: **954 ngày giao
dịch, đúng 150 mã mỗi ngày không thiếu ngày nào**, lưu ở `code/data/processed/universe_daily.csv`
(không lên GitHub, tái tạo được bằng cách chạy lại module với cùng cache giá).

## 2. VN30 theo từng kỳ xét duyệt

`vnrank/universe/build_universe.load_vn30_membership` + `vn30_members_asof(date)` tra đúng 30 mã có hiệu
lực tại một ngày bất kỳ, dùng kỳ gần nhất có `effective_date <= date` (không dùng danh sách hiện tại cho
quá khứ). **Kiểm chéo:** toàn bộ 8 kỳ x 30 mã VN30 (kể cả các kỳ 2023) đều có mặt trong danh sách 707 mã
hiện đang niêm yết — nghĩa là chưa từng có mã VN30 nào trong mẫu bị hủy niêm yết, bộ lọc VN30 không bị
ảnh hưởng bởi giới hạn survivorship (khác với universe rộng $U_t$, xem mục 3). Lát cắt $U^{VN30}_t$ dùng
hàm này khi báo cáo kết quả theo tập con VN30 ở Giai đoạn 4 (D56), không thuộc phạm vi tạo thêm bảng ở
bước này.

## 3. Ước lượng số mã bị thiếu do universe không giữ được mã đã hủy niêm yết (D71)

Không có cách đo trực tiếp (không còn dữ liệu giá của các mã đã hủy niêm yết qua Entrade/DNSE). Ước
lượng gián tiếp bằng hai bước, tra qua WebSearch 04/11/2026 (nguồn: tin tức tài chính, không có thống kê
tổng hợp chính thức đầy đủ theo năm cho cả giai đoạn — đã nêu rõ khoảng không chắc chắn):

1. **Số mã HOSE+HNX đã hủy niêm yết trong giai đoạn mẫu (01/2023-11/2026), tự tổng hợp từ nguồn rời
   rạc:** ước lượng khoảng **45-70 mã** (điểm giữa ~55). Căn cứ: 2023 không có số tổng hợp đầy đủ, chỉ
   xác nhận được một số trường hợp cụ thể (ví dụ FLC, hủy niêm yết hiệu lực 20/02/2023, theo Quyết định
   51/QĐ-SGDHCM 13/02/2023); 2024 có số khá rõ (~10 HOSE + 12-16 HNX, các nguồn không khớp nhau hoàn
   toàn); 2025 có ít nhất vài trường hợp xác nhận trên HNX (VE8, CVN, KSQ) và HOSE (DPC); 2026 (đến
   11/2026) có thêm VNE, BCG, TCD, HVX, CLG và một số khác đang xem xét (KPF, HBC). Đây là ước lượng
   trật tự độ lớn (order-of-magnitude), KHÔNG phải thống kê chính thức đầy đủ.
2. **Tỷ lệ các mã này có thể đã đủ thanh khoản để lọt vào $U_t$ (N=150) tại một thời điểm nào đó trước
   khi hủy niêm yết:** phần lớn mã bị hủy niêm yết bắt buộc là do vi phạm công bố thông tin hoặc lỗ lũy
   kế vượt vốn điều lệ — thường đi kèm thanh khoản đã suy giảm từ trước, nên nhiều khả năng đã rơi khỏi
   universe thanh khoản cao trước khi bị hủy hẳn. Tuy vậy có ít nhất 3 trường hợp xác nhận cụ thể từng
   là cổ phiếu vốn hóa lớn, thanh khoản cao trong phần lớn giai đoạn mẫu trước khi hủy niêm yết: **FLC**
   (02/2023), **HBC** (Hòa Bình Construction, 09/2024), **HNG** (HAGL Agrico, 09/2024). Ước lượng tỷ lệ
   hợp lý: 5%-15% của tổng số mã hủy niêm yết.

**Kết quả (dùng `estimate_delisted_undercount(55, (0.05, 0.15))`):** khoảng **3-8 mã** (điểm giữa ~6) có
thể đã xuất hiện trong $U_t$ tại một thời điểm nào đó trong mẫu nếu còn dữ liệu giá của chúng, tương
đương khoảng **2-5% kích thước universe** (N=150) tại một thời điểm bất kỳ, hoặc dưới 1% nếu tính trên
707 mã nền hiện có. Đây là ước lượng sơ bộ, không phải số đếm chính xác; ghi vào phần "Giới hạn" của
luận văn cùng với giới hạn đã nêu ở D71.

## 4. Module và test

`code/vnrank/universe/build_universe.py`: `load_listed_stocks`, `fetch_price_cache`, `load_price_panel`,
`load_vn30_membership`, `vn30_members_asof`, `compute_liquidity_features`, `build_daily_universe`,
`estimate_delisted_undercount`. Test ở `code/tests/test_build_universe.py` (7 test, không gọi mạng thật,
dùng file tạm/DataFrame nhỏ tự dựng; tổng 28 test PASS cục bộ cả dự án).

## 5. Việc không làm ở bước này (ngoài phạm vi 3.4)

- Không cắt universe theo `test_end` (chưa chốt, để bước dùng tới universe này tự xử lý theo đúng
  `data_split` của `config.yaml`).
- Không xử lý vấn đề giá thô/khóa trần-sàn (đã giải quyết riêng ở 3.3, D73).
- Không tạo sổ đăng ký feature hay bảng nhãn (3.5, 3.6).
