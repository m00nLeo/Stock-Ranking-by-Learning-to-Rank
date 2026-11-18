# Báo cáo kiểm tra bước 3.2: nguồn dữ liệu giá

**Ngày:** 03/11/2026. **Quyết định liên quan:** D71 (`decision_log.md`). **Trạng thái:** Xong, có giới hạn đã ghi nhận.

## 1. vnstock không dùng được

`vnstock` (và phụ thuộc bắt buộc `vnai`) bị quản trị viên PyPI cách ly từ 24/09/2026, chờ rà soát bảo mật, áp dụng cho toàn bộ các bản kể cả 3.x cũ. Mã nguồn GitHub (`thinh-vu/vnstock`) tự ghi là "lưu trữ phục vụ tham khảo lịch sử"; bản đang phát triển ("Thế hệ 5") chuyển sang chỉ mục gói riêng của tác giả, giấy phép thương mại. `vnai` ("Vnstock Analytics Interface") không có mã nguồn mở, được import bắt buộc và bọc quanh gần như mọi hàm lấy dữ liệu thật trong `vnstock` (qua decorator `@optimize_execution`) ở khoảng 40 file, nên không thể tách rời dễ dàng. Quyết định: không dùng `vnstock`.

## 2. Nguồn thay thế: API công khai Entrade/DNSE

Thử nghiệm trực tiếp `https://services.entrade.com.vn/chart-api/v2/ohlcs/{stock|index}`, không cần giả lập trình duyệt hay vượt rào kiểm tra chống bot:

| Kiểm tra | Kết quả |
|---|---|
| Giá VNM, 01/2022 | Có dữ liệu, số liệu hợp lý |
| VNINDEX, 01/2022 (tham số `symbol_type=index`) | Có dữ liệu, mức điểm khớp VN-Index thực tế đầu 2022 (~1500-1530) |
| VNM, đến 30/10/2026 | Có dữ liệu mới nhất |
| ROS, FLC (mã đã hủy niêm yết năm 2023), giai đoạn 2023 còn giao dịch | Không có: API báo "invalid symbol" |

## 3. Bốn điều cần xác nhận của 3.2

| # | Điều | Kết quả |
|---|---|---|
| 1 | Độ phủ 01/2022 đến khoảng 10/2026 | ĐẠT cho mã đang niêm yết và VN-Index (xác nhận có dữ liệu đến 30/10/2026) |
| 2 | Giá điều chỉnh đúng | CHƯA XÁC NHẬN raw hay đã điều chỉnh split/cổ tức. Cần kiểm ở 3.3 bằng một ngày có sự kiện doanh nghiệp đã xác minh từ nguồn chính thức (không suy đoán từ trí nhớ) |
| 3 | Dữ liệu mã đã hủy niêm yết | KHÔNG CÓ qua nguồn này; không tìm được nguồn công khai, hợp lệ nào khác trong phạm vi phiên này. Người dùng chấp nhận giới hạn survivorship bias (D71), ghi vào phần "Giới hạn" của luận văn |
| 4 | VN30 theo thời điểm, VN-Index | VN-Index: ĐẠT. VN30 theo kỳ: không có API; nguồn xác nhận là thông báo định kỳ của HOSE (hiện 2 kỳ/năm, trước 07/2025 theo quý); cần tự tổng hợp khoảng 6-8 kỳ từ 01/2023 đến nay, thực hiện ở 3.4 |

## 4. Hệ quả cho thiết kế

- `code/vnrank/data/entrade_client.py`: client tự viết, chỉ phụ thuộc `requests`, không qua `vnstock`/`vnai`.
- `code/config/config.yaml`: thêm mục `data_source`; `universe.keep_delisted` đổi `true` → `false`.
- `03_Giai_doan_2_Cu_the_hoa/2_Bai_toan_hinh_thuc/bai_toan_hinh_thuc.md` và `6_Chia_du_lieu_bias/giao_thuc_chia_du_lieu_bias.md` (rủi ro #3, #6): đã chú thích [SỬA 3.2, D71], không xóa nội dung cũ.
- Việc còn lại trước khi 3.2 coi là khép hoàn toàn: kiểm giá điều chỉnh ở 3.3; tổng hợp VN30 theo kỳ ở 3.4; ước lượng số mã bị thiếu do bỏ qua mã hủy niêm yết ở 3.4.

## 5. Một việc dừng lại, không tự làm

Trong lúc dò thêm nguồn cho mã hủy niêm yết, có một lần định gọi API khác (VNDirect) kèm header giả lập trình duyệt để qua kiểm tra chống bot; hệ thống chặn hành động này ("lách kiểm tra chống bot") và đã dừng theo đúng hướng dẫn, không tìm cách khác để lách.
