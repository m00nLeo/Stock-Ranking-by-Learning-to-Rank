# Báo cáo bước 3.3: làm sạch dữ liệu, kiểm giá điều chỉnh

**Ngày:** 04/11/2026. **Trạng thái:** Xong (D73). Mục 4 cập nhật: vấn đề thiếu giá thô đã giải quyết, không còn chờ quyết định.

## 1. Giá điều chỉnh hay giá thô: ĐÃ XÁC NHẬN LÀ ĐÃ ĐIỀU CHỈNH

Dùng sự kiện HPG trả cổ tức bằng cổ phiếu tỷ lệ 10% (100 cổ phiếu nhận 10 cổ phiếu mới), ngày giao dịch không hưởng quyền (GDKHQ) 25/05/2026, ngày đăng ký cuối cùng 26/05/2026. Nguồn: thông báo điều chỉnh của chứng quyền HPG.08.PHS (ghi ngày thực hiện điều chỉnh 25/05/2026 do sự kiện này) và tin tức tài chính (index.vn, doanhnhan.baophapluat.vn), đối chiếu qua WebSearch ngày 04/11/2026.

Một cổ tức cổ phiếu 10% về lý thuyết làm giá tham chiếu bị cắt khoảng 1 − 1/1,10 ≈ 9,1% (tự tính) vào đúng phiên GDKHQ nếu dùng giá thô. Dữ liệu thực tế lấy từ `entrade_client.fetch_ohlcv('HPG', ...)`:

| Ngày | Đóng cửa |
|---|---|
| 2026-05-22 (phiên trước GDKHQ) | 23,95 |
| 2026-05-25 (GDKHQ) | 24,10 |

Không có khoảng trống giảm ~9%, giá còn tăng nhẹ. Kết luận: **giá trả về bởi API Entrade/DNSE đã được điều chỉnh (adjusted) cho sự kiện doanh nghiệp**, không phải giá thô.

## 2. Hệ quả: thiếu giá thô cho kiểm khóa trần, sàn

`giao_thuc_chia_du_lieu_bias.md` rủi ro #4 yêu cầu **cả hai loại giá**: giá điều chỉnh (tính lợi suất, nhãn) và giá thô (so với giá tham chiếu và biên độ sàn, để phát hiện khóa trần/sàn). Nguồn hiện tại (Entrade/DNSE) chỉ cho giá đã điều chỉnh. Đây là một giới hạn dữ liệu MỚI, chưa có trong báo cáo 3.2. Xem mục 4 câu hỏi cho người dùng.

## 3. Ngày giao dịch, tạm ngừng, khối lượng bằng 0, giá trị bất thường

- **VNINDEX thiếu một vài phiên có thật:** so lịch giao dịch của VNINDEX với VNM (2022-2025) qua API này, phát hiện VNINDEX thiếu 07/04/2023 và 12/08/2024, dù VNM có dữ liệu bình thường hai ngày đó. Tra tin tức xác nhận VN-Index vẫn có điểm đóng cửa ngày 07/04/2023 (1069,71 điểm, theo tổng hợp tin thị trường), tức đây là **lỗi hụt dữ liệu của nguồn, không phải thị trường nghỉ**. Vì vậy module `cleaning.trading_calendar_from_symbols` hợp nhất lịch giao dịch từ nhiều mã (không dùng riêng VNINDEX làm chuẩn).
- **Khối lượng bằng 0, OHLC bất thường:** kiểm VNM, HPG, VIC giai đoạn 2022-2024 (748 phiên mỗi mã): không có phiên nào khối lượng bằng 0 hay OHLC vô lý (high < giá khác, giá âm...). Đây là ba mã rất thanh khoản, chưa đại diện cho mã thanh khoản thấp hoặc từng bị tạm ngừng; sẽ kiểm thêm khi dựng universe thật ở 3.4.
- **Mã tạm ngừng giao dịch:** chưa thử nghiệm với một mã có lịch sử tạm ngừng đã xác minh, vì chưa cần thiết cho phạm vi 3.3 (đủ để biết `flag_missing_calendar_days` xử lý được tình huống này).

## 4. Giải quyết thiếu giá thô (D73, 04/11/2026)

Thay vì tìm nguồn giá thô hay tự suy ngược mức giá tuyệt đối, dùng tính chất OHLC suy biến: phiên bị khóa trần/sàn thì `open = high = low = close`, tính chất này không đổi khi điều chỉnh. Chỉ coi là khóa khi mức thay đổi so với phiên trước khớp biên độ sàn (±7% HOSE, ±10% HNX, dung sai 0,5 điểm phần trăm). Đã kiểm chứng bằng mã L10 (HOSE): chuỗi 6 phiên liên tiếp (20-27/10/2026) O=H=L=C, mỗi phiên tăng đúng khoảng +7%, khớp tin tức công khai về việc L10 từng tăng trần nhiều phiên. Hàm `cleaning.detect_limit_lock` chạy đúng trên dữ liệu này, phát hiện cả 6 phiên.

Không cần nguồn dữ liệu nào khác ngoài Entrade/DNSE. Vấn đề thiếu giá thô cho 3.6 coi như đã giải quyết.

## 5. Module đã viết

`code/vnrank/data/cleaning.py`: `validate_ohlc_integrity`, `flag_zero_volume`, `trading_calendar_from_symbols`, `flag_missing_calendar_days`. Test ở `code/tests/test_cleaning.py` (4 test, cộng 13 test cũ, tổng 17 PASS cục bộ).
