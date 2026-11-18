# MANIFEST — bộ dữ liệu universe/feature/label/split

**Phiên bản:** `v20261107-ce5d2e5c`. **Ngày tạo:** 07/11/2026. **Căn cứ:** `code/docs/3_9_dataset_card.md`.

Mã băm 8 ký tự = 8 ký tự đầu của SHA256 nối tiếp nội dung bốn file sau, ĐÚNG theo thứ tự liệt kê (dùng để
xác nhận dataset chưa bị đổi, không phải để giải mã ngược nội dung):

| File | SHA256 (16 ký tự đầu) | Số dòng (không tính header) | Kích thước |
|---|---|---|---|
| `universe_daily.csv` | `23a5a7fbe86a0c96` | 143.100 | 2.289.613 byte |
| `features_daily.csv` | `252415f4ec746f2a` | 143.100 | 33.151.297 byte |
| `labels_daily.csv` | `4180f3831fa62d5b` | 423.934 | 59.506.257 byte |
| `splits_daily.csv` | `5beaddbb63d5770a` | 954 | 17.639 byte |
| `labels_raw.csv` (trung gian, không tính vào mã băm phiên bản) | `b8fd2cf6ebaadec6` | 988.995 | 111.505.798 byte |

**Quyết định liên quan:** D74 (universe), D76/D77 (feature), D78 (nhãn), D79 (split), D80 (kiểm tra
leakage), D81 (đóng gói dataset).

**Để xác nhận file chưa bị đổi**, tính lại SHA256 từng file (ví dụ `sha256sum <file>` hoặc
`Get-FileHash <file> -Algorithm SHA256` trên PowerShell) và so 16 ký tự đầu với bảng trên.
