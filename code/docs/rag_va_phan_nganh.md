# Mở rộng hệ thống minh họa: RAG tra cứu báo cáo + phân tích theo ngành

08/11/2026 (D91), cập nhật 11/11/2026 (D92: dữ liệu ngành thật). **Nguồn gốc:** cô hướng dẫn yêu
cầu/gợi ý bổ sung hai phần này vào hệ thống minh họa (không ảnh hưởng phần benchmark xếp hạng và
đánh giá đang làm ở Giai đoạn 4).

## 1. Phạm vi

Đây là phần MỞ RỘNG của hệ thống minh họa (nhánh S, charter mục 7), KHÔNG đổi phần lõi đã chốt ở
D43 (benchmark xếp hạng + dự báo xu hướng là trọng tâm chính). Hai phần bổ sung:

1. **RAG (Retrieval-Augmented Generation) cho chatbot:** chatbot có thể tra cứu nội dung các báo
   cáo kỹ thuật đã làm (`code/docs/*.md`), để trả lời câu hỏi về PHƯƠNG PHÁP đã dùng — khác với ba
   hàm đọc `xep_hang`/`xu_huong`/`ket_qua` (D64, S1) vốn chỉ trả số liệu, không giải thích cách làm.
2. **Phân tích theo nhóm ngành:** với một ngành cụ thể, liệt kê các mã thuộc ngành đó kèm mức độ
   ổn định và xu hướng gần nhất, để người dùng lọc theo ngành quan tâm.

## 2. RAG: thiết kế và đã làm được

`code/vnrank/system/rag.py` — quy trình ba bước kinh điển của RAG, tự cài bằng TF-IDF/cosine
thuần Python + numpy (không phụ thuộc scikit-learn hay vector-DB, nhất quán với `harness.py`):

1. **Nạp (load)**: `load_corpus(docs_dir)` đọc mọi file `.md` trong một thư mục, cắt theo từng mục
   cấp 2 ("## ...") thành các đoạn (chunk), mỗi đoạn gắn tên file nguồn và tiêu đề mục.
2. **Đánh chỉ mục (index)**: `build_index(chunks)` tính trọng số TF-IDF cho từng đoạn.
3. **Truy xuất (retrieve)**: `search(query, index, top_k)` tính TF-IDF của câu hỏi, so cosine với
   mọi đoạn, trả về các đoạn liên quan nhất.

Phần SINH câu trả lời (đưa đoạn truy xuất được vào một LLM để viết câu trả lời cuối) nằm NGOÀI
module này — module chỉ làm phần truy xuất; LLM thật ở chatbot sẽ nhận các đoạn làm ngữ cảnh,
không tự bịa ngoài đó (cùng nguyên tắc "LLM không tự sinh số liệu" đã có ở `system_architecture.md`).

**Chạy thử trên kho báo cáo thật của dự án** (13 file, 93 đoạn, `code/docs/*.md`):

| Câu hỏi | Đoạn truy xuất đứng đầu |
|---|---|
| "VN30 lấy từ nguồn nào" | `3_4_universe.md` — mục "VN30 theo từng kỳ xét duyệt" (điểm 0,346) |
| "pipeline as-of dùng để làm gì" | `3_12_pipeline_asof.md` — mở đầu (điểm 0,417) |
| "làm sao kiểm tra không có rò rỉ dữ liệu" | `3_8_kiem_tra_leakage.md` — mục "Việc không làm ở bước này" (điểm 0,265) |

Hai câu đầu truy xuất đúng mục cần. Câu thứ ba cho thấy một hạn chế thật của truy xuất theo từ
khóa thuần túy: mục "việc không làm" của `3_8` nhắc nhiều lần từ "rò rỉ" (dù đang nói về phạm vi
KHÔNG làm), nên được xếp hạng cao hơn mục chính xác hơn có thể ở nơi khác trong cùng file. Đây là
hạn chế đã biết của TF-IDF (chỉ khớp từ, không hiểu ngữ cảnh); nếu cần chính xác hơn, có thể nâng
cấp sang embedding ngữ nghĩa sau — không làm ngay vì kho tài liệu hiện còn nhỏ (93 đoạn) và chưa
cần thiết (nguyên tắc "cần và đủ", D63).

Test: `code/tests/test_rag.py` (6 test, dữ liệu tổng hợp nhỏ trong `tmp_path`, không đọc
`code/docs/` thật trong test để không phụ thuộc nội dung báo cáo có thể đổi).

## 3. Phân tích theo ngành: cơ chế VÀ dữ liệu thật đã có (D92, 11/11/2026)

`code/vnrank/system/sector_analysis.py` — với một ngày `as_of` (point-in-time, chỉ dùng dữ liệu
`<= as_of`):
- `stability_and_trend_by_symbol`: với mỗi mã, tính độ ổn định (nghịch đảo độ lệch chuẩn lợi suất
  log trong cửa sổ gần nhất) và xu hướng (dấu lợi suất tích lũy cùng cửa sổ).
- `sector_ranking`: gộp với một bảng ánh xạ mã-ngành, xếp hạng độ ổn định TRONG TỪNG NGÀNH riêng
  (không so giữa các ngành khác nhau, vì mức độ biến động "bình thường" khác nhau giữa các ngành).
- `filter_by_sector`: lọc bảng đã xếp hạng còn đúng một ngành.
- `build_sector_mapping_from_vietstock`: đọc 11 file Excel Vietstock "Cổ phiếu theo Ngành
  *.xlsx" (người dùng cung cấp 11/11/2026, xem mục 4) thành một bảng `symbol`, `exchange`,
  `sector`.

Test: `code/tests/test_sector_analysis.py` (6 test): 5 test cho cơ chế tính toán (dữ liệu tổng hợp
tự dựng — nhận diện đúng mã ổn định hơn trong cùng ngành, đúng chiều xu hướng, chỉ dùng dữ liệu
đến `as_of`, xếp hạng tách riêng theo ngành), 1 test cho `build_sector_mapping_from_vietstock`
(file Excel tổng hợp mô phỏng, loại đúng mã quỹ ETF).

**Chạy thử trên dữ liệu thật** (707 mã, `as_of` = 03/11/2026, cửa sổ 63 phiên):

| Ngành | Mã ổn định nhất | Xu hướng |
|---|---|---|
| Công nghệ thông tin | CMG | giảm |
| Tài chính | VAB | giảm |

699/707 mã xếp hạng được (8 mã chưa đủ 63 phiên lịch sử gần `as_of` hoặc có độ lệch chuẩn bằng 0,
bị loại đúng theo thiết kế). Bảng đầy đủ xem `code/data/processed/sector_mapping.csv`.

## 4. Nguồn dữ liệu ngành: người dùng cung cấp, khớp tuyệt đối với danh sách mã đang dùng

Người dùng cung cấp 11 file Excel Vietstock "Cổ phiếu theo Ngành <tên ngành>.xlsx" (xuất
11/11/2026, cùng khuôn mẫu file danh sách mã đã dùng ở 3.4), theo đúng chuẩn **GICS (Global
Industry Classification Standard)** mà HOSE dùng cho VNAllShare (đã xác nhận ở báo cáo lần trước
qua factsheet chính thức). 11 ngành (đủ cả 11 ngành GICS chuẩn, nhiều hơn 10 ngành thấy được qua
factsheet tổng hợp trước đó — factsheet thiếu "Dịch vụ truyền thông"): Năng lượng, Nguyên vật
liệu, Công nghiệp, Tiêu dùng không thiết yếu, "thiết yếu" (Hàng tiêu dùng thiết yếu — tên file bị
cắt), Chăm sóc sức khỏe, Tài chính, Công nghệ thông tin, Dịch vụ truyền thông, Tiện ích, Bất động
sản.

**Đối chiếu:** gộp 11 file được 731 mã; loại 24 mã quỹ ETF (cùng 24 mã đã loại ở `load_listed_
stocks`, 3.4) còn lại ĐÚNG 707 mã — khớp TUYỆT ĐỐI với danh sách niêm yết đang dùng xuyên suốt dự
án (707 = 408 HOSE + 299 HNX, D74), không thiếu không thừa một mã nào. Độ chi tiết: một cấp (ngành
GICS cấp 1, 11 nhóm), không có phân ngành con (GICS còn 3 cấp sâu hơn — industry group, industry,
sub-industry — nhưng các file này không có); nếu cần sâu hơn phải tìm nguồn khác.

**Phạm vi:** toàn thị trường (707/707 mã đang theo dõi), không chỉ VN30 — vượt mục tiêu tối thiểu
ban đầu ("có thể làm trong mốc VN30").

## 5. Dữ liệu mã hủy niêm yết: đã trích xuất đầy đủ, KẾT LUẬN CHÍNH THỨC (D94)

Người dùng cung cấp hai file PDF ("Hủy niêm yết sàn HOSE.pdf", "Hủy niêm yết sàn HNX.pdf"). Đã
trích xuất đầy đủ cả hai bằng `pdftotext -layout -enc UTF-8` (poppler), ghép theo cặp liên kết
mã-cổ-phiếu và dòng dữ liệu sàn/ngày/khối lượng:
- **HOSE: 50/50 dòng — ĐỦ.** File PDF do Vietstock xuất chỉ chứa trang 1/2 phân trang gốc (tổng
  89 bản ghi), 39 dòng còn lại nằm ở trang 2 mà file PDF người dùng cung cấp không có — đây là giới
  hạn của chính file nguồn, không phải giới hạn trích xuất.
- **HNX: 33/34 dòng.** Dòng STT 31 biến mất khỏi lớp văn bản của PDF ở cả hai chế độ trích xuất
  (có `-layout` và không), nhiều khả năng do một banner tiêu đề lặp lại khi in trang che mất nội
  dung dòng này trong bản PDF gốc — không thể khôi phục từ file này.

**Sửa một hiểu lầm đã ghi ở D92:** đọc lại cẩn thận thứ tự cột tiêu đề bảng (`Sàn GD hiện tại` rồi
mới đến `Sàn GD trước khi hủy`) cho thấy **file "Hủy niêm yết sàn HOSE.pdf" lọc theo "Sàn GD hiện
tại" = HOSE**, không phải "các mã đã rời HOSE". Nói cách khác, toàn bộ 50 dòng trong file này là
các mã ĐANG giao dịch trên HOSE, từng có đăng ký ở một sàn khác (chủ yếu UPCOM, một số ít HNX)
trước đó — và ngược lại cho file HNX. Ví dụ ACB: "Sàn GD hiện tại = HOSE, Sàn GD trước khi hủy =
HNX, ngày 01/12/2020" khớp đúng sự kiện thật ACB chuyển từ HNX sang HOSE tháng 12/2020. Câu ở D92
("BCM rời HOSE năm 2020, niêm yết lại sau đó") ĐẢO NGƯỢC chiều thật: BCM hủy đăng ký UPCOM ngày
19/08/2020 khi được chuyển LÊN HOSE, và đã giao dịch liên tục trên HOSE từ đó đến nay — không có
chuyện rời rồi quay lại. Hai file này là **nhật ký các lần chuyển sàn (promotion)**, không phải
nhật ký các mã rời sàn.

**Đối chiếu với danh sách 707 mã hiện hành** (`code/data/processed/sector_mapping.csv`): cả 83 mã
trích xuất được (50 HOSE + 33 HNX) đều **CÓ MẶT** trong 707 mã — khớp 100%, không mã nào thiếu.
ROS và FLC (hai mã đã xác nhận "invalid symbol" ở 3.2) không xuất hiện trong cả hai file — đúng
như dự kiến, vì hai mã này đã ngừng giao dịch hoàn toàn nên không có "sàn hiện tại" nào để lọc vào
loại danh sách này.

**Kết luận chính thức:** hai file "Hủy niêm yết sàn HOSE/HNX" của Vietstock KHÔNG liên quan đến câu
hỏi khảo sát sống sót (survivorship bias) mà D71/D74/D89 đặt ra — chúng ghi lại việc chuyển sàn,
không phải việc công ty ngừng hoạt động hay dữ liệu biến mất. Ước lượng **45-70 mã có thể thiếu do
bỏ mã hủy niêm yết (D71/D74/D89) GIỮ NGUYÊN, không có căn cứ để sửa** — hai nguồn dữ liệu đơn giản
không nói về cùng một hiện tượng. Đóng hẳn câu hỏi còn treo từ D92.

## 6. Việc chưa làm, để lại

- Tích hợp RAG và phân tích ngành vào chatbot thật (S9-S12) — hiện mới có hai module lõi, chưa nối
  vào luồng hỏi-đáp Gradio.
- Chọn LLM cụ thể cho bước sinh câu trả lời (chưa chốt ở S1; có thể dùng chung nhà cung cấp với
  Trường hợp 3 nếu phù hợp).
- Phân ngành sâu hơn cấp 1 (nếu cần cho RAG/chatbot sau này) — chưa có nguồn.

## 7. Test

`code/tests/test_rag.py` (6 test), `code/tests/test_sector_analysis.py` (6 test, thêm 1 so với báo
cáo trước). Toàn bộ suite dự án: 134 PASS (từ 122 trước D91, 133 sau D91).
