"""Kiểm tất định cho câu trả lời của chatbot (bước 5, `system_architecture.md` mục 5):

- Nhánh JSON số liệu (`xep_hang`/`xu_huong`/`ket_qua`): (a) mọi số nêu trong câu trả lời khớp một
  giá trị trong JSON sau khi làm tròn theo quy tắc hiển thị; (b) mọi mã có trong JSON (nếu JSON có
  danh sách mã xếp hạng) đều được liệt kê, đúng thứ tự hạng; (c) không chứa cụm từ trong danh sách
  chặn.
- Nhánh RAG (`tra_cuu_phuong_phap`): câu trả lời là văn bản tự do dựa trên đoạn truy xuất, KHÔNG
  phải JSON số liệu, nên CHỈ áp dụng kiểm (c) — không áp kiểm (a)/(b). Đây là điểm
  `system_architecture.md` mục 5 ("Mở rộng chưa có trong đặc tả gốc") đã ghi là cần thiết kế khi
  triển khai S12 (D110).

Hỏng một trong các kiểm áp dụng thì chatbot hiển thị bảng mẫu dựng từ JSON (`messages.bang_mau_tu_json`)
thay câu của LLM, và ghi cờ `roi_ve_bang_mau` vào log.
"""

from __future__ import annotations

import re
import unicodedata

_SO_CHU_SO_MAC_DINH = 4
_SO_CHU_SO_XU_HUONG = 2

_NGAY_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_NGAY_VN_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
_SO_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def _chuan_hoa(text: str) -> str:
    """Chữ thường, bỏ dấu, để so cụm từ chặn không phân biệt hoa/thường hay cách gõ dấu."""
    nfkd = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def khong_chua_cum_tu_chan(cau_tra_loi: str, cum_tu_chan: list[str]) -> tuple[bool, list[str]]:
    """Trả (đạt, danh sách cụm từ chặn tìm thấy). Không phân biệt hoa/thường và dấu tiếng Việt."""
    chuan = _chuan_hoa(cau_tra_loi)
    loat = [c for c in cum_tu_chan if _chuan_hoa(c) in chuan]
    return (len(loat) == 0, loat)


def _flatten_so(du_lieu) -> list[float]:
    """Duyệt đệ quy dict/list, gom mọi giá trị số (int/float, không phải bool) thành một danh
    sách phẳng — dùng để kiểm "mọi số trong câu trả lời khớp một giá trị trong JSON"."""
    ra: list[float] = []
    if isinstance(du_lieu, dict):
        for v in du_lieu.values():
            ra.extend(_flatten_so(v))
    elif isinstance(du_lieu, list):
        for v in du_lieu:
            ra.extend(_flatten_so(v))
    elif isinstance(du_lieu, bool):
        pass
    elif isinstance(du_lieu, (int, float)):
        ra.append(float(du_lieu))
    return ra


def _flatten_ngay(du_lieu) -> set[str]:
    """Gom mọi giá trị chuỗi dạng YYYY-MM-DD xuất hiện trong JSON, để đối chiếu ngày nêu trong câu
    trả lời (có thể viết theo định dạng khác, ví dụ dd/mm/yyyy)."""
    ra: set[str] = set()
    if isinstance(du_lieu, dict):
        for v in du_lieu.values():
            ra |= _flatten_ngay(v)
    elif isinstance(du_lieu, list):
        for v in du_lieu:
            ra |= _flatten_ngay(v)
    elif isinstance(du_lieu, str) and _NGAY_ISO_RE.fullmatch(du_lieu):
        ra.add(du_lieu)
    return ra


def moi_so_khop_json(cau_tra_loi: str, du_lieu: dict) -> tuple[bool, list[str]]:
    """Kiểm (a): mọi số (và mọi ngày) nêu trong `cau_tra_loi` khớp một giá trị trong `du_lieu` sau
    khi làm tròn theo quy tắc hiển thị (4 chữ số thường, 2 chữ số riêng cho số đã làm tròn 2 chữ
    số — kiểm cả hai mức vì không biết số đang nhắc tới trường nào). Trả (đạt, danh sách số không
    khớp)."""
    ngay_hop_le = _flatten_ngay(du_lieu)
    text = cau_tra_loi

    # Mã băm (ví dụ "e9f3a7b2") là chuỗi hex, không phải số — bỏ khỏi text trước khi quét số,
    # tránh các đoạn chữ số lẫn trong đó (ví dụ "a7b2" không chứa số, nhưng "d10f8e" thì có "10")
    # bị hiểu nhầm thành số cần đối chiếu.
    ma_bam = du_lieu.get("ma_bam")
    if isinstance(ma_bam, str) and ma_bam:
        text = text.replace(ma_bam, " ")

    # Loại bỏ các đoạn ngày khỏi text trước khi quét số thường, đối chiếu ngày riêng.
    khong_khop: list[str] = []

    def _thay_ngay_iso(m: re.Match) -> str:
        if m.group(0) not in ngay_hop_le:
            khong_khop.append(m.group(0))
        return " "

    def _thay_ngay_vn(m: re.Match) -> str:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        iso = f"{y}-{int(mo):02d}-{int(d):02d}"
        if iso not in ngay_hop_le:
            khong_khop.append(m.group(0))
        return " "

    text_khong_ngay = _NGAY_ISO_RE.sub(_thay_ngay_iso, text)
    text_khong_ngay = _NGAY_VN_RE.sub(_thay_ngay_vn, text_khong_ngay)

    # Giá trị hợp lệ = chính các số trong JSON, làm tròn theo quy tắc hiển thị (4 chữ số thường,
    # 2 chữ số riêng cho diem_xu_huong — không biết số trong câu đang nhắc trường nào nên chấp
    # nhận cả hai mức). KHÔNG làm tròn thêm số đã trích từ câu trả lời: nếu LLM viết ít chữ số
    # thập phân hơn mức hiển thị quy định, coi là không khớp (rơi về bảng mẫu), tránh tình huống
    # một số bịa đặt tình cờ làm tròn trùng một giá trị nguyên nhỏ (ví dụ hạng 1, 2) trong JSON.
    so_hop_le = _flatten_so(du_lieu)
    muc_lam_tron = (_SO_CHU_SO_MAC_DINH, _SO_CHU_SO_XU_HUONG)
    tap_hop_le = {round(so, c) for so in so_hop_le for c in muc_lam_tron}

    for m in _SO_RE.finditer(text_khong_ngay):
        raw = m.group(0)
        try:
            gia_tri = float(raw.replace(",", "."))
        except ValueError:
            continue
        if not any(abs(gia_tri - hl) < 1e-9 for hl in tap_hop_le):
            khong_khop.append(raw)

    return (len(khong_khop) == 0, khong_khop)


def ma_dung_thu_tu_hang(cau_tra_loi: str, du_lieu: dict) -> tuple[bool, list[str]]:
    """Kiểm (b): nếu `du_lieu` có `danh_sach` (danh sách mã kèm `hang`, như `xep_hang` trả về), mọi
    mã phải xuất hiện trong `cau_tra_loi`, và thứ tự xuất hiện đầu tiên của mỗi mã không được đảo
    ngược so với thứ tự `hang`. Nếu `du_lieu` không có `danh_sach` (ví dụ `xu_huong`, `ket_qua`),
    không có gì để kiểm, coi là đạt."""
    danh_sach = du_lieu.get("danh_sach")
    if not danh_sach:
        return (True, [])

    thieu = []
    vi_tri = []
    for r in sorted(danh_sach, key=lambda r: r["hang"]):
        idx = cau_tra_loi.find(r["ma"])
        if idx == -1:
            thieu.append(r["ma"])
        else:
            vi_tri.append((r["hang"], idx))

    if thieu:
        return (False, [f"thiếu mã {ma}" for ma in thieu])

    for (hang_a, idx_a), (hang_b, idx_b) in zip(vi_tri, vi_tri[1:]):
        if idx_a > idx_b:
            return (False, [f"sai thứ tự giữa hạng {hang_a} và hạng {hang_b}"])
    return (True, [])


def kiem_tat_dinh(cau_tra_loi: str, du_lieu: dict, cum_tu_chan: list[str]) -> dict:
    """Áp cả ba kiểm cho nhánh JSON số liệu. Trả `{"dat": bool, "ly_do": [...]}`."""
    dat_so, loi_so = moi_so_khop_json(cau_tra_loi, du_lieu)
    dat_ma, loi_ma = ma_dung_thu_tu_hang(cau_tra_loi, du_lieu)
    dat_chan, cum_tim_thay = khong_chua_cum_tu_chan(cau_tra_loi, cum_tu_chan)

    ly_do = []
    if not dat_so:
        ly_do.append(f"số không khớp JSON: {loi_so}")
    if not dat_ma:
        ly_do.append(f"mã sai: {loi_ma}")
    if not dat_chan:
        ly_do.append(f"chứa cụm từ chặn: {cum_tim_thay}")

    return {"dat": dat_so and dat_ma and dat_chan, "ly_do": ly_do}


def kiem_tat_dinh_rag(cau_tra_loi: str, cum_tu_chan: list[str]) -> dict:
    """Nhánh RIÊNG cho câu trả lời RAG (`tra_cuu_phuong_phap`): câu trả lời là văn bản tự do, CHỈ
    kiểm không chứa cụm từ chặn — không áp kiểm số/mã vì không phải JSON số liệu."""
    dat_chan, cum_tim_thay = khong_chua_cum_tu_chan(cau_tra_loi, cum_tu_chan)
    ly_do = [] if dat_chan else [f"chứa cụm từ chặn: {cum_tim_thay}"]
    return {"dat": dat_chan, "ly_do": ly_do}
