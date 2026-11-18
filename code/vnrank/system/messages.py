"""Câu chữ cố định cho chatbot (bước S12, `system_architecture.md` mục 5): câu từ chối, câu cảnh
báo, danh sách cụm từ chặn khuyến nghị mua/bán/nắm giữ, và bảng hiển thị dự phòng dựng thẳng từ
JSON khi bước kiểm tất định (`deterministic_checks.py`) thất bại. KHÔNG do LLM sinh ra — mọi câu
ở đây là văn bản cố định do mã gắn, đúng tinh thần "LLM không tự sinh số liệu" của dự án.
"""

from __future__ import annotations

# --- Câu từ chối mẫu (bước 2, mục 5): khi LLM không gọi hàm nào -------------------------------

CAU_TU_CHOI = (
    "Xin lỗi, câu hỏi này nằm ngoài phạm vi hệ thống. Hệ thống chỉ trả lời về xếp hạng cổ phiếu, "
    "dự báo xu hướng, kết quả nghiên cứu trên tập kiểm tra đã đóng băng, và phương pháp đã dùng "
    "trong luận văn. Hệ thống không đưa ra khuyến nghị mua, bán hay nắm giữ chứng khoán."
)

# --- Câu cảnh báo cố định (bước 6, mục 5): gắn kèm ngày dữ liệu, chỉ cho câu có hiển thị số liệu
# từ xep_hang/xu_huong/ket_qua (không áp dụng cho câu trả lời RAG — xem docstring deterministic_checks) ---


def cau_canh_bao(ngay_du_lieu: str) -> str:
    """Nguyên văn đã chốt ở `system_architecture.md` mục 5."""
    return (
        f"Đây là kết quả của mô hình nghiên cứu trên dữ liệu lịch sử (ngày dữ liệu: {ngay_du_lieu}), "
        "phục vụ mục đích học thuật. Không phải khuyến nghị mua, bán hay nắm giữ chứng khoán."
    )


# --- Danh sách cụm từ chặn khuyến nghị mua/bán/nắm giữ (bước 5c, mục 5) -----------------------
# Khởi điểm đề xuất ở D110; bổ sung thêm nếu chạy thử bộ 40 câu thấy lọt cụm từ nào.

CUM_TU_CHAN: list[str] = [
    "nên mua",
    "nên bán",
    "khuyến nghị mua",
    "khuyến nghị bán",
    "nắm giữ",
    "chốt lời",
    "cắt lỗ",
    "đáng mua",
    "đáng bán",
    "mua vào",
    "bán ra",
    "mua ngay",
    "bán ngay",
    "tư vấn đầu tư",
]

# --- Câu báo khi RAG không tìm thấy đoạn tài liệu nào liên quan --------------------------------

CAU_RAG_KHONG_TIM_THAY = (
    "Không tìm thấy nội dung liên quan trong các báo cáo kỹ thuật của dự án cho câu hỏi này."
)


# --- Bảng hiển thị dự phòng (bước 5, mục 5): dựng THẲNG từ JSON khi kiểm tất định thất bại -----


def _so(x, chu_so: int = 4) -> str:
    """Định dạng số thực `chu_so` chữ số thập phân, dấu phẩy làm dấu thập phân (quy ước dự án)."""
    return f"{x:.{chu_so}f}".replace(".", ",")


def bang_mau_xep_hang(du_lieu: dict) -> str:
    dong = [
        f"Ngày dữ liệu: {du_lieu['ngay_du_lieu']} · khung {du_lieu['h']} phiên · "
        f"top {du_lieu['K']} · cấu hình {du_lieu['cau_hinh']} · mã băm {du_lieu['ma_bam']}"
    ]
    for r in du_lieu["danh_sach"]:
        dong.append(
            f"{r['hang']}. {r['ma']} ({r['san']}) — điểm xếp hạng {_so(r['diem_xep_hang'])}, "
            f"điểm xu hướng {_so(r['diem_xu_huong'], 2)}"
        )
    return "\n".join(dong)


def bang_mau_xu_huong(du_lieu: dict) -> str:
    return (
        f"Ngày dữ liệu: {du_lieu['ngay_du_lieu']} · mã {du_lieu['ma']} · khung {du_lieu['h']} phiên "
        f"· mã băm {du_lieu['ma_bam']}\n"
        f"P(tăng) = {_so(du_lieu['p_tang'])}, P(đi ngang) = {_so(du_lieu['p_ngang'])}, "
        f"P(giảm) = {_so(du_lieu['p_giam'])}\n"
        f"Điểm xu hướng = {_so(du_lieu['diem_xu_huong'], 2)} · hạng {du_lieu['hang']} trong ngày "
        "(điểm của mô hình, không phải xác suất đã hiệu chuẩn)"
    )


def bang_mau_ket_qua(du_lieu: dict) -> str:
    dong = [
        f"Khung {du_lieu['h']} phiên · K={du_lieu['K']} · mức chi phí {du_lieu['muc_chi_phi']} · "
        f"giai đoạn test {du_lieu['tu_ngay']} đến {du_lieu['den_ngay']} · mã băm {du_lieu['ma_bam']}"
    ]
    for r in du_lieu["bang"]:
        dong.append(
            f"{r['cau_hinh']}: Rank IC {_so(r['rank_ic'])}, precision@K {_so(r['precision_at_k'])}, "
            f"lợi suất năm hóa ròng {_so(r['loi_suat_nam_hoa_rong'])}, Sharpe ròng {_so(r['sharpe_rong'])}, "
            f"MDD {_so(r['mdd'])}, turnover {_so(r['turnover'])}"
        )
    return "\n".join(dong)


_BANG_MAU_THEO_HAM = {
    "xep_hang": bang_mau_xep_hang,
    "xu_huong": bang_mau_xu_huong,
    "ket_qua": bang_mau_ket_qua,
}


def bang_mau_tu_json(ten_ham: str, du_lieu: dict) -> str:
    """Dựng bảng hiển thị dự phòng từ JSON thật của `ten_ham` (`xep_hang`/`xu_huong`/`ket_qua`),
    dùng khi bước kiểm tất định thất bại (bước 5, mục 5) — thay cho câu của LLM."""
    return _BANG_MAU_THEO_HAM[ten_ham](du_lieu)
