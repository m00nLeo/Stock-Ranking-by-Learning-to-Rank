"""Test cho `vnrank/system/messages.py` (bước S12, D110): câu mẫu cố định và bảng hiển thị dự
phòng dựng từ JSON."""
from __future__ import annotations

from vnrank.system import messages as msg


def test_cau_tu_choi_khong_rong_va_neu_ro_khong_khuyen_nghi():
    # CAU_TU_CHOI la van ban CO DINH do du an viet (khong phai LLM sinh ra), nen duoc phep nhac
    # cum "khuyen nghi mua/ban/nam giu" trong cau PHU DINH ("khong dua ra..."); danh sach cum tu
    # chan (deterministic_checks.khong_chua_cum_tu_chan) chi ap dung cho van ban LLM sinh, khong
    # ap cho chinh cac cau mau nay.
    assert msg.CAU_TU_CHOI
    assert "không đưa ra khuyến nghị" in msg.CAU_TU_CHOI.lower()


def test_cau_canh_bao_co_dung_ngay_va_khong_khuyen_nghi():
    cau = msg.cau_canh_bao("2025-07-15")
    assert "2025-07-15" in cau
    assert "không phải khuyến nghị" in cau.lower()


def test_cum_tu_chan_khong_rong_va_la_chu_thuong():
    assert len(msg.CUM_TU_CHAN) >= 10
    assert all(c == c.lower() for c in msg.CUM_TU_CHAN)


def test_bang_mau_xep_hang_dung_du_truong():
    du_lieu = {
        "ngay_du_lieu": "2025-07-15", "h": 5, "K": 2, "cau_hinh": "A1", "ma_bam": "abcd1234",
        "danh_sach": [
            {"hang": 1, "ma": "AAA", "san": "HOSE", "diem_xep_hang": 0.123456, "diem_xu_huong": 0.2},
            {"hang": 2, "ma": "BBB", "san": "HNX", "diem_xep_hang": 0.1, "diem_xu_huong": -0.05},
        ],
    }
    text = msg.bang_mau_tu_json("xep_hang", du_lieu)
    assert "AAA" in text and "BBB" in text
    assert text.index("AAA") < text.index("BBB")
    assert "0,1235" in text  # 0.123456 lam tron 4 chu so


def test_bang_mau_xu_huong_dung_du_truong():
    du_lieu = {
        "ngay_du_lieu": "2025-07-15", "ma": "AAA", "h": 5, "ma_bam": "abcd1234",
        "p_tang": 0.4512, "p_ngang": 0.3, "p_giam": 0.2488, "diem_xu_huong": 0.2024, "hang": 3,
    }
    text = msg.bang_mau_tu_json("xu_huong", du_lieu)
    assert "AAA" in text
    assert "0,20" in text  # diem_xu_huong lam tron 2 chu so


def test_bang_mau_ket_qua_dung_du_truong():
    du_lieu = {
        "h": 5, "K": 10, "muc_chi_phi": "cơ sở", "tu_ngay": "2025-04-01", "den_ngay": "2026-10-02",
        "ma_bam": "abcd1234",
        "bang": [
            {"cau_hinh": "A1", "rank_ic": -0.04, "precision_at_k": 0.45, "loi_suat_nam_hoa_rong": -0.1,
             "sharpe_rong": -0.28, "mdd": -0.2, "turnover": 0.5},
        ],
    }
    text = msg.bang_mau_tu_json("ket_qua", du_lieu)
    assert "A1" in text
    assert "cơ sở" in text
