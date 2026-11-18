"""Test cho `vnrank/system/deterministic_checks.py` (bước S12, D110): ba kiểm cho nhánh JSON số
liệu, và nhánh riêng cho câu trả lời RAG. Dữ liệu tổng hợp nhỏ, không gọi LLM thật."""
from __future__ import annotations

from vnrank.system import deterministic_checks as dc

XEP_HANG_MAU = {
    "ngay_du_lieu": "2025-07-15", "h": 5, "K": 2, "cau_hinh": "A1", "ma_bam": "abcd1234",
    "danh_sach": [
        {"hang": 1, "ma": "AAA", "san": "HOSE", "diem_xep_hang": 0.123456, "diem_xu_huong": 0.2},
        {"hang": 2, "ma": "BBB", "san": "HNX", "diem_xep_hang": 0.1, "diem_xu_huong": -0.05},
    ],
}

XU_HUONG_MAU = {
    "ngay_du_lieu": "2025-07-15", "ma": "AAA", "h": 5, "ma_bam": "abcd1234",
    "p_tang": 0.4512, "p_ngang": 0.3, "p_giam": 0.2488, "diem_xu_huong": 0.2024, "hang": 3,
}


# --- khong_chua_cum_tu_chan ---------------------------------------------------------------------


def test_khong_chua_cum_tu_chan_phat_hien_dung_cum_tu():
    dat, loat = dc.khong_chua_cum_tu_chan("Bạn nên mua mã này ngay", ["nên mua"])
    assert not dat
    assert loat == ["nên mua"]


def test_khong_chua_cum_tu_chan_khong_phan_biet_hoa_thuong_va_dau():
    dat, _ = dc.khong_chua_cum_tu_chan("Nên Mua mã này", ["nên mua"])
    assert not dat


def test_khong_chua_cum_tu_chan_dat_khi_khong_co_cum_nao():
    dat, loat = dc.khong_chua_cum_tu_chan("Đây là thông tin xếp hạng.", ["nên mua", "nên bán"])
    assert dat
    assert loat == []


# --- moi_so_khop_json ---------------------------------------------------------------------------


def test_moi_so_khop_json_dat_khi_so_dung_tu_json():
    cau = "Mã AAA đứng hạng 1 với điểm xếp hạng 0,1235 và điểm xu hướng 0,20."
    dat, loi = dc.moi_so_khop_json(cau, XEP_HANG_MAU)
    assert dat, loi


def test_moi_so_khop_json_phat_hien_so_bia_dat():
    cau = "Mã AAA có điểm xếp hạng 0,9999, một con số bịa đặt."
    dat, loi = dc.moi_so_khop_json(cau, XEP_HANG_MAU)
    assert not dat
    assert "0,9999" in loi


def test_moi_so_khop_json_bo_qua_ma_bam():
    cau = f"Mã băm cấu hình là {XEP_HANG_MAU['ma_bam']}."
    dat, loi = dc.moi_so_khop_json(cau, XEP_HANG_MAU)
    assert dat, loi


def test_moi_so_khop_json_doi_chieu_ngay_dinh_dang_khac():
    cau = "Ngày dữ liệu là 15/07/2025."
    dat, loi = dc.moi_so_khop_json(cau, XEP_HANG_MAU)
    assert dat, loi


def test_moi_so_khop_json_phat_hien_ngay_sai():
    cau = "Ngày dữ liệu là 16/07/2025."
    dat, loi = dc.moi_so_khop_json(cau, XEP_HANG_MAU)
    assert not dat


def test_moi_so_khop_json_cho_xu_huong_2_chu_so():
    cau = "P(tăng) là 0,45, P(đi ngang) là 0,30, P(giảm) là 0,25, điểm xu hướng 0,20."
    dat, loi = dc.moi_so_khop_json(cau, XU_HUONG_MAU)
    assert dat, loi


# --- ma_dung_thu_tu_hang --------------------------------------------------------------------


def test_ma_dung_thu_tu_hang_dat_khi_dung_thu_tu():
    cau = "Hạng 1 là AAA, hạng 2 là BBB."
    dat, loi = dc.ma_dung_thu_tu_hang(cau, XEP_HANG_MAU)
    assert dat, loi


def test_ma_dung_thu_tu_hang_phat_hien_thieu_ma():
    cau = "Hạng 1 là AAA."
    dat, loi = dc.ma_dung_thu_tu_hang(cau, XEP_HANG_MAU)
    assert not dat
    assert "BBB" in loi[0]


def test_ma_dung_thu_tu_hang_phat_hien_sai_thu_tu():
    cau = "Hạng 2 là BBB, hạng 1 là AAA."
    dat, loi = dc.ma_dung_thu_tu_hang(cau, XEP_HANG_MAU)
    assert not dat


def test_ma_dung_thu_tu_hang_dat_khi_khong_co_danh_sach():
    dat, loi = dc.ma_dung_thu_tu_hang("P(tăng) là 0,45.", XU_HUONG_MAU)
    assert dat
    assert loi == []


# --- kiem_tat_dinh / kiem_tat_dinh_rag ------------------------------------------------------


def test_kiem_tat_dinh_dat_khi_dung_het():
    cau = "Mã AAA đứng hạng 1 (điểm xếp hạng 0,1235, điểm xu hướng 0,20), mã BBB đứng hạng 2."
    kq = dc.kiem_tat_dinh(cau, XEP_HANG_MAU, ["nên mua"])
    assert kq["dat"], kq["ly_do"]


def test_kiem_tat_dinh_hong_khi_co_cum_tu_chan():
    cau = "Mã AAA đứng hạng 1, mã BBB đứng hạng 2, nên mua ngay."
    kq = dc.kiem_tat_dinh(cau, XEP_HANG_MAU, ["nên mua"])
    assert not kq["dat"]
    assert any("cụm từ chặn" in ly for ly in kq["ly_do"])


def test_kiem_tat_dinh_rag_chi_kiem_cum_tu_chan():
    kq_dat = dc.kiem_tat_dinh_rag("Universe mỗi ngày lấy 150 mã thanh khoản cao nhất.", ["nên mua"])
    assert kq_dat["dat"]
    kq_hong = dc.kiem_tat_dinh_rag("Bạn nên mua mã này.", ["nên mua"])
    assert not kq_hong["dat"]
