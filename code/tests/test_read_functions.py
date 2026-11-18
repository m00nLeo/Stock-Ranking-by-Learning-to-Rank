"""Test cho `vnrank/system/read_functions.py` (ba hàm đọc của hệ thống minh họa S2).

Dùng dữ liệu mẫu giả (`nap_du_lieu_mau`) nên test không phụ thuộc artefact thật hay mô hình đã
huấn luyện; chỉ kiểm hợp đồng JSON ở `system_architecture.md` mục 4 và thứ tự kiểm tham số.
"""
from __future__ import annotations

import pytest

from vnrank.system import read_functions as rf


@pytest.fixture(scope="module")
def _artefact_mau():
    """Sinh artefact mẫu một lần cho cả module (sinh dữ liệu giả khá chậm)."""
    return rf.tao_du_lieu_mau(seed=42)


@pytest.fixture(autouse=True)
def _nap_mau(_artefact_mau):
    """Mỗi test bắt đầu với artefact mẫu đã nạp, kết thúc thì xóa để test độc lập."""
    rf.nap_artefact(_artefact_mau)
    yield
    rf._xoa_artefact()


def _la_loi(kq) -> bool:
    return isinstance(kq, dict) and set(kq.keys()) == {"loi"}


# --- Hằng số đọc từ config.yaml (không lặp lại trong code) ---


def test_horizons_from_config():
    assert rf._HORIZONS == [5, 10, 21]


def test_K_hop_le_la_hop_cua_K_va_K_sensitivity():
    assert rf._K_HOP_LE == [5, 10, 20]


def test_muc_chi_phi_la_ba_nhan_chuoi_theo_dac_ta():
    assert rf._MUC_CHI_PHI == ["thấp", "cơ sở", "cao"]


# --- xep_hang ---


def test_xep_hang_tra_du_K_phan_tu_theo_hang_tang_dan():
    kq = rf.xep_hang("2025-07-15", 5, 10)
    assert "loi" not in kq
    assert kq["h"] == 5
    assert kq["K"] == 10
    assert kq["cau_hinh"] == "A1"
    assert len(kq["danh_sach"]) == 10
    hang = [d["hang"] for d in kq["danh_sach"]]
    assert hang == list(range(1, 11))
    for d in kq["danh_sach"]:
        assert set(d.keys()) == {"hang", "ma", "san", "diem_xep_hang", "diem_xu_huong"}
        assert d["san"] in {"HOSE", "HNX"}


def test_xep_hang_ngay_du_lieu_khong_sau_ngay_asof():
    kq = rf.xep_hang("2025-07-15", 5, 10)
    assert kq["ngay_du_lieu"] <= "2025-07-15"


def test_xep_hang_ma_bam_8_ky_tu():
    assert len(rf.xep_hang("2025-07-15", 5, 10)["ma_bam"]) == 8


@pytest.mark.parametrize(
    "ngay,h,K,loi_con",
    [
        ("15/07/2025", 5, 10, "ngay phải có dạng YYYY-MM-DD"),
        ("2020-01-01", 5, 10, "ngay nằm ngoài giai đoạn test"),
        ("2030-01-01", 5, 10, "ngay nằm ngoài giai đoạn test"),
        ("2025-07-15", 7, 10, "h phải thuộc {5, 10, 21}"),
        ("2025-07-15", 5, 15, "K phải thuộc {5, 10, 20}"),
    ],
)
def test_xep_hang_loi(ngay, h, K, loi_con):
    kq = rf.xep_hang(ngay, h, K)
    assert _la_loi(kq)
    assert loi_con in kq["loi"]


def test_xep_hang_loi_ngay_truoc_ngay_quyet_dinh_dau_tien():
    # Ngày đầu giai đoạn test nhưng trước lưới quyết định đầu tiên của h (mục 8.5).
    kq = rf.xep_hang("2025-04-01", 5, 10)
    assert _la_loi(kq)
    assert "trước ngày quyết định đầu tiên" in kq["loi"]


# --- xu_huong ---


def test_xu_huong_tra_cau_truc_dung():
    ma = rf.xep_hang("2025-07-15", 5, 10)["danh_sach"][0]["ma"]
    kq = rf.xu_huong(ma, "2025-07-15", 5)
    assert "loi" not in kq
    assert set(kq.keys()) == {
        "ngay_du_lieu",
        "ma",
        "h",
        "ma_bam",
        "p_tang",
        "p_ngang",
        "p_giam",
        "diem_xu_huong",
        "hang",
    }
    assert kq["ma"] == ma
    assert kq["h"] == 5
    # diem_xu_huong = p_tang - p_giam
    assert abs(kq["diem_xu_huong"] - (kq["p_tang"] - kq["p_giam"])) < 1e-12


def test_xu_huong_hang_trung_voi_xep_hang():
    top = rf.xep_hang("2025-07-15", 5, 10)["danh_sach"][0]
    kq = rf.xu_huong(top["ma"], "2025-07-15", 5)
    assert kq["hang"] == top["hang"] == 1


@pytest.mark.parametrize(
    "ma,ngay,h,loi_con",
    [
        ("", "2025-07-15", 5, "ma phải là chuỗi mã cổ phiếu"),
        (123, "2025-07-15", 5, "ma phải là chuỗi mã cổ phiếu"),
        ("ABC", "not-a-date", 5, "ngay phải có dạng YYYY-MM-DD"),
        ("ABC", "2025-07-15", 7, "h phải thuộc {5, 10, 21}"),
        ("ZZZ", "2025-07-15", 5, "không thuộc tập mã"),
    ],
)
def test_xu_huong_loi(ma, ngay, h, loi_con):
    kq = rf.xu_huong(ma, ngay, h)
    assert _la_loi(kq)
    assert loi_con in kq["loi"]


# --- ket_qua ---


def test_ket_qua_tra_bang_3_cau_hinh_dung_6_thuoc_do():
    kq = rf.ket_qua(5, 10, "cơ sở")
    assert "loi" not in kq
    assert kq["h"] == 5
    assert kq["K"] == 10
    assert kq["muc_chi_phi"] == "cơ sở"
    assert len(kq["bang"]) == 3
    cau_hinh = [b["cau_hinh"] for b in kq["bang"]]
    assert cau_hinh[0] == "A1"
    for b in kq["bang"]:
        assert set(b.keys()) == {
            "cau_hinh",
            "rank_ic",
            "precision_at_k",
            "loi_suat_nam_hoa_rong",
            "sharpe_rong",
            "mdd",
            "turnover",
        }


def test_ket_qua_tu_ngay_den_ngay_tu_artefact():
    kq = rf.ket_qua(5, 10, "cơ sở")
    assert kq["tu_ngay"] == rf.TEST_START
    assert kq["den_ngay"] == rf.TEST_END


@pytest.mark.parametrize(
    "h,K,muc_chi_phi,loi_con",
    [
        (7, 10, "cơ sở", "h phải thuộc {5, 10, 21}"),
        (5, 15, "cơ sở", "K phải thuộc {5, 10, 20}"),
        (5, 10, "trung bình", "muc_chi_phi phải là một trong: thấp, cơ sở, cao"),
    ],
)
def test_ket_qua_loi(h, K, muc_chi_phi, loi_con):
    kq = rf.ket_qua(h, K, muc_chi_phi)
    assert _la_loi(kq)
    assert loi_con in kq["loi"]


# --- Hợp đồng lỗi chung ---


def test_moi_loi_chi_co_dung_mot_khoa_loi():
    # Không có mã lỗi số, không đoán giá trị (mục 4.1).
    kq = rf.xep_hang("2025-07-15", 7, 10)
    assert kq == {"loi": "h phải thuộc {5, 10, 21}"}


def test_ham_tra_loi_khi_chua_nap_artefact():
    rf._xoa_artefact()
    kq = rf.xep_hang("2025-07-15", 5, 10)
    assert _la_loi(kq)
    assert "chưa nạp artefact" in kq["loi"]


# --- Artefact thật một phần (S3, D107): A1/topk_a1 chỉ ở h=5, test_results rỗng ------------------


def _artefact_mot_phan(_artefact_mau) -> dict:
    """Mô phỏng artefact thật của S3: `topk_a1` chỉ còn h=5 (A1 không được xây ở h=10,21),
    `test_results` rỗng (S4 chưa làm) — giữ nguyên `rank_scores`/`trend_scores` của mock."""
    partial = dict(_artefact_mau)
    partial["topk_a1"] = _artefact_mau["topk_a1"][_artefact_mau["topk_a1"]["h"] == 5].reset_index(drop=True)
    partial["test_results"] = _artefact_mau["test_results"].iloc[0:0]
    return partial


def test_xep_hang_loi_ro_rang_khi_khong_co_du_lieu_a1_cho_h(_artefact_mau):
    rf.nap_artefact(_artefact_mot_phan(_artefact_mau))
    kq = rf.xep_hang("2025-07-15", 10, 10)
    assert _la_loi(kq)
    assert "h=10" in kq["loi"]
    assert rf.xep_hang("2025-07-15", 5, 10) == rf.xep_hang("2025-07-15", 5, 10)
    assert "loi" not in rf.xep_hang("2025-07-15", 5, 10)


def test_xu_huong_loi_ro_rang_khi_khong_co_du_lieu_a1_cho_h(_artefact_mau):
    rf.nap_artefact(_artefact_mot_phan(_artefact_mau))
    kq = rf.xu_huong("AAA", "2025-07-15", 21)
    assert _la_loi(kq)
    assert "h=21" in kq["loi"]


def test_ket_qua_loi_ro_rang_khi_test_results_rong(_artefact_mau):
    rf.nap_artefact(_artefact_mot_phan(_artefact_mau))
    kq = rf.ket_qua(5, 10, "cơ sở")
    assert _la_loi(kq)
    assert "S8" in kq["loi"]


def test_ket_qua_loi_ro_rang_khi_test_results_chi_co_h5(_artefact_mau):
    # Mô phỏng artefact thật sau S8 (D109, D113): test_results CHỈ có h=5 (A1 chỉ thật ở h=5,
    # D107) — ket_qua(h=10 hoặc 21, ...) phải báo lỗi rõ ràng, giống xep_hang/xu_huong, thay vì
    # trả bang=[] rỗng (phá hợp đồng "mảng 3 phần tử").
    partial = dict(_artefact_mau)
    partial["test_results"] = _artefact_mau["test_results"][_artefact_mau["test_results"]["h"] == 5].reset_index(drop=True)
    rf.nap_artefact(partial)

    kq10 = rf.ket_qua(10, 10, "cơ sở")
    assert _la_loi(kq10)
    assert "h=10" in kq10["loi"]

    kq21 = rf.ket_qua(21, 10, "cơ sở")
    assert _la_loi(kq21)
    assert "h=21" in kq21["loi"]

    kq5 = rf.ket_qua(5, 10, "cơ sở")
    assert "loi" not in kq5
    assert len(kq5["bang"]) == 3


# --- Tái lập (H2) ---


def test_tao_du_lieu_mau_tat_dinh_cung_seed():
    a1 = rf.tao_du_lieu_mau(seed=42)
    a2 = rf.tao_du_lieu_mau(seed=42)
    assert a1["rank_scores"].equals(a2["rank_scores"])
    assert a1["test_results"].equals(a2["test_results"])


def test_cung_dau_vao_cung_dau_ra():
    rf.nap_du_lieu_mau()
    kq1 = rf.ket_qua(21, 20, "cao")
    kq2 = rf.ket_qua(21, 20, "cao")
    assert kq1 == kq2