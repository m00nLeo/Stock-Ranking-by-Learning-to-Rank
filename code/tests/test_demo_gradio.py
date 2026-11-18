"""Test cho `vnrank/system/demo_gradio.py` (giao diện Gradio S2): quy tắc hiển thị (làm tròn,
chú thích xu hướng chưa hiệu chuẩn), truyền lỗi từ ba hàm đọc nguyên văn, dựng app thật (không giả
lập `gradio`, cần `gradio`/`huggingface_hub` cài đúng trong `code/.venv`)."""
from __future__ import annotations

import gradio as gr
import pytest

from vnrank.system import demo_gradio as dg
from vnrank.system import read_functions as rf


@pytest.fixture(scope="module")
def _artefact_mau():
    return rf.tao_du_lieu_mau(seed=42)


@pytest.fixture(autouse=True)
def _nap_mau(_artefact_mau):
    rf.nap_artefact(_artefact_mau)
    yield
    rf._xoa_artefact()


# --- build_app: dựng được giao diện Gradio thật ------------------------------------------------


def test_build_app_returns_gradio_blocks():
    app = dg.build_app()
    assert isinstance(app, gr.Blocks)


def test_build_app_default_loads_mock_data():
    # Mac dinh (du_lieu_that=False) phai goi nap_du_lieu_mau, GIU NGUYEN hanh vi S2 cu — kiem bang
    # cach doc lai artefact hien hanh ngay sau build_app() (mau giai 30 ma, xem tao_du_lieu_mau).
    dg.build_app()
    assert len(rf._artefact_hien_hanh()["symbols"]) == 30


def test_build_app_du_lieu_that_calls_nap_artefact_that(monkeypatch):
    # du_lieu_that=True phai goi dung artifact_loader.nap_artefact_that() (S3/S4/S8, D113) —
    # gia lap de khong chay lai walk-forward that trong unit test.
    import vnrank.system.artifact_loader as al

    goi = {"count": 0}

    def _gia_lap():
        goi["count"] += 1
        rf.nap_artefact(rf.tao_du_lieu_mau(seed=1))

    monkeypatch.setattr(al, "nap_artefact_that", _gia_lap)
    app = dg.build_app(du_lieu_that=True)
    assert isinstance(app, gr.Blocks)
    assert goi["count"] == 1


# --- _bang_xep_hang: làm tròn đúng quy tắc hiển thị (4 chữ số, diem_xu_huong 2 chữ số) ----------


def test_bang_xep_hang_rounds_display_columns():
    header, df = dg._bang_xep_hang("2025-07-15", 5, 10)
    assert "Ngày dữ liệu" in header
    assert len(df) == 10
    for v in df["diem_xu_huong"]:
        assert round(v, 2) == v
    for v in df["diem_xep_hang"]:
        assert round(v, 4) == v


def test_bang_xep_hang_passes_through_error_verbatim():
    header, df = dg._bang_xep_hang("2025-07-15", 7, 10)
    assert header == "Lỗi: h phải thuộc {5, 10, 21}"
    assert df.empty


# --- _bang_xu_huong: chú thích xác suất chưa hiệu chuẩn, làm tròn 2 chữ số ----------------------


def test_bang_xu_huong_rounds_diem_xu_huong_to_two_decimals():
    ma = rf.xep_hang("2025-07-15", 5, 10)["danh_sach"][0]["ma"]
    header, df = dg._bang_xu_huong(ma, "2025-07-15", 5)
    assert f"Mã: {ma}" in header
    assert round(df["diem_xu_huong"].iloc[0], 2) == df["diem_xu_huong"].iloc[0]


def test_bang_xu_huong_passes_through_error_verbatim():
    header, df = dg._bang_xu_huong("ZZZ", "2025-07-15", 5)
    assert "không thuộc tập mã" in header
    assert df.empty


# --- _bang_ket_qua: ba cấu hình, sáu thước đo, làm tròn 4 chữ số --------------------------------


def test_bang_ket_qua_rounds_all_metric_columns():
    header, df = dg._bang_ket_qua(5, 10, "cơ sở")
    assert "Giai đoạn test" in header
    assert len(df) == 3
    for col in ("rank_ic", "precision_at_k", "loi_suat_nam_hoa_rong", "sharpe_rong", "mdd", "turnover"):
        for v in df[col]:
            assert round(v, 4) == v


def test_bang_ket_qua_passes_through_error_verbatim():
    header, df = dg._bang_ket_qua(5, 10, "trung bình")
    assert "muc_chi_phi phải là một trong" in header
    assert df.empty
