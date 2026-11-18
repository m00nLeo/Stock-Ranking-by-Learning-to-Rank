"""Test cho `vnrank/system/chatbot.py` (bước S12, D110): đủ 7 bước ở `system_architecture.md`
mục 5, dùng CLIENT GIẢ LẬP (không gọi API Gemini thật — đúng quy ước mục 7.2: unit test không cần
LLM, chỉ script riêng mới chạy thật 40 câu). Artefact dùng `tao_du_lieu_mau` (dữ liệu mẫu giả, đã
có từ S2), không phụ thuộc dữ liệu thật của dự án.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from vnrank.system import chatbot as cb
from vnrank.system import messages as msg
from vnrank.system import read_functions as rf


class _FakeFunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class _FakeContent:
    """Giả lập `candidates[0].content` — chatbot.py chỉ chuyển tiếp nguyên object này sang lượt
    gọi thứ hai, không đọc bên trong, nên một placeholder bất kỳ là đủ cho test."""


class _FakeResponse:
    def __init__(self, function_calls=None, text=None):
        self.function_calls = function_calls or []
        self.text = text
        self.candidates = [type("C", (), {"content": _FakeContent()})()]


class _FakeModels:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.models = _FakeModels(responses)


@pytest.fixture(scope="module", autouse=True)
def _artefact_mau():
    rf.nap_artefact(rf.tao_du_lieu_mau(seed=42))
    yield
    rf._xoa_artefact()


def _log_path(tmp_path) -> Path:
    return tmp_path / "chatbot_log.jsonl"


# --- Bước 2: không gọi hàm nào -> từ chối -------------------------------------------------------


def test_khong_goi_ham_tra_cau_tu_choi(tmp_path):
    client = _FakeClient([_FakeResponse(function_calls=[], text="(không nên hiển thị)")])
    ra = cb.tra_loi("Lãi suất tiết kiệm hiện nay là bao nhiêu?", client=client, log_path=_log_path(tmp_path))
    assert ra["dau_ra_hien_thi"] == msg.CAU_TU_CHOI
    assert ra["loi_goi_ham"] is None
    assert client.models.calls and len(client.models.calls) == 1  # khong goi LLM lan hai


def test_ten_ham_la_gia_tra_cau_tu_choi(tmp_path):
    goi = _FakeFunctionCall("ham_khong_ton_tai", {})
    client = _FakeClient([_FakeResponse(function_calls=[goi])])
    ra = cb.tra_loi("Câu hỏi bất kỳ", client=client, log_path=_log_path(tmp_path))
    assert ra["dau_ra_hien_thi"] == msg.CAU_TU_CHOI


# --- Bước 3: lỗi tham số -> trả thông báo mẫu, KHÔNG gọi LLM lần hai ----------------------------


def test_loi_tham_so_tra_thong_bao_mau_khong_goi_llm_lan_hai(tmp_path):
    goi = _FakeFunctionCall("xep_hang", {"ngay": "2025-07-15", "h": 7, "K": 10})  # h=7 khong hop le
    client = _FakeClient([_FakeResponse(function_calls=[goi])])
    ra = cb.tra_loi("Top 10 mã ngày 15/7 khung 7 phiên?", client=client, log_path=_log_path(tmp_path))
    assert "loi" in ra["json_ham"]
    assert ra["dau_ra_hien_thi"] == ra["json_ham"]["loi"]
    assert len(client.models.calls) == 1


# --- Bước 4-6: goi ham thanh cong, LLM tra loi dat kiem -> hien thi + canh bao ------------------


def test_xep_hang_thanh_cong_va_dat_kiem_gan_canh_bao(tmp_path):
    goi = _FakeFunctionCall("xep_hang", {"ngay": "2025-09-01", "h": 5, "K": 5})
    kq_that = rf.xep_hang("2025-09-01", 5, 5)
    assert "loi" not in kq_that
    cau_dung = "Top 5 mã hạng cao nhất là " + ", ".join(r["ma"] for r in kq_that["danh_sach"]) + "."
    client = _FakeClient([
        _FakeResponse(function_calls=[goi]),
        _FakeResponse(text=cau_dung),
    ])
    ra = cb.tra_loi("Top 5 mã ngày 1/9/2025 khung 5 phiên?", client=client, log_path=_log_path(tmp_path))
    assert ra["ket_qua_kiem"]["dat"], ra["ket_qua_kiem"]
    assert not ra["roi_ve_bang_mau"]
    assert cau_dung in ra["dau_ra_hien_thi"]
    assert "Không phải khuyến nghị" in ra["dau_ra_hien_thi"]
    assert kq_that["ngay_du_lieu"] in ra["dau_ra_hien_thi"]
    assert len(client.models.calls) == 2


def test_xep_hang_llm_bia_so_lieu_roi_ve_bang_mau(tmp_path):
    goi = _FakeFunctionCall("xep_hang", {"ngay": "2025-09-01", "h": 5, "K": 5})
    client = _FakeClient([
        _FakeResponse(function_calls=[goi]),
        _FakeResponse(text="Mã đứng đầu có điểm xếp hạng 999,9999, một con số bịa đặt."),
    ])
    ra = cb.tra_loi("Top 5 mã ngày 1/9/2025 khung 5 phiên?", client=client, log_path=_log_path(tmp_path))
    assert not ra["ket_qua_kiem"]["dat"]
    assert ra["roi_ve_bang_mau"] is True
    assert "999,9999" not in ra["dau_ra_hien_thi"]  # bang mau dung thang tu JSON, khong con cau bia


def test_xep_hang_llm_khuyen_nghi_roi_ve_bang_mau(tmp_path):
    goi = _FakeFunctionCall("xep_hang", {"ngay": "2025-09-01", "h": 5, "K": 5})
    client = _FakeClient([
        _FakeResponse(function_calls=[goi]),
        _FakeResponse(text="Bạn nên mua các mã này ngay."),
    ])
    ra = cb.tra_loi("Top 5 mã ngày 1/9/2025 khung 5 phiên?", client=client, log_path=_log_path(tmp_path))
    assert ra["roi_ve_bang_mau"] is True
    assert "nên mua" not in ra["dau_ra_hien_thi"].lower()


# --- Nhánh RAG: chỉ kiểm cụm từ chặn, không gắn câu cảnh báo "ngày dữ liệu" ----------------------


def test_rag_thanh_cong_khong_gan_canh_bao_ngay_du_lieu(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "tra_cuu_phuong_phap", lambda cau_hoi, top_k=3: [
        {"source": "3_4_universe.md", "section": "Universe mỗi ngày", "text": "Universe lấy 150 mã thanh khoản cao nhất mỗi ngày."}
    ])
    goi = _FakeFunctionCall("tra_cuu_phuong_phap", {"cau_hoi": "Universe lấy bao nhiêu mã mỗi ngày?"})
    client = _FakeClient([
        _FakeResponse(function_calls=[goi]),
        _FakeResponse(text="Universe lấy 150 mã có thanh khoản cao nhất mỗi ngày."),
    ])
    ra = cb.tra_loi("Universe lấy bao nhiêu mã mỗi ngày?", client=client, log_path=_log_path(tmp_path))
    assert ra["ket_qua_kiem"]["dat"]
    assert "ngày dữ liệu" not in ra["dau_ra_hien_thi"].lower()


def test_rag_khong_tim_thay_doan_lien_quan(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "tra_cuu_phuong_phap", lambda cau_hoi, top_k=3: [])
    goi = _FakeFunctionCall("tra_cuu_phuong_phap", {"cau_hoi": "Một câu hỏi không liên quan gì"})
    client = _FakeClient([_FakeResponse(function_calls=[goi])])
    ra = cb.tra_loi("Một câu hỏi không liên quan gì", client=client, log_path=_log_path(tmp_path))
    assert ra["dau_ra_hien_thi"] == msg.CAU_RAG_KHONG_TIM_THAY
    assert len(client.models.calls) == 1


def test_rag_llm_khuyen_nghi_roi_ve_doan_truy_xuat_tho(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "tra_cuu_phuong_phap", lambda cau_hoi, top_k=3: [
        {"source": "x.md", "section": "y", "text": "nội dung gốc"}
    ])
    goi = _FakeFunctionCall("tra_cuu_phuong_phap", {"cau_hoi": "abc"})
    client = _FakeClient([
        _FakeResponse(function_calls=[goi]),
        _FakeResponse(text="Bạn nên mua mã này."),
    ])
    ra = cb.tra_loi("abc", client=client, log_path=_log_path(tmp_path))
    assert ra["roi_ve_bang_mau"] is True
    assert "nội dung gốc" in ra["dau_ra_hien_thi"]
    assert "nên mua" not in ra["dau_ra_hien_thi"].lower()


# --- Bước 7: ghi log JSONL -----------------------------------------------------------------


def test_ghi_log_mot_dong_moi_luot(tmp_path):
    log_path = _log_path(tmp_path)
    client1 = _FakeClient([_FakeResponse(function_calls=[])])
    client2 = _FakeClient([_FakeResponse(function_calls=[])])
    cb.tra_loi("câu 1", client=client1, log_path=log_path)
    cb.tra_loi("câu 2", client=client2, log_path=log_path)
    dong = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(dong) == 2
    import json
    d0 = json.loads(dong[0])
    assert set(d0.keys()) == {
        "cau_hoi", "loi_goi_ham", "json_ham", "tra_loi_llm", "ket_qua_kiem",
        "dau_ra_hien_thi", "phien_ban_llm", "roi_ve_bang_mau",
    }
    assert d0["phien_ban_llm"] == cb.TEN_MODEL
