"""Test cho rag.py (truy xuất tăng cường cho chatbot). Dữ liệu tổng hợp nhỏ tự dựng trong
tmp_path, không đọc code/docs/ thật của dự án (giữ test độc lập với nội dung báo cáo có thể
thay đổi)."""
from __future__ import annotations

from vnrank.system.rag import build_index, load_corpus, search


def _write_doc(path, content):
    path.write_text(content, encoding="utf-8")


def test_load_corpus_splits_by_level2_heading(tmp_path):
    _write_doc(
        tmp_path / "a.md",
        "# Tiêu đề A\n\nĐoạn mở đầu.\n\n## Mục 1\n\nNội dung mục 1.\n\n## Mục 2\n\nNội dung mục 2.\n",
    )
    chunks = load_corpus(tmp_path)
    assert len(chunks) == 3
    assert chunks[0]["section"] == "(mở đầu)"
    assert "Đoạn mở đầu" in chunks[0]["text"]
    assert chunks[1]["section"].startswith("Mục 1")
    assert "Nội dung mục 1" in chunks[1]["text"]
    assert chunks[2]["section"].startswith("Mục 2")


def test_load_corpus_reads_multiple_files(tmp_path):
    _write_doc(tmp_path / "a.md", "## X\n\nNội dung X.\n")
    _write_doc(tmp_path / "b.md", "## Y\n\nNội dung Y.\n")
    chunks = load_corpus(tmp_path)
    sources = {c["source"] for c in chunks}
    assert sources == {"a.md", "b.md"}


def test_search_finds_relevant_chunk_by_distinctive_terms(tmp_path):
    _write_doc(
        tmp_path / "leakage.md",
        "## Kiểm tra rò rỉ\n\nXáo trộn nhãn và đo Rank IC trên tập validation, kết quả gần 0, "
        "không có dấu hiệu rò rỉ dữ liệu.\n\n"
        "## Kiểm tra giá\n\nGiá đóng cửa đã điều chỉnh cổ tức, xác nhận bằng sự kiện cổ tức HPG.\n",
    )
    chunks = load_corpus(tmp_path)
    index = build_index(chunks)

    results = search("rò rỉ dữ liệu Rank IC", index, top_k=1)
    assert len(results) == 1
    assert results[0]["section"].startswith("Kiểm tra rò rỉ")
    assert results[0]["score"] > 0


def test_search_returns_empty_when_no_shared_terms(tmp_path):
    _write_doc(tmp_path / "a.md", "## Mục\n\nMèo và chó là động vật nuôi phổ biến.\n")
    chunks = load_corpus(tmp_path)
    index = build_index(chunks)
    results = search("phương trình vi phân đạo hàm riêng", index, top_k=3)
    assert results == []


def test_search_ranks_more_relevant_chunk_higher(tmp_path):
    _write_doc(
        tmp_path / "a.md",
        "## Momentum\n\nMomentum là lợi suất quá khứ. Momentum dùng làm đặc trưng. Momentum "
        "có ba cửa sổ.\n\n"
        "## Khác\n\nMột câu nhắc tới momentum một lần, còn lại nói về chủ đề khác hẳn.\n",
    )
    chunks = load_corpus(tmp_path)
    index = build_index(chunks)
    results = search("momentum", index, top_k=2)
    assert len(results) == 2
    assert results[0]["section"].startswith("Momentum")
    assert results[0]["score"] > results[1]["score"]


def test_build_index_vectors_are_unit_normalized(tmp_path):
    _write_doc(tmp_path / "a.md", "## Mục\n\nMột hai ba bốn năm sáu bảy.\n")
    chunks = load_corpus(tmp_path)
    index = build_index(chunks)
    import math

    norm = math.sqrt(sum(w * w for w in index["vectors"][0].values()))
    assert norm == 1.0 or abs(norm - 1.0) < 1e-9
