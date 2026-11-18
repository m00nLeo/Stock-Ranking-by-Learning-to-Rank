"""Truy xuất tăng cường cho chatbot (RAG — Retrieval-Augmented Generation): cho phép chatbot
tra cứu nội dung các báo cáo kỹ thuật đã có trong `code/docs/` (sản phẩm của các bước trước),
để trả lời câu hỏi về PHƯƠNG PHÁP và KẾT QUẢ đã làm (khác ba hàm đọc `xep_hang`/`xu_huong`/
`ket_qua` vốn chỉ trả số liệu, không giải thích phương pháp).

Thiết kế: chỉ dùng list/dict thuần Python và một ít toán học cơ bản (TF-IDF, cosine) tự cài,
không phụ thuộc scikit-learn hay bất kỳ thư viện embedding/vector-DB nào — nhất quán với cách
`leakage_checks.py`/`harness.py` đã làm, và tránh phụ thuộc nặng khó cài trên môi trường cục bộ
(xem ghi chú 4.3 về scikit-learn/lightgbm/xgboost). Có thể nâng cấp sang embedding ngữ nghĩa sau
nếu truy xuất từ khóa (lexical) không đủ tốt — đây là lựa chọn "cần và đủ" cho quy mô kho tài
liệu hiện tại (một vài chục file markdown).

Quy trình (ba bước RAG kinh điển: nạp — đánh chỉ mục — truy xuất):
1. `load_corpus`: đọc toàn bộ file .md trong một thư mục, cắt theo từng mục "## " (heading cấp 2)
   thành các đoạn (chunk) nhỏ, mỗi đoạn gắn kèm tên file nguồn và tiêu đề mục.
2. `build_index`: tính TF-IDF cho từng đoạn, lưu thành một chỉ mục tra cứu được.
3. `search`: với một câu hỏi, tính TF-IDF của câu hỏi, so cosine với mọi đoạn trong chỉ mục,
   trả về các đoạn liên quan nhất kèm điểm số.

Phần SINH CÂU TRẢ LỜI (đưa đoạn truy xuất được vào một LLM để viết câu trả lời) nằm NGOÀI module
này — module chỉ làm phần truy xuất (retrieval), LLM cụ thể dùng ở chatbot thật sẽ nhận các đoạn
`search` trả về làm ngữ cảnh, không tự bịa nội dung ngoài các đoạn đó (cùng nguyên tắc "LLM không
tự sinh số liệu" đã có ở `system_architecture.md`).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Tách từ đơn giản: chữ (giữ nguyên dấu tiếng Việt) và số, bỏ dấu câu, chuyển thường."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def load_corpus(docs_dir: str | Path) -> list[dict]:
    """Đọc mọi file `.md` trong `docs_dir`, cắt theo heading cấp 2 ("## ...") thành các đoạn.

    Phần trước heading cấp 2 đầu tiên (thường là tiêu đề file và đoạn mở đầu) được giữ làm một
    đoạn riêng. Trả về list[dict] với khóa "source" (tên file), "section" (tiêu đề mục, hoặc
    "(mở đầu)"), "text" (nội dung đoạn, đã bỏ khoảng trắng thừa ở hai đầu).
    """
    chunks = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        content = path.read_text(encoding="utf-8")
        parts = re.split(r"(?m)^## ", content)
        intro = parts[0].strip()
        if intro:
            chunks.append({"source": path.name, "section": "(mở đầu)", "text": intro})
        for part in parts[1:]:
            lines = part.split("\n", 1)
            title = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            if body:
                chunks.append({"source": path.name, "section": title, "text": body})
    return chunks


def build_index(chunks: list[dict]) -> dict:
    """Tính TF-IDF cho từng đoạn trong `chunks`. Trả về dict {"chunks", "idf", "vectors"}:
    "vectors" là list song song với `chunks`, mỗi phần tử là dict {từ: trọng số TF-IDF đã chuẩn
    hóa về vector đơn vị (norm 1), để cosine similarity chỉ còn là tích vô hướng}."""
    n = len(chunks)
    doc_freq: Counter = Counter()
    token_lists = []
    for c in chunks:
        tokens = _tokenize(c["text"])
        token_lists.append(tokens)
        doc_freq.update(set(tokens))

    idf = {term: math.log((1 + n) / (1 + df)) + 1.0 for term, df in doc_freq.items()}

    vectors = []
    for tokens in token_lists:
        tf = Counter(tokens)
        weights = {term: count * idf[term] for term, count in tf.items()}
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        vectors.append({term: w / norm for term, w in weights.items()})

    return {"chunks": chunks, "idf": idf, "vectors": vectors}


def _query_vector(query: str, idf: dict) -> dict:
    tokens = _tokenize(query)
    tf = Counter(tokens)
    weights = {term: count * idf[term] for term, count in tf.items() if term in idf}
    norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
    return {term: w / norm for term, w in weights.items()}


def _cosine(a: dict, b: dict) -> float:
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    return sum(w * longer.get(term, 0.0) for term, w in shorter.items())


def search(query: str, index: dict, top_k: int = 3) -> list[dict]:
    """Truy vấn `index` (từ `build_index`), trả về tối đa `top_k` đoạn có điểm cosine TF-IDF cao
    nhất với `query`, mỗi kết quả kèm "score". Bỏ qua đoạn có điểm 0 (không chung từ khóa nào)."""
    qvec = _query_vector(query, index["idf"])
    scored = []
    for chunk, vec in zip(index["chunks"], index["vectors"]):
        score = _cosine(qvec, vec)
        if score > 0:
            scored.append({**chunk, "score": score})
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:top_k]
