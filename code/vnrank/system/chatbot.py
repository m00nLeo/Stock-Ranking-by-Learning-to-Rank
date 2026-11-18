"""Chatbot hoàn chỉnh (bước S12, `system_architecture.md` mục 5): LLM qua API có gọi hàm (Google
Gemini, quyết định người dùng D110), BỐN công cụ — ba hàm đọc đã có (`xep_hang`, `xu_huong`,
`ket_qua`, dùng thẳng `read_functions.py`, không sửa) và THÊM `tra_cuu_phuong_phap` (bọc
`rag.search()` trên kho `code/docs/*.md`, trả đoạn văn bản thay vì JSON số liệu — phần mở rộng đã
ghi ở mục 5 "Mở rộng chưa có trong đặc tả gốc", thiết kế khi triển khai S12/D110). Không có bộ nhớ
hội thoại, mỗi câu hỏi độc lập, temperature 0.

**Mã băm/model**: `TEN_MODEL` cố định `gemini-3.8-flash`. [SỬA, D112] Ban đầu chọn `gemini-2.5-flash`
(bản GA ổn định tại thời điểm viết kế hoạch D110/D111) nhưng khi chạy thật lần đầu (D112), API trả
lỗi 404 "model này không còn cấp cho người dùng mới", đề nghị đổi sang `gemini-3.8-flash` — đã xác
nhận model này vẫn hỗ trợ đúng luồng `generate_content(tools=...)` hiện có (không bắt buộc đổi sang
Interactions API) qua tài liệu Google AI chính thức, đổi tên model, CHẠY LẠI THẬT xác nhận hoạt
động đúng. Nếu model này bị khai tử, xác nhận lại trên ai.google.dev trước khi đổi, ghi vào
decision_log.

**Lưu ý triển khai chưa kiểm được bằng test giả lập**: cách gửi kết quả hàm về cho LLM ở lượt gọi
thứ hai (vai trò `role="user"` kèm `Part.from_function_response`, theo tài liệu hiện hành) chỉ có
thể xác nhận đầy đủ khi chạy thật với `GEMINI_API_KEY` — xem `scripts/chay_40_cau_that.py`.

Đủ 7 bước theo mục 5:
1. Gửi câu hỏi + mô tả bốn công cụ + system prompt cho LLM.
2. LLM KHÔNG gọi hàm thì trả câu từ chối mẫu, KHÔNG hiển thị văn bản tự do của LLM.
3. LLM gọi hàm thì chạy hàm thật (`read_functions.py` đã tự kiểm tham số, trả `{"loi": ...}` nếu
   sai — không cần một lớp kiểm tham số riêng ở đây).
4. LLM nhận JSON/đoạn văn bản, viết câu trả lời tiếng Việt CHỈ dựa trên đó (gọi LLM lần hai).
5. Kiểm tất định (`deterministic_checks.py`): hỏng thì hiển thị bảng mẫu (`messages.py`) thay câu
   của LLM.
6. Gắn câu cảnh báo cố định + ngày dữ liệu — CHỈ cho nhánh JSON số liệu thành công (không áp cho
   RAG, vì câu cảnh báo nói về "kết quả của mô hình nghiên cứu", không phải tra cứu phương pháp).
7. Ghi một dòng log JSONL mỗi lượt, CỤC BỘ (file trong `code/data/processed/`, chưa nối Google
   Drive — để lại cho lúc chạy trên Colab thật).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from google import genai
from google.genai import types

from vnrank.system import deterministic_checks as checks
from vnrank.system import messages as msg
from vnrank.system import rag
from vnrank.system import read_functions as rf

CODE_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = CODE_ROOT / "docs"
LOG_PATH = CODE_ROOT / "data" / "processed" / "chatbot_log.jsonl"

TEN_MODEL = "gemini-3.8-flash"

_CAC_HAM = {"xep_hang", "xu_huong", "ket_qua", "tra_cuu_phuong_phap"}

SYSTEM_PROMPT = (
    "Bạn là một trợ lý trả lời câu hỏi về một hệ thống xếp hạng cổ phiếu Việt Nam, cho một luận "
    "văn thạc sĩ. Phạm vi CHỈ gồm: xếp hạng cổ phiếu, dự báo xu hướng, bảng kết quả nghiên cứu "
    "trên tập kiểm tra đã đóng băng, và phương pháp đã dùng trong dự án (qua công cụ "
    "tra_cuu_phuong_phap). Nếu câu hỏi ngoài phạm vi này, KHÔNG gọi công cụ nào cả. "
    "Khi trả lời từ kết quả của một công cụ, CHỈ dùng đúng số liệu hoặc nội dung công cụ trả về, "
    "không tự bịa thêm số liệu nào khác, không đoán hay suy diễn ngoài dữ liệu đã cho. "
    "KHÔNG được đưa ra khuyến nghị mua, bán, hay nắm giữ bất kỳ cổ phiếu nào, và KHÔNG giải thích "
    "lý do vì sao một mã được xếp hạng cao hay thấp. Viết câu trả lời bằng tiếng Việt tự nhiên, "
    "dùng dấu phẩy làm dấu thập phân khi viết số (ví dụ 0,1235)."
)


def _khai_bao_cong_cu() -> types.Tool:
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="xep_hang",
                description=(
                    "Trả về top-K mã cổ phiếu xếp hạng cao nhất tại ngày quyết định gần nhất "
                    "<= ngay, theo một khung thời gian h."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "ngay": types.Schema(type="STRING", description="Ngày dạng YYYY-MM-DD, trong giai đoạn test"),
                        "h": types.Schema(type="INTEGER", description="Khung thời gian, số phiên"),
                        "K": types.Schema(type="INTEGER", description="Số mã muốn lấy từ đầu bảng xếp hạng"),
                    },
                    required=["ngay", "h", "K"],
                ),
            ),
            types.FunctionDeclaration(
                name="xu_huong",
                description=(
                    "Trả về xác suất tăng, đi ngang, giảm của một mã cổ phiếu tại ngày quyết định "
                    "gần nhất <= ngay, theo một khung thời gian h."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "ma": types.Schema(type="STRING", description="Mã cổ phiếu"),
                        "ngay": types.Schema(type="STRING", description="Ngày dạng YYYY-MM-DD, trong giai đoạn test"),
                        "h": types.Schema(type="INTEGER", description="Khung thời gian, số phiên"),
                    },
                    required=["ma", "ngay", "h"],
                ),
            ),
            types.FunctionDeclaration(
                name="ket_qua",
                description=(
                    "Trả về bảng kết quả nghiên cứu trên toàn bộ giai đoạn test đã đóng băng "
                    "(không phải một truy vấn theo ngày), cho một khung thời gian h, số mã K, và "
                    "một mức chi phí giao dịch."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "h": types.Schema(type="INTEGER", description="Khung thời gian, số phiên"),
                        "K": types.Schema(type="INTEGER", description="Số mã trong danh mục"),
                        "muc_chi_phi": types.Schema(
                            type="STRING", enum=["thấp", "cơ sở", "cao"], description="Mức chi phí giao dịch"
                        ),
                    },
                    required=["h", "K", "muc_chi_phi"],
                ),
            ),
            types.FunctionDeclaration(
                name="tra_cuu_phuong_phap",
                description=(
                    "Tìm đoạn văn bản liên quan trong các báo cáo kỹ thuật của dự án, để trả lời "
                    "câu hỏi về PHƯƠNG PHÁP đã dùng (khác với số liệu xếp hạng/xu hướng/kết quả)."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={"cau_hoi": types.Schema(type="STRING", description="Câu hỏi về phương pháp cần tra cứu")},
                    required=["cau_hoi"],
                ),
            ),
        ]
    )


_INDEX_RAG: dict | None = None


def _index_rag(docs_dir: Path = DOCS_DIR) -> dict:
    global _INDEX_RAG
    if _INDEX_RAG is None:
        _INDEX_RAG = rag.build_index(rag.load_corpus(docs_dir))
    return _INDEX_RAG


def tra_cuu_phuong_phap(cau_hoi: str, top_k: int = 3) -> list[dict]:
    """Công cụ RAG thứ tư: bọc `rag.search()` trên kho `code/docs/*.md`."""
    return rag.search(cau_hoi, _index_rag(), top_k=top_k)


def _goi_ham_that(ten_ham: str, tham_so: dict):
    if ten_ham == "xep_hang":
        return rf.xep_hang(tham_so.get("ngay"), tham_so.get("h"), tham_so.get("K"))
    if ten_ham == "xu_huong":
        return rf.xu_huong(tham_so.get("ma"), tham_so.get("ngay"), tham_so.get("h"))
    if ten_ham == "ket_qua":
        return rf.ket_qua(tham_so.get("h"), tham_so.get("K"), tham_so.get("muc_chi_phi"))
    if ten_ham == "tra_cuu_phuong_phap":
        return tra_cuu_phuong_phap(tham_so.get("cau_hoi", ""))
    raise ValueError(f"tên hàm không hợp lệ: {ten_ham}")


def _dinh_dang_rag(doan_truy_xuat: list[dict]) -> str:
    return "\n\n".join(f"[{c['source']} - {c['section']}]\n{c['text']}" for c in doan_truy_xuat)


def _ghi_log(dong: dict, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dong, ensure_ascii=False) + "\n")


def tra_loi(cau_hoi: str, client=None, log_path: Path = LOG_PATH) -> dict:
    """Thực hiện đủ 7 bước ở mục 5 cho MỘT câu hỏi độc lập. Trả về dict đủ 8 trường log
    (`cau_hoi`, `loi_goi_ham`, `json_ham`, `tra_loi_llm`, `ket_qua_kiem`, `dau_ra_hien_thi`,
    `phien_ban_llm`, `roi_ve_bang_mau`) — cũng là dòng được ghi vào `log_path`.

    `client`: đối tượng `google.genai.Client` (hoặc giả lập có cùng giao diện
    `.models.generate_content(model=, contents=, config=)`) — truyền vào để test không cần gọi
    API thật. Mặc định dựng từ biến môi trường `GEMINI_API_KEY`."""
    if client is None:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    dong_log = {
        "cau_hoi": cau_hoi,
        "loi_goi_ham": None,
        "json_ham": None,
        "tra_loi_llm": None,
        "ket_qua_kiem": None,
        "dau_ra_hien_thi": None,
        "phien_ban_llm": TEN_MODEL,
        "roi_ve_bang_mau": False,
    }

    config_1 = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[_khai_bao_cong_cu()],
        temperature=0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # Bước 1: gửi câu hỏi.
    phan_hoi_1 = client.models.generate_content(model=TEN_MODEL, contents=cau_hoi, config=config_1)
    goi_ham = list(getattr(phan_hoi_1, "function_calls", None) or [])

    # Bước 2: không gọi hàm nào -> từ chối, không hiển thị văn bản tự do của LLM.
    if not goi_ham:
        dong_log["dau_ra_hien_thi"] = msg.CAU_TU_CHOI
        _ghi_log(dong_log, log_path)
        return dong_log

    goi = goi_ham[0]
    ten_ham = goi.name
    tham_so = dict(goi.args or {})
    dong_log["loi_goi_ham"] = {"ham": ten_ham, "tham_so": tham_so}

    if ten_ham not in _CAC_HAM:
        dong_log["dau_ra_hien_thi"] = msg.CAU_TU_CHOI
        _ghi_log(dong_log, log_path)
        return dong_log

    # Bước 3: chạy hàm thật (read_functions.py đã tự kiểm tham số, trả {"loi": ...} nếu sai).
    ket_qua_ham = _goi_ham_that(ten_ham, tham_so)
    dong_log["json_ham"] = ket_qua_ham

    la_rag = ten_ham == "tra_cuu_phuong_phap"
    la_loi = (not la_rag) and isinstance(ket_qua_ham, dict) and "loi" in ket_qua_ham
    khong_tim_thay_rag = la_rag and not ket_qua_ham

    if la_loi:
        dong_log["dau_ra_hien_thi"] = ket_qua_ham["loi"]
        _ghi_log(dong_log, log_path)
        return dong_log
    if khong_tim_thay_rag:
        dong_log["dau_ra_hien_thi"] = msg.CAU_RAG_KHONG_TIM_THAY
        _ghi_log(dong_log, log_path)
        return dong_log

    noi_dung_ham = _dinh_dang_rag(ket_qua_ham) if la_rag else json.dumps(ket_qua_ham, ensure_ascii=False)

    # Bước 4: gọi LLM lần hai, kèm kết quả hàm, để viết câu trả lời tiếng Việt.
    lich_su = [
        types.Content(role="user", parts=[types.Part(text=cau_hoi)]),
        phan_hoi_1.candidates[0].content,
        types.Content(
            role="user",
            parts=[types.Part.from_function_response(name=ten_ham, response={"content": noi_dung_ham})],
        ),
    ]
    config_2 = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0)
    phan_hoi_2 = client.models.generate_content(model=TEN_MODEL, contents=lich_su, config=config_2)
    tra_loi_llm = (getattr(phan_hoi_2, "text", None) or "").strip()
    dong_log["tra_loi_llm"] = tra_loi_llm

    # Bước 5: kiểm tất định.
    kiem = (
        checks.kiem_tat_dinh_rag(tra_loi_llm, msg.CUM_TU_CHAN)
        if la_rag
        else checks.kiem_tat_dinh(tra_loi_llm, ket_qua_ham, msg.CUM_TU_CHAN)
    )
    dong_log["ket_qua_kiem"] = kiem

    if kiem["dat"]:
        hien_thi = tra_loi_llm
    else:
        dong_log["roi_ve_bang_mau"] = True
        hien_thi = noi_dung_ham if la_rag else msg.bang_mau_tu_json(ten_ham, ket_qua_ham)

    # Bước 6: gắn cảnh báo cố định + ngày dữ liệu — CHỈ cho nhánh JSON số liệu.
    if not la_rag:
        ngay_du_lieu = ket_qua_ham.get("ngay_du_lieu") or ket_qua_ham.get("den_ngay")
        hien_thi = f"{hien_thi}\n\n{msg.cau_canh_bao(ngay_du_lieu)}"

    dong_log["dau_ra_hien_thi"] = hien_thi

    # Bước 7: ghi log.
    _ghi_log(dong_log, log_path)
    return dong_log
