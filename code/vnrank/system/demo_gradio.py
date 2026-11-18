"""Giao diện Gradio cho hệ thống minh họa S2 (ba hàm đọc `read_functions.py`).

Giao diện chỉ **hiển thị** dữ liệu từ artefact đã đóng băng thông qua ba hàm đọc
`xep_hang`, `xu_huong`, `ket_qua`; không tự tính lại gì. Mọi quy tắc hiển thị theo
`system_architecture.md` mục 4:

- `diem_xu_huong` chỉ hiển thị làm tròn 2 chữ số, kèm chú thích "điểm của mô hình,
  không phải xác suất đã hiệu chuẩn" (mục 4.3 / D60).
- Khi hàm trả `{"loi": "..."}`, giao diện hiển thị nguyên văn lỗi đó.

`build_app()` trả về một `gradio.Blocks` để notebook Colab (chạy `07_system_demo.ipynb`)
có thể `.launch()`; tách khỏi lời gọi launch để test được logic trước khi đẩy lên Colab.
"""
from __future__ import annotations

import gradio as gr
import pandas as pd

from vnrank.system import read_functions as rf

_DONG_CHU_THICH_XU_HUONG = "điểm của mô hình, không phải xác suất đã hiệu chuẩn"


def _dinh_dang_loi(kq) -> tuple[bool, str]:
    """Nếu `kq` là lỗi thì trả (True, mô tả lỗi); ngược lại (False, chuỗi rỗng)."""
    if isinstance(kq, dict) and set(kq.keys()) == {"loi"}:
        return True, kq["loi"]
    return False, ""


def _bang_xep_hang(ngay: str, h: int, K: int) -> tuple[str, pd.DataFrame]:
    """Trả (dòng mô tả tiêu đề, bảng danh_sách) cho tab Xếp hạng."""
    kq = rf.xep_hang(ngay, h, K)
    la_loi, loi = _dinh_dang_loi(kq)
    if la_loi:
        return f"Lỗi: {loi}", pd.DataFrame()
    header = (
        f"Ngày dữ liệu: {kq['ngay_du_lieu']} · Cấu hình: {kq['cau_hinh']} · "
        f"mã băm: {kq['ma_bam']}"
    )
    df = pd.DataFrame(kq["danh_sach"])
    df["diem_xep_hang"] = df["diem_xep_hang"].round(4)
    df["diem_xu_huong"] = df["diem_xu_huong"].round(2)
    return header, df


def _bang_xu_huong(ma: str, ngay: str, h: int) -> tuple[str, pd.DataFrame]:
    """Trả (dòng mô tả tiêu đề, bảng một dòng) cho tab Xu hướng."""
    kq = rf.xu_huong(ma, ngay, h)
    la_loi, loi = _dinh_dang_loi(kq)
    if la_loi:
        return f"Lỗi: {loi}", pd.DataFrame()
    header = (
        f"Ngày dữ liệu: {kq['ngay_du_lieu']} · Mã: {kq['ma']} · Hạng: {kq['hang']} · "
        f"mã băm: {kq['ma_bam']}"
    )
    df = pd.DataFrame(
        [
            {
                "p_tang": round(kq["p_tang"], 4),
                "p_ngang": round(kq["p_ngang"], 4),
                "p_giam": round(kq["p_giam"], 4),
                "diem_xu_huong": round(kq["diem_xu_huong"], 2),
            }
        ]
    )
    return header, df


def _bang_ket_qua(h: int, K: int, muc_chi_phi: str) -> tuple[str, pd.DataFrame]:
    """Trả (dòng mô tả tiêu đề, bảng kết quả) cho tab Kết quả."""
    kq = rf.ket_qua(h, K, muc_chi_phi)
    la_loi, loi = _dinh_dang_loi(kq)
    if la_loi:
        return f"Lỗi: {loi}", pd.DataFrame()
    header = (
        f"Giai đoạn test: {kq['tu_ngay']} → {kq['den_ngay']} · mã băm: {kq['ma_bam']}"
    )
    df = pd.DataFrame(kq["bang"])
    for col in ("rank_ic", "precision_at_k", "loi_suat_nam_hoa_rong", "sharpe_rong", "mdd", "turnover"):
        df[col] = df[col].round(4)
    return header, df


def build_app(du_lieu_that: bool = False) -> gr.Blocks:
    """Dựng giao diện Gradio. Notebook gọi `app = build_app(); app.launch()`.

    `du_lieu_that=False` (mặc định, giữ nguyên hành vi S2): nạp dữ liệu MẪU GIẢ tất định
    (`nap_du_lieu_mau`), dùng khi artefact thật chưa có hoặc chỉ muốn xem khung giao diện.
    `du_lieu_that=True` (S3/S4/S8, D107/D109/D113): nạp artefact THẬT qua
    `artifact_loader.nap_artefact_that()` — `rank_scores`/`trend_scores`/`topk_a1` của L1/T2/A1 và
    `test_results` của `ket_qua` đều là kết quả mô hình thật đã đóng băng, không phải số liệu giả."""
    if du_lieu_that:
        from vnrank.system.artifact_loader import nap_artefact_that

        nap_artefact_that()
        dong_du_lieu = (
            "Dữ liệu THẬT từ mô hình đã huấn luyện và đóng băng (benchmark 4.1-4.13) — "
            "xem mục \"Độ tin cậy H1-H4\" trong dashboard trước khi đọc số liệu, không phải "
            "mô hình đã được chứng minh có lợi thế dự báo."
        )
    else:
        rf.nap_du_lieu_mau()  # artefact mẫu tất định cho demo S2
        dong_du_lieu = "Dữ liệu mẫu (mock) — số liệu chỉ để minh họa giao diện, không phải kết quả thực tế."

    with gr.Blocks(title="Hệ thống minh họa xếp hạng cổ phiếu") as app:
        gr.Markdown(f"# Hệ thống minh họa xếp hạng cổ phiếu (S2)\n{dong_du_lieu}")

        with gr.Tab("Xếp hạng"):
            gr.Markdown("Top-`K` của cấu hình A1 tại ngày quyết định gần nhất ≤ ngày nhập.")
            with gr.Row():
                ngay_xh = gr.Textbox(label="Ngày (YYYY-MM-DD)", value="2025-07-15")
                h_xh = gr.Dropdown(choices=rf._HORIZONS, value=5, label="h (ngày)")
                K_xh = gr.Dropdown(choices=rf._K_HOP_LE, value=10, label="K")
            nut_xh = gr.Button("Xem xếp hạng")
            tieu_de_xh = gr.Markdown()
            bang_xh = gr.Dataframe(headers=["hang", "ma", "san", "diem_xep_hang", "diem_xu_huong"])
            nut_xh.click(_bang_xep_hang, [ngay_xh, h_xh, K_xh], [tieu_de_xh, bang_xh])

        with gr.Tab("Xu hướng"):
            gr.Markdown("Xu hướng của một mã cổ phiếu tại ngày quyết định gần nhất ≤ ngày nhập.")
            with gr.Row():
                ma_xh_trend = gr.Textbox(label="Mã cổ phiếu", value="AAA")
                ngay_trend = gr.Textbox(label="Ngày (YYYY-MM-DD)", value="2025-07-15")
                h_trend = gr.Dropdown(choices=rf._HORIZONS, value=5, label="h (ngày)")
            nut_trend = gr.Button("Xem xu hướng")
            tieu_de_trend = gr.Markdown()
            bang_trend = gr.Dataframe(
                headers=["p_tang", "p_ngang", "p_giam", "diem_xu_huong"]
            )
            gr.Markdown(f"Ghi chú: `diem_xu_huong` — {_DONG_CHU_THICH_XU_HUONG}.")
            nut_trend.click(_bang_xu_huong, [ma_xh_trend, ngay_trend, h_trend], [tieu_de_trend, bang_trend])

        with gr.Tab("Kết quả"):
            gr.Markdown("Bảng kết quả trên toàn giai đoạn test đã đóng băng (không tính lại).")
            with gr.Row():
                h_kq = gr.Dropdown(choices=rf._HORIZONS, value=5, label="h (ngày)")
                K_kq = gr.Dropdown(choices=rf._K_HOP_LE, value=10, label="K")
                mcp_kq = gr.Dropdown(choices=rf._MUC_CHI_PHI, value="cơ sở", label="muc_chi_phi")
            nut_kq = gr.Button("Xem kết quả")
            tieu_de_kq = gr.Markdown()
            bang_kq = gr.Dataframe(
                headers=[
                    "cau_hinh",
                    "rank_ic",
                    "precision_at_k",
                    "loi_suat_nam_hoa_rong",
                    "sharpe_rong",
                    "mdd",
                    "turnover",
                ]
            )
            nut_kq.click(_bang_ket_qua, [h_kq, K_kq, mcp_kq], [tieu_de_kq, bang_kq])

    return app


if __name__ == "__main__":
    build_app().launch()