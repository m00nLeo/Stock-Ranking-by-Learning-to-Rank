"""Bước 5.1 (Giai đoạn 5, D115): sinh TOÀN BỘ bảng và hình được trích dẫn trong văn xuôi của bản
thảo 5.2 (`05_Giai_doan_4_Thuc_nghiem/1_Ket_qua_va_thao_luan/tong_hop_ket_qua_h1_h4.md` và
`do_tin_cay_h1_h4_va_ap_dung_he_thong.md`), để luận văn TÁI LẬP ĐƯỢC. CHỈ ĐỌC các file CSV đã đóng
băng (4.1-4.13, SHAP 4.9) — KHÔNG tính toán lại bất kỳ số liệu nào.

**Quyết định người dùng (D115, trước khi viết module này):**
1. Đầu ra: script xuất THẲNG ra một file `.docx` (không phải PNG/markdown rời) — dùng `python-docx`.
2. Thư viện vẽ hình: `matplotlib` (thêm vào `requirements.txt`/`requirements.lock.txt`; phát hiện
   và sửa cùng kiểu xung đột D104: bản mặc định mới nhất kéo theo numpy 2.x phá vỡ numpy==1.26.4
   đã pin — ghim lại `matplotlib==3.9.2`).

**Danh sách bảng/hình** (đúng những con số/bảng đã trích dẫn trong văn xuôi của hai bản thảo trên,
tiêu chí chọn theo yêu cầu người dùng — không thêm thước đo nào khác):
1. Bảng tổng hợp H1-H4 (chỉ số chính, p-giá trị, ngưỡng Holm, kết luận).
2. Bảng chi tiết H1 (Rank IC L1 so R2, validation/test, theo $h$).
3. Bảng chi tiết H2 (Rank IC F0/F1/F2).
4. Bảng chi tiết H3 (macro-F1/balanced accuracy T2 so D0/D1/D2 theo $h$, kèm Brier/BSS/ECE).
5. Bảng chi tiết H4 (precision@K A0-A3).
6. Bảng độ nhạy RQ3 (4.8, đủ 4 yếu tố).
7. Hình cột SHAP importance (4.9, 11 đặc trưng) + bảng số liệu kèm theo.

Xuất ra MỘT file `.docx` (mặc định `05_Giai_doan_4_Thuc_nghiem/2_Bang_hinh/bang_hinh_ket_qua.docx`,
đúng quy ước thư mục số thứ tự đã có trong `05_Giai_doan_4_Thuc_nghiem/`), có cả 7 bảng/hình trên,
theo đúng thứ tự xuất hiện trong văn xuôi bản thảo 5.2. Đây là tài liệu TRUNG GIAN để tái lập và để
người dùng tự chọn đoạn/bảng dán vào luận văn — KHÔNG phải bản thân file luận văn chính thức.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # không cần hiển thị, chỉ lưu file — tránh phụ thuộc GUI backend
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.shared import Inches

CODE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = CODE_ROOT / "data" / "processed"

FEATURE_VI_LABELS = {
    "beta_63": "beta_63 (rủi ro hệ thống)",
    "avg_value_l": "avg_value_l (thanh khoản)",
}


def _fmt(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and x != x):  # NaN
        return "—"
    if isinstance(x, str):
        return x
    return f"{x:.{nd}f}".replace(".", ",")


# --- Mục 1: bảng tổng hợp H1-H4 ------------------------------------------------------------------


def build_summary_table(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Bảng tổng hợp bốn giả thuyết: chỉ số chính, p-giá trị, ngưỡng Holm (thứ tự tăng dần theo p,
    đúng cách Holm step-down xét), kết luận. Nguồn: `hypotheses_summary_4_7.csv`."""
    summary = pd.read_csv(data_dir / "hypotheses_summary_4_7.csv")
    chi_so_chinh = {
        "H1": "Rank IC (L1 so R2, h=5): -0,0299",
        "H2": "Rank IC (F0 so F1, h=5): -0,0213",
        "H3": "macro-F1 (T2 so D0), giao-hợp 3 h: max(p)",
        "H4": "precision@K (A1 so A0, h=5): -0,0037",
    }
    summary = summary.sort_values("p_value").reset_index(drop=True)
    n = len(summary)
    rows = []
    for rank, (_, r) in enumerate(summary.iterrows(), start=1):
        nguong_holm = 0.05 / (n - rank + 1)
        rows.append({
            "Giả thuyết": r["hypothesis"],
            "Chỉ số chính": chi_so_chinh.get(r["hypothesis"], ""),
            "p-giá trị": _fmt(r["p_value"]),
            "Thứ hạng p (tăng dần)": rank,
            "Ngưỡng Holm": _fmt(nguong_holm),
            "Bác bỏ H0?": "Có" if bool(r["holm_rejected"]) else "Không",
            "Ghi chú": r["note"],
        })
    return pd.DataFrame(rows)


# --- Mục 2: H1 chi tiết ---------------------------------------------------------------------------


def build_h1_table(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Rank IC validation/test của L1 và R2 theo từng h, cộng kiểm định H1 ở h=5.
    Nguồn: `ranking_lambdarank_results_4_4.csv`, `regression_topk_results_4_3.csv`, `h1_test_4_4.csv`."""
    l1 = pd.read_csv(data_dir / "ranking_lambdarank_results_4_4.csv")
    reg = pd.read_csv(data_dir / "regression_topk_results_4_3.csv")
    r2 = reg[reg["model"] == "R2_lightgbm"]
    h1_test = pd.read_csv(data_dir / "h1_test_4_4.csv").iloc[0]

    rows = []
    for h in sorted(l1["h"].unique()):
        l1_row = l1[l1["h"] == h].iloc[0]
        r2_row = r2[r2["h"] == h].iloc[0]
        rows.append({
            "h": int(h),
            "Rank IC validation (L1)": _fmt(l1_row["rank_ic_validation"]),
            "Rank IC test (L1)": _fmt(l1_row["ic_mean"]),
            "se (L1)": _fmt(l1_row["se"]),
            "Rank IC validation (R2)": _fmt(r2_row["rank_ic_validation"]),
            "Rank IC test (R2)": _fmt(r2_row["ic_mean"]),
        })
    detail = pd.DataFrame(rows)
    kiem_dinh = pd.DataFrame([{
        "h": 5, "mean(d_t)=IC(L1)-IC(R2)": _fmt(h1_test["mean"]), "se": _fmt(h1_test["se"]),
        "t": _fmt(h1_test["t_stat"]), "p (một phía)": _fmt(h1_test["p_value"]),
    }])
    return detail, kiem_dinh


# --- Mục 3: H2 chi tiết ---------------------------------------------------------------------------


def build_h2_table(data_dir: Path = DATA_DIR):
    """F0 (gộp cố định)/F1 (tốt nhất theo horizon)/F2 (từng horizon của L1 trên nhãn h=5).
    Nguồn: `horizon_aggregation_results_4_5.csv`, `h2_test_4_5.csv`."""
    horizon = pd.read_csv(data_dir / "horizon_aggregation_results_4_5.csv")
    h2_test = pd.read_csv(data_dir / "h2_test_4_5.csv").iloc[0]

    rows = []
    for _, r in horizon.iterrows():
        rows.append({
            "Cấu hình": r["model"],
            "Rank IC test": _fmt(r["ic_mean"]),
            "ic_std": _fmt(r["ic_std"]),
            "ICIR": _fmt(r["icir"]),
        })
    detail = pd.DataFrame(rows)
    kiem_dinh = pd.DataFrame([{
        "δ_IC (biên không kém)": _fmt(h2_test["delta_ic"]),
        "mean(d_t)=IC(F0)-IC(F1)": _fmt(h2_test["mean"]), "se": _fmt(h2_test["se"]),
        "t": _fmt(h2_test["t_stat"]), "p (một phía)": _fmt(h2_test["p_value"]),
    }])
    return detail, kiem_dinh


# --- Mục 4: H3 chi tiết ---------------------------------------------------------------------------


def build_h3_table(data_dir: Path = DATA_DIR):
    """macro-F1/balanced accuracy của T2 so D0/D1/D2 theo h, cộng Brier/BSS/ECE, cộng p theo h.
    Nguồn: `trend_classifier_results_4_10.csv`, `h3_test_4_10.csv`."""
    tc = pd.read_csv(data_dir / "trend_classifier_results_4_10.csv")
    h3_test = pd.read_csv(data_dir / "h3_test_4_10.csv")

    rows = []
    for h in sorted(tc["h"].unique()):
        sub = tc[tc["h"] == h]
        t2 = sub[sub["model"] == "T2_lightgbm"].iloc[0]
        d0 = sub[sub["model"] == "D0_lop_da_so"].iloc[0]
        d2 = sub[sub["model"] == "D2_ngau_nhien"].iloc[0]
        bss_row = sub[sub["model"] == "H4d_brier_bss_ece"].iloc[0]
        p_h = h3_test[h3_test["h"] == h].iloc[0]
        rows.append({
            "h": int(h),
            "macro-F1 (T2)": _fmt(t2["macro_f1"], 6),
            "macro-F1 (D0)": _fmt(d0["macro_f1"], 6),
            "macro-F1 (D2 ngẫu nhiên)": _fmt(d2["macro_f1"], 6),
            "balanced accuracy (T2)": _fmt(t2["balanced_accuracy"]),
            "BSS (T2 so D1)": _fmt(bss_row["bss_t2_vs_d1"]),
            "ECE (T2)": _fmt(bss_row["ece_t2"]),
            "p (bootstrap khối)": _fmt(p_h["p_value"]),
        })
    return pd.DataFrame(rows)


# --- Mục 5: H4 chi tiết ---------------------------------------------------------------------------


def build_h4_table(data_dir: Path = DATA_DIR):
    """precision@K của A0-A3, cộng kiểm định H4 (A1 so A0) và H4c (A1 so A2).
    Nguồn: `trend_integration_results_4_13.csv`, `h4_test_4_13.csv`, `h4c_test_4_13.csv`."""
    ti = pd.read_csv(data_dir / "trend_integration_results_4_13.csv")
    h4_test = pd.read_csv(data_dir / "h4_test_4_13.csv").iloc[0]
    h4c_test = pd.read_csv(data_dir / "h4c_test_4_13.csv").iloc[0]

    rows = []
    for _, r in ti.iterrows():
        rows.append({
            "Cấu hình": r["ablation"],
            "precision@K": _fmt(r["precision_mean"]),
            "Độ lệch chuẩn": _fmt(r["precision_sd"]) if r["precision_sd"] == r["precision_sd"] else "—",
        })
    detail = pd.DataFrame(rows)
    kiem_dinh = pd.DataFrame([
        {
            "Kiểm định": "H4 (A1 so A0, không kém, δ=0,02)",
            "mean(d_t)": _fmt(h4_test["mean"]), "se": _fmt(h4_test["se"]),
            "t": _fmt(h4_test["t_stat"]), "p": _fmt(h4_test["p_value"]),
        },
        {
            "Kiểm định": "H4c (A1 so A2, hai phía)",
            "mean(d_t)": _fmt(h4c_test["mean"]), "se": _fmt(h4c_test["se"]),
            "t": _fmt(h4c_test["t_stat"]), "p": _fmt(h4c_test["p_value_two_sided"]),
        },
    ])
    return detail, kiem_dinh


# --- Mục 6: độ nhạy RQ3 (4.8) ----------------------------------------------------------------------


def build_sensitivity_table(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Bảng độ nhạy đầy đủ (chi phí, K, ngưỡng thanh khoản, giai đoạn thị trường).
    Nguồn: `sensitivity_analysis_4_8.csv`."""
    sens = pd.read_csv(data_dir / "sensitivity_analysis_4_8.csv")
    rows = []
    for _, r in sens.iterrows():
        rows.append({
            "Yếu tố": r["yeu_to"], "Mức": r["muc"],
            "Rank IC": _fmt(r["ic"]), "Sharpe ròng năm hóa": _fmt(r["sharpe"]),
            "macro-F1": _fmt(r["f1"]),
            "ΔRank IC": _fmt(r["delta_ic"]), "ΔSharpe": _fmt(r["delta_sharpe"]), "Δmacro-F1": _fmt(r["delta_f1"]),
        })
    return pd.DataFrame(rows)


# --- Mục 7: SHAP (4.9) ------------------------------------------------------------------------------


def build_shap_figure(data_dir: Path = DATA_DIR, output_path: Path | None = None) -> Path:
    """Hình cột |SHAP| trung bình của 11 đặc trưng, sắp giảm dần theo hạng. Nguồn: `shap_importance_4_9.csv`."""
    shap_df = pd.read_csv(data_dir / "shap_importance_4_9.csv").sort_values("rank")
    labels = [FEATURE_VI_LABELS.get(f, f) for f in shap_df["feature"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(labels[::-1], (shap_df["ty_trong"] * 100).to_numpy()[::-1], color="#146e62")
    ax.set_xlabel("Tỷ trọng |SHAP| trung bình (%)")
    ax.set_title("Mức độ quan trọng của đặc trưng (SHAP, mô hình L1, h=5)")
    fig.tight_layout()

    if output_path is None:
        output_path = data_dir.parents[2] / "05_Giai_doan_4_Thuc_nghiem" / "2_Bang_hinh" / "shap_importance_4_9.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def build_shap_table(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    shap_df = pd.read_csv(data_dir / "shap_importance_4_9.csv").sort_values("rank")
    rows = []
    for _, r in shap_df.iterrows():
        rows.append({
            "Hạng": int(r["rank"]),
            "Đặc trưng": FEATURE_VI_LABELS.get(r["feature"], r["feature"]),
            "Tỷ trọng |SHAP| (%)": _fmt(r["ty_trong"] * 100, 2),
            "mean_shap (chiều)": _fmt(r["mean_shap"], 6),
        })
    return pd.DataFrame(rows)


# --- Ghép vào một file .docx -------------------------------------------------------------------


def _add_table(document: Document, df: pd.DataFrame, title: str | None = None) -> None:
    if title:
        document.add_heading(title, level=3)
    table = document.add_table(rows=1, cols=len(df.columns))
    table.style = "Light Grid Accent 1"
    for i, col in enumerate(df.columns):
        table.rows[0].cells[i].text = str(col)
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)


def build_report_docx(data_dir: Path = DATA_DIR, output_path: Path | None = None) -> Path:
    """Sinh file `.docx` đầy đủ 7 bảng/hình, đúng thứ tự xuất hiện trong văn xuôi bản thảo 5.2.
    CHỈ ĐỌC CSV đã đóng băng, KHÔNG tính lại gì."""
    if output_path is None:
        output_path = data_dir.parents[2] / "05_Giai_doan_4_Thuc_nghiem" / "2_Bang_hinh" / "bang_hinh_ket_qua.docx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = Document()
    document.add_heading("Bảng và hình kết quả Giai đoạn 4 (bước 5.1)", level=1)
    document.add_paragraph(
        "Sinh tự động từ các file CSV đã đóng băng (4.1-4.13, SHAP 4.9) bằng "
        "code/vnrank/evaluation/report_figures.py — không tính lại số liệu nào. "
        "Tài liệu trung gian để tái lập và tham khảo khi viết luận văn, không phải bản luận văn chính thức."
    )

    document.add_heading("1. Tổng hợp kiểm định H1-H4", level=2)
    _add_table(document, build_summary_table(data_dir))

    document.add_heading("2. H1 — Rank IC L1 so R2", level=2)
    h1_detail, h1_test = build_h1_table(data_dir)
    _add_table(document, h1_detail, "Rank IC theo h")
    _add_table(document, h1_test, "Kiểm định H1 (h=5, Newey-West)")

    document.add_heading("3. H2 — Gộp đa horizon (F0/F1/F2)", level=2)
    h2_detail, h2_test = build_h2_table(data_dir)
    _add_table(document, h2_detail, "Rank IC F0/F1/F2")
    _add_table(document, h2_test, "Kiểm định H2 (không kém, Newey-West)")

    document.add_heading("4. H3 — Phân loại xu hướng T2 so D0/D1/D2", level=2)
    _add_table(document, build_h3_table(data_dir))

    document.add_heading("5. H4 — Tích hợp xu hướng vào Top-K (A0-A3)", level=2)
    h4_detail, h4_test = build_h4_table(data_dir)
    _add_table(document, h4_detail, "precision@K theo cấu hình")
    _add_table(document, h4_test, "Kiểm định H4/H4c")

    document.add_heading("6. Phân tích độ nhạy (RQ3, bước 4.8)", level=2)
    _add_table(document, build_sensitivity_table(data_dir))

    document.add_heading("7. SHAP — Giải thích đóng góp đặc trưng (L1, h=5, bước 4.9)", level=2)
    shap_png = build_shap_figure(data_dir, output_path.parent / "shap_importance_4_9.png")
    document.add_picture(str(shap_png), width=Inches(6))
    _add_table(document, build_shap_table(data_dir))

    document.save(output_path)
    return output_path


if __name__ == "__main__":
    path = build_report_docx()
    print(f"Da luu bao cao bang/hinh vao: {path}")
