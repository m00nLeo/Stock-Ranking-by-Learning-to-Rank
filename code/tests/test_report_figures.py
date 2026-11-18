"""Test cho bước 5.1 (D115): sinh bảng/hình báo cáo từ CSV đã đóng băng. Dữ liệu tổng hợp nhỏ tự
dựng (ghi vào tmp_path), không đọc dữ liệu thật của dự án."""
from __future__ import annotations

import pandas as pd

from vnrank.evaluation.report_figures import (
    build_h1_table,
    build_h2_table,
    build_h3_table,
    build_h4_table,
    build_report_docx,
    build_sensitivity_table,
    build_shap_figure,
    build_shap_table,
    build_summary_table,
)


def _write_csv(tmp_path, name: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(tmp_path / name, index=False)


def _write_minimal_dataset(tmp_path) -> None:
    _write_csv(tmp_path, "hypotheses_summary_4_7.csv", [
        {"hypothesis": "H1", "p_value": 0.9957, "note": "L1 so R2", "holm_rejected": False},
        {"hypothesis": "H2", "p_value": 0.9605, "note": "F0 so F1", "holm_rejected": False},
        {"hypothesis": "H3", "p_value": 1.0, "note": "T2 so D0", "holm_rejected": False},
        {"hypothesis": "H4", "p_value": 0.0986, "note": "A1 so A0", "holm_rejected": False},
    ])
    _write_csv(tmp_path, "ranking_lambdarank_results_4_4.csv", [
        {"model": "L1_lambdarank", "h": 5, "rank_ic_validation": 0.011, "ic_mean": -0.0405, "se": 0.0156},
        {"model": "L1_lambdarank", "h": 10, "rank_ic_validation": 0.036, "ic_mean": -0.0326, "se": 0.0171},
    ])
    _write_csv(tmp_path, "regression_topk_results_4_3.csv", [
        {"model": "R2_lightgbm", "h": 5, "rank_ic_validation": 0.019, "ic_mean": -0.0106},
        {"model": "R2_lightgbm", "h": 10, "rank_ic_validation": 0.027, "ic_mean": -0.0130},
    ])
    _write_csv(tmp_path, "h1_test_4_4.csv", [
        {"mean": -0.0299, "se": 0.0114, "t_stat": -2.629, "p_value": 0.9957},
    ])
    _write_csv(tmp_path, "horizon_aggregation_results_4_5.csv", [
        {"model": "F0", "ic_mean": -0.0320, "ic_std": 0.153, "icir": -0.208},
        {"model": "F1_R2_lightgbm", "ic_mean": -0.0106, "ic_std": 0.150, "icir": -0.071},
    ])
    _write_csv(tmp_path, "h2_test_4_5.csv", [
        {"delta_ic": 0.0038, "mean": -0.0213, "se": 0.0100, "t_stat": -1.756, "p_value": 0.9605},
    ])
    _write_csv(tmp_path, "trend_classifier_results_4_10.csv", [
        {"model": "T2_lightgbm", "h": 5, "macro_f1": 0.2566, "balanced_accuracy": 0.3333},
        {"model": "D0_lop_da_so", "h": 5, "macro_f1": 0.2566, "balanced_accuracy": 0.3333},
        {"model": "D2_ngau_nhien", "h": 5, "macro_f1": 0.3324, "balanced_accuracy": 0.3326},
        {"model": "H4d_brier_bss_ece", "h": 5, "bss_t2_vs_d1": 0.0040, "ece_t2": 0.0074},
    ])
    _write_csv(tmp_path, "h3_test_4_10.csv", [{"h": 5, "p_value": 1.0}])
    _write_csv(tmp_path, "trend_integration_results_4_13.csv", [
        {"ablation": "A0", "precision_mean": 0.4512, "precision_sd": None},
        {"ablation": "A1", "precision_mean": 0.4475, "precision_sd": None},
        {"ablation": "A2", "precision_mean": 0.4357, "precision_sd": None},
        {"ablation": "A3", "precision_mean": 0.4445, "precision_sd": 0.0056},
    ])
    _write_csv(tmp_path, "h4_test_4_13.csv", [
        {"mean": -0.0037, "se": 0.0126, "t_stat": 1.290, "p_value": 0.0986},
    ])
    _write_csv(tmp_path, "h4c_test_4_13.csv", [
        {"mean": 0.0117, "se": 0.0120, "t_stat": 0.976, "p_value_two_sided": 0.3289},
    ])
    _write_csv(tmp_path, "sensitivity_analysis_4_8.csv", [
        {"yeu_to": "co_so", "muc": "co_so", "ic": -0.0405, "sharpe": -0.2782, "f1": 0.2566,
         "delta_ic": 0.0, "delta_sharpe": 0.0, "delta_f1": 0.0},
        {"yeu_to": "chi_phi", "muc": "thap", "ic": -0.0405, "sharpe": -0.0092, "f1": 0.2566,
         "delta_ic": 0.0, "delta_sharpe": 0.269, "delta_f1": 0.0},
    ])
    _write_csv(tmp_path, "shap_importance_4_9.csv", [
        {"feature": "beta_63", "mean_abs_shap": 0.0185, "mean_shap": 0.000139, "ty_trong": 0.2019, "rank": 1},
        {"feature": "ret_63", "mean_abs_shap": 0.0132, "mean_shap": -0.000273, "ty_trong": 0.1447, "rank": 2},
    ])


# --- Mỗi bảng: đúng hình dạng và giữ nguyên số liệu từ CSV ---------------------------------------


def test_summary_table_sorted_by_p_and_holm_threshold(tmp_path):
    _write_minimal_dataset(tmp_path)
    table = build_summary_table(tmp_path)
    assert list(table["Giả thuyết"]) == ["H4", "H2", "H1", "H3"]  # sắp theo p tăng dần
    assert table.iloc[0]["Ngưỡng Holm"] == "0,0125"  # 0,05/4
    assert table.iloc[-1]["Ngưỡng Holm"] == "0,0500"  # 0,05/1


def test_h1_table_has_both_horizons_and_test_row(tmp_path):
    _write_minimal_dataset(tmp_path)
    detail, kiem_dinh = build_h1_table(tmp_path)
    assert list(detail["h"]) == [5, 10]
    assert detail.iloc[0]["Rank IC test (L1)"] == "-0,0405"
    assert kiem_dinh.iloc[0]["p (một phía)"] == "0,9957"


def test_h2_table_has_f0_f1(tmp_path):
    _write_minimal_dataset(tmp_path)
    detail, kiem_dinh = build_h2_table(tmp_path)
    assert set(detail["Cấu hình"]) == {"F0", "F1_R2_lightgbm"}
    assert kiem_dinh.iloc[0]["δ_IC (biên không kém)"] == "0,0038"


def test_h3_table_macro_f1_matches_six_decimals(tmp_path):
    _write_minimal_dataset(tmp_path)
    table = build_h3_table(tmp_path)
    assert table.iloc[0]["macro-F1 (T2)"] == table.iloc[0]["macro-F1 (D0)"] == "0,256600"


def test_h4_table_four_configs_and_two_tests(tmp_path):
    _write_minimal_dataset(tmp_path)
    detail, kiem_dinh = build_h4_table(tmp_path)
    assert list(detail["Cấu hình"]) == ["A0", "A1", "A2", "A3"]
    assert detail.iloc[3]["Độ lệch chuẩn"] == "0,0056"
    assert detail.iloc[0]["Độ lệch chuẩn"] == "—"  # NaN -> gạch ngang, không phải "nan"
    assert len(kiem_dinh) == 2


def test_sensitivity_table_rows_match_input(tmp_path):
    _write_minimal_dataset(tmp_path)
    table = build_sensitivity_table(tmp_path)
    assert len(table) == 2
    assert table.iloc[0]["Yếu tố"] == "co_so"


def test_shap_table_two_features_in_rank_order(tmp_path):
    _write_minimal_dataset(tmp_path)
    table = build_shap_table(tmp_path)
    assert list(table["Hạng"]) == [1, 2]
    assert "beta_63" in table.iloc[0]["Đặc trưng"]


def test_shap_figure_writes_a_real_png(tmp_path):
    _write_minimal_dataset(tmp_path)
    out = build_shap_figure(tmp_path, tmp_path / "shap.png")
    assert out.exists()
    assert out.stat().st_size > 0
    with open(out, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"  # chữ ký PNG thật


# --- build_report_docx: ghép đủ bảy mục thành một file .docx thật ------------------------------


def test_build_report_docx_has_all_seven_sections(tmp_path):
    _write_minimal_dataset(tmp_path)
    out = build_report_docx(tmp_path, tmp_path / "report.docx")
    assert out.exists()

    from docx import Document

    doc = Document(str(out))
    headings = [p.text for p in doc.paragraphs if p.style.name == "Heading 2"]
    assert headings == [
        "1. Tổng hợp kiểm định H1-H4",
        "2. H1 — Rank IC L1 so R2",
        "3. H2 — Gộp đa horizon (F0/F1/F2)",
        "4. H3 — Phân loại xu hướng T2 so D0/D1/D2",
        "5. H4 — Tích hợp xu hướng vào Top-K (A0-A3)",
        "6. Phân tích độ nhạy (RQ3, bước 4.8)",
        "7. SHAP — Giải thích đóng góp đặc trưng (L1, h=5, bước 4.9)",
    ]
    assert len(doc.tables) == 10  # 1+2+2+1+2+1+1 bang (xem report_figures.build_report_docx)
    assert len(doc.inline_shapes) == 1  # hinh SHAP da chen
