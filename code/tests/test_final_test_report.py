"""Test cho tổng hợp kết quả test cuối cùng (bước 4.7): gộp bốn bảng kết quả Rank IC rời rạc, danh
sách cấu hình đã đóng băng, tổng hợp p-giá trị ba giả thuyết. Dữ liệu tổng hợp nhỏ tự dựng, không
đọc dữ liệu thật."""
from __future__ import annotations

import math

import pandas as pd

from vnrank.evaluation.final_test_report import (
    consolidate_rank_ic_results,
    frozen_configuration_summary,
    hypotheses_summary,
)


def _baseline_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"baseline": "B2_momentum", "h": 5, "chon_tren_validation": "(21, 0)", "rank_ic_validation": 0.03, "ic_mean": -0.02, "p_value": 0.85},
            {"baseline": "B0_dong_trong_so", "h": 5, "chon_tren_validation": None, "rank_ic_validation": None, "ic_mean": None, "p_value": None},
        ]
    )


def _regression_results() -> pd.DataFrame:
    return pd.DataFrame(
        [{"model": "R2_lightgbm", "h": 5, "chon_tren_validation": "{'num_leaves': 15}", "rank_ic_validation": 0.019, "ic_mean": -0.011, "p_value": 0.78}]
    )


def _ranking_results() -> pd.DataFrame:
    return pd.DataFrame(
        [{"model": "L1_lambdarank", "h": 5, "chon_tren_validation": "{'num_leaves': 15}", "rank_ic_validation": 0.011, "ic_mean": -0.041, "p_value": 0.995}]
    )


def _horizon_results() -> pd.DataFrame:
    # F0/F1/F2 (4.5) KHÔNG có cột chon_tren_validation/p_value (tự chúng không chọn cấu hình riêng).
    return pd.DataFrame(
        [
            {"model": "F0", "h": 5, "ic_mean": -0.032, "icir": -0.21},
            {"model": "F1_R2_lightgbm", "h": 5, "ic_mean": -0.011, "icir": -0.07},
        ]
    )


def _trend_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "T2_lightgbm", "h": 5, "chon_tren_validation": "{'num_leaves': 15}", "macro_f1_validation": 0.255, "macro_f1": 0.257},
            {"model": "D0_lop_da_so", "h": 5, "chon_tren_validation": None, "macro_f1_validation": None, "macro_f1": 0.257},
        ]
    )


# --- consolidate_rank_ic_results: gộp đúng cột, chuẩn hóa tên cột họ mô hình --------------------


def test_consolidate_rank_ic_results_normalizes_family_column_name():
    frames = {
        "baseline": _baseline_results(),
        "regression_topk": _regression_results(),
        "ranking_lambdarank": _ranking_results(),
        "horizon_aggregation": _horizon_results(),
    }
    out = consolidate_rank_ic_results(frames)
    assert set(out["nhom"]) == {"baseline", "regression_topk", "ranking_lambdarank", "horizon_aggregation"}
    assert set(out["family"]) == {"B2_momentum", "B0_dong_trong_so", "R2_lightgbm", "L1_lambdarank", "F0", "F1_R2_lightgbm"}


def test_consolidate_rank_ic_results_keeps_every_row_and_fills_missing_columns_with_nan():
    frames = {
        "baseline": _baseline_results(),
        "regression_topk": _regression_results(),
        "ranking_lambdarank": _ranking_results(),
        "horizon_aggregation": _horizon_results(),
    }
    out = consolidate_rank_ic_results(frames)
    assert len(out) == 2 + 1 + 1 + 2
    # horizon_aggregation không có cột chon_tren_validation/p_value -> NaN sau khi gộp
    f0_row = out[(out["nhom"] == "horizon_aggregation") & (out["family"] == "F0")].iloc[0]
    assert "chon_tren_validation" in out.columns
    assert pd.isna(f0_row["chon_tren_validation"])
    assert pd.isna(f0_row["p_value"])


# --- frozen_configuration_summary: chỉ giữ dòng có cấu hình (bỏ B0/B1/F0-F2/D0-D2) --------------


def test_frozen_configuration_summary_drops_rows_without_config():
    frames = {
        "baseline": _baseline_results(),
        "regression_topk": _regression_results(),
        "ranking_lambdarank": _ranking_results(),
        "horizon_aggregation": _horizon_results(),
    }
    rank_ic_df = consolidate_rank_ic_results(frames)
    trend_df = _trend_results()
    out = frozen_configuration_summary(rank_ic_df, trend_df)

    assert set(out["family"]) == {"B2_momentum", "R2_lightgbm", "L1_lambdarank", "T2_lightgbm"}
    assert "B0_dong_trong_so" not in set(out["family"])
    assert "F0" not in set(out["family"])
    assert "D0_lop_da_so" not in set(out["family"])


def test_frozen_configuration_summary_tags_metric_name_correctly():
    frames = {
        "baseline": _baseline_results(),
        "regression_topk": _regression_results(),
        "ranking_lambdarank": _ranking_results(),
        "horizon_aggregation": _horizon_results(),
    }
    rank_ic_df = consolidate_rank_ic_results(frames)
    trend_df = _trend_results()
    out = frozen_configuration_summary(rank_ic_df, trend_df)

    rank_ic_rows = out[out["nhom"] != "trend_classifier"]
    trend_rows = out[out["nhom"] == "trend_classifier"]
    assert (rank_ic_rows["metric_name"] == "rank_ic").all()
    assert (trend_rows["metric_name"] == "macro_f1").all()
    t2_row = trend_rows[trend_rows["family"] == "T2_lightgbm"].iloc[0]
    assert t2_row["validation_metric"] == 0.255


# --- hypotheses_summary: H1-H3 có p-giá trị, H4 còn để trống (NaN) ------------------------------


def test_hypotheses_summary_has_three_resolved_and_one_pending():
    out = hypotheses_summary({"p_value": 0.9957}, {"p_value": 0.9605}, h3_p_value=1.0)
    assert len(out) == 4
    resolved = out[out["hypothesis"].isin(["H1", "H2", "H3"])]
    assert resolved["p_value"].notna().all()
    h4_row = out[out["hypothesis"] == "H4"].iloc[0]
    assert math.isnan(h4_row["p_value"])
    assert "4.13" in h4_row["note"]
