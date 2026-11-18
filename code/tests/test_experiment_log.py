"""Test cho tổng hợp log thử nghiệm (bước 4.6): gộp log lưới rời rạc của 4.2/4.3/4.4 thành một log
thống nhất và đếm N theo h; và bảng mở rộng gồm cả T2 (4.10)/A0-A3 (4.13), D116/D118. Dữ liệu tổng
hợp nhỏ tự dựng, không đọc dữ liệu thật."""
from __future__ import annotations

import pandas as pd

from vnrank.evaluation.experiment_log import consolidate_frames, consolidate_frames_all, summarize_by_h


def _baseline_log() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"baseline": "B2_momentum", "h": 5, "config": "(21, 0)", "rank_ic_validation": 0.03, "chon": True},
            {"baseline": "B2_momentum", "h": 5, "config": "(63, 0)", "rank_ic_validation": 0.00, "chon": False},
            {"baseline": "B2_momentum", "h": 10, "config": "(21, 0)", "rank_ic_validation": 0.01, "chon": True},
        ]
    )


def _regression_log() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "R1_ridge", "h": 5, "config": "{'alpha': 1}", "rank_ic_validation": -0.01, "chon": False},
            {"model": "R2_lightgbm", "h": 5, "config": "{'num_leaves': 15}", "rank_ic_validation": 0.02, "chon": True},
        ]
    )


def _ranking_log() -> pd.DataFrame:
    return pd.DataFrame(
        [{"model": "L1_lambdarank", "h": 5, "config": "{'num_leaves': 15}", "rank_ic_validation": 0.01, "chon": True}]
    )


# --- consolidate_frames: gộp đúng cột, chuẩn hóa tên cột họ mô hình ----------------------------


def test_consolidate_frames_normalizes_family_column_name():
    out = consolidate_frames({"4.2": _baseline_log(), "4.3": _regression_log(), "4.4": _ranking_log()})
    assert list(out.columns) == ["buoc", "family", "h", "config", "rank_ic_validation", "chon"]
    assert set(out["family"]) == {"B2_momentum", "R1_ridge", "R2_lightgbm", "L1_lambdarank"}


def test_consolidate_frames_keeps_every_row_tagged_with_its_buoc():
    out = consolidate_frames({"4.2": _baseline_log(), "4.3": _regression_log(), "4.4": _ranking_log()})
    assert len(out) == 3 + 2 + 1
    assert out[out["buoc"] == "4.4"]["family"].tolist() == ["L1_lambdarank"]
    assert out[out["buoc"] == "4.2"]["chon"].sum() == 2  # B2_momentum h=5 va h=10 deu chon=True


# --- summarize_by_h: dem dung N theo h va theo buoc --------------------------------------------


def test_summarize_by_h_counts_match_source_rows():
    log_df = consolidate_frames({"4.2": _baseline_log(), "4.3": _regression_log(), "4.4": _ranking_log()})
    summary = summarize_by_h(log_df)
    row_h5 = summary[summary["h"] == 5].iloc[0]
    assert row_h5["4.2"] == 2
    assert row_h5["4.3"] == 2
    assert row_h5["4.4"] == 1
    assert row_h5["tong"] == 5

    row_h10 = summary[summary["h"] == 10].iloc[0]
    assert row_h10["4.2"] == 1
    assert row_h10["tong"] == 1


# --- consolidate_frames_all: bảng mở rộng T2 (4.10)/A0-A3 (4.13), D116/D118 --------------------


def _trend_classifier_log() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "T2_lightgbm", "h": 5, "config": "{'num_leaves': 15}", "macro_f1_validation": 0.25, "chon": True},
            {"model": "T2_lightgbm", "h": 5, "config": "{'num_leaves': 31}", "macro_f1_validation": 0.24, "chon": False},
        ]
    )


def _trend_integration_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ablation": "A0", "h": 5, "precision_mean": 0.451},
            {"ablation": "A1", "h": 5, "precision_mean": 0.447},
            {"ablation": "A2", "h": 5, "precision_mean": 0.436},
            {"ablation": "A3", "h": 5, "precision_mean": 0.444},
        ]
    )


def test_consolidate_frames_all_has_generic_metric_columns():
    out = consolidate_frames_all(
        {"4.2": _baseline_log(), "4.3": _regression_log(), "4.4": _ranking_log()},
        _trend_classifier_log(),
        _trend_integration_results(),
    )
    assert list(out.columns) == ["buoc", "family", "h", "config", "metric_name", "metric_value", "chon"]
    assert len(out) == (3 + 2 + 1) + 2 + 4


def test_consolidate_frames_all_keeps_rank_ic_values_for_old_sources():
    out = consolidate_frames_all(
        {"4.2": _baseline_log(), "4.3": _regression_log(), "4.4": _ranking_log()},
        _trend_classifier_log(),
        _trend_integration_results(),
    )
    rank_ic_rows = out[out["metric_name"] == "rank_ic"]
    assert len(rank_ic_rows) == 6
    assert rank_ic_rows["metric_value"].tolist() == [0.03, 0.00, 0.01, -0.01, 0.02, 0.01]


def test_consolidate_frames_all_tags_t2_with_macro_f1():
    out = consolidate_frames_all(
        {"4.2": _baseline_log(), "4.3": _regression_log(), "4.4": _ranking_log()},
        _trend_classifier_log(),
        _trend_integration_results(),
    )
    t2_rows = out[out["buoc"] == "4.10"]
    assert set(t2_rows["metric_name"]) == {"macro_f1"}
    assert t2_rows["chon"].sum() == 1


def test_consolidate_frames_all_tags_a0_a3_with_precision_test_and_only_a1_chosen():
    out = consolidate_frames_all(
        {"4.2": _baseline_log(), "4.3": _regression_log(), "4.4": _ranking_log()},
        _trend_classifier_log(),
        _trend_integration_results(),
    )
    a_rows = out[out["buoc"] == "4.13"].sort_values("config")
    assert a_rows["config"].tolist() == ["A0", "A1", "A2", "A3"]
    assert set(a_rows["metric_name"]) == {"precision_test"}
    assert a_rows.set_index("config")["chon"].to_dict() == {"A0": False, "A1": True, "A2": False, "A3": False}


def test_summarize_by_h_total_equals_full_log_length():
    log_df = consolidate_frames({"4.2": _baseline_log(), "4.3": _regression_log(), "4.4": _ranking_log()})
    summary = summarize_by_h(log_df)
    assert summary["tong"].sum() == len(log_df)
