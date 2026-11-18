"""Test cho tích hợp xu hướng vào top-K (bước 4.13): A0-A3, kiểm định H4/H4c. Dữ liệu tổng hợp nhỏ
tự dựng, không đọc dữ liệu thật."""
from __future__ import annotations

import numpy as np
import pandas as pd

from vnrank.features.feature_registry import FEATURE_COLUMNS
from vnrank.models.trend_integration import (
    a3_precision_distribution,
    build_labeled_features_sign,
    h4_test,
    h4c_test,
    precision_at_k_for_selection,
    two_stage_select,
)


def _zero_features_row(date, symbol) -> dict:
    row = {"date": date, "symbol": symbol}
    for col in FEATURE_COLUMNS:
        row[col] = 0.0
    return row


# --- build_labeled_features_sign: y_sign = sign(r), r<=0 -> -1 ---------------------------------


def test_build_labeled_features_sign_assigns_minus_one_to_zero_return():
    features = pd.DataFrame([_zero_features_row("2024-01-01", "AAA"), _zero_features_row("2024-01-01", "BBB")])
    features["date"] = pd.to_datetime(features["date"])
    labels_h = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "symbol": ["AAA", "BBB"],
            "r": [0.0, 0.05],
            "exit_date_resolved": pd.to_datetime(["2024-01-05", "2024-01-05"]),
        }
    )
    out = build_labeled_features_sign(features, labels_h)
    y_by_symbol = dict(zip(out["symbol"], out["y_sign"]))
    assert y_by_symbol["AAA"] == -1  # r == 0 -> lop -1
    assert y_by_symbol["BBB"] == 1


def test_build_labeled_features_sign_drops_missing_r():
    features = pd.DataFrame([_zero_features_row("2024-01-01", "AAA"), _zero_features_row("2024-01-01", "BBB")])
    features["date"] = pd.to_datetime(features["date"])
    labels_h = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "symbol": ["AAA", "BBB"],
            "r": [0.02, np.nan],
            "exit_date_resolved": pd.to_datetime(["2024-01-05", None]),
        }
    )
    out = build_labeled_features_sign(features, labels_h)
    assert list(out["symbol"]) == ["AAA"]


# --- two_stage_select: top-M theo score, roi top-K theo rerank_col -----------------------------


def test_two_stage_select_picks_top_k_within_top_m():
    dates = pd.to_datetime(["2024-01-01"] * 6)
    l1_scores = pd.DataFrame(
        {"date": dates, "symbol": ["A", "B", "C", "D", "E", "F"], "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]}
    )
    rerank = pd.DataFrame(
        {"date": dates, "symbol": ["A", "B", "C", "D", "E", "F"], "p_diff": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]}
    )
    # top-M=4 theo score: A,B,C,D. Trong do top-K=2 theo p_diff: B (0.9), D (0.8).
    out = two_stage_select(l1_scores, rerank, "p_diff", m=4, k=2)
    assert set(out["symbol"]) == {"B", "D"}


def test_two_stage_select_uses_all_candidates_when_fewer_than_m():
    dates = pd.to_datetime(["2024-01-01"] * 3)
    l1_scores = pd.DataFrame({"date": dates, "symbol": ["A", "B", "C"], "score": [0.9, 0.5, 0.1]})
    rerank = pd.DataFrame({"date": dates, "symbol": ["A", "B", "C"], "p_diff": [0.1, 0.9, 0.5]})
    out = two_stage_select(l1_scores, rerank, "p_diff", m=20, k=2)
    assert set(out["symbol"]) == {"B", "C"}  # top-M = ca 3 ma, top-K=2 theo p_diff: B, C


# --- precision_at_k_for_selection: bo ngay thieu K ma, tinh dung ty le r_excess>0 ---------------


def test_precision_at_k_for_selection_computes_correct_ratio():
    selected = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 3), "symbol": ["A", "B", "C"]})
    labels_h = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-01"] * 3), "symbol": ["A", "B", "C"], "r_excess": [0.01, -0.02, 0.03]}
    )
    out = precision_at_k_for_selection(selected, labels_h, k=3)
    assert out.iloc[0] == 2 / 3


def test_precision_at_k_for_selection_skips_day_with_too_few_symbols():
    selected = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-01"]), "symbol": ["A", "B"]})
    labels_h = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-01", "2024-01-01"]), "symbol": ["A", "B"], "r_excess": [0.01, -0.02]}
    )
    out = precision_at_k_for_selection(selected, labels_h, k=3)
    assert len(out) == 0


# --- a3_precision_distribution: xao ngau nhien, deterministic theo seed ------------------------


def test_a3_precision_distribution_is_deterministic_given_seed():
    dates = pd.to_datetime(["2024-01-01"] * 5)
    l1_scores = pd.DataFrame({"date": dates, "symbol": list("ABCDE"), "score": [0.9, 0.8, 0.7, 0.6, 0.5]})
    labels_h = pd.DataFrame({"date": dates, "symbol": list("ABCDE"), "r_excess": [0.01, -0.01, 0.02, -0.02, 0.0]})
    a = a3_precision_distribution(l1_scores, labels_h, m=5, k=2, n_repeat=50, seed=7)
    b = a3_precision_distribution(l1_scores, labels_h, m=5, k=2, n_repeat=50, seed=7)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 50
    assert np.all((a >= 0) & (a <= 1))


def test_a3_precision_distribution_nan_when_fewer_than_k_candidates():
    dates = pd.to_datetime(["2024-01-01"])
    l1_scores = pd.DataFrame({"date": dates, "symbol": ["A"], "score": [0.9]})
    labels_h = pd.DataFrame({"date": dates, "symbol": ["A"], "r_excess": [0.01]})
    out = a3_precision_distribution(l1_scores, labels_h, m=5, k=2, n_repeat=10, seed=1)
    assert np.all(np.isnan(out))


# --- h4_test / h4c_test: bien delta va hai phia ------------------------------------------------


def test_h4_test_not_rejected_when_a1_clearly_better_than_margin():
    dates = pd.bdate_range("2024-01-01", periods=30)
    rng = np.random.default_rng(0)
    prec_a1 = pd.Series(0.6 + rng.normal(scale=0.02, size=30), index=dates)
    prec_a0 = pd.Series(0.3 + rng.normal(scale=0.02, size=30), index=dates)
    out = h4_test(prec_a1, prec_a0, delta=0.02, h=5)
    assert out["mean"] > 0
    assert out["p_value"] < 0.05  # A1 vuot han bien, bac bo H0 (khong kem hon)


def test_h4c_test_two_sided_p_value_high_when_no_real_difference():
    dates = pd.bdate_range("2024-01-01", periods=30)
    rng = np.random.default_rng(1)
    prec_a1 = pd.Series(0.5 + rng.normal(scale=0.02, size=30), index=dates)
    prec_a2 = pd.Series(0.5 + rng.normal(scale=0.02, size=30), index=dates)  # cung phan phoi, khac nhieu doc lap
    out = h4c_test(prec_a1, prec_a2, h=5)
    assert out["p_value_two_sided"] > 0.05
    assert 0.0 <= out["p_value_two_sided"] <= 1.0
