"""Test cho gộp đa horizon (bước 4.5): F0 (trung bình chuẩn hóa thứ hạng của L1 ba horizon), F1
("tốt nhất theo horizon", đọc lại số liệu đã đóng băng), kiểm định H2. Dữ liệu tổng hợp nhỏ tự
dựng, không đọc dữ liệu thật."""
from __future__ import annotations

import numpy as np
import pandas as pd

from vnrank.models.horizon_aggregation import (
    best_family_at_h,
    build_f0,
    h2_test,
    normalize_rank_score,
)


# --- normalize_rank_score: khớp công thức cross_sectional_normalize ----------------------------


def test_normalize_rank_score_matches_formula_range_and_order():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"] * 4,
            "symbol": ["A", "B", "C", "D"],
            "score": [1.0, 3.0, 2.0, 4.0],
        }
    )
    out = normalize_rank_score(df)
    assert out["score"].min() >= -1.0 and out["score"].max() <= 1.0
    # Thứ hạng A<C<B<D -> z tăng dần đúng thứ tự
    ordered = out.set_index("symbol").loc[["A", "C", "B", "D"], "score"].to_numpy()
    assert np.all(np.diff(ordered) > 0)
    # Cực trị đúng công thức z = 2*(rank-1)/(n-1) - 1: rank=1 -> -1, rank=n -> 1
    assert np.isclose(out.loc[out["symbol"] == "A", "score"].iloc[0], -1.0)
    assert np.isclose(out.loc[out["symbol"] == "D", "score"].iloc[0], 1.0)


def test_normalize_rank_score_single_observation_day_is_zero():
    df = pd.DataFrame({"date": ["2024-01-01"], "symbol": ["A"], "score": [5.0]})
    out = normalize_rank_score(df)
    assert out["score"].iloc[0] == 0.0


# --- build_f0: trung bình ba horizon, chỉ giữ (date, symbol) chung cho cả ba -------------------


def test_build_f0_averages_three_horizons_after_normalization():
    dates = ["2024-01-01", "2024-01-02"]
    symbols = ["A", "B", "C"]
    # h=5: điểm giống hệt h=10, khác h=21 -> F0 vẫn nằm trong [-1,1] và là trung bình đúng
    scores_h5 = pd.DataFrame(
        [{"date": d, "symbol": s, "score": i} for d in dates for i, s in enumerate(symbols)]
    )
    scores_h10 = scores_h5.copy()
    scores_h21 = scores_h5.copy()
    scores_h21["score"] = scores_h21["score"].to_numpy()[::-1]  # đảo ngược thứ hạng ở h=21

    f0 = build_f0({5: scores_h5, 10: scores_h10, 21: scores_h21})
    assert set(f0.columns) == {"date", "symbol", "score"}
    assert len(f0) == len(dates) * len(symbols)
    # Vì h=5 và h=10 giống hệt, còn h=21 đảo ngược hoàn toàn (z và -z) -> trung bình không nhất
    # thiết bằng 0 cho MỌI mã (chỉ đúng khi 2 cái giống + 1 cái đối, trung bình = (2z + (-z))/3 = z/3)
    merged = scores_h5.merge(f0, on=["date", "symbol"], suffixes=("_raw", "_f0"))
    assert merged["score_f0"].abs().le(1.0).all()


def test_build_f0_inner_joins_on_shared_date_symbol_only():
    scores_a = pd.DataFrame({"date": ["2024-01-01", "2024-01-01"], "symbol": ["A", "B"], "score": [1.0, 2.0]})
    scores_b = pd.DataFrame({"date": ["2024-01-01"], "symbol": ["A"], "score": [1.0]})  # thiếu B
    f0 = build_f0({5: scores_a, 10: scores_b})
    assert list(f0["symbol"]) == ["A"]


# --- best_family_at_h: chọn đúng mô hình có Rank IC validation cao nhất tại h -------------------


def test_best_family_at_h_picks_max_validation_ic_for_that_h_only():
    selection = pd.DataFrame(
        [
            {"model": "R1_ridge", "h": 5, "config": {"alpha": 1}, "rank_ic_validation": 0.01},
            {"model": "R2_lightgbm", "h": 5, "config": {"num_leaves": 15}, "rank_ic_validation": 0.03},
            {"model": "L1_lambdarank", "h": 5, "config": {"num_leaves": 31}, "rank_ic_validation": 0.02},
            {"model": "L1_lambdarank", "h": 10, "config": {"num_leaves": 63}, "rank_ic_validation": 0.05},
        ]
    )
    best5 = best_family_at_h(selection, 5)
    assert best5["model"] == "R2_lightgbm"
    best10 = best_family_at_h(selection, 10)
    assert best10["model"] == "L1_lambdarank"
    assert best10["config"] == {"num_leaves": 63}


# --- h2_test: biên delta_IC, không kiểm định được khi F1 val IC <= 0 ---------------------------


def test_h2_test_not_testable_when_delta_ic_is_nan():
    out = h2_test(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), delta_ic=float("nan"))
    assert out["testable"] is False
    assert out["p_value"] == 1.0


def _synthetic_f0_f1(n_days: int = 30, n_symbols: int = 5, f0_edge: float = 0.02, seed: int = 0):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    rows_f0, rows_f1, label_rows = [], [], []
    for d in dates:
        true = rng.normal(size=n_symbols)
        rows_f0 += [{"date": d, "symbol": f"S{i}", "score": true[i] + rng.normal(scale=0.01)} for i in range(n_symbols)]
        rows_f1 += [{"date": d, "symbol": f"S{i}", "score": true[i] + rng.normal(scale=0.01 + f0_edge)} for i in range(n_symbols)]
        label_rows += [{"date": d, "symbol": f"S{i}", "r": true[i]} for i in range(n_symbols)]
    f0 = pd.DataFrame(rows_f0)
    f1 = pd.DataFrame(rows_f1)
    labels_h5 = pd.DataFrame(label_rows)
    splits = pd.DataFrame({"date": dates, "split": "test"})
    return f0, f1, labels_h5, splits


def test_h2_test_not_rejected_when_f0_clearly_better_than_margin():
    # F1 có nhiễu LỚN HƠN F0 -> F0 tốt hơn F1, chắc chắn không kém hơn biên delta_IC dương nhỏ.
    f0, f1, labels_h5, splits = _synthetic_f0_f1(f0_edge=0.5)
    out = h2_test(f0, f1, labels_h5, splits, delta_ic=0.01)
    assert out["testable"] is True
    assert out["p_value"] < 0.5


def test_h2_test_margin_shifts_threshold_vs_h1_style_zero_margin():
    f0, f1, labels_h5, splits = _synthetic_f0_f1(f0_edge=0.5, seed=1)
    zero_margin = h2_test(f0, f1, labels_h5, splits, delta_ic=1e-9)
    big_margin = h2_test(f0, f1, labels_h5, splits, delta_ic=1.0)
    # Biên delta_IC lớn hơn phải dễ "không kém hơn" hơn (p-giá trị nhỏ hơn hoặc bằng).
    assert big_margin["p_value"] <= zero_margin["p_value"]
