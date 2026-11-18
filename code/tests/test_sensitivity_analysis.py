"""Test cho phân tích độ nhạy (bước 4.8): co giãn chi phí, lọc theo ngưỡng thanh khoản, lọc theo
giai đoạn thị trường. Dữ liệu tổng hợp nhỏ tự dựng, không đọc dữ liệu thật (phần đọc dữ liệu thật
để tính ngưỡng thanh khoản "chặt hơn" chỉ chạy trong `__main__`, không test ở đây)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from vnrank.evaluation.sensitivity_analysis import (
    C_BUY_BASE,
    C_RT_BASE,
    C_SELL_BASE,
    macro_f1_under_filter,
    market_state_rebalance_dates,
    rank_ic_mean_under_filter,
    scaled_costs,
    sharpe_for_config,
)


# --- scaled_costs: giữ nguyên tỷ trọng mua/bán, cộng đúng bằng c_rt_target ----------------------


def test_scaled_costs_preserves_ratio_and_sums_to_target():
    c_buy, c_sell = scaled_costs(0.0120)
    assert abs((c_buy + c_sell) - 0.0120) < 1e-12
    assert abs(c_buy / c_sell - C_BUY_BASE / C_SELL_BASE) < 1e-9


def test_scaled_costs_identity_at_base_rt():
    c_buy, c_sell = scaled_costs(C_RT_BASE)
    assert abs(c_buy - C_BUY_BASE) < 1e-12
    assert abs(c_sell - C_SELL_BASE) < 1e-12


# --- rank_ic_mean_under_filter: lọc thanh khoản làm thay đổi tập ứng viên -----------------------


def _synthetic_scores_and_labels():
    dates = pd.bdate_range("2024-01-01", periods=10)
    rng = np.random.default_rng(0)
    rows_scored, rows_labels, rows_liquidity = [], [], []
    for d in dates:
        true = rng.normal(size=6)
        for i in range(6):
            rows_scored.append({"date": d, "symbol": f"S{i}", "score": true[i] + rng.normal(scale=0.01)})
            rows_labels.append({"date": d, "symbol": f"S{i}", "r": true[i], "y": 1 if true[i] > 0 else -1})
            # S0-S2 thanh khoan cao, S3-S5 thanh khoan thap
            rows_liquidity.append({"date": d, "symbol": f"S{i}", "avg_value_l": 200.0 if i < 3 else 50.0})
    scored = pd.DataFrame(rows_scored)
    labels_h = pd.DataFrame(rows_labels)
    splits = pd.DataFrame({"date": dates, "split": "test"})
    liquidity_table = pd.DataFrame(rows_liquidity)
    return scored, labels_h, splits, liquidity_table


def test_rank_ic_mean_under_filter_without_filter_uses_all_symbols():
    scored, labels_h, splits, _ = _synthetic_scores_and_labels()
    ic = rank_ic_mean_under_filter(scored, labels_h, splits)
    assert -1.0 <= ic <= 1.0


def test_rank_ic_mean_under_filter_with_stricter_threshold_restricts_sample():
    scored, labels_h, splits, liquidity_table = _synthetic_scores_and_labels()
    merged = scored.merge(liquidity_table, on=["date", "symbol"])
    n_before = len(merged)
    n_after = len(merged[merged["avg_value_l"] >= 100.0])
    assert n_after < n_before  # loc chat hon phai bot ma (S3-S5 thanh khoan thap bi loai)
    ic_strict = rank_ic_mean_under_filter(scored, labels_h, splits, liquidity_table, threshold=100.0)
    assert -1.0 <= ic_strict <= 1.0


# --- macro_f1_under_filter: tuong tu, tren nhan phan loai ---------------------------------------


def test_macro_f1_under_filter_runs_and_bounded():
    scored, labels_h, splits, liquidity_table = _synthetic_scores_and_labels()
    pred_scored = scored.rename(columns={"score": "pred"}).copy()
    pred_scored["pred"] = np.sign(pred_scored["pred"]).replace(0, 1).astype(int)
    f1 = macro_f1_under_filter(pred_scored, labels_h, splits)
    assert 0.0 <= f1 <= 1.0
    f1_strict = macro_f1_under_filter(pred_scored, labels_h, splits, liquidity_table, threshold=100.0)
    assert 0.0 <= f1_strict <= 1.0


# --- sharpe_for_config: K khac nhau cho ket qua khac nhau ---------------------------------------


def test_sharpe_for_config_changes_with_k():
    dates = pd.bdate_range("2024-01-01", periods=20)
    rng = np.random.default_rng(1)
    rows_scored, rows_labels = [], []
    for d in dates:
        for i in range(10):
            score = rng.normal()
            rows_scored.append({"date": d, "symbol": f"S{i}", "score": score})
            rows_labels.append({"date": d, "symbol": f"S{i}", "r": rng.normal(scale=0.02)})
    scored = pd.DataFrame(rows_scored)
    labels_h = pd.DataFrame(rows_labels)
    rebalance_dates = list(dates[::5])

    sharpe_k5 = sharpe_for_config(scored, labels_h, rebalance_dates, k=5, c_buy=C_BUY_BASE, c_sell=C_SELL_BASE, h=5)
    sharpe_k10 = sharpe_for_config(scored, labels_h, rebalance_dates, k=10, c_buy=C_BUY_BASE, c_sell=C_SELL_BASE, h=5)
    assert sharpe_k5 == sharpe_k5  # khong NaN (du mau)
    assert sharpe_k10 == sharpe_k10


# --- market_state_rebalance_dates: chỉ giữ ngày đúng giai đoạn ----------------------------------


def test_market_state_rebalance_dates_filters_by_regime():
    dates = pd.bdate_range("2024-01-01", periods=6)
    market_state = pd.DataFrame({"date": dates, "state": ["tang", "tang", "giam", "giam", "di_ngang", "di_ngang"]})
    out = market_state_rebalance_dates(list(dates), market_state, "giam")
    assert out == [dates[2], dates[3]]


def test_market_state_rebalance_dates_empty_when_no_match():
    dates = pd.bdate_range("2024-01-01", periods=3)
    market_state = pd.DataFrame({"date": dates, "state": ["tang", "tang", "tang"]})
    out = market_state_rebalance_dates(list(dates), market_state, "giam")
    assert out == []
