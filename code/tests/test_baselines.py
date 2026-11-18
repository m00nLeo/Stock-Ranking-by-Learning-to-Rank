"""Test cho baseline không ML (bước 4.2). Dữ liệu tổng hợp nhỏ tự dựng."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vnrank.models.baselines import (
    amihud_illiquidity,
    equal_weight_portfolio_returns,
    imom_score,
    momentum_score,
    vn30_index_returns,
)

# --- B2: momentum ------------------------------------------------------------------------------


def test_momentum_score_skip_zero_matches_plain_log_return():
    close = pd.Series([100.0, 101.0, 102.0, 105.0, 103.0, 108.0])
    s = momentum_score(close, window=3, skip=0)
    expected = np.log(close / close.shift(3))
    pd.testing.assert_series_equal(s, expected)


def test_momentum_score_skip_nonzero_excludes_recent_sessions():
    close = pd.Series([100.0, 110.0, 120.0, 90.0, 95.0, 130.0])
    # tại t=5 (index 5, gia tri 130): window=4, skip=2 -> ln(C[3]/C[1]) = ln(90/110)
    s = momentum_score(close, window=4, skip=2)
    assert s.iloc[5] == pytest.approx(np.log(90.0 / 110.0))


# --- B3: IMOM ------------------------------------------------------------------------------


def test_imom_score_near_zero_when_stock_is_pure_beta_times_market():
    rng = np.random.default_rng(0)
    n = 260
    mkt_ret = pd.Series(rng.normal(0, 0.01, n))
    beta_true = 1.7
    stock_ret = beta_true * mkt_ret  # không có phần riêng (idiosyncratic) nào
    mkt_close = pd.Series(100.0 * np.exp(mkt_ret.cumsum()))
    stock_close = pd.Series(50.0 * np.exp(stock_ret.cumsum()))

    imom = imom_score(
        stock_close, stock_ret, mkt_close, mkt_ret, window=63, skip=21, estimation_window=126
    )
    # sau khi beta cuộn hội tụ (đủ estimation_window), IMOM phải gần 0 vì không có phần dư
    assert abs(imom.iloc[-1]) < 1e-9


def test_imom_score_positive_when_stock_outperforms_beta_adjusted_market():
    rng = np.random.default_rng(1)
    n = 260
    mkt_ret = pd.Series(rng.normal(0, 0.01, n))
    beta_true = 1.0
    extra_ret = pd.Series(np.zeros(n))
    extra_ret.iloc[-50:-21] = 0.01  # phần riêng dương, trong cửa sổ momentum nhưng ngoài skip
    stock_ret = beta_true * mkt_ret + extra_ret
    mkt_close = pd.Series(100.0 * np.exp(mkt_ret.cumsum()))
    stock_close = pd.Series(50.0 * np.exp(stock_ret.cumsum()))

    imom = imom_score(
        stock_close, stock_ret, mkt_close, mkt_ret, window=63, skip=21, estimation_window=126
    )
    assert imom.iloc[-1] > 0.1


# --- B4: Amihud ----------------------------------------------------------------------------


def test_amihud_illiquidity_matches_manual_calculation():
    close = pd.Series([10.0, 10.5, 10.0, 11.0])
    volume = pd.Series([1000.0, 1000.0, 1000.0, 1000.0])
    out = amihud_illiquidity(close, volume, window=3)
    daily_ret = np.log(close / close.shift(1))
    dollar_vol = close * volume
    manual_last = (daily_ret.abs() / dollar_vol).iloc[1:4].mean()
    assert out.iloc[3] == pytest.approx(manual_last)
    assert out.iloc[:3].isna().all()  # chưa đủ window=3 ngày lợi suất


def test_amihud_illiquidity_higher_for_lower_dollar_volume():
    close = pd.Series([10.0] * 5 + [10.5] * 5)
    high_volume = pd.Series([1_000_000.0] * 10)
    low_volume = pd.Series([1_000.0] * 10)
    illiq_high_vol = amihud_illiquidity(close, high_volume, window=5).iloc[-1]
    illiq_low_vol = amihud_illiquidity(close, low_volume, window=5).iloc[-1]
    assert illiq_low_vol > illiq_high_vol


# --- B0: đồng trọng số -----------------------------------------------------------------------


def test_equal_weight_portfolio_returns_averages_across_symbols_same_day():
    labels = pd.DataFrame(
        {
            "date": ["d1", "d1", "d1", "d2"],
            "symbol": ["AAA", "BBB", "CCC", "AAA"],
            "h": [5, 5, 5, 5],
            "r": [0.01, 0.03, -0.02, 0.05],
        }
    )
    universe = pd.DataFrame({"date": ["d1", "d1", "d2"], "symbol": ["AAA", "BBB", "AAA"]})
    out = equal_weight_portfolio_returns(labels, universe, h=5)
    assert out["d1"] == pytest.approx((0.01 + 0.03) / 2)  # CCC khong thuoc universe ngay d1
    assert out["d2"] == pytest.approx(0.05)


def test_equal_weight_portfolio_returns_filters_by_horizon():
    labels = pd.DataFrame(
        {"date": ["d1", "d1"], "symbol": ["AAA", "AAA"], "h": [5, 10], "r": [0.01, 0.99]}
    )
    universe = pd.DataFrame({"date": ["d1"], "symbol": ["AAA"]})
    out = equal_weight_portfolio_returns(labels, universe, h=5)
    assert out["d1"] == pytest.approx(0.01)


# --- B1: VN30 ------------------------------------------------------------------------------


def test_vn30_index_returns_entry_t_plus_1_exit_t_plus_1_plus_h():
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "close": [100.0, 102.0, 104.0, 108.0, 110.0, 120.0],
        }
    )
    out = vn30_index_returns(prices, h=3)
    # t = index 0: entry = close[1] = 102, exit = close[1+3] = close[4] = 110
    assert out.iloc[0] == pytest.approx(110.0 / 102.0 - 1.0)


def test_vn30_index_returns_nan_near_end_of_series():
    prices = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=4, freq="D"), "close": [100.0, 101.0, 102.0, 103.0]}
    )
    out = vn30_index_returns(prices, h=5)
    assert out.isna().all()
