"""Test cho feature_registry.py (bước 3.5). Chỉ dùng dữ liệu tổng hợp nhỏ, không gọi mạng thật."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vnrank.features.feature_registry import (
    FEATURE_COLUMNS,
    compute_features,
    cross_sectional_normalize,
    fill_market_calendar_gaps,
    macd_hist,
    past_return,
    rolling_beta,
    rsi,
    sma_ratio,
    volume_ratio,
)


def _price_df(closes, volumes=None, start="2024-01-01"):
    n = len(closes)
    dates = pd.date_range(start, periods=n, freq="D")
    volumes = volumes or [1000] * n
    return pd.DataFrame({"date": dates, "open": closes, "high": closes, "low": closes, "close": closes, "volume": volumes})


def test_past_return_log_and_nan_before_window():
    df = _price_df([10, 11, 12, 13, 14, 20])
    out = past_return(df, window=3)
    assert out.iloc[:3].isna().all()
    assert out.iloc[3] == pytest.approx(np.log(13 / 10))
    assert out.iloc[5] == pytest.approx(np.log(20 / 12))


def test_sma_ratio_matches_manual_mean():
    df = _price_df([10, 10, 10, 20])
    out = sma_ratio(df, window=3)
    assert out.iloc[:2].isna().all()
    manual = 20 / ((10 + 10 + 20) / 3) - 1
    assert out.iloc[3] == pytest.approx(manual)


def test_rsi_bounds_and_trend_direction():
    up = _price_df([10 + i for i in range(30)])
    down = _price_df([40 - i for i in range(30)])
    rsi_up = rsi(up, window=14)
    rsi_down = rsi(down, window=14)
    assert rsi_up.iloc[-1] > 90
    assert rsi_down.iloc[-1] < 10
    assert rsi_up.iloc[:14].isna().all()


def test_macd_hist_zero_for_flat_price():
    df = _price_df([100.0] * 40)
    out = macd_hist(df)
    valid = out.dropna()
    assert (valid.abs() < 1e-9).all()


def test_volume_ratio_excludes_current_day():
    df = _price_df([10] * 4, volumes=[100, 100, 100, 400])
    out = volume_ratio(df, window=3)
    # trung bình 3 phiên LIỀN TRƯỚC (không gồm phiên hiện tại): (100+100+100)/3 = 100
    manual = 400 / 100 - 1
    assert out.iloc[3] == pytest.approx(manual)
    assert out.iloc[:3].isna().all()


def test_rolling_beta_recovers_known_slope():
    rng = np.random.default_rng(0)
    market_ret = pd.Series(rng.normal(0, 0.01, 200))
    stock_ret = 2.0 * market_ret  # beta thật = 2, không nhiễu
    beta = rolling_beta(stock_ret, market_ret, window=63)
    assert beta.iloc[-1] == pytest.approx(2.0, abs=1e-6)
    assert beta.iloc[:62].isna().all()


def test_fill_market_calendar_gaps_forward_fills():
    market_df = pd.DataFrame({"date": ["2023-01-02", "2023-01-04"], "close": [100.0, 110.0]})
    calendar = ["2023-01-02", "2023-01-03", "2023-01-04"]
    out = fill_market_calendar_gaps(market_df, calendar)
    assert out.loc[out["date"] == "2023-01-03", "close"].iloc[0] == 100.0


def test_compute_features_no_lookahead():
    """giao_thuc_chia_du_lieu_bias.md rủi ro #1: dịch giá tương lai không được làm đổi đặc trưng
    tại ngày t. Thay đổi close/volume CHỈ ở các phiên SAU một mốc t_cut, feature tại t_cut trở về
    trước phải giữ nguyên hệt."""
    rng = np.random.default_rng(1)
    n = 160
    closes = list(10 + np.cumsum(rng.normal(0, 0.1, n)))
    panel_a = _price_df(closes).assign(symbol="AAA")
    market_df = _price_df([100 + i * 0.05 for i in range(n)])[["date", "close"]]

    panel_b = panel_a.copy()
    t_cut = 120
    future_mask = panel_b.index > t_cut
    panel_b.loc[future_mask, "close"] = panel_b.loc[future_mask, "close"] * 5.0
    panel_b.loc[future_mask, "high"] = panel_b.loc[future_mask, "close"]
    panel_b.loc[future_mask, "low"] = panel_b.loc[future_mask, "close"]
    panel_b.loc[future_mask, "volume"] = panel_b.loc[future_mask, "volume"] * 50

    out_a = compute_features(panel_a, market_df)
    out_b = compute_features(panel_b, market_df)
    past = out_a["date"] <= out_a.loc[t_cut, "date"]
    pd.testing.assert_frame_equal(
        out_a.loc[past].reset_index(drop=True), out_b.loc[past].reset_index(drop=True)
    )


def test_compute_features_returns_expected_columns():
    closes = list(range(10, 150))
    panel = pd.concat(
        [
            _price_df(closes).assign(symbol="AAA"),
            _price_df([c * 1.5 for c in closes]).assign(symbol="BBB"),
        ],
        ignore_index=True,
    )
    market_df = _price_df([100 + i * 0.1 for i in range(len(closes))])[["date", "close"]]
    out = compute_features(panel, market_df)
    assert list(out.columns) == ["symbol", "date"] + FEATURE_COLUMNS
    assert set(out["symbol"]) == {"AAA", "BBB"}
    assert len(out) == len(panel)


def test_cross_sectional_normalize_maps_rank_to_minus1_plus1():
    dates = pd.to_datetime(["2024-01-01"] * 3)
    features = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["A", "B", "C"],
            "ret_21": [1.0, 2.0, 3.0],
        }
    )
    universe = features[["date", "symbol"]]
    out = cross_sectional_normalize(features, universe, feature_cols=["ret_21"])
    ranked = out.set_index("symbol")["ret_21"]
    assert ranked["A"] == pytest.approx(-1.0)
    assert ranked["B"] == pytest.approx(0.0)
    assert ranked["C"] == pytest.approx(1.0)


def test_cross_sectional_normalize_fills_missing_with_median():
    dates = pd.to_datetime(["2024-01-01"] * 3)
    features = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["A", "B", "C"],
            "ret_21": [1.0, np.nan, 3.0],
        }
    )
    universe = features[["date", "symbol"]]
    out = cross_sectional_normalize(features, universe, feature_cols=["ret_21"])
    # B được thay bằng trung vị (2.0, trùng giá trị không có trong tập), rank giữa A và C -> 0
    assert out.set_index("symbol").loc["B", "ret_21"] == pytest.approx(0.0)


def test_cross_sectional_normalize_all_missing_day_is_zero():
    dates = pd.to_datetime(["2024-01-01"] * 2)
    features = pd.DataFrame({"date": dates, "symbol": ["A", "B"], "ret_21": [np.nan, np.nan]})
    universe = features[["date", "symbol"]]
    out = cross_sectional_normalize(features, universe, feature_cols=["ret_21"])
    assert (out["ret_21"] == 0.0).all()
