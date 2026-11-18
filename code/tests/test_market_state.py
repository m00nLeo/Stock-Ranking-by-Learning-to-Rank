"""Test cho market_state.py (bước 3.11). Chỉ dùng dữ liệu tổng hợp nhỏ, không gọi mạng thật."""
from __future__ import annotations

import pandas as pd

from vnrank.labels.market_state import compute_market_state


def _df(dates, closes):
    n = len(dates)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000] * n,
        }
    )


def _calendar(n, start="2024-01-01"):
    return list(pd.date_range(start, periods=n, freq="D"))


def test_tang_when_close_and_short_sma_above_long_sma():
    # giá tăng đều: tại ngày đủ 5 phiên, close > SMA_long và SMA_short > SMA_long
    closes = [10.0 + i for i in range(8)]
    out = compute_market_state(_df(_calendar(8), closes), sma_short=3, sma_long=5)
    assert out["state"].iloc[:4].isna().all()
    assert (out["state"].iloc[4:] == "tang").all()


def test_giam_when_close_and_short_sma_below_long_sma():
    # giá giảm đều: tại ngày đủ 5 phiên, close < SMA_long và SMA_short < SMA_long
    closes = [20.0 - i for i in range(8)]
    out = compute_market_state(_df(_calendar(8), closes), sma_short=3, sma_long=5)
    assert out["state"].iloc[:4].isna().all()
    assert (out["state"].iloc[4:] == "giam").all()


def test_di_ngang_for_remaining_cases():
    # giá phẳng: không thỏa tang cũng không thỏa giam -> di_ngang
    closes = [10.0] * 8
    out = compute_market_state(_df(_calendar(8), closes), sma_short=3, sma_long=5)
    assert out["state"].iloc[:4].isna().all()
    assert (out["state"].iloc[4:] == "di_ngang").all()


def test_state_nan_before_enough_history_for_long_sma():
    # SMA200 mặc định cần 200 phiên: 199 ngày đầu chưa đủ -> state None/NaN
    closes = [100.0 + i * 0.1 for i in range(210)]
    out = compute_market_state(_df(_calendar(210), closes))
    assert out["state"].iloc[:199].isna().all()
    assert out["state"].iloc[199:].notna().all()