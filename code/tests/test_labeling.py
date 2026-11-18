"""Test cho labeling.py (bước 3.6). Chỉ dùng dữ liệu tổng hợp nhỏ, không gọi mạng thật."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vnrank.labels.labeling import (
    C_RT_BASE,
    add_excess_and_relevance,
    build_trading_calendar,
    compute_raw_labels,
    compute_symbol_labels,
    ewma_sigma,
)


def _df(dates, closes, highs=None, lows=None, opens=None, volumes=None):
    n = len(dates)
    highs = highs or closes
    lows = lows or closes
    opens = opens or closes
    volumes = volumes or [1000] * n
    return pd.DataFrame(
        {"date": pd.to_datetime(dates), "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}
    )


def _calendar(n, start="2024-01-01"):
    return list(pd.date_range(start, periods=n, freq="D"))


def test_build_trading_calendar_unions_dates():
    panel = pd.concat(
        [
            _df(["2024-01-01", "2024-01-03"], [10, 11]).assign(symbol="A"),
            _df(["2024-01-02", "2024-01-03"], [20, 21]).assign(symbol="B"),
        ]
    )
    cal = build_trading_calendar(panel)
    assert cal == list(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))


def test_ewma_sigma_zero_for_flat_price_after_warmup():
    closes = [100.0] * 25
    log_ret = np.log(pd.Series(closes) / pd.Series(closes).shift(1))
    sigma = ewma_sigma(log_ret, span=20)
    # log_ret[0] luôn NaN (shift); cần đủ 20 giá trị lợi suất hợp lệ (vị trí 1..20) nên kết quả đầu
    # tiên xuất hiện ở vị trí 20, không phải 19
    assert sigma.iloc[:20].isna().all()
    assert sigma.iloc[20] == pytest.approx(0.0)


def test_entry_skipped_when_next_session_has_no_trade():
    cal = _calendar(6)
    # mã không có dữ liệu ở cal[1] (phiên vào lệnh của t=cal[0])
    dates = [cal[0], cal[2], cal[3], cal[4], cal[5]]
    df = _df(dates, [10, 10, 10, 10, 10])
    out = compute_symbol_labels(df, exchange="HOSE", calendar=cal, horizons=(1,))
    assert out[out["date"] == cal[0]].empty


def test_entry_skipped_when_ceiling_locked():
    cal = _calendar(6)
    closes = [10.0, 10.7, 10.7, 10.7, 10.7, 10.7]  # cal[1] tăng đúng +7%, O=H=L=C -> khóa trần
    df = _df(cal, closes)
    out = compute_symbol_labels(df, exchange="HOSE", calendar=cal, horizons=(1,))
    assert out[out["date"] == cal[0]].empty


def test_exit_extended_when_floor_locked_at_nominal_exit():
    cal = _calendar(6)
    # t = cal[0], entry = cal[1] (giá 10), thoát dự kiến h=2 -> cal[3]
    # cal[3] khóa sàn so với cal[2] (giảm đúng 7%); cal[4] không khóa -> dời sang cal[4]
    closes = [10.0, 10.0, 10.0, 9.3, 9.35, 9.4]
    df = _df(cal, closes)
    out = compute_symbol_labels(df, exchange="HOSE", calendar=cal, horizons=(2,))
    row = out[(out["date"] == cal[0]) & (out["h"] == 2)].iloc[0]
    assert row["exit_date_resolved"] == cal[4]
    assert row["exit_extended_khoa_san"]
    assert row["r"] == pytest.approx(9.35 / 10.0 - 1.0)


def test_trend_label_touches_upper_barrier_at_expected_date():
    cal = _calendar(27)
    closes = [100.0] * 22 + [100.0, 100.3, 100.7, 100.5, 100.6]
    df = _df(cal, closes)
    out = compute_symbol_labels(df, exchange="HOSE", calendar=cal, horizons=(5,))
    t = cal[20]
    row = out[(out["date"] == t) & (out["h"] == 5)].iloc[0]
    assert row["y"] == 1
    assert row["t1"] == cal[24]
    assert row["san_chi_phi"]  # sigma = 0 nên rào bị sàn chi phí 0,60% chặn


def test_trend_label_no_touch_returns_zero():
    cal = _calendar(27)
    closes = [100.0] * 26 + [100.1]  # biến động quá nhỏ, không chạm rào nào
    df = _df(cal, closes)
    out = compute_symbol_labels(df, exchange="HOSE", calendar=cal, horizons=(5,))
    t = cal[20]
    row = out[(out["date"] == t) & (out["h"] == 5)].iloc[0]
    assert row["y"] == 0
    assert row["t1"] == cal[26]


def test_labels_do_not_depend_on_prices_far_beyond_the_window():
    """giao_thuc_chia_du_lieu_bias.md rủi ro #1 áp dụng tương tự cho nhãn: thay đổi giá ở các phiên
    nằm NGOÀI cửa sổ cần dùng cho một nhãn (sau t+1+h) không được làm đổi nhãn đó."""
    cal = _calendar(30)
    closes_a = [100.0] * 22 + [100.0, 100.3, 100.7, 100.5, 100.6, 100.6, 100.6, 100.6]
    closes_b = list(closes_a)
    closes_b[27:] = [999.0, 1.0, 500.0]  # xáo trộn mạnh các phiên xa sau cửa sổ của t=cal[20], h=5

    out_a = compute_symbol_labels(_df(cal, closes_a), "HOSE", cal, horizons=(5,))
    out_b = compute_symbol_labels(_df(cal, closes_b), "HOSE", cal, horizons=(5,))
    row_a = out_a[(out_a["date"] == cal[20]) & (out_a["h"] == 5)].iloc[0]
    row_b = out_b[(out_b["date"] == cal[20]) & (out_b["h"] == 5)].iloc[0]
    assert row_a["y"] == row_b["y"]
    assert row_a["t1"] == row_b["t1"]
    assert row_a["r"] == pytest.approx(row_b["r"])


def test_barrier_b_never_below_cost_floor_even_with_high_volatility():
    """3.13 (1): rào b = max(k*sigma*sqrt(h), C_RT_BASE) theo thiết kế không bao giờ thấp hơn sàn chi
    phí 0,60%, kể cả khi biến động cao làm k*sigma*sqrt(h) < sàn ở các phiên đầu (sigma chưa đủ dữ liệu
    hoặc biến động thấp bất thường)."""
    rng = np.random.default_rng(7)
    cal = _calendar(120)
    # giá đi ngẫu nhiên biên độ lớn xen giá đi ngang dài, để vừa có đoạn sigma cao vừa có đoạn sigma thấp
    log_rets = np.concatenate([np.zeros(40), rng.normal(0, 0.05, 80)])
    closes = 100.0 * np.exp(np.cumsum(log_rets))
    out = compute_symbol_labels(_df(cal, list(closes)), "HOSE", cal, horizons=(5, 10, 21))
    b = out["b"].dropna()
    assert len(b) > 0
    assert (b >= C_RT_BASE - 1e-12).all()


def test_compute_raw_labels_combines_symbols():
    cal = _calendar(6)
    panel = pd.concat(
        [
            _df(cal, [10.0] * 6).assign(symbol="AAA"),
            _df(cal, [20.0] * 6).assign(symbol="BBB"),
        ],
        ignore_index=True,
    )
    stocks = pd.DataFrame({"symbol": ["AAA", "BBB"], "exchange": ["HOSE", "HNX"]})
    out = compute_raw_labels(panel, stocks, horizons=(1,))
    assert set(out["symbol"]) == {"AAA", "BBB"}


def test_add_excess_and_relevance_uses_only_universe_and_valid_rows():
    dates = pd.to_datetime(["2024-01-01"] * 3)
    raw = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "date": dates,
            "h": [5, 5, 5],
            "r": [0.10, 0.20, np.nan],  # C không bán được (r = NaN)
            "exit_date_resolved": [pd.Timestamp("2024-01-10")] * 3,
            "exit_extended_khoa_san": [False, False, True],
            "y": [1, 1, 0],
            "t1": dates,
            "san_chi_phi": [False, False, False],
            "khoa_tran_san": [False, False, False],
            "tam_ngung": [False, False, False],
        }
    )
    universe = raw[["date", "symbol"]]
    out = add_excess_and_relevance(raw, universe, quantiles=5)
    a = out[out["symbol"] == "A"].iloc[0]
    b = out[out["symbol"] == "B"].iloc[0]
    c = out[out["symbol"] == "C"].iloc[0]
    # trung bình chỉ tính trên A, B (có r hợp lệ): mean = 0.15
    assert a["r_excess"] == pytest.approx(0.10 - 0.15)
    assert b["r_excess"] == pytest.approx(0.20 - 0.15)
    assert pd.isna(c["r_excess"])
