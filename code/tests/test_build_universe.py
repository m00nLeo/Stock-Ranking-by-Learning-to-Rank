"""Test khung cho build_universe.py (bước 3.4). Không đọc file dữ liệu thật của dự án
(data/raw/universe_inputs/ không lên GitHub); mọi test dùng file tạm hoặc DataFrame nhỏ tự dựng,
và không gọi mạng thật (entrade_client được giả lập)."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from vnrank.universe.build_universe import (
    build_daily_universe,
    compute_liquidity_features,
    estimate_delisted_undercount,
    fetch_price_cache,
    load_listed_stocks,
    load_vn30_membership,
    vn30_members_asof,
)


def _write_listing_xlsx(path):
    # Mô phỏng đúng cấu trúc file Vietstock: vài dòng trang trí trước dòng tiêu đề cột.
    rows = [
        [None, None, None, None],
        [None, None, None, "DỮ LIỆU DOANH NGHIỆP"],
        [None, None, None, "Ngày cập nhật: 04/11/2026"],
        [None, None, None, None],
        ["STT", "Mã CK", "Tên công ty", "Sàn CK"],
        [1, "AAA", "CTCP Nhựa An Phát Xanh", "HOSE"],
        [2, "BBB", "CTCP B", "HNX"],
        [3, "FUEVFVND", "Quỹ ETF DCVFMVN DIAMOND", "HOSE"],
        [4, "CCC", "CTCP C", "HOSE"],
    ]
    pd.DataFrame(rows).to_excel(path, sheet_name="Sheet1", header=False, index=False)


def test_load_listed_stocks_excludes_funds(tmp_path):
    path = tmp_path / "listing.xlsx"
    _write_listing_xlsx(path)
    df = load_listed_stocks(path)
    assert sorted(df["symbol"]) == ["AAA", "BBB", "CCC"]
    assert set(df["exchange"]) == {"HOSE", "HNX"}


def _write_vn30_csv(path):
    rows = []
    for period, eff, tickers in [
        ("2023-01", "2023-02-06", ["A", "B", "C"]),
        ("2023-07", "2023-08-07", ["A", "B", "D"]),
    ]:
        for t in tickers:
            rows.append({"review_period": period, "announcement_date": eff, "effective_date": eff, "ticker": t})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_vn30_members_asof_before_first_period_is_empty(tmp_path):
    path = tmp_path / "vn30.csv"
    _write_vn30_csv(path)
    df = load_vn30_membership(path)
    assert vn30_members_asof(df, "2022-12-01") == set()


def test_vn30_members_asof_picks_latest_effective_period(tmp_path):
    path = tmp_path / "vn30.csv"
    _write_vn30_csv(path)
    df = load_vn30_membership(path)
    assert vn30_members_asof(df, "2023-03-01") == {"A", "B", "C"}
    assert vn30_members_asof(df, "2023-08-07") == {"A", "B", "D"}
    assert vn30_members_asof(df, "2026-01-01") == {"A", "B", "D"}


def test_fetch_price_cache_skips_existing_file(tmp_path):
    out_dir = tmp_path / "prices"
    out_dir.mkdir()
    (out_dir / "AAA.csv").write_text("date,open,high,low,close,volume\n", encoding="utf-8")

    with patch("vnrank.universe.build_universe.fetch_ohlcv") as mock_fetch:
        mock_fetch.return_value = pd.DataFrame(
            {"date": ["2022-01-03"], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [100]}
        )
        status = fetch_price_cache(["AAA", "BBB"], "2022-01-01", "2022-01-31", out_dir=out_dir, max_workers=2)
    assert status["AAA"] == "skip"
    assert (out_dir / "BBB.csv").exists()


def _panel():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    rows = []
    for d, close, vol in zip(dates, [10, 10, 10, 10, 10, 10], [100, 100, 0, 100, 100, 100]):
        rows.append({"symbol": "AAA", "date": d, "open": close, "high": close, "low": close, "close": close, "volume": vol})
    for d, close, vol in zip(dates, [1, 1, 1, 1, 1, 1], [1000, 1000, 1000, 1000, 1000, 1000]):
        rows.append({"symbol": "PENNY", "date": d, "open": close, "high": close, "low": close, "close": close, "volume": vol})
    return pd.DataFrame(rows)


def test_compute_liquidity_features_uses_lag_and_window():
    panel = _panel()
    out = compute_liquidity_features(panel, window_l=3, lag_sessions=1)
    aaa = out[out["symbol"] == "AAA"].reset_index(drop=True)
    # window=3, lag=1: giá trị tại dòng cuối (index 5) = trung bình traded_value của 3 phiên
    # [index 2,3,4] (close*vol = 10*0, 10*100, 10*100 = 0,1000,1000) dịch lên 1 dòng -> dùng ở index 5
    assert aaa.loc[5, "avg_value_l"] == pytest.approx((0 + 1000 + 1000) / 3)
    assert aaa.loc[5, "active_sessions_l"] == 2  # 2 trong 3 phiên có volume > 0
    assert aaa.loc[5, "lagged_close"] == aaa.loc[4, "close"]


def test_build_daily_universe_filters_penny_and_ranks_by_liquidity():
    panel = _panel()
    features = compute_liquidity_features(panel, window_l=3, lag_sessions=1)
    listed = pd.DataFrame({"symbol": ["AAA", "PENNY"], "exchange": ["HOSE", "HOSE"]})
    universe = build_daily_universe(features, listed, p_min=5.0, v_min=1.0, n_min=1, n_size=10)
    # PENNY có giá 1 < p_min=5 nên bị loại ở mọi ngày đủ dữ liệu
    assert "PENNY" not in set(universe["symbol"])
    assert "AAA" in set(universe["symbol"])


def test_estimate_delisted_undercount_arithmetic():
    out = estimate_delisted_undercount(confirmed_delisted_count=20, plausible_liquid_fraction=(0.2, 0.5))
    assert out["so_ma_huy_niem_yet"] == 20
    assert out["uoc_luong_thap"] == 4
    assert out["uoc_luong_cao"] == 10
