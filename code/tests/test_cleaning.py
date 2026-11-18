"""Test khung cho cleaning.py (bước 3.3). Không gọi mạng thật, trừ hàm mocked riêng."""
from unittest.mock import patch

import pandas as pd

from vnrank.data.cleaning import (
    detect_limit_lock,
    flag_missing_calendar_days,
    flag_zero_volume,
    trading_calendar_from_symbols,
    validate_ohlc_integrity,
)


def _df(rows):
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])


def test_validate_ohlc_integrity_flags_bad_rows():
    df = _df(
        [
            ["2022-01-03", 10.0, 10.5, 9.8, 10.2, 1000],  # hợp lệ
            ["2022-01-04", 10.0, 9.0, 9.8, 10.2, 1000],   # high < open: không hợp lệ
            ["2022-01-05", 10.0, 10.5, 10.6, 10.2, 1000],  # low > high: không hợp lệ
            ["2022-01-06", -1.0, 10.5, 9.8, 10.2, 1000],  # giá âm: không hợp lệ
        ]
    )
    out = validate_ohlc_integrity(df)
    assert out["ohlc_hop_le"].tolist() == [True, False, False, False]


def test_flag_zero_volume():
    df = _df(
        [
            ["2022-01-03", 10.0, 10.5, 9.8, 10.2, 1000],
            ["2022-01-04", 10.0, 10.5, 9.8, 10.2, 0],
        ]
    )
    out = flag_zero_volume(df)
    assert out["khoi_luong_bang_khong"].tolist() == [False, True]


def test_flag_missing_calendar_days():
    calendar = ["2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06"]
    present = ["2022-01-03", "2022-01-05"]
    missing = flag_missing_calendar_days(present, calendar)
    assert missing == ["2022-01-04", "2022-01-06"]


def test_detect_limit_lock_flags_ceiling_streak_matching_band():
    # mô phỏng L10 thật: mỗi phiên O=H=L=C, tăng đúng +7% (biên độ HOSE) so với phiên trước
    closes = [23.90, 25.59, 27.38, 29.30, 31.35, 33.54]
    rows = [[f"2026-10-{19+i}", c, c, c, c, 1000] for i, c in enumerate(closes)]
    df = _df(rows)
    out = detect_limit_lock(df, san="HOSE")
    # phiên đầu không có đóng cửa trước nên không đánh giá được
    assert out.loc[0, "khoa_bien_do"] == False  # noqa: E712
    assert out.loc[1:, "khoa_bien_do"].all()
    assert (out.loc[1:, "huong_khoa"] == "tran").all()


def test_detect_limit_lock_floor_direction():
    df = _df(
        [
            ["2026-01-02", 30.0, 30.0, 30.0, 30.0, 500],
            ["2026-01-03", 27.9, 27.9, 27.9, 27.9, 500],  # giảm đúng 7%
        ]
    )
    out = detect_limit_lock(df, san="HOSE")
    assert out.loc[1, "khoa_bien_do"]
    assert out.loc[1, "huong_khoa"] == "san"


def test_detect_limit_lock_ignores_degenerate_day_off_band():
    # suy biến (một lệnh khớp duy nhất) nhưng mức thay đổi không khớp biên độ nào
    df = _df(
        [
            ["2026-01-02", 30.0, 30.0, 30.0, 30.0, 100],
            ["2026-01-03", 30.6, 30.6, 30.6, 30.6, 100],  # chỉ +2%, không phải khóa biên độ
        ]
    )
    out = detect_limit_lock(df, san="HOSE")
    assert not out.loc[1, "khoa_bien_do"]
    assert out.loc[1, "huong_khoa"] is None


def test_detect_limit_lock_ignores_non_degenerate_day():
    df = _df(
        [
            ["2026-01-02", 30.0, 30.0, 30.0, 30.0, 100],
            ["2026-01-03", 32.0, 32.1, 31.5, 32.0, 100],  # +7% nhưng không suy biến
        ]
    )
    out = detect_limit_lock(df, san="HOSE")
    assert not out.loc[1, "khoa_bien_do"]


def test_trading_calendar_from_symbols_unions_dates():
    def fake_fetch(symbol, start, end, symbol_type="stock"):
        data = {
            "VNINDEX": ["2022-01-03", "2022-01-05"],
            "VNM": ["2022-01-03", "2022-01-04", "2022-01-05"],
        }
        return pd.DataFrame({"date": data[symbol]})

    with patch("vnrank.data.cleaning.fetch_ohlcv", side_effect=fake_fetch):
        calendar = trading_calendar_from_symbols(
            [("VNINDEX", "index"), ("VNM", "stock")], "2022-01-01", "2022-01-10"
        )
    assert calendar == ["2022-01-03", "2022-01-04", "2022-01-05"]
