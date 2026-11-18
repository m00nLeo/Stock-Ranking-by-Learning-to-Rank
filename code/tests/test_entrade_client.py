"""Test khung cho entrade_client (bước 3.2, D71). Không gọi mạng thật: giả lập response."""
from unittest.mock import MagicMock, patch

import pandas as pd

from vnrank.data.entrade_client import fetch_ohlcv


def _mock_response(status_code=200, json_data=None):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data or {}
    mock.raise_for_status = MagicMock()
    return mock


def test_fetch_ohlcv_parses_valid_payload():
    payload = {
        "t": [1640995200, 1641081600],
        "o": [100.0, 101.0],
        "h": [102.0, 103.0],
        "l": [99.0, 100.5],
        "c": [101.0, 102.5],
        "v": [1000, 2000],
    }
    with patch("vnrank.data.entrade_client.requests.get", return_value=_mock_response(200, payload)):
        df = fetch_ohlcv("VNM", "2022-01-01", "2022-01-02")
    assert len(df) == 2
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert df["close"].tolist() == [101.0, 102.5]


def test_fetch_ohlcv_invalid_symbol_returns_empty_not_error():
    with patch("vnrank.data.entrade_client.requests.get", return_value=_mock_response(400)):
        df = fetch_ohlcv("ROS", "2023-01-01", "2023-06-30")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_fetch_ohlcv_requires_matching_symbol_type():
    url_calls = []

    def fake_get(url, headers, params, timeout):
        url_calls.append(url)
        return _mock_response(400)

    with patch("vnrank.data.entrade_client.requests.get", side_effect=fake_get):
        fetch_ohlcv("VNINDEX", "2022-01-01", "2022-01-05", symbol_type="stock")
    assert url_calls[0].endswith("/ohlcs/stock")


def test_fetch_ohlcv_dedupes_same_calendar_date_keeping_later_timestamp():
    # mô phỏng đúng sự cố 27/12/2022: hai mốc UTC khác nhau cùng đổi ra một ngày VN,
    # giữ bản ghi mốc timestamp lớn hơn (02:00 UTC, khớp quy ước xuyên suốt chuỗi)
    payload = {
        "t": [1672099200, 1672106400, 1672192800],  # 00:00, 02:00 UTC (27/12), rồi 28/12
        "o": [6.16, 6.19, 6.37],
        "h": [6.26, 6.34, 6.41],
        "l": [6.15, 6.16, 6.29],
        "c": [6.19, 6.33, 6.37],
        "v": [1196500, 1606700, 1152100],
    }
    with patch("vnrank.data.entrade_client.requests.get", return_value=_mock_response(200, payload)):
        df = fetch_ohlcv("AAA", "2022-12-27", "2022-12-28")
    assert len(df) == 2
    row = df[df["date"] == "2022-12-27"].iloc[0]
    assert row["close"] == 6.33
    assert row["volume"] == 1606700


def test_fetch_ohlcv_index_type_builds_correct_url():
    url_calls = []

    def fake_get(url, headers, params, timeout):
        url_calls.append(url)
        return _mock_response(200, {"t": [], "o": [], "h": [], "l": [], "c": [], "v": []})

    with patch("vnrank.data.entrade_client.requests.get", side_effect=fake_get):
        fetch_ohlcv("VNINDEX", "2022-01-01", "2022-01-05", symbol_type="index")
    assert url_calls[0].endswith("/ohlcs/index")
