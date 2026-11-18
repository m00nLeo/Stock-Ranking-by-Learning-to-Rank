"""Test cho sector_analysis.py. Dữ liệu tổng hợp nhỏ (ánh xạ mã-ngành TỔNG HỢP, không đọc
data/raw/sector_inputs/ thật của dự án, cùng quy ước test_build_universe.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from vnrank.system.sector_analysis import (
    build_sector_mapping_from_vietstock,
    filter_by_sector,
    sector_ranking,
    stability_and_trend_by_symbol,
)


def _write_sector_xlsx(path, rows):
    # Mô phỏng đúng cấu trúc file Vietstock, cùng khuôn mẫu test_build_universe._write_listing_xlsx.
    header = [
        [None, None, None, None],
        [None, None, None, "DỮ LIỆU DOANH NGHIỆP"],
        [None, None, None, "Ngày cập nhật: 11/11/2026"],
        [None, None, None, None],
        ["STT", "Mã CK", "Tên công ty", "Sàn CK"],
    ]
    pd.DataFrame(header + rows).to_excel(path, sheet_name="Sheet1", header=False, index=False)


def test_build_sector_mapping_from_vietstock_merges_files_and_excludes_funds(tmp_path):
    _write_sector_xlsx(
        tmp_path / "Cổ phiếu theo Ngành Tài chính.xlsx",
        [[1, "AAA", "CTCP A", "HOSE"], [2, "FUEVFVND", "Quỹ ETF DCVFMVN DIAMOND", "HOSE"]],
    )
    _write_sector_xlsx(
        tmp_path / "Cổ phiếu theo Ngành Công nghệ thông tin.xlsx",
        [[1, "BBB", "CTCP B", "HNX"]],
    )
    mapping = build_sector_mapping_from_vietstock(tmp_path)
    assert set(mapping["symbol"]) == {"AAA", "BBB"}  # quỹ ETF đã bị loại
    assert mapping.set_index("symbol").loc["AAA", "sector"] == "Tài chính"
    assert mapping.set_index("symbol").loc["BBB", "sector"] == "Công nghệ thông tin"


def _calendar(n):
    return pd.to_datetime(pd.date_range("2023-01-02", periods=n, freq="B"))


def _panel(symbol_specs, n_days=80):
    """symbol_specs: dict symbol -> (drift nhật, do_bien_dong)."""
    dates = _calendar(n_days)
    rng = np.random.default_rng(7)
    rows = []
    for symbol, (drift, vol) in symbol_specs.items():
        rets = rng.normal(drift, vol, n_days)
        close = 10.0 * np.exp(np.cumsum(rets))
        for d, c in zip(dates, close):
            rows.append((symbol, d, c))
    return pd.DataFrame(rows, columns=["symbol", "date", "close"])


def test_stability_and_trend_identifies_stable_and_volatile():
    panel = _panel({"STBL": (0.0005, 0.002), "VOLA": (0.0005, 0.05)})
    as_of = panel["date"].max()
    out = stability_and_trend_by_symbol(panel, as_of, lookback_sessions=63)
    stbl = out.set_index("symbol").loc["STBL", "stability_score"]
    vola = out.set_index("symbol").loc["VOLA", "stability_score"]
    assert stbl > vola


def test_stability_and_trend_detects_trend_direction():
    panel = _panel({"UP": (0.01, 0.002), "DOWN": (-0.01, 0.002), "FLAT": (0.0, 0.001)})
    as_of = panel["date"].max()
    out = stability_and_trend_by_symbol(panel, as_of, lookback_sessions=63).set_index("symbol")
    assert out.loc["UP", "trend"] == "tang"
    assert out.loc["DOWN", "trend"] == "giam"
    assert out.loc["FLAT", "trend"] == "di_ngang"


def test_stability_and_trend_only_uses_data_up_to_as_of():
    panel = _panel({"A": (0.0, 0.002)}, n_days=80)
    as_of_early = panel["date"].iloc[65]
    out = stability_and_trend_by_symbol(panel, as_of_early, lookback_sessions=63)
    assert (out["symbol"] == "A").any()
    # không có dòng nào của ngày sau as_of lọt vào tính toán: kiểm gián tiếp bằng cách so sánh
    # với kết quả tính trên panel đã cắt vật lý chỉ còn <= as_of_early.
    truncated = panel[panel["date"] <= as_of_early]
    out_truncated = stability_and_trend_by_symbol(truncated, as_of_early, lookback_sessions=63)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), out_truncated.reset_index(drop=True))


def test_sector_ranking_ranks_within_each_sector_separately():
    panel = _panel(
        {
            "A1": (0.0005, 0.001),  # ổn định nhất ngành X
            "A2": (0.0005, 0.01),
            "B1": (0.0005, 0.02),  # ổn định nhất ngành Y
            "B2": (0.0005, 0.05),
        }
    )
    mapping = pd.DataFrame(
        {"symbol": ["A1", "A2", "B1", "B2"], "sector": ["X", "X", "Y", "Y"]}
    )
    as_of = panel["date"].max()
    ranking = sector_ranking(panel, mapping, as_of, lookback_sessions=63)

    rank_a1 = ranking.set_index("symbol").loc["A1", "rank_in_sector"]
    rank_b1 = ranking.set_index("symbol").loc["B1", "rank_in_sector"]
    assert rank_a1 == 1  # ổn định nhất trong ngành X
    assert rank_b1 == 1  # ổn định nhất trong ngành Y, dù kém ổn định hơn A1 (khác ngành)


def test_filter_by_sector_keeps_only_requested_sector():
    panel = _panel({"A1": (0.0, 0.01), "B1": (0.0, 0.01)})
    mapping = pd.DataFrame({"symbol": ["A1", "B1"], "sector": ["X", "Y"]})
    as_of = panel["date"].max()
    ranking = sector_ranking(panel, mapping, as_of, lookback_sessions=63)
    only_x = filter_by_sector(ranking, "X")
    assert set(only_x["symbol"]) == {"A1"}
