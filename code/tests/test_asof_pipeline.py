"""Test cho pipeline chế độ as-of (bước 3.12). Dữ liệu tổng hợp nhỏ tự dựng (không đọc
data/raw/universe_inputs/ hay data/raw/prices/ thật của dự án, không gọi mạng thật), ghi ra
tmp_path để mô phỏng cache cục bộ."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vnrank.asof.pipeline import _load_panel_asof, get_features, get_universe
from vnrank.features.feature_registry import (
    FEATURE_COLUMNS,
    compute_features,
    cross_sectional_normalize,
    fill_market_calendar_gaps,
)
from vnrank.universe.build_universe import build_daily_universe, compute_liquidity_features

SYMBOLS = [("AAA", "HOSE"), ("BBB", "HOSE"), ("CCC", "HNX"), ("DDD", "HNX"), ("EEE", "HOSE")]
N_DAYS = 190  # đủ cho cửa sổ dài nhất (ret_126/sma_ratio_126) cộng biên cho vài ngày as_of và dữ
# liệu "tương lai" thêm vào sau as_of ở test điểm-tại-thời-điểm.


def _calendar(n_days: int = N_DAYS) -> list[pd.Timestamp]:
    return list(pd.bdate_range("2023-01-02", periods=n_days))


def _make_panel(calendar: list[pd.Timestamp]) -> pd.DataFrame:
    """Giá đóng cửa đi bộ ngẫu nhiên nhỏ quanh 20-40 (nghìn đồng, luôn > p_min=5), khối lượng không
    đổi đủ lớn để luôn qua ngưỡng thanh khoản (v_min=100.000, n_min=15 trên window 21)."""
    frames = []
    for i, (symbol, _exchange) in enumerate(SYMBOLS):
        rng = np.random.default_rng(100 + i)
        rets = rng.normal(0, 0.01, len(calendar))
        close = 20.0 * (1 + i * 0.3) * np.exp(np.cumsum(rets))
        frames.append(
            pd.DataFrame(
                {
                    "date": calendar,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 200_000,
                }
            ).assign(symbol=symbol)
        )
    return pd.concat(frames, ignore_index=True)[["symbol", "date", "open", "high", "low", "close", "volume"]]


def _make_market(calendar: list[pd.Timestamp]) -> pd.DataFrame:
    rng = np.random.default_rng(999)
    rets = rng.normal(0, 0.008, len(calendar))
    close = 1000.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"date": calendar, "open": close, "high": close, "low": close, "close": close, "volume": 1})


def _write_listing_xlsx(path: Path) -> None:
    rows = [
        ["STT", "Mã CK", "Tên công ty", "Sàn CK"],
    ] + [[i + 1, sym, f"CTCP {sym}", exch] for i, (sym, exch) in enumerate(SYMBOLS)]
    pd.DataFrame(rows).to_excel(path, sheet_name="Sheet1", header=False, index=False)


def _write_cache(cache_dir: Path, panel: pd.DataFrame, market: pd.DataFrame) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for symbol, group in panel.groupby("symbol"):
        group.drop(columns="symbol").to_csv(cache_dir / f"{symbol}.csv", index=False)
    market.to_csv(cache_dir / "VNINDEX.csv", index=False)


def _offline_reference(panel: pd.DataFrame, market: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tính universe/feature theo đúng công thức 3.4/3.5 trực tiếp trên `panel`/`market` đã cho
    (không qua module asof), dùng làm đối chứng độc lập với hàm đang kiểm tra."""
    liquidity = compute_liquidity_features(panel, window_l=21, lag_sessions=0)
    universe = build_daily_universe(liquidity, listed=None, p_min=5.0, v_min=100_000, n_min=15, n_size=150)
    universe_asof = universe[universe["date"] == as_of].reset_index(drop=True)

    calendar = sorted(panel["date"].unique())
    market_filled = fill_market_calendar_gaps(market[["date", "close"]], calendar)
    raw_features = compute_features(panel, market_filled)
    normalized = cross_sectional_normalize(raw_features, universe)
    features_asof = normalized[normalized["date"] == as_of].reset_index(drop=True)
    return universe_asof, features_asof


@pytest.fixture
def cache(tmp_path):
    calendar = _calendar()
    panel = _make_panel(calendar)
    market = _make_market(calendar)
    listing_path = tmp_path / "listing.xlsx"
    _write_listing_xlsx(listing_path)
    cache_dir = tmp_path / "prices"
    _write_cache(cache_dir, panel, market)
    return {"calendar": calendar, "panel": panel, "market": market, "listing_path": listing_path, "cache_dir": cache_dir}


# Ba ngày as_of mẫu: sớm (vừa đủ 126 phiên lịch sử), giữa, gần cuối calendar tổng hợp.
AS_OF_SAMPLE_INDICES = [130, 150, 165]


@pytest.mark.parametrize("idx", AS_OF_SAMPLE_INDICES)
def test_get_universe_and_features_match_offline_reconstruction(cache, idx):
    as_of = cache["calendar"][idx]
    ref_universe, ref_features = _offline_reference(cache["panel"], cache["market"], as_of)

    got_universe = get_universe(as_of, price_cache_dir=cache["cache_dir"], listing_path=cache["listing_path"])
    got_features = get_features(as_of, price_cache_dir=cache["cache_dir"], listing_path=cache["listing_path"])

    pd.testing.assert_frame_equal(
        got_universe.sort_values("symbol").reset_index(drop=True),
        ref_universe.sort_values("symbol").reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        got_features.sort_values("symbol").reset_index(drop=True)[["date", "symbol"] + FEATURE_COLUMNS],
        ref_features.sort_values("symbol").reset_index(drop=True)[["date", "symbol"] + FEATURE_COLUMNS],
        check_exact=False,
        rtol=1e-9,
    )


def test_load_panel_asof_never_returns_rows_after_as_of(cache):
    """Kiểm tra trực tiếp hợp đồng point-in-time của `_load_panel_asof`: dù cache có đủ dữ liệu đến
    hết lịch (190 phiên), panel/market trả về cho một as_of giữa chừng không được có dòng nào có
    `date > as_of`. Đây là phần việc do code tự cắt, không phải hệ quả tình cờ của công thức 3.4/3.5
    (công thức rolling/ewm vốn đã chỉ nhìn về quá khứ nên bản thân không bị ảnh hưởng bởi dữ liệu sau
    as_of có mặt hay không trong panel đưa vào - xem test dưới), nên cần test riêng cho chính việc cắt.
    """
    as_of = cache["calendar"][150]
    panel, market = _load_panel_asof(as_of, cache["cache_dir"], cache["listing_path"])
    assert panel["date"].max() <= as_of
    assert market["date"].max() <= as_of
    assert len(panel) > 0 and len(market) > 0


@pytest.mark.parametrize("idx", AS_OF_SAMPLE_INDICES)
def test_get_universe_and_features_stable_when_cache_has_shocked_future_rows(tmp_path, idx):
    """Cache có sẵn dữ liệu SAU as_of (như một hệ thống thật đã tải trước), với các phiên sau as_of
    bị chỉnh thành giá trị cực đoan (giá x1000, khối lượng x1000). Kết quả ngày as_of phải giữ
    nguyên, khớp với kết quả tính trên panel bị cắt vật lý chỉ còn <= as_of. Lưu ý: vì mọi công thức
    3.4/3.5 đều chỉ nhìn về quá khứ (rolling/shift/ewm, chuẩn hóa chéo theo từng ngày riêng), việc cắt
    dữ liệu ở `_load_panel_asof` (kiểm trực tiếp ở test trên) đã đủ đảm bảo tính chất này; test này
    xác nhận thêm ở mức đầu ra của get_universe/get_features cho một kịch bản cache thực tế."""
    calendar = _calendar()
    as_of = calendar[idx]

    panel_full = _make_panel(calendar)
    market_full = _make_market(calendar)

    future_mask = panel_full["date"] > as_of
    panel_shocked = panel_full.copy()
    panel_shocked.loc[future_mask, ["open", "high", "low", "close"]] *= 1000.0
    panel_shocked.loc[future_mask, "volume"] *= 1000

    market_shocked = market_full.copy()
    future_mask_mkt = market_shocked["date"] > as_of
    market_shocked.loc[future_mask_mkt, ["open", "high", "low", "close"]] *= 1000.0

    listing_path = tmp_path / "listing.xlsx"
    _write_listing_xlsx(listing_path)
    cache_dir = tmp_path / "prices"
    _write_cache(cache_dir, panel_shocked, market_shocked)

    panel_truncated = panel_full[panel_full["date"] <= as_of].reset_index(drop=True)
    market_truncated = market_full[market_full["date"] <= as_of].reset_index(drop=True)
    ref_universe, ref_features = _offline_reference(panel_truncated, market_truncated, as_of)

    got_universe = get_universe(as_of, price_cache_dir=cache_dir, listing_path=listing_path)
    got_features = get_features(as_of, price_cache_dir=cache_dir, listing_path=listing_path)

    pd.testing.assert_frame_equal(
        got_universe.sort_values("symbol").reset_index(drop=True),
        ref_universe.sort_values("symbol").reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        got_features.sort_values("symbol").reset_index(drop=True)[["date", "symbol"] + FEATURE_COLUMNS],
        ref_features.sort_values("symbol").reset_index(drop=True)[["date", "symbol"] + FEATURE_COLUMNS],
        check_exact=False,
        rtol=1e-9,
    )


def test_get_universe_empty_before_enough_liquidity_history(cache):
    """Ngày đầu calendar chưa đủ 21 phiên cho cửa sổ thanh khoản -> universe rỗng, không suy đoán."""
    as_of = cache["calendar"][5]
    got = get_universe(as_of, price_cache_dir=cache["cache_dir"], listing_path=cache["listing_path"])
    assert got.empty
