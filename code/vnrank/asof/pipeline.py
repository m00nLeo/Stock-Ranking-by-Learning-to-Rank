"""Pipeline chế độ as-of (bước 3.12): mô phỏng đúng những gì một hệ thống chạy thật sẽ thấy tại
một ngày `as_of` bất kỳ, chỉ dùng dữ liệu giá đến hết ngày đó (point-in-time), rồi tính universe và
đặc trưng theo đúng công thức đã dùng ở 3.4 (`build_universe.py`) và 3.5 (`feature_registry.py`).

Không tính lại thuật toán: gọi thẳng các hàm đã có, chỉ cắt dữ liệu đầu vào còn <= `as_of` trước khi
gọi, để pipeline không bao giờ đọc được phần dữ liệu sau `as_of` dù cache cục bộ có sẵn đến tận hiện
tại. Đây là hạ tầng cho S2 (khung giao diện minh họa) nên ưu tiên đúng và rõ ràng hơn tối ưu tốc độ:
mỗi lần gọi tính lại toàn bộ lịch sử đến `as_of`, không cache giữa các lần gọi.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from vnrank.features.feature_registry import (
    FEATURE_COLUMNS,
    compute_features,
    cross_sectional_normalize,
    fill_market_calendar_gaps,
)
from vnrank.universe.build_universe import (
    LISTING_XLSX,
    PRICE_CACHE_DIR,
    build_daily_universe,
    compute_liquidity_features,
    load_listed_stocks,
    load_price_panel,
)

CODE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = CODE_ROOT / "config" / "config.yaml"


def _load_universe_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["universe"]


def _load_panel_asof(
    as_of: pd.Timestamp, price_cache_dir: Path, listing_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Đọc panel giá cổ phiếu và VN-Index đã cache, cắt còn <= `as_of`. Dữ liệu sau `as_of` có thể
    đã có sẵn trong cache (toàn bộ dự án tải một lần) nhưng không được đọc tới ở đây."""
    stocks = load_listed_stocks(listing_path)
    panel = load_price_panel(stocks["symbol"].tolist(), cache_dir=price_cache_dir)
    panel = panel[panel["date"] <= as_of].reset_index(drop=True)

    market_raw = pd.read_csv(price_cache_dir / "VNINDEX.csv", parse_dates=["date"])
    market_raw = market_raw[market_raw["date"] <= as_of].reset_index(drop=True)
    calendar = sorted(panel["date"].unique())
    market = fill_market_calendar_gaps(market_raw, calendar)
    return panel, market


def get_universe(
    as_of: str | pd.Timestamp,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    listing_path: Path = LISTING_XLSX,
) -> pd.DataFrame:
    """Universe $U_t$ tại ngày `as_of`, chỉ dùng dữ liệu giá <= `as_of` (point-in-time).

    Trả về DataFrame cột `date`, `symbol` cho đúng ngày `as_of` (rỗng nếu `as_of` không phải phiên
    giao dịch, hoặc chưa đủ `liquidity_window_sessions` phiên lịch sử để tính thanh khoản).
    """
    as_of = pd.Timestamp(as_of)
    cfg = _load_universe_config()
    stocks = load_listed_stocks(listing_path)
    panel, _ = _load_panel_asof(as_of, price_cache_dir, listing_path)

    liquidity = compute_liquidity_features(
        panel,
        window_l=cfg["liquidity_window_sessions"],
        lag_sessions=cfg["liquidity_lag_sessions"],
    )
    universe = build_daily_universe(
        liquidity,
        stocks,
        p_min=cfg["price_min_thousand_vnd"],
        v_min=cfg["avg_value_min_thousand_vnd"],
        n_min=cfg["min_active_sessions"],
        n_size=cfg["n_size"],
    )
    return universe[universe["date"] == as_of].reset_index(drop=True)


def get_features(
    as_of: str | pd.Timestamp,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    listing_path: Path = LISTING_XLSX,
) -> pd.DataFrame:
    """Đặc trưng đã chuẩn hóa chéo của $U_t$ tại ngày `as_of`, chỉ dùng dữ liệu giá <= `as_of`.

    Trả về DataFrame cột `date`, `symbol`, cộng các cột trong `FEATURE_COLUMNS`, giới hạn trong
    universe của đúng ngày `as_of` (rỗng nếu universe ngày đó rỗng).
    """
    as_of = pd.Timestamp(as_of)
    universe_asof = get_universe(as_of, price_cache_dir, listing_path)
    if universe_asof.empty:
        return pd.DataFrame(columns=["date", "symbol"] + FEATURE_COLUMNS)

    panel, market = _load_panel_asof(as_of, price_cache_dir, listing_path)
    raw_features = compute_features(panel, market)
    normalized = cross_sectional_normalize(raw_features, universe_asof)
    return normalized[normalized["date"] == as_of].reset_index(drop=True)
