"""Kiểm tra rò rỉ trên pipeline GHÉP (bước 3.8, mục (1)): feature_registry + labeling cùng chạy trên
một bộ dữ liệu, dịch giá ở các phiên tương lai không được làm đổi đặc trưng HAY nhãn ở những ngày
trước đó. Bổ sung cho các test đơn vị riêng lẻ đã có ở 3.5 (`test_compute_features_no_lookahead`) và
3.6 (`test_labels_do_not_depend_on_prices_far_beyond_the_window`), ở đây hai bước chạy NỐI TIẾP nhau
đúng như trong pipeline thật, không gọi mạng thật."""
from __future__ import annotations

import numpy as np
import pandas as pd

from vnrank.features.feature_registry import FEATURE_COLUMNS, compute_features, fill_market_calendar_gaps
from vnrank.labels.labeling import HORIZONS, compute_raw_labels


def _price_df(dates, closes, volumes=None):
    n = len(dates)
    volumes = volumes or [1000] * n
    return pd.DataFrame(
        {"date": dates, "open": closes, "high": closes, "low": closes, "close": closes, "volume": volumes}
    )


def _build_panel_and_market(seed, n_days=160, tweak_from=None, tweak_scale=5.0):
    rng = np.random.default_rng(seed)
    dates = list(pd.date_range("2023-01-01", periods=n_days, freq="D"))
    closes_a = list(10 + np.cumsum(rng.normal(0, 0.1, n_days)))
    closes_b = list(15 + np.cumsum(rng.normal(0, 0.1, n_days)))
    if tweak_from is not None:
        for i in range(tweak_from, n_days):
            closes_a[i] *= tweak_scale
            closes_b[i] /= tweak_scale
    panel = pd.concat(
        [
            _price_df(dates, closes_a).assign(symbol="AAA"),
            _price_df(dates, closes_b).assign(symbol="BBB"),
        ],
        ignore_index=True,
    )
    market_closes = list(100 + np.cumsum(rng.normal(0, 0.1, n_days)))
    market_df = _price_df(dates, market_closes)[["date", "close"]]
    return panel, market_df, dates


def _run_pipeline(panel, market_df, calendar, stocks):
    market_filled = fill_market_calendar_gaps(market_df, calendar)
    features = compute_features(panel, market_filled)
    labels = compute_raw_labels(panel, stocks, horizons=HORIZONS)
    return features, labels


def test_feature_and_label_columns_are_disjoint():
    """Bảo đảm không cột nhãn nào vô tình lọt vào danh sách cột đặc trưng (rò rỉ nhãn vào feature)."""
    label_cols = {"r", "y", "t1", "b", "relevance", "r_excess", "exit_date_resolved", "san_chi_phi", "khoa_tran_san", "tam_ngung"}
    assert set(FEATURE_COLUMNS).isdisjoint(label_cols)


def test_combined_pipeline_no_lookahead_for_features_and_labels():
    tweak_from = 130
    stocks = pd.DataFrame({"symbol": ["AAA", "BBB"], "exchange": ["HOSE", "HNX"]})

    panel_a, market_a, dates = _build_panel_and_market(seed=7, tweak_from=None)
    panel_b, market_b, _ = _build_panel_and_market(seed=7, tweak_from=tweak_from, tweak_scale=8.0)

    features_a, labels_a = _run_pipeline(panel_a, market_a, dates, stocks)
    features_b, labels_b = _run_pipeline(panel_b, market_b, dates, stocks)

    cutoff = dates[tweak_from - 1]  # ngày cuối cùng CHƯA bị xáo trộn phía tương lai

    feat_a_past = features_a[features_a["date"] <= cutoff].reset_index(drop=True)
    feat_b_past = features_b[features_b["date"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(feat_a_past, feat_b_past)

    # Nhãn tại một ngày quyết định t chỉ "an toàn" nếu toàn bộ cửa sổ cần dùng (đến t+1+h, h lớn nhất)
    # cũng nằm trước mốc xáo trộn; so sánh trên tập con đó.
    safe_label_dates = [d for d in dates if dates.index(d) + 1 + max(HORIZONS) < tweak_from]
    lab_a_safe = labels_a[labels_a["date"].isin(safe_label_dates)].sort_values(["symbol", "date", "h"]).reset_index(drop=True)
    lab_b_safe = labels_b[labels_b["date"].isin(safe_label_dates)].sort_values(["symbol", "date", "h"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(lab_a_safe, lab_b_safe)
