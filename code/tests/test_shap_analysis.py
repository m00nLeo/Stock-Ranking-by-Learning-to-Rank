"""Test cho SHAP (bước 4.9): giải thích đóng góp đặc trưng của L1 bằng `pred_contrib=True` của
LightGBM (Tree SHAP chính xác, không cài thêm thư viện `shap`). Dữ liệu tổng hợp nhỏ tự dựng."""
from __future__ import annotations

import numpy as np
import pandas as pd

from vnrank.evaluation.shap_analysis import global_shap_importance, walk_forward_shap_contributions
from vnrank.features.feature_registry import FEATURE_COLUMNS
from vnrank.models.ranking_lambdarank import _train_lambdarank_booster

GBDT_COMMON = {
    "learning_rate": 0.1,
    "max_trees": 50,
    "early_stopping_rounds": 5,
    "subsample_row": 0.8,
    "subsample_col": 0.8,
}


def _zero_features_row(date, symbol) -> dict:
    row = {"date": date, "symbol": symbol}
    for col in FEATURE_COLUMNS:
        row[col] = 0.0
    return row


def _linear_synthetic_window(n_dates: int = 30, n_symbols: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    rows = []
    for d in dates:
        signal = rng.normal(size=n_symbols)
        relevance = pd.Series(signal).rank(method="first").astype(int) - 1
        for i in range(n_symbols):
            row = {"date": d, "symbol": f"S{i}"}
            for col in FEATURE_COLUMNS:
                row[col] = 0.0
            row["ret_21"] = signal[i]
            row["relevance"] = int(min(relevance[i], 4))
            rows.append(row)
    return pd.DataFrame(rows)


def _no_leak_dataset(n_days: int = 40):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    symbols = ["AAA", "BBB"]
    rows = []
    for d in dates:
        for s in symbols:
            row = _zero_features_row(d, s)
            row["ret_21"] = 0.1 if s == "AAA" else -0.1
            row["relevance"] = 4 if s == "AAA" else 0
            row["exit_date_resolved"] = d + pd.Timedelta(days=1)
            rows.append(row)
    return pd.DataFrame(rows), list(dates)


# --- Tính CỘNG (additivity) của Tree SHAP: tổng cột shap_* cộng bias = dự báo thô -----------------


def test_shap_contributions_sum_to_raw_prediction():
    window = _linear_synthetic_window()
    predict_df = window[window["date"] == window["date"].max()].reset_index(drop=True)
    params = {"num_leaves": 15, "min_data_in_leaf": 5}

    booster = _train_lambdarank_booster(window, params, GBDT_COMMON, seed=42, fraction=0.2)
    raw_pred = booster.predict(predict_df[FEATURE_COLUMNS].to_numpy(), num_iteration=booster.best_iteration)
    contrib = booster.predict(
        predict_df[FEATURE_COLUMNS].to_numpy(), pred_contrib=True, num_iteration=booster.best_iteration
    )
    np.testing.assert_allclose(contrib.sum(axis=1), raw_pred, atol=1e-6)


# --- walk_forward_shap_contributions: phủ hết ngày test, cộng đúng với score -------------------


def test_walk_forward_shap_contributions_covers_every_test_day_and_sums_to_score():
    labeled, dates = _no_leak_dataset()
    train_start_date, test_start_date, test_end_date = dates[0], dates[20], dates[-1]
    contributions = walk_forward_shap_contributions(
        labeled, {"num_leaves": 15, "min_data_in_leaf": 2}, train_start_date, test_start_date, test_end_date, dates,
        refit_frequency=10, gbdt_common=GBDT_COMMON, seed=42, fraction=0.2,
    )
    expected_dates = set(d for d in dates if test_start_date <= d <= test_end_date)
    assert set(contributions["date"]) == expected_dates
    assert len(contributions) == len(expected_dates) * 2

    shap_cols = [f"shap_{c}" for c in FEATURE_COLUMNS] + ["shap_base"]
    np.testing.assert_allclose(contributions[shap_cols].sum(axis=1).to_numpy(), contributions["score"].to_numpy())


# --- global_shap_importance: xếp đúng theo |SHAP|, đặc trưng có tín hiệu thật đứng đầu ----------


def test_global_shap_importance_ranks_signal_feature_first():
    labeled, dates = _no_leak_dataset()
    train_start_date, test_start_date, test_end_date = dates[0], dates[20], dates[-1]
    contributions = walk_forward_shap_contributions(
        labeled, {"num_leaves": 15, "min_data_in_leaf": 2}, train_start_date, test_start_date, test_end_date, dates,
        refit_frequency=10, gbdt_common=GBDT_COMMON, seed=42, fraction=0.2,
    )
    importance = global_shap_importance(contributions)

    assert list(importance["feature"]) == FEATURE_COLUMNS  # mọi đặc trưng có mặt đúng một lần
    assert importance.iloc[0]["feature"] == "ret_21"  # đặc trưng duy nhất mang tín hiệu (AAA/BBB)
    assert (importance["mean_abs_shap"].to_numpy()[:-1] >= importance["mean_abs_shap"].to_numpy()[1:]).all()
    np.testing.assert_allclose(importance["ty_trong"].sum(), 1.0)
    assert list(importance["rank"]) == list(range(1, len(importance) + 1))
