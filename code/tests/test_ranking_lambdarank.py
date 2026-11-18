"""Test cho xếp hạng L1 (bước 4.4): LightGBM LambdaRank, mỗi ngày là một truy vấn, nhãn
`relevance` (Q=5). Dữ liệu tổng hợp nhỏ tự dựng, không đọc dữ liệu thật."""
from __future__ import annotations

import numpy as np
import pandas as pd

from vnrank.features.feature_registry import FEATURE_COLUMNS
from vnrank.models.ranking_lambdarank import (
    _fit_predict_lambdarank,
    _groups_by_date,
    build_candidate_configs,
    build_labeled_features,
    h1_test,
    select_config_on_validation,
    walk_forward_test_scores,
)

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


# --- build_candidate_configs: lưới CHUNG với R2 -------------------------------------------------


def test_build_candidate_configs_matches_lightgbm_grid():
    model_grid = {
        "lightgbm_configs": [
            {"num_leaves": 15, "min_data_in_leaf": 200},
            {"num_leaves": 31, "min_data_in_leaf": 100},
            {"num_leaves": 63, "min_data_in_leaf": 50},
        ]
    }
    configs = build_candidate_configs(model_grid)
    assert len(configs) == 3
    assert configs["nl31_md100"] == {"num_leaves": 31, "min_data_in_leaf": 100}


# --- build_labeled_features ----------------------------------------------------------------


def test_build_labeled_features_drops_unresolved_labels():
    features = pd.DataFrame([_zero_features_row("2024-01-01", "AAA"), _zero_features_row("2024-01-01", "BBB")])
    features["date"] = pd.to_datetime(features["date"])
    labels_h = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "symbol": ["AAA", "BBB"],
            "relevance": [3, np.nan],  # BBB chưa hiện thực nhãn (đuôi dữ liệu, 3.6)
            "r": [0.01, np.nan],
            "exit_date_resolved": pd.to_datetime(["2024-01-05", None]),
        }
    )
    out = build_labeled_features(features, labels_h)
    assert list(out["symbol"]) == ["AAA"]


# --- _groups_by_date: kích thước truy vấn đúng theo ngày, liền khối ----------------------------


def test_groups_by_date_matches_rows_per_day_in_order():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-03"]),
            "symbol": ["AAA", "BBB", "AAA", "AAA", "BBB", "CCC"],
        }
    )
    groups = _groups_by_date(df)
    assert groups == [2, 1, 3]
    assert sum(groups) == len(df)


# --- fit_predict: chạy được, bắt được tín hiệu thứ hạng rõ ràng --------------------------------


def _linear_synthetic_window(n_dates: int = 30, n_symbols: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    rows = []
    for d in dates:
        signal = rng.normal(size=n_symbols)
        relevance = pd.Series(signal).rank(method="first").astype(int) - 1  # 0..n_symbols-1 duy nhất mỗi ngày
        for i in range(n_symbols):
            row = {"date": d, "symbol": f"S{i}"}
            for col in FEATURE_COLUMNS:
                row[col] = 0.0
            row["ret_21"] = signal[i]
            row["relevance"] = int(min(relevance[i], 4))  # cắt về thang 0..4 (Q=5)
            rows.append(row)
    return pd.DataFrame(rows)


def test_fit_predict_lambdarank_recovers_rank_signal():
    window = _linear_synthetic_window()
    predict_df = window[window["date"] == window["date"].max()]
    params = {"num_leaves": 15, "min_data_in_leaf": 5}
    preds = _fit_predict_lambdarank(window, params, GBDT_COMMON, seed=42, fraction=0.2, predict_df=predict_df)
    assert preds.shape == (len(predict_df),)
    corr = np.corrcoef(preds, predict_df["ret_21"].to_numpy())[0, 1]
    assert corr > 0.5, f"LambdaRank: dự báo không bắt được tín hiệu thứ hạng rõ ràng (corr={corr:.3f})"


# --- select_config_on_validation: chọn đúng cấu hình có tín hiệu thật --------------------------


def test_select_config_on_validation_prefers_config_with_real_signal():
    window = _linear_synthetic_window(n_dates=40, n_symbols=6, seed=1)
    train_start_date = window["date"].min()
    train_end_date = window["date"].quantile(0.6, interpolation="lower")

    labeled = window.assign(exit_date_resolved=window["date"] + pd.Timedelta(days=1))
    splits = pd.DataFrame({"date": sorted(labeled["date"].unique())})
    splits["split"] = np.where(splits["date"] <= train_end_date, "train", "validation")
    labels_h = labeled[["date", "symbol", "ret_21"]].rename(columns={"ret_21": "r"})

    configs = {"good": {"num_leaves": 15, "min_data_in_leaf": 5}, "degenerate": {"num_leaves": 15, "min_data_in_leaf": 10_000}}
    best_key, val_ic = select_config_on_validation(
        labeled, configs, labels_h, splits, train_start_date, train_end_date, GBDT_COMMON, seed=42, fraction=0.2,
    )
    assert best_key == "good"
    assert val_ic["good"] == val_ic["good"]  # không NaN


# --- walk_forward_test_scores: phủ hết ngày test, không rò rỉ nhãn chưa hiện thực --------------


def _no_leak_dataset(n_days: int = 40):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    symbols = ["AAA", "BBB"]
    feat_rows = []
    for d in dates:
        for s in symbols:
            row = _zero_features_row(d, s)
            row["ret_21"] = 0.1 if s == "AAA" else -0.1
            feat_rows.append(row)
    features = pd.DataFrame(feat_rows)

    label_rows = []
    for d in dates:
        for s in symbols:
            label_rows.append(
                {
                    "date": d,
                    "symbol": s,
                    "relevance": 4 if s == "AAA" else 0,
                    "r": 0.02 if s == "AAA" else -0.02,
                    "exit_date_resolved": d + pd.Timedelta(days=1),
                }
            )
    labels = pd.DataFrame(label_rows)
    labeled = features.merge(labels, on=["date", "symbol"])
    return labeled, list(dates)


def test_walk_forward_test_scores_covers_every_test_day():
    labeled, dates = _no_leak_dataset()
    train_start_date, test_start_date, test_end_date = dates[0], dates[20], dates[-1]
    scores = walk_forward_test_scores(
        labeled, {"num_leaves": 15, "min_data_in_leaf": 2}, train_start_date, test_start_date, test_end_date, dates,
        refit_frequency=10, gbdt_common=GBDT_COMMON, seed=42, fraction=0.2,
    )
    expected_dates = set(d for d in dates if test_start_date <= d <= test_end_date)
    assert set(scores["date"]) == expected_dates
    assert len(scores) == len(expected_dates) * 2  # 2 mã mỗi ngày


def test_walk_forward_test_scores_does_not_leak_unresolved_future_label():
    """Như `test_regression_topk.py`: mẫu trong cửa sổ huấn luyện của lần refit đầu nhưng
    `exit_date_resolved` rơi SAU mốc refit đầu tiên phải bị purge loại hẳn."""
    labeled, dates = _no_leak_dataset()
    train_start_date, test_start_date, test_end_date = dates[0], dates[20], dates[-1]
    common_kwargs = dict(
        train_start_date=train_start_date, test_start_date=test_start_date, test_end_date=test_end_date,
        calendar=dates, refit_frequency=10, gbdt_common=GBDT_COMMON, seed=42, fraction=0.2,
    )
    params = {"num_leaves": 15, "min_data_in_leaf": 2}
    leak_mask = (labeled["date"] == dates[5]) & (labeled["symbol"] == "AAA")

    labeled_dropped = labeled[~leak_mask].reset_index(drop=True)
    labeled_leaked = labeled.copy()
    labeled_leaked.loc[leak_mask, "relevance"] = 0  # cực đoan trái ngược AAA, lộ ngay nếu rò rỉ
    labeled_leaked.loc[leak_mask, "exit_date_resolved"] = dates[30]  # chưa hiện thực tại refit đầu

    scores_dropped = walk_forward_test_scores(labeled_dropped, params, **common_kwargs)
    scores_leaked = walk_forward_test_scores(labeled_leaked, params, **common_kwargs)

    first_day_dropped = scores_dropped[scores_dropped["date"] == test_start_date].sort_values("symbol")["score"].to_numpy()
    first_day_leaked = scores_leaked[scores_leaked["date"] == test_start_date].sort_values("symbol")["score"].to_numpy()
    np.testing.assert_allclose(first_day_dropped, first_day_leaked)


# --- h1_test: d_t = IC_L1 - IC_R2, Newey-West -----------------------------------------------


def test_h1_test_detects_l1_better_than_r2():
    dates = pd.bdate_range("2024-01-01", periods=30)
    rows_l1, rows_r2, label_rows = [], [], []
    rng = np.random.default_rng(0)
    for d in dates:
        true = rng.normal(size=5)
        rows_l1 += [{"date": d, "symbol": f"S{i}", "score": true[i] + rng.normal(scale=0.01)} for i in range(5)]
        rows_r2 += [{"date": d, "symbol": f"S{i}", "score": rng.normal()} for i in range(5)]  # độc lập, không có tín hiệu
        label_rows += [{"date": d, "symbol": f"S{i}", "r": true[i]} for i in range(5)]
    l1_scores = pd.DataFrame(rows_l1)
    r2_scores = pd.DataFrame(rows_r2)
    labels_h = pd.DataFrame(label_rows)
    splits = pd.DataFrame({"date": dates, "split": "test"})

    out = h1_test(l1_scores, r2_scores, labels_h, splits, h=5)
    assert out["mean"] > 0
    assert out["n_days"] == 30
