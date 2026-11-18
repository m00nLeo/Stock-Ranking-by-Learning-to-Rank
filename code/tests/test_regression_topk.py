"""Test cho hồi quy rồi lấy Top-K (bước 4.3): R1 Ridge, R2 LightGBM, R3 XGBoost. Dữ liệu tổng hợp
nhỏ tự dựng, không đọc dữ liệu thật."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vnrank.features.feature_registry import FEATURE_COLUMNS
from vnrank.models.regression_topk import (
    _split_early_stopping,
    build_candidate_configs,
    build_labeled_features,
    fit_predict,
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


# --- build_candidate_configs -------------------------------------------------------------------


def test_build_candidate_configs_respects_three_model_budget():
    model_grid = {
        "regression_topk": {"R1_ridge": {"alpha": [0.1, 1, 10]}, "R3_xgboost": {"max_depth": [3, 5, 7]}},
        "lightgbm_configs": [
            {"num_leaves": 15, "min_data_in_leaf": 200},
            {"num_leaves": 31, "min_data_in_leaf": 100},
            {"num_leaves": 63, "min_data_in_leaf": 50},
        ],
    }
    configs = build_candidate_configs(model_grid)
    assert set(configs.keys()) == {"R1_ridge", "R2_lightgbm", "R3_xgboost"}
    for family, grid in configs.items():
        assert len(grid) <= 3, f"{family} vượt ngân sách 3 cấu hình (2.3 mục 6)"
    assert configs["R1_ridge"]["alpha_10"] == {"alpha": 10}
    assert configs["R3_xgboost"]["depth_5"] == {"max_depth": 5}


# --- build_labeled_features ----------------------------------------------------------------


def test_build_labeled_features_drops_unresolved_labels():
    features = pd.DataFrame([_zero_features_row("2024-01-01", "AAA"), _zero_features_row("2024-01-01", "BBB")])
    features["date"] = pd.to_datetime(features["date"])
    labels_h = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "symbol": ["AAA", "BBB"],
            "r_excess": [0.01, np.nan],  # BBB chưa hiện thực nhãn (đuôi dữ liệu, 3.6)
            "exit_date_resolved": pd.to_datetime(["2024-01-05", None]),
        }
    )
    out = build_labeled_features(features, labels_h)
    assert list(out["symbol"]) == ["AAA"]


# --- _split_early_stopping ---------------------------------------------------------------------


def test_split_early_stopping_uses_last_fraction_by_time():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    window = pd.DataFrame({"date": dates, "symbol": "AAA"})
    fit_part, eval_part = _split_early_stopping(window, fraction=0.2)
    assert fit_part["date"].max() < eval_part["date"].min()
    assert len(eval_part) == 2  # 20% cuối của 10 ngày


def test_split_early_stopping_keeps_at_least_one_date_each_side_when_window_small():
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    window = pd.DataFrame({"date": dates, "symbol": "AAA"})
    fit_part, eval_part = _split_early_stopping(window, fraction=0.2)
    assert len(fit_part) >= 1
    assert len(eval_part) >= 1


# --- fit_predict: mỗi họ mô hình chạy được và khôi phục tín hiệu tuyến tính đơn giản -----------


def _linear_synthetic_window(n_dates: int = 30, n_symbols: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    rows = []
    for d in dates:
        for i in range(n_symbols):
            row = {"date": d, "symbol": f"S{i}"}
            for col in FEATURE_COLUMNS:
                row[col] = rng.normal()
            row["r_excess"] = 0.5 * row["ret_21"] + rng.normal(scale=0.01)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("model_family", ["R1_ridge", "R2_lightgbm", "R3_xgboost"])
def test_fit_predict_recovers_linear_signal(model_family):
    window = _linear_synthetic_window()
    predict_df = window[window["date"] == window["date"].max()]
    params = {"alpha": 1.0, "num_leaves": 15, "min_data_in_leaf": 5, "max_depth": 3}
    preds = fit_predict(model_family, window, params, GBDT_COMMON, seed=42, fraction=0.2, predict_df=predict_df)
    assert preds.shape == (len(predict_df),)
    corr = np.corrcoef(preds, predict_df["ret_21"].to_numpy())[0, 1]
    assert corr > 0.5, f"{model_family}: dự báo không bắt được tín hiệu tuyến tính rõ ràng (corr={corr:.3f})"


# --- select_config_on_validation: chọn đúng cấu hình có tín hiệu thật --------------------------


def test_select_config_on_validation_prefers_config_with_real_signal():
    window = _linear_synthetic_window(n_dates=40, n_symbols=6, seed=1)
    train_start_date = window["date"].min()
    train_end_date = window["date"].quantile(0.6, interpolation="lower")
    validation_dates = window.loc[window["date"] > train_end_date, "date"].unique()

    labeled = window.assign(exit_date_resolved=window["date"] + pd.Timedelta(days=1))
    splits = pd.DataFrame(
        {
            "date": sorted(labeled["date"].unique()),
        }
    )
    splits["split"] = np.where(splits["date"] <= train_end_date, "train", "validation")
    # Rank IC được tính theo thứ hạng trong ngày; cộng một hằng số mỗi ngày không đổi thứ hạng, nên
    # dùng thẳng r_excess làm "r" (cột harness yêu cầu) không ảnh hưởng kết quả kiểm định ở đây.
    labels_h = labeled[["date", "symbol", "r_excess"]].rename(columns={"r_excess": "r"}).assign(r_excess=labeled["r_excess"])

    # min_data_in_leaf cực lớn (vượt số mẫu mỗi cửa sổ) buộc LightGBM chỉ còn một lá (hằng số),
    # khác hẳn Ridge (shrink đều mọi hệ số nên không đổi thứ hạng dự báo dù alpha lớn tới đâu).
    configs = {"good": {"num_leaves": 15, "min_data_in_leaf": 5}, "degenerate": {"num_leaves": 15, "min_data_in_leaf": 10_000}}
    best_key, val_ic = select_config_on_validation(
        "R2_lightgbm", labeled, configs, labels_h, splits,
        train_start_date, train_end_date, GBDT_COMMON, seed=42, fraction=0.2,
    )
    assert best_key == "good"
    assert val_ic["good"] == val_ic["good"]  # không NaN
    assert not (val_ic["degenerate"] == val_ic["degenerate"])  # NaN: điểm hằng số (một lá) mỗi ngày


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
            r_excess = 0.2 * (0.1 if s == "AAA" else -0.1)
            label_rows.append(
                {"date": d, "symbol": s, "r_excess": r_excess, "exit_date_resolved": d + pd.Timedelta(days=1)}
            )
    labels = pd.DataFrame(label_rows)
    labeled = features.merge(labels, on=["date", "symbol"])
    return labeled, list(dates)


def test_walk_forward_test_scores_covers_every_test_day():
    labeled, dates = _no_leak_dataset()
    train_start_date, test_start_date, test_end_date = dates[0], dates[20], dates[-1]
    scores = walk_forward_test_scores(
        "R1_ridge", labeled, {"alpha": 1.0}, train_start_date, test_start_date, test_end_date, dates,
        refit_frequency=10, gbdt_common=GBDT_COMMON, seed=42, fraction=0.2,
    )
    expected_dates = set(d for d in dates if test_start_date <= d <= test_end_date)
    assert set(scores["date"]) == expected_dates
    assert len(scores) == len(expected_dates) * 2  # 2 mã mỗi ngày


def test_walk_forward_test_scores_does_not_leak_unresolved_future_label():
    """Mẫu ở ngày dates[5] (trong cửa sổ huấn luyện của lần refit đầu) nhưng `exit_date_resolved`
    rơi vào SAU mốc refit đầu tiên (test_start_date=dates[20]) -> chưa hiện thực tại thời điểm refit,
    purge phải loại hẳn mẫu này. Nếu purge đúng, dự báo ngày đầu test phải giống HỆT trường hợp mẫu
    đó không hề tồn tại trong dữ liệu (kiểm bằng cách so với dữ liệu đã xóa hẳn dòng đó), dù giá trị
    `r_excess` bị đặt thành một số cực đoan (999.0) để lộ ngay nếu còn rò rỉ vào huấn luyện."""
    labeled, dates = _no_leak_dataset()
    train_start_date, test_start_date, test_end_date = dates[0], dates[20], dates[-1]
    common_kwargs = dict(
        train_start_date=train_start_date, test_start_date=test_start_date, test_end_date=test_end_date,
        calendar=dates, refit_frequency=10, gbdt_common=GBDT_COMMON, seed=42, fraction=0.2,
    )
    leak_mask = (labeled["date"] == dates[5]) & (labeled["symbol"] == "AAA")

    labeled_dropped = labeled[~leak_mask].reset_index(drop=True)
    labeled_leaked = labeled.copy()
    labeled_leaked.loc[leak_mask, "r_excess"] = 999.0
    labeled_leaked.loc[leak_mask, "exit_date_resolved"] = dates[30]  # chưa hiện thực tại refit đầu

    scores_dropped = walk_forward_test_scores("R1_ridge", labeled_dropped, {"alpha": 1.0}, **common_kwargs)
    scores_leaked = walk_forward_test_scores("R1_ridge", labeled_leaked, {"alpha": 1.0}, **common_kwargs)

    first_day_dropped = scores_dropped[scores_dropped["date"] == test_start_date].sort_values("symbol")["score"].to_numpy()
    first_day_leaked = scores_leaked[scores_leaked["date"] == test_start_date].sort_values("symbol")["score"].to_numpy()
    np.testing.assert_allclose(first_day_dropped, first_day_leaked)
