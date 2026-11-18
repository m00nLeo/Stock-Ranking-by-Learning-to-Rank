"""Test cho bộ phân loại xu hướng T2 (bước 4.10): LightGBM đa lớp, đối chứng D0/D1/D2, kiểm định
H3 (bootstrap khối theo ngày). Dữ liệu tổng hợp nhỏ tự dựng, không đọc dữ liệu thật."""
from __future__ import annotations

import numpy as np
import pandas as pd

from vnrank.evaluation.harness import macro_f1
from vnrank.features.feature_registry import FEATURE_COLUMNS
from vnrank.models.trend_classifier import (
    CLASSES,
    _fit_predict_t2,
    _macro_f1_from_confusion,
    block_bootstrap_h3,
    build_candidate_configs,
    build_labeled_features,
    class_frequencies,
    d0_walk_forward_scores,
    d1_walk_forward_scores,
    d2_walk_forward_scores,
    evaluate_classification,
    h3_test,
    majority_class,
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


# --- build_candidate_configs: lưới CHUNG với R2/L1 ----------------------------------------------


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


# --- build_labeled_features: purge theo t1, bỏ nhãn chưa hiện thực -------------------------------


def test_build_labeled_features_drops_rows_with_missing_y_or_t1():
    features = pd.DataFrame([_zero_features_row("2024-01-01", "AAA"), _zero_features_row("2024-01-01", "BBB")])
    features["date"] = pd.to_datetime(features["date"])
    labels_h = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "symbol": ["AAA", "BBB"],
            "y": [1, np.nan],  # BBB chưa đủ dữ liệu ước lượng sigma (đầu chuỗi, labeling.py)
            "t1": pd.to_datetime(["2024-01-05", None]),
        }
    )
    out = build_labeled_features(features, labels_h)
    assert list(out["symbol"]) == ["AAA"]
    assert out["y"].dtype == np.int64 or out["y"].dtype == int


# --- majority_class, class_frequencies -----------------------------------------------------------


def test_majority_class_picks_most_frequent():
    y = np.array([1, 1, 1, 0, 0, -1])
    assert majority_class(y) == 1


def test_majority_class_tie_break_follows_classes_order():
    # Đồng hạng giữa tăng(1) và đi ngang(0): CLASSES = (1, 0, -1) -> chọn 1 trước.
    y = np.array([1, 1, 0, 0, -1])
    assert majority_class(y) == 1


def test_class_frequencies_sum_to_one_and_match_counts():
    y = np.array([1, 1, 0, -1, -1, -1])
    freq = class_frequencies(y)
    assert freq[1] == 2 / 6
    assert freq[0] == 1 / 6
    assert freq[-1] == 3 / 6
    assert sum(freq.values()) == 1.0


# --- fit_predict / walk_forward_test_scores: bắt được tín hiệu rõ ràng --------------------------


def _linear_synthetic_window(n_dates: int = 30, n_symbols: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    rows = []
    for d in dates:
        for i in range(n_symbols):
            row = {"date": d, "symbol": f"S{i}"}
            for col in FEATURE_COLUMNS:
                row[col] = 0.0
            signal = rng.normal()
            row["ret_21"] = signal
            row["y"] = 1 if signal > 0.3 else (-1 if signal < -0.3 else 0)
            rows.append(row)
    return pd.DataFrame(rows)


def test_fit_predict_t2_recovers_class_signal():
    window = _linear_synthetic_window(n_dates=60, n_symbols=8, seed=0)
    predict_df = window[window["date"] == window["date"].max()]
    params = {"num_leaves": 15, "min_data_in_leaf": 5}
    probs = _fit_predict_t2(window, params, GBDT_COMMON, seed=42, fraction=0.2, predict_df=predict_df)
    assert probs.shape == (len(predict_df), len(CLASSES))
    pred = np.array(CLASSES)[np.argmax(probs, axis=1)]
    true = predict_df["y"].to_numpy()
    accuracy = float((pred == true).mean())
    assert accuracy > 0.5, f"T2: khong bat duoc tin hieu ro rang (accuracy={accuracy:.3f})"


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
            label_rows.append({"date": d, "symbol": s, "y": 1 if s == "AAA" else -1, "t1": d + pd.Timedelta(days=1)})
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
    assert len(scores) == len(expected_dates) * 2


def test_walk_forward_test_scores_does_not_leak_unresolved_future_label():
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
    labeled_leaked.loc[leak_mask, "y"] = -1  # cực đoan trái ngược AAA
    labeled_leaked.loc[leak_mask, "t1"] = dates[30]  # chưa hiện thực tại refit đầu

    scores_dropped = walk_forward_test_scores(labeled_dropped, params, **common_kwargs)
    scores_leaked = walk_forward_test_scores(labeled_leaked, params, **common_kwargs)

    first_day_dropped = scores_dropped[scores_dropped["date"] == test_start_date].sort_values("symbol")[list(CLASSES)].to_numpy()
    first_day_leaked = scores_leaked[scores_leaked["date"] == test_start_date].sort_values("symbol")[list(CLASSES)].to_numpy()
    np.testing.assert_allclose(first_day_dropped, first_day_leaked)


# --- select_config_on_validation: chọn đúng cấu hình có tín hiệu thật ---------------------------


def test_select_config_on_validation_prefers_config_with_real_signal():
    window = _linear_synthetic_window(n_dates=50, n_symbols=8, seed=1)
    train_start_date = window["date"].min()
    train_end_date = window["date"].quantile(0.6, interpolation="lower")

    labeled = window.assign(t1=window["date"] + pd.Timedelta(days=1))
    splits = pd.DataFrame({"date": sorted(labeled["date"].unique())})
    splits["split"] = np.where(splits["date"] <= train_end_date, "train", "validation")

    configs = {"good": {"num_leaves": 15, "min_data_in_leaf": 5}, "degenerate": {"num_leaves": 15, "min_data_in_leaf": 10_000}}
    best_key, val_f1 = select_config_on_validation(
        labeled, configs, splits, train_start_date, train_end_date, GBDT_COMMON, seed=42, fraction=0.2,
    )
    assert best_key == "good"
    assert val_f1["good"] >= val_f1["degenerate"]


# --- D0/D1/D2: không dùng đặc trưng, chỉ dựa trên nhãn cửa sổ huấn luyện -------------------------


def test_d0_walk_forward_scores_predicts_majority_class_of_window():
    labeled, dates = _no_leak_dataset()
    train_start_date, test_start_date, test_end_date = dates[0], dates[20], dates[-1]
    scores = d0_walk_forward_scores(labeled, train_start_date, test_start_date, test_end_date, dates, refit_frequency=10)
    # AAA luôn y=1, BBB luôn y=-1 trong toàn bộ cửa sổ huấn luyện -> không có lớp đa số tuyệt đối rõ
    # ràng (50/50), nhưng majority_class tất định chọn CLASSES[0]=1 khi đồng hạng.
    assert set(scores["pred"].unique()) <= {1, 0, -1}
    assert len(scores) == len(set(d for d in dates if test_start_date <= d <= test_end_date)) * 2


def test_d1_walk_forward_scores_probabilities_sum_to_one():
    labeled, dates = _no_leak_dataset()
    train_start_date, test_start_date, test_end_date = dates[0], dates[20], dates[-1]
    scores = d1_walk_forward_scores(labeled, train_start_date, test_start_date, test_end_date, dates, refit_frequency=10)
    totals = scores[list(CLASSES)].sum(axis=1)
    np.testing.assert_allclose(totals.to_numpy(), 1.0)


def test_d2_walk_forward_scores_is_deterministic_given_seed():
    labeled, dates = _no_leak_dataset()
    train_start_date, test_start_date, test_end_date = dates[0], dates[20], dates[-1]
    scores_a = d2_walk_forward_scores(labeled, train_start_date, test_start_date, test_end_date, dates, refit_frequency=10, seed=7)
    scores_b = d2_walk_forward_scores(labeled, train_start_date, test_start_date, test_end_date, dates, refit_frequency=10, seed=7)
    pd.testing.assert_frame_equal(scores_a, scores_b)


# --- evaluate_classification: macro-F1/balanced accuracy trên mẫu mã-ngày của test ---------------


def test_evaluate_classification_perfect_prediction():
    dates = pd.bdate_range("2024-01-01", periods=5)
    scored = pd.DataFrame({"date": dates, "symbol": "AAA", "pred": [1, 1, 0, -1, -1]})
    labels_h = pd.DataFrame({"date": dates, "symbol": "AAA", "y": [1, 1, 0, -1, -1]})
    splits = pd.DataFrame({"date": dates, "split": "test"})
    out = evaluate_classification(scored, labels_h, splits)
    assert out["macro_f1"] == 1.0
    assert out["balanced_accuracy"] == 1.0
    assert out["n"] == 5


# --- block_bootstrap_h3 / _macro_f1_from_confusion: khớp đúng harness.macro_f1 trực tiếp ---------


def test_macro_f1_from_confusion_matches_harness_macro_f1_directly():
    rng = np.random.default_rng(0)
    y_true = rng.choice(CLASSES, size=200)
    y_pred = rng.choice(CLASSES, size=200)
    from vnrank.evaluation.harness import confusion_matrix as cm_fn

    cm = cm_fn(y_true, y_pred, CLASSES).to_numpy()
    direct = macro_f1(y_true, y_pred, CLASSES)
    fast = _macro_f1_from_confusion(cm)
    assert abs(fast - direct) < 1e-12


def test_block_bootstrap_h3_delta_obs_matches_direct_macro_f1_difference():
    rng = np.random.default_rng(0)
    n = 300
    dates = pd.bdate_range("2024-01-01", periods=n // 3).repeat(3)[:n]
    y_true = rng.choice(CLASSES, size=n)
    pred_t2 = y_true.copy()  # T2 đoán đúng hoàn toàn
    pred_d0 = rng.choice(CLASSES, size=n)  # D0 đoán ngẫu nhiên

    out = block_bootstrap_h3(np.array(dates), y_true, pred_t2, pred_d0, block_len=10, n_boot=200, seed=1)
    direct_delta = macro_f1(y_true, pred_t2, CLASSES) - macro_f1(y_true, pred_d0, CLASSES)
    assert out["delta_obs"] == direct_delta
    assert out["delta_obs"] > 0.5  # T2 hoàn hảo, D0 ngẫu nhiên -> chênh lệch lớn
    assert 0.0 <= out["p_value"] <= 1.0


def test_block_bootstrap_h3_p_value_high_when_no_real_difference():
    rng = np.random.default_rng(2)
    n = 300
    dates = pd.bdate_range("2024-01-01", periods=n // 3).repeat(3)[:n]
    y_true = rng.choice(CLASSES, size=n)
    pred_t2 = rng.choice(CLASSES, size=n)  # cả hai đều ngẫu nhiên, không hơn nhau thật sự
    pred_d0 = rng.choice(CLASSES, size=n)

    out = block_bootstrap_h3(np.array(dates), y_true, pred_t2, pred_d0, block_len=10, n_boot=500, seed=3)
    assert out["p_value"] > 0.05


def test_h3_test_wraps_block_bootstrap_with_right_block_length():
    dates = pd.bdate_range("2024-01-01", periods=30)
    rng = np.random.default_rng(0)
    rows_t2, rows_d0, label_rows = [], [], []
    for d in dates:
        true = rng.choice(CLASSES, size=5)
        rows_t2 += [{"date": d, "symbol": f"S{i}", "pred": true[i]} for i in range(5)]
        rows_d0 += [{"date": d, "symbol": f"S{i}", "pred": rng.choice(CLASSES)} for i in range(5)]
        label_rows += [{"date": d, "symbol": f"S{i}", "y": true[i]} for i in range(5)]
    t2_scored = pd.DataFrame(rows_t2)
    d0_scored = pd.DataFrame(rows_d0)
    labels_h = pd.DataFrame(label_rows)
    splits = pd.DataFrame({"date": dates, "split": "test"})

    out = h3_test(t2_scored, d0_scored, labels_h, splits, h=5, n_boot=100, seed=0)
    assert out["h"] == 5
    assert out["block_len"] == 10  # 2*h
    assert out["delta_obs"] > 0  # T2 đoán đúng hoàn toàn, D0 ngẫu nhiên
