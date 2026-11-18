"""Test cho leakage_checks.py (bước 3.8). Dữ liệu tổng hợp, không gọi mạng thật."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vnrank.evaluation.leakage_checks import (
    auc_one_vs_rest,
    macro_f1,
    ovr_ridge_classifier_fit,
    ovr_ridge_classifier_predict,
    rank_ic_by_day,
    ridge_fit,
    ridge_predict,
    run_label_leakage_check,
    run_label_leakage_check_repeated,
    run_trend_leakage_check,
    run_trend_leakage_check_repeated,
)


def test_ridge_fit_recovers_known_linear_relationship():
    rng = np.random.default_rng(0)
    n, d = 2000, 3
    X = rng.normal(0, 1, (n, d))
    true_beta = np.array([2.0, -1.0, 0.5])
    y = X @ true_beta + 1.0  # + intercept 1.0, không nhiễu
    beta = ridge_fit(X, y, alpha=0.01)
    assert beta[0] == pytest.approx(1.0, abs=1e-4)
    assert beta[1:] == pytest.approx(true_beta, abs=1e-4)
    pred = ridge_predict(X, beta)
    assert pred == pytest.approx(y, abs=1e-3)


def test_rank_ic_by_day_perfect_and_zero_correlation():
    dates = pd.to_datetime(["2024-01-01"] * 4 + ["2024-01-02"] * 4)
    scores_perfect = [1, 2, 3, 4, 10, 20, 30, 40]
    true_perfect = [1, 2, 3, 4, 100, 200, 300, 400]
    out = rank_ic_by_day(dates, np.array(scores_perfect), np.array(true_perfect))
    assert out["ic_mean"] == pytest.approx(1.0)
    assert out["n_days"] == 2

    scores_inverse = [4, 3, 2, 1, 40, 30, 20, 10]
    out2 = rank_ic_by_day(dates, np.array(scores_inverse), np.array(true_perfect))
    assert out2["ic_mean"] == pytest.approx(-1.0)


def test_rank_ic_by_day_skips_degenerate_days():
    dates = pd.to_datetime(["2024-01-01"] * 2)  # chỉ 2 dòng, dưới ngưỡng 3
    out = rank_ic_by_day(dates, np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert out["n_days"] == 0
    assert np.isnan(out["ic_mean"])


def _synthetic_panel(rng, n_days=60, n_symbols=40, n_features=5):
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rows = []
    true_w = rng.normal(0, 1, n_features)
    for d in dates:
        X = rng.normal(0, 1, (n_symbols, n_features))
        y = X @ true_w + rng.normal(0, 0.1, n_symbols)  # tín hiệu thật, nhiễu nhỏ
        for i in range(n_symbols):
            rows.append({"date": d, "symbol": i, **{f"f{j}": X[i, j] for j in range(n_features)}, "y": y[i]})
    return pd.DataFrame(rows), [f"f{j}" for j in range(n_features)]


def test_leakage_check_real_signal_gives_high_ic_sanity():
    rng = np.random.default_rng(1)
    df, feat_cols = _synthetic_panel(rng)
    train = df[df["date"] < "2023-02-01"]
    val = df[df["date"] >= "2023-02-01"]
    out = run_label_leakage_check(
        train[feat_cols].to_numpy(), train["y"].to_numpy(),
        val["date"], val[feat_cols].to_numpy(), val["y"].to_numpy(),
    )
    assert out["ic_mean"] > 0.5  # tín hiệu thật rõ ràng, mô hình phải bắt được


def test_leakage_check_shuffled_labels_give_near_zero_ic_when_repeated():
    rng = np.random.default_rng(1)
    df, feat_cols = _synthetic_panel(rng)
    train = df[df["date"] < "2023-02-01"].copy()
    val = df[df["date"] >= "2023-02-01"]
    train_y = train["y"].to_numpy()
    out = run_label_leakage_check_repeated(
        train[feat_cols].to_numpy(),
        lambda: rng.permutation(train_y),
        val["date"], val[feat_cols].to_numpy(), val["y"].to_numpy(),
        n_repeats=20,
    )
    # một lần xáo trộn đơn lẻ có thể lệch khỏi 0 do trùng hợp (ít đặc trưng); lặp lại 20 lần độc lập
    # rồi lấy trung bình phải triệt tiêu hiệu ứng đó, cho IC gần 0 đáng tin cậy hơn
    assert abs(out["ic_mean"]) < 0.15


def test_leakage_check_pure_random_labels_give_near_zero_ic_when_repeated():
    rng = np.random.default_rng(1)
    df, feat_cols = _synthetic_panel(rng)
    train = df[df["date"] < "2023-02-01"].copy()
    val = df[df["date"] >= "2023-02-01"]
    n_train = len(train)
    out = run_label_leakage_check_repeated(
        train[feat_cols].to_numpy(),
        lambda: rng.normal(0, 1, n_train),
        val["date"], val[feat_cols].to_numpy(), val["y"].to_numpy(),
        n_repeats=20,
    )
    assert abs(out["ic_mean"]) < 0.15


# --- Bước 3.13: kiểm tra rò rỉ cho nhãn xu hướng (phân loại 3 lớp) ---


def test_auc_one_vs_rest_perfect_and_reversed_separation():
    scores = np.array([1, 2, 3, 4, 5, 6])
    y_perfect = np.array([0, 0, 0, 1, 1, 1])
    assert auc_one_vs_rest(scores, y_perfect) == pytest.approx(1.0)
    assert auc_one_vs_rest(-scores, y_perfect) == pytest.approx(0.0)


def test_auc_one_vs_rest_single_class_is_nan():
    assert np.isnan(auc_one_vs_rest(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1])))


def test_macro_f1_perfect_prediction_is_one():
    y = np.array([-1, -1, 0, 0, 1, 1])
    assert macro_f1(y, y, classes=(-1, 0, 1)) == pytest.approx(1.0)


def test_macro_f1_known_confusion():
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0])
    # lớp 0: tp=2, fp=1, fn=1 -> precision=2/3, recall=2/3, F1=2/3
    # lớp 1: tp=1, fp=1, fn=1 -> precision=1/2, recall=1/2, F1=1/2
    expected = (2 / 3 + 1 / 2) / 2
    assert macro_f1(y_true, y_pred, classes=(0, 1)) == pytest.approx(expected)


def _synthetic_trend_panel(rng, n_days=80, n_symbols=50, n_features=6):
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rows = []
    true_w = rng.normal(0, 1, n_features)
    for d in dates:
        X = rng.normal(0, 1, (n_symbols, n_features))
        latent = X @ true_w + rng.normal(0, 0.3, n_symbols)
        y = np.where(latent > 0.5, 1, np.where(latent < -0.5, -1, 0))
        for i in range(n_symbols):
            rows.append({"date": d, "symbol": i, **{f"f{j}": X[i, j] for j in range(n_features)}, "y": y[i]})
    return pd.DataFrame(rows), [f"f{j}" for j in range(n_features)]


def test_trend_leakage_check_real_signal_beats_random_sanity():
    rng = np.random.default_rng(2)
    df, feat_cols = _synthetic_trend_panel(rng)
    train = df[df["date"] < "2023-03-01"]
    val = df[df["date"] >= "2023-03-01"]
    out = run_trend_leakage_check(
        train[feat_cols].to_numpy(), train["y"].to_numpy(),
        val[feat_cols].to_numpy(), val["y"].to_numpy(),
    )
    assert out["auc_macro"] > 0.65  # tín hiệu thật rõ ràng, phải rõ rệt hơn 0,5
    assert out["macro_f1"] > 0.4


def test_trend_leakage_check_shuffled_labels_give_near_random_auc_when_repeated():
    rng = np.random.default_rng(2)
    df, feat_cols = _synthetic_trend_panel(rng)
    train = df[df["date"] < "2023-03-01"].copy()
    val = df[df["date"] >= "2023-03-01"]
    train_y = train["y"].to_numpy()
    out = run_trend_leakage_check_repeated(
        train[feat_cols].to_numpy(),
        lambda: rng.permutation(train_y),
        val[feat_cols].to_numpy(), val["y"].to_numpy(),
        n_repeats=15,
    )
    assert abs(out["auc_mean"] - 0.5) < 0.1
