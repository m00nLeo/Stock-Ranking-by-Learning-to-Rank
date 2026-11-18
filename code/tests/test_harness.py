"""Test cho lõi bộ đánh giá (bước 4.1, harness.py). Dữ liệu tổng hợp nhỏ tự dựng, không đọc dữ
liệu thật của dự án."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vnrank.evaluation.harness import (
    balanced_accuracy,
    brier_multiclass,
    brier_skill_score,
    circular_block_bootstrap_mean_ci,
    confusion_matrix,
    expected_calibration_error,
    f1_per_class,
    icir,
    icir_nonoverlapping,
    macro_f1,
    newey_west_se,
    newey_west_test,
    precision_at_k_by_day,
    rank_ic_series,
)
from vnrank.evaluation.multiple_testing import holm_step_down

H = 5  # kỳ hạn dùng trong các test liên quan đến độ trễ/cửa sổ


def _calendar(n):
    return pd.to_datetime(pd.date_range("2023-01-01", periods=n, freq="B"))


def _synthetic_panel(n_days=120, n_stocks=20, seed=123):
    """Mỗi ngày: lợi suất = 0,5*quality + nhiễu; tín hiệu momentum ~ quality (tương quan thật với
    lợi suất); tín hiệu ngẫu nhiên độc lập với lợi suất."""
    rng = np.random.default_rng(seed)
    dates = _calendar(n_days)
    rows = []
    for d in dates:
        quality = rng.normal(0, 1, n_stocks)
        returns = 0.5 * quality + rng.normal(0, 1, n_stocks)
        momentum_signal = quality + rng.normal(0, 0.3, n_stocks)
        random_signal = rng.normal(0, 1, n_stocks)
        for i in range(n_stocks):
            rows.append((d, returns[i], momentum_signal[i], random_signal[i]))
    return pd.DataFrame(rows, columns=["date", "ret", "mom", "rand"])


# --- Rank IC, ICIR -----------------------------------------------------------------------------


def test_rank_ic_series_perfect_rank_correlation_each_day():
    dates = pd.Series(["d1"] * 4 + ["d2"] * 4)
    scores = pd.Series([1, 2, 3, 4, 4, 3, 2, 1])
    true = pd.Series([10, 20, 30, 40, 40, 30, 20, 10])
    ic = rank_ic_series(dates, scores.to_numpy(), true.to_numpy())
    assert list(ic.index) == ["d1", "d2"]
    assert ic.loc["d1"] == pytest.approx(1.0)
    assert ic.loc["d2"] == pytest.approx(1.0)


def test_rank_ic_series_skips_day_with_too_few_or_constant_values():
    dates = pd.Series(["d1", "d1", "d2", "d2", "d2"])
    scores = pd.Series([1, 2, 5, 5, 5])  # d1 chỉ 2 quan sát; d2 score không đổi
    true = pd.Series([1, 2, 1, 2, 3])
    ic = rank_ic_series(dates, scores.to_numpy(), true.to_numpy())
    assert ic.empty


def test_icir_matches_manual_mean_std():
    ic = pd.Series([0.1, 0.2, 0.0, -0.1, 0.3])
    out = icir(ic)
    assert out["ic_mean"] == pytest.approx(ic.mean())
    assert out["ic_std"] == pytest.approx(ic.std(ddof=1))
    assert out["icir"] == pytest.approx(ic.mean() / ic.std(ddof=1))
    assert out["n"] == 5


def test_icir_nonoverlapping_subsamples_every_h_th_day():
    ic = pd.Series(range(10), dtype=float)  # 0..9
    out = icir_nonoverlapping(ic, h=3)
    # lấy chỉ số 0, 3, 6, 9 -> trung bình (0+3+6+9)/4 = 4.5
    assert out["n"] == 4
    assert out["ic_mean"] == pytest.approx(4.5)


# --- precision@K ---------------------------------------------------------------------------


def test_precision_at_k_by_day_counts_positive_returns_in_top_k():
    dates = pd.Series(["d1"] * 4)
    scores = pd.Series([4, 3, 2, 1])  # top-2 theo score: index 0, 1
    returns = pd.Series([0.01, -0.02, 0.03, 0.04])  # top-2 có 1/2 dương
    out = precision_at_k_by_day(dates, scores.to_numpy(), returns.to_numpy(), k=2)
    assert out["d1"] == pytest.approx(0.5)


def test_precision_at_k_by_day_skips_day_with_fewer_than_k_rows():
    dates = pd.Series(["d1", "d1"])
    out = precision_at_k_by_day(dates, np.array([1.0, 2.0]), np.array([0.1, -0.1]), k=5)
    assert out.empty


# --- Newey-West ----------------------------------------------------------------------------


def test_newey_west_se_with_lag_zero_equals_naive_standard_error():
    # lag=0: Newey-West rút về sai số chuẩn thường của trung bình, nhưng dùng chuẩn hóa 1/n
    # (phương sai tổng thể, ddof=0) cho mọi gamma_l kể cả gamma_0, theo đúng quy ước HAC gốc
    # (Newey-West, 1987), không phải 1/(n-1) như độ lệch chuẩn mẫu thông thường.
    rng = np.random.default_rng(1)
    d = rng.normal(0, 1, 200)
    se_nw = newey_west_se(d, lag=0)
    se_naive = d.std(ddof=0) / np.sqrt(len(d))
    assert se_nw == pytest.approx(se_naive, rel=1e-9)


def test_newey_west_test_margin_shifts_null_hypothesis():
    d = np.full(50, 0.01)  # chuỗi hằng số dương nhỏ, se=0 -> t_stat vô hạn
    out0 = newey_west_test(d, lag=4, margin=0.0)
    out_margin = newey_west_test(d, lag=4, margin=-0.02)  # H0: E[d] <= 0.02, d=0.01 không vượt qua
    assert out0["p_value"] < 1e-6
    assert out_margin["p_value"] > 0.5


def test_newey_west_random_signal_not_rejected_momentum_signal_rejected():
    df = _synthetic_panel()
    ic_rand = rank_ic_series(df["date"], df["rand"].to_numpy(), df["ret"].to_numpy())
    ic_mom = rank_ic_series(df["date"], df["mom"].to_numpy(), df["ret"].to_numpy())

    nw_rand = newey_west_test(ic_rand.to_numpy(), lag=H - 1)
    nw_mom = newey_west_test(ic_mom.to_numpy(), lag=H - 1)

    assert abs(nw_rand["mean"]) < 0.1
    assert nw_rand["p_value"] > 0.05  # không bác bỏ H0 (không có tín hiệu)

    assert nw_mom["mean"] > 0.2
    assert nw_mom["p_value"] < 0.001  # bác bỏ rõ ràng (có tín hiệu thật)


def test_holm_does_not_reject_random_signals_but_rejects_momentum():
    """Mô phỏng 4 'giả thuyết' H1-H4 cùng kiểm định trên tín hiệu ngẫu nhiên (độc lập) và trên tín
    hiệu momentum, xác nhận Holm không bác bỏ giả thuyết nào ở tín hiệu ngẫu nhiên, nhưng bác bỏ hết
    ở tín hiệu momentum có tương quan thật."""
    p_values_random = []
    p_values_momentum = []
    for seed in (10, 20, 30, 40):
        df = _synthetic_panel(seed=seed)
        ic_rand = rank_ic_series(df["date"], df["rand"].to_numpy(), df["ret"].to_numpy())
        ic_mom = rank_ic_series(df["date"], df["mom"].to_numpy(), df["ret"].to_numpy())
        p_values_random.append(newey_west_test(ic_rand.to_numpy(), lag=H - 1)["p_value"])
        p_values_momentum.append(newey_west_test(ic_mom.to_numpy(), lag=H - 1)["p_value"])

    assert holm_step_down(p_values_random, alpha=0.05) == [False, False, False, False]
    assert holm_step_down(p_values_momentum, alpha=0.05) == [True, True, True, True]


# --- Bootstrap khối tròn ---------------------------------------------------------------------


def test_circular_block_bootstrap_ci_excludes_zero_for_clear_momentum_signal():
    df = _synthetic_panel()
    ic_mom = rank_ic_series(df["date"], df["mom"].to_numpy(), df["ret"].to_numpy())
    out = circular_block_bootstrap_mean_ci(ic_mom.to_numpy(), block_len=2 * H, n_boot=2000, seed=7)
    assert out["ci_low"] > 0.0  # cận dưới một phía vẫn dương -> có tín hiệu vững


def test_circular_block_bootstrap_ci_includes_zero_for_random_signal():
    df = _synthetic_panel()
    ic_rand = rank_ic_series(df["date"], df["rand"].to_numpy(), df["ret"].to_numpy())
    out = circular_block_bootstrap_mean_ci(ic_rand.to_numpy(), block_len=2 * H, n_boot=2000, seed=7)
    assert out["ci_low"] < 0.0  # không có bằng chứng tín hiệu dương vững


def test_circular_block_bootstrap_is_reproducible_with_same_seed():
    d = np.array([0.1, -0.2, 0.3, 0.05, -0.1, 0.2, 0.15, -0.05] * 5)
    out1 = circular_block_bootstrap_mean_ci(d, block_len=4, n_boot=500, seed=42)
    out2 = circular_block_bootstrap_mean_ci(d, block_len=4, n_boot=500, seed=42)
    assert out1 == out2


# --- macro-F1 / balanced accuracy / ma trận nhầm lẫn ------------------------------------------


def test_balanced_accuracy_perfect_prediction():
    y_true = np.array([-1, -1, 0, 0, 1, 1])
    y_pred = y_true.copy()
    assert balanced_accuracy(y_true, y_pred, classes=(-1, 0, 1)) == pytest.approx(1.0)


def test_balanced_accuracy_majority_class_baseline():
    y_true = np.array([-1, 0, 0, 0, 1])  # lớp đa số là 0
    y_pred = np.array([0, 0, 0, 0, 0])  # luôn đoán lớp đa số
    # recall lớp -1 = 0, lớp 0 = 1, lớp 1 = 0 -> trung bình 1/3
    assert balanced_accuracy(y_true, y_pred, classes=(-1, 0, 1)) == pytest.approx(1 / 3)


def test_confusion_matrix_counts_correctly():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    cm = confusion_matrix(y_true, y_pred, classes=(0, 1))
    assert cm.loc[0, 0] == 2  # hai lần thật=0, đoán=0
    assert cm.loc[1, 0] == 1  # thật=1, đoán=0
    assert cm.loc[1, 1] == 1  # thật=1, đoán=1


def test_macro_f1_perfect_prediction_is_one():
    y_true = np.array([-1, -1, 0, 0, 1, 1])
    y_pred = y_true.copy()
    assert macro_f1(y_true, y_pred, classes=(-1, 0, 1)) == pytest.approx(1.0)


def test_macro_f1_majority_class_baseline_is_low():
    # Luôn đoán lớp đa số (0): lớp -1 và 1 có F1 = 0 (recall 0), lớp 0 precision < 1 vì có FP.
    y_true = np.array([-1, 0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0, 0])
    per_class = f1_per_class(y_true, y_pred, classes=(-1, 0, 1))
    assert per_class[-1] == pytest.approx(0.0)
    assert per_class[1] == pytest.approx(0.0)
    assert 0.0 < per_class[0] < 1.0
    assert macro_f1(y_true, y_pred, classes=(-1, 0, 1)) == pytest.approx(np.mean(list(per_class.values())))


def test_macro_f1_ignores_class_absent_from_true_labels():
    y_true = np.array([0, 0, 1, 1])  # lớp -1 không xuất hiện trong y_true
    y_pred = np.array([0, 0, 1, -1])
    per_class = f1_per_class(y_true, y_pred, classes=(-1, 0, 1))
    assert -1 not in per_class
    assert set(per_class) == {0, 1}


# --- Brier, BSS, ECE (mô tả) -------------------------------------------------------------------


def test_brier_multiclass_zero_for_perfect_confident_prediction():
    y_true = np.array([1, 0])
    probs = pd.DataFrame({0: [0.0, 1.0], 1: [1.0, 0.0]})
    assert brier_multiclass(y_true, probs, classes=(0, 1)) == pytest.approx(0.0)


def test_brier_skill_score_positive_when_model_better_than_baseline():
    assert brier_skill_score(brier_model=0.1, brier_baseline=0.2) == pytest.approx(0.5)


def test_expected_calibration_error_zero_when_perfectly_calibrated():
    # 10 mẫu, xác suất dự báo lớp 1 luôn đúng bằng tỷ lệ đúng thực tế trong nhóm đó (ở đây tầm thường:
    # dự báo tuyệt đối đúng, confidence = 1.0 và luôn đúng -> ECE = 0).
    y_true = np.array([1] * 10)
    probs = pd.DataFrame({0: [0.0] * 10, 1: [1.0] * 10})
    assert expected_calibration_error(y_true, probs, classes=(0, 1), n_bins=5) == pytest.approx(0.0)


def test_expected_calibration_error_positive_when_overconfident():
    rng = np.random.default_rng(5)
    y_true = (rng.random(200) < 0.5).astype(int)  # thật ra chỉ đúng 50% số lần
    probs = pd.DataFrame({0: [0.1] * 200, 1: [0.9] * 200})  # nhưng luôn tự tin 90% vào lớp 1
    ece = expected_calibration_error(y_true, probs, classes=(0, 1), n_bins=5)
    assert ece > 0.2  # lệch lớn giữa độ tự tin (0,9) và tỷ lệ đúng thật (~0,5)
