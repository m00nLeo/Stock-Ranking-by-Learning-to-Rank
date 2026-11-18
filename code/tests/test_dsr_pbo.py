"""Test cho công thức DSR/PBO (bước 4.14). Đối chiếu DSR với ví dụ số đầy đủ trong bài báo gốc
(decision_log.md D116, Bailey & López de Prado 2014 trang 9-10) để xác nhận cách cài ĐÚNG đơn vị
(V/SR0 năm hóa, sr_test không năm hóa). PBO/estimate_effective_n dùng dữ liệu tổng hợp tự dựng."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vnrank.evaluation.dsr_pbo import (
    cscv_pbo,
    deflated_sharpe_ratio,
    estimate_effective_n,
    sharpe_raw,
)


# --- sharpe_raw -----------------------------------------------------------------------------


def test_sharpe_raw_matches_manual_mean_over_std():
    r = np.array([0.01, 0.02, -0.01, 0.03, 0.0])
    expected = r.mean() / r.std(ddof=1)
    assert sharpe_raw(r) == pytest.approx(expected)


def test_sharpe_raw_nan_when_zero_variance():
    assert np.isnan(sharpe_raw(np.array([0.01, 0.01, 0.01])))


# --- deflated_sharpe_ratio: khớp ví dụ số D116 -----------------------------------------------


def test_deflated_sharpe_ratio_matches_worked_example():
    """D116: N=100, V=0,5 (Sharpe NĂM HÓA), T=1250, gamma3=-3, gamma4=10, SR năm hóa=2,5
    -> SR0~0,1132, DSR~0,9004 (SR0/sr_test ở đơn vị KHÔNG năm hóa, chia SR năm hóa cho căn(250))."""
    ann_factor = np.sqrt(250)
    sr_test = 2.5 / ann_factor
    out = deflated_sharpe_ratio(
        sr_test=sr_test, n_trials=100, var_sr_trials_annualized=0.5, ann_factor=ann_factor,
        t_obs=1250, skew=-3, kurt=10,
    )
    assert out["sr0"] == pytest.approx(0.1132, abs=1e-3)
    assert out["dsr"] == pytest.approx(0.9004, abs=1e-3)


def test_deflated_sharpe_ratio_more_trials_deflates_more():
    """N lớn hơn (nhiều thử nghiệm hơn) phải kéo SR0 lên (càng khó vượt qua), DSR giảm, mọi tham
    số khác giữ nguyên — đúng trực giác chọn mẫu càng nhiều càng dễ gặp may."""
    kwargs = dict(sr_test=0.2, var_sr_trials_annualized=0.3, ann_factor=np.sqrt(250), t_obs=500, skew=0.0, kurt=3.0)
    small_n = deflated_sharpe_ratio(n_trials=5, **kwargs)
    large_n = deflated_sharpe_ratio(n_trials=200, **kwargs)
    assert large_n["sr0"] > small_n["sr0"]
    assert large_n["dsr"] < small_n["dsr"]


def test_deflated_sharpe_ratio_single_trial_sr0_is_zero():
    out = deflated_sharpe_ratio(
        sr_test=0.1, n_trials=1, var_sr_trials_annualized=0.3, ann_factor=np.sqrt(250), t_obs=300, skew=0.0, kurt=3.0,
    )
    assert out["sr0"] == 0.0


# --- estimate_effective_n --------------------------------------------------------------------


def test_estimate_effective_n_perfectly_correlated_trials_equals_one():
    rng = np.random.default_rng(0)
    base = rng.normal(size=100)
    trials = {"a": base, "b": base * 2.0, "c": base + 5.0}  # tuong quan 1 (bien doi tuyen tinh)
    out = estimate_effective_n(trials)
    assert out["rho_hat"] == pytest.approx(1.0, abs=1e-9)
    assert out["n_hat"] == pytest.approx(1.0, abs=1e-9)
    assert out["m"] == 3


def test_estimate_effective_n_independent_trials_equals_m():
    rng = np.random.default_rng(1)
    trials = {f"t{i}": rng.normal(size=5000) for i in range(5)}
    out = estimate_effective_n(trials)
    assert out["rho_hat"] == pytest.approx(0.0, abs=0.05)
    assert out["n_hat"] == pytest.approx(5.0, abs=0.3)


def test_estimate_effective_n_single_trial_returns_m():
    out = estimate_effective_n({"only": np.array([0.1, 0.2, 0.3])})
    assert out["m"] == 1
    assert out["n_hat"] == 1.0


# --- cscv_pbo --------------------------------------------------------------------------------


def test_cscv_pbo_rejects_odd_s():
    mat = pd.DataFrame(np.random.default_rng(0).normal(size=(64, 3)))
    with pytest.raises(ValueError):
        cscv_pbo(mat, s=7)


def test_cscv_pbo_high_when_best_is_config_is_pure_is_noise():
    """Dựng N cấu hình là nhiễu thuần túy, độc lập, KHÔNG có cấu hình nào thật sự tốt hơn — cấu
    hình thắng IS chỉ do may rủi, nên PBO phải cao (gần 0,5, đúng kỳ vọng lý thuyết CSCV khi không
    có tín hiệu thật: hạng OOS của cấu hình thắng IS phân bố đều quanh trung vị)."""
    rng = np.random.default_rng(42)
    t_obs, n_cfg = 480, 10
    mat = pd.DataFrame(rng.normal(scale=0.01, size=(t_obs, n_cfg)))
    out = cscv_pbo(mat, s=8)
    assert 0.3 <= out["pbo"] <= 0.7
    assert out["n_combos"] == 70  # C(8,4)


def test_cscv_pbo_low_when_one_config_dominates_in_and_out_of_sample():
    """Một cấu hình có lợi suất kỳ vọng dương RÕ RỆT và NHẤT QUÁN trên mọi khối (IS lẫn OOS) —
    cấu hình này luôn thắng IS và luôn hạng cao OOS, PBO phải thấp."""
    rng = np.random.default_rng(7)
    t_obs, n_cfg = 480, 6
    mat = rng.normal(scale=0.01, size=(t_obs, n_cfg))
    mat[:, 0] += 0.02  # cot 0: vuot troi moi phien, moi khoi
    out = cscv_pbo(pd.DataFrame(mat), s=8)
    assert out["pbo"] < 0.2


def test_cscv_pbo_block_size_and_sessions_used():
    mat = pd.DataFrame(np.random.default_rng(0).normal(size=(188, 4)))
    out = cscv_pbo(mat, s=8)
    assert out["block_size"] == 23  # 188 // 8
    assert out["n_sessions_used"] == 184
    assert out["n_sessions_total"] == 188
