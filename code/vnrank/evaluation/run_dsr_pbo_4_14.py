"""Chạy DSR/PBO thật trên dữ liệu dự án (bước 4.14). Lắp ráp `trial_portfolios.py` (dựng lại danh
mục validation cho từng cấu hình) với `dsr_pbo.py` (công thức). Hai nhóm tìm kiếm (D116, D118):
"l1_pool" (N=21, h=5, chiến lược chính = L1) và "a1_pool" (N=4, h=5, chiến lược chính = A1).

Chạy: `python -m vnrank.evaluation.run_dsr_pbo_4_14` từ `code/`, dùng `code/.venv` (D96).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kurtosis, skew

from vnrank.costs.portfolio_backtest import preselected_portfolio_returns, sharpe_annualized, top_k_portfolio_returns
from vnrank.evaluation.dsr_pbo import cscv_pbo, deflated_sharpe_ratio, estimate_effective_n, sharpe_raw
from vnrank.evaluation.trial_portfolios import (
    CODE_ROOT,
    H,
    K,
    L1_FROZEN_CONFIG,
    T2_FROZEN_CONFIG,
    build_a0_a3_selections_validation,
    build_baseline_scores_validation,
    build_l1_scores_validation,
    build_regression_scores_validation,
    build_staggered_daily_series,
    load_common,
    preselected_trial,
    top_k_trial,
)
from vnrank.models import trend_classifier as t2
from vnrank.models.ranking_lambdarank import build_labeled_features as l1_build_labeled_features
from vnrank.models.ranking_lambdarank import walk_forward_test_scores as l1_walk_forward_test_scores
from vnrank.models.regression_topk import REALIZATION_COL as REG_REALIZATION_COL
from vnrank.models.trend_integration import build_labeled_features_sign, two_stage_select
from vnrank.splits.split_protocol import build_refit_schedule

ANN_FACTOR = np.sqrt(250 / H)


def _pool_h_period_and_daily(
    scores_by_name: dict[str, pd.DataFrame], common: dict, rebalance_dates_phase0: list, calendar: list, is_preselected: bool, k_for_preselected: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, pd.Series]]:
    period_returns, daily_series = {}, {}
    for name, scored in scores_by_name.items():
        if is_preselected:
            r_net_df = preselected_trial(scored, common, rebalance_dates_phase0, k_for_preselected)
            build_fn = lambda rb, s=scored: preselected_trial(s, common, rb, k_for_preselected)
        else:
            r_net_df = top_k_trial(scored, common, rebalance_dates_phase0)
            build_fn = lambda rb, s=scored: top_k_trial(s, common, rb)
        period_returns[name] = r_net_df["r_net"].to_numpy()
        daily_series[name] = build_staggered_daily_series(build_fn, calendar, H)
        print(f"  [validation] {name}: {len(period_returns[name])} ky, Sharpe tho={sharpe_raw(period_returns[name]):.4f}")
    return period_returns, daily_series


def build_l1_test_series(common: dict) -> np.ndarray:
    """Chuỗi lợi suất kỳ TEST (K=10, chi phí cơ sở) của L1 (cấu hình đã đóng băng), để tính
    sr_test/T/skew/kurt của DSR (nhóm l1_pool)."""
    calendar = sorted(common["splits"]["date"].unique())
    test_start = common["splits"].loc[common["splits"]["split"] == "test", "date"].min()
    test_end = common["splits"].loc[common["splits"]["split"] == "test", "date"].max()
    refit_frequency = common["cfg"]["data_split"]["refit_frequency_sessions"]

    labeled = l1_build_labeled_features(common["features"], common["labels_h5"])
    test_scores = l1_walk_forward_test_scores(
        labeled, L1_FROZEN_CONFIG, common["train_start_date"], test_start, test_end, calendar,
        refit_frequency, common["gbdt_common"], common["seed"], common["fraction"],
    )
    test_scores = test_scores.merge(common["splits"], on="date", how="inner")
    test_scores = test_scores[test_scores["split"] == "test"][["date", "symbol", "score"]]

    rebalance_dates = build_refit_schedule(test_start, test_end, calendar, frequency=H)
    r_net = top_k_portfolio_returns(test_scores, common["labels_h5"], rebalance_dates, K, common["c_buy"], common["c_sell"])
    return r_net["r_net"].to_numpy(), test_scores


def build_a1_test_series(common: dict, l1_test_scores: pd.DataFrame) -> np.ndarray:
    """Chuỗi lợi suất kỳ TEST (K=10, chi phí cơ sở) của A1 (tích hợp xu hướng), đúng quy trình
    `trend_integration.py` __main__ (D103)."""
    calendar = sorted(common["splits"]["date"].unique())
    test_start = common["splits"].loc[common["splits"]["split"] == "test", "date"].min()
    test_end = common["splits"].loc[common["splits"]["split"] == "test", "date"].max()
    refit_frequency = common["cfg"]["data_split"]["refit_frequency_sessions"]
    top_m = common["cfg"]["trend_integration"]["top_M"]
    k = common["cfg"]["trend_integration"]["K"]

    t2_labeled = t2.build_labeled_features(common["features"], common["labels_h5"])
    t2_scores = t2.walk_forward_test_scores(
        t2_labeled, T2_FROZEN_CONFIG, common["train_start_date"], test_start, test_end, calendar,
        refit_frequency, common["gbdt_common"], common["seed"], common["fraction"],
    )
    t2_scores = t2_scores.copy()
    t2_scores["p_diff"] = t2_scores[1] - t2_scores[-1]

    a1_selected = two_stage_select(l1_test_scores, t2_scores[["date", "symbol", "p_diff"]], "p_diff", top_m, k)
    rebalance_dates = build_refit_schedule(test_start, test_end, calendar, frequency=H)
    r_net = preselected_portfolio_returns(a1_selected, common["labels_h5"], rebalance_dates, k, common["c_buy"], common["c_sell"])
    return r_net["r_net"].to_numpy()


def run_pool(name: str, period_returns: dict[str, np.ndarray], daily_series: dict[str, pd.Series], chosen_key: str, test_returns: np.ndarray, pbo_s: int) -> dict:
    sr_test = sharpe_raw(test_returns)
    t_obs = len(test_returns)
    gamma3 = float(skew(test_returns, bias=False))
    gamma4 = float(kurtosis(test_returns, fisher=False, bias=False))

    n_info = estimate_effective_n(period_returns)
    sr_ann_by_trial = {k: sharpe_annualized(v, H) for k, v in period_returns.items()}
    var_sr_ann = float(np.nanvar(list(sr_ann_by_trial.values()), ddof=1))

    dsr_out = deflated_sharpe_ratio(
        sr_test=sr_test, n_trials=n_info["n_hat"], var_sr_trials_annualized=var_sr_ann,
        ann_factor=ANN_FACTOR, t_obs=t_obs, skew=gamma3, kurt=gamma4,
    )

    daily_matrix = pd.concat(daily_series, axis=1).dropna(how="any")
    pbo_out = cscv_pbo(daily_matrix, s=pbo_s)

    return {
        "pool": name, "chosen": chosen_key, "m_trials_log": n_info["m"], "rho_hat": n_info["rho_hat"],
        "n_hat": n_info["n_hat"], "var_sharpe_annualized": var_sr_ann, "sr_test_raw": sr_test,
        "sr_test_annualized": sr_test * ANN_FACTOR, "gamma3_skew": gamma3, "gamma4_kurt": gamma4,
        **{f"dsr_{k2}": v for k2, v in dsr_out.items() if k2 not in ("sr_test",)},
        **{f"pbo_{k2}": v for k2, v in pbo_out.items()},
    }


def main() -> None:
    common = load_common()
    calendar = common["validation_dates"]
    rebalance_dates_phase0 = calendar[0::H]
    pbo_s = common["cfg"]["hypotheses"]["pbo_S"]

    print("=== Nhom l1_pool (N=21, h=5) ===")
    baseline_scores = build_baseline_scores_validation(common)
    print(f"  baseline: {len(baseline_scores)} cau hinh")
    reg_scores = build_regression_scores_validation(common)
    print(f"  regression: {len(reg_scores)} cau hinh")
    l1_scores, l1_frozen_scored = build_l1_scores_validation(common)
    print(f"  L1: {len(l1_scores)} cau hinh")
    l1_pool_scores = {**baseline_scores, **reg_scores, **l1_scores}
    assert len(l1_pool_scores) == 21, f"ky vong 21 cau hinh, co {len(l1_pool_scores)}"

    l1_period, l1_daily = _pool_h_period_and_daily(l1_pool_scores, common, rebalance_dates_phase0, calendar, is_preselected=False)

    l1_test_returns, l1_test_scores = build_l1_test_series(common)
    print(f"  L1 test: {len(l1_test_returns)} ky, Sharpe nam hoa={sharpe_annualized(l1_test_returns, H):.4f}")

    l1_pool_result = run_pool("l1_pool", l1_period, l1_daily, "L1_num_leaves=15_min_data_in_leaf=200", l1_test_returns, pbo_s)

    print("\n=== Nhom a1_pool (N=4, h=5) ===")
    selections = build_a0_a3_selections_validation(common, l1_frozen_scored)
    k_trend = common["cfg"]["trend_integration"]["K"]

    a1_pool_scores = {"A0": selections["A0"], "A1": selections["A1"], "A2": selections["A2"]}
    a1_period, a1_daily = _pool_h_period_and_daily(a1_pool_scores, common, rebalance_dates_phase0, calendar, is_preselected=True, k_for_preselected=k_trend)

    print(f"  A3: trung binh {len(selections['A3_replicates'])} lan lap")
    a3_period_list, a3_daily_list = [], []
    for rep_sel in selections["A3_replicates"]:
        r_net_df = preselected_trial(rep_sel, common, rebalance_dates_phase0, k_trend)
        a3_period_list.append(r_net_df["r_net"].to_numpy())
        build_fn = lambda rb, s=rep_sel: preselected_trial(s, common, rb, k_trend)
        a3_daily_list.append(build_staggered_daily_series(build_fn, calendar, H))
    a1_period["A3"] = np.mean(np.vstack(a3_period_list), axis=0)
    a1_daily["A3"] = pd.concat(a3_daily_list, axis=1).mean(axis=1)
    print(f"  A3: Sharpe tho trung binh={sharpe_raw(a1_period['A3']):.4f}")

    a1_test_returns = build_a1_test_series(common, l1_test_scores)
    print(f"  A1 test: {len(a1_test_returns)} ky, Sharpe nam hoa={sharpe_annualized(a1_test_returns, H):.4f} (doi chieu system_results_s8: -0.9013)")

    a1_pool_result = run_pool("a1_pool", a1_period, a1_daily, "A1", a1_test_returns, pbo_s)

    results_df = pd.DataFrame([l1_pool_result, a1_pool_result])
    out_path = common["data_dir"] / "dsr_pbo_results_4_14.csv"
    results_df.to_csv(out_path, index=False)
    print("\n" + results_df.to_string(index=False))
    print(f"Da luu {out_path}")

    trial_rows = []
    for name, r in l1_period.items():
        trial_rows.append({"pool": "l1_pool", "config": name, "sharpe_raw_validation": sharpe_raw(r), "sharpe_annualized_validation": sharpe_annualized(r, H), "n_ky": len(r)})
    for name, r in a1_period.items():
        trial_rows.append({"pool": "a1_pool", "config": name, "sharpe_raw_validation": sharpe_raw(r), "sharpe_annualized_validation": sharpe_annualized(r, H), "n_ky": len(r)})
    trials_df = pd.DataFrame(trial_rows)
    trials_path = common["data_dir"] / "dsr_pbo_trials_4_14.csv"
    trials_df.to_csv(trials_path, index=False)
    print(f"Da luu {trials_path}")


if __name__ == "__main__":
    main()
