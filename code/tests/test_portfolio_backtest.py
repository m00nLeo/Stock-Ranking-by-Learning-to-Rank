"""Test cho mô phỏng danh mục top-K (bước 4.8): chọn top-K, turnover, chi phí, Sharpe năm hóa. Dữ
liệu tổng hợp nhỏ tự dựng, không đọc dữ liệu thật."""
from __future__ import annotations

import numpy as np
import pandas as pd

from vnrank.costs.portfolio_backtest import (
    annualized_net_return,
    build_rebalance_schedule,
    max_drawdown,
    preselected_portfolio_returns,
    select_top_k,
    sharpe_annualized,
    top_k_portfolio_returns,
)


# --- build_rebalance_schedule: cứ h phiên một lần, không chồng lấn -----------------------------


def test_build_rebalance_schedule_every_h_sessions():
    calendar = pd.bdate_range("2024-01-01", periods=30).tolist()
    test_start, test_end = calendar[0], calendar[-1]
    schedule = build_rebalance_schedule(test_start, test_end, calendar, h=5)
    assert schedule[0] == calendar[0]
    assert schedule[1] == calendar[5]
    assert len(schedule) == 6  # 30 ngay / 5


# --- select_top_k: chọn đúng K mã điểm cao nhất, xử lý khi thiếu ứng viên -----------------------


def test_select_top_k_picks_highest_scores():
    candidates = pd.DataFrame({"symbol": ["A", "B", "C", "D"], "score": [0.1, 0.5, 0.3, 0.9]})
    assert select_top_k(candidates, 2) == ["D", "B"]


def test_select_top_k_returns_all_when_fewer_than_k_candidates():
    candidates = pd.DataFrame({"symbol": ["A", "B"], "score": [0.1, 0.5]})
    assert set(select_top_k(candidates, 5)) == {"A", "B"}


# --- top_k_portfolio_returns: turnover và chi phí đúng công thức -------------------------------


def test_top_k_portfolio_returns_first_period_is_all_buy():
    dates = pd.bdate_range("2024-01-01", periods=1)
    scored = pd.DataFrame({"date": [dates[0]] * 3, "symbol": ["A", "B", "C"], "score": [0.9, 0.5, 0.1]})
    labels_h = pd.DataFrame({"date": [dates[0]] * 3, "symbol": ["A", "B", "C"], "r": [0.02, 0.04, -0.01]})
    out = top_k_portfolio_returns(scored, labels_h, [dates[0]], k=2, c_buy=0.0025, c_sell=0.0035)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["turnover_buy"] == 1.0
    assert row["turnover_sell"] == 0.0
    expected_gross = (0.02 + 0.04) / 2  # A, B co diem cao nhat
    expected_cost = 0.0025 * 1.0
    assert row["r_net"] == expected_gross - expected_cost


def test_top_k_portfolio_returns_computes_turnover_between_periods():
    dates = pd.bdate_range("2024-01-01", periods=2)
    # Ky 1: chon A,B (diem cao nhat). Ky 2: diem doi, chon B,C -> A roi danh muc, C moi vao.
    scored = pd.DataFrame(
        {
            "date": [dates[0]] * 3 + [dates[1]] * 3,
            "symbol": ["A", "B", "C"] * 2,
            "score": [0.9, 0.8, 0.1, 0.1, 0.8, 0.9],
        }
    )
    labels_h = pd.DataFrame(
        {
            "date": [dates[0]] * 3 + [dates[1]] * 3,
            "symbol": ["A", "B", "C"] * 2,
            "r": [0.01, 0.02, -0.01, 0.00, 0.03, 0.05],
        }
    )
    out = top_k_portfolio_returns(scored, labels_h, list(dates), k=2, c_buy=0.0025, c_sell=0.0035)
    row2 = out.iloc[1]
    # ky 2 chon B, C (diem cao nhat) -> A roi (ban), C moi vao (mua), B giu nguyen.
    assert row2["turnover_buy"] == 0.5  # 1 ma moi / K=2
    assert row2["turnover_sell"] == 0.5  # 1 ma roi / K=2
    expected_gross = (0.03 + 0.05) / 2
    expected_cost = 0.0025 * 0.5 + 0.0035 * 0.5
    assert row2["r_net"] == expected_gross - expected_cost


def test_top_k_portfolio_returns_excludes_symbols_without_valid_r():
    # Ma co diem cao nhat (Z) khong co dong r (khoa tran o phien vao, 3.6) -> phai bi loai.
    dates = pd.bdate_range("2024-01-01", periods=1)
    scored = pd.DataFrame({"date": [dates[0]] * 3, "symbol": ["Z", "A", "B"], "score": [0.99, 0.5, 0.4]})
    labels_h = pd.DataFrame({"date": [dates[0]] * 2, "symbol": ["A", "B"], "r": [0.02, 0.01]})
    out = top_k_portfolio_returns(scored, labels_h, [dates[0]], k=2, c_buy=0.0, c_sell=0.0)
    assert len(out) == 1
    assert out.iloc[0]["n_held"] == 2
    assert out.iloc[0]["r_net"] == (0.02 + 0.01) / 2


# --- sharpe_annualized: đúng công thức mean/std*sqrt(250/h) -------------------------------------


def test_sharpe_annualized_matches_formula():
    r = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
    h = 5
    expected = r.mean() / r.std(ddof=1) * np.sqrt(250 / h)
    assert sharpe_annualized(r, h) == expected


def test_sharpe_annualized_nan_for_too_few_observations():
    assert np.isnan(sharpe_annualized(np.array([0.01]), h=5))


def test_sharpe_annualized_nan_for_zero_variance():
    assert np.isnan(sharpe_annualized(np.array([0.01, 0.01, 0.01]), h=5))


# --- annualized_net_return: gộp lãi kép rồi quy đổi 250/(h*n) -----------------------------------


def test_annualized_net_return_matches_compounded_formula():
    r = np.array([0.02, -0.01, 0.015])
    h = 5
    growth = np.prod(1.0 + r)
    expected = growth ** (250.0 / (h * len(r))) - 1.0
    assert annualized_net_return(r, h) == expected


def test_annualized_net_return_nan_for_empty():
    assert np.isnan(annualized_net_return(np.array([]), h=5))


def test_annualized_net_return_single_period_matches_itself_annualized():
    # Mot ky duy nhat h=5 phien -> nam hoa don gian bang (1+r)^(250/5) - 1.
    r = np.array([0.1])
    out = annualized_net_return(r, h=5)
    assert out == (1.1 ** 50) - 1.0


# --- max_drawdown: sut giam lon nhat tu dinh tren duong von luy ke ------------------------------


def test_max_drawdown_zero_for_monotonic_gains():
    r = np.array([0.01, 0.02, 0.03])
    assert max_drawdown(r) == 0.0


def test_max_drawdown_matches_manual_peak_to_trough():
    # Von: 1 -> 1.10 (dinh) -> 0.88 (day) -> 0.968. Sut giam = (1.10-0.88)/1.10.
    r = np.array([0.10, -0.20, 0.10])
    expected = (1.10 - 1.10 * 0.80) / 1.10
    assert abs(max_drawdown(r) - expected) < 1e-12


def test_max_drawdown_is_nonnegative():
    r = np.array([0.05, 0.03, -0.1, 0.02])
    assert max_drawdown(r) >= 0.0


# --- preselected_portfolio_returns: nhu top_k_portfolio_returns nhung danh sach da chon san -----


def test_preselected_portfolio_returns_matches_top_k_for_equivalent_selection():
    dates = pd.bdate_range("2024-01-01", periods=2)
    scored = pd.DataFrame(
        {
            "date": [dates[0]] * 3 + [dates[1]] * 3,
            "symbol": ["A", "B", "C"] * 2,
            "score": [0.9, 0.8, 0.1, 0.1, 0.8, 0.9],
        }
    )
    labels_h = pd.DataFrame(
        {
            "date": [dates[0]] * 3 + [dates[1]] * 3,
            "symbol": ["A", "B", "C"] * 2,
            "r": [0.01, 0.02, -0.01, 0.00, 0.03, 0.05],
        }
    )
    via_top_k = top_k_portfolio_returns(scored, labels_h, list(dates), k=2, c_buy=0.0025, c_sell=0.0035)

    # Danh sach da "chon san" giong het top-2 theo score moi ngay (mo phong A1 da qua hai giai doan).
    preselected = pd.concat(
        [
            scored[scored["date"] == dates[0]].nlargest(2, "score")[["date", "symbol"]],
            scored[scored["date"] == dates[1]].nlargest(2, "score")[["date", "symbol"]],
        ],
        ignore_index=True,
    )
    via_preselected = preselected_portfolio_returns(preselected, labels_h, list(dates), k=2, c_buy=0.0025, c_sell=0.0035)

    pd.testing.assert_frame_equal(via_top_k.reset_index(drop=True), via_preselected.reset_index(drop=True))


def test_preselected_portfolio_returns_turnover_uses_nominal_k_not_actual_count():
    # Ky 1: chon {A, B} (du k=2). Ky 2: danh sach da chon san CHI con {A} (it hon k=2 danh nghia,
    # vd top-M nho hon k) -> B roi danh muc. turnover_sell phai chia cho k=2 DANH NGHIA (0,5), khong
    # phai chia cho so ma THUC NHAN duoc o ky 2 (1 ma, se ra 1,0) -> nhat quan voi top_k_portfolio_returns.
    dates = pd.bdate_range("2024-01-01", periods=2)
    preselected = pd.DataFrame(
        {"date": [dates[0], dates[0], dates[1]], "symbol": ["A", "B", "A"]}
    )
    labels_h = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1]],
            "symbol": ["A", "B", "A"],
            "r": [0.01, 0.02, 0.015],
        }
    )
    out = preselected_portfolio_returns(preselected, labels_h, list(dates), k=2, c_buy=0.0, c_sell=0.0)
    assert out.iloc[1]["turnover_sell"] == 0.5  # 1 ma roi / k=2 danh nghia, khong phai /1
    assert out.iloc[1]["turnover_buy"] == 0.0
