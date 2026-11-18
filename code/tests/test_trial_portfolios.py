"""Test cho các hàm THUẦN TÚY (không fit mô hình, không đọc dữ liệu thật) của
`trial_portfolios.py`, bước 4.14: lịch tái cân bằng so le và cách trải lợi suất kỳ ra ngày dùng
cho chuỗi NGÀY của CSCV/PBO. Các hàm cần fit mô hình/đọc CSV thật (`build_*_scores_validation`,
`build_a0_a3_selections_validation`) được xác nhận bằng cách CHẠY THẬT một lần
(`run_dsr_pbo_4_14.py`, `decision_log.md` D118), không lặp lại ở đây (đúng quy ước dự án — xem
`test_experiment_log.py`: test tổng hợp dùng dữ liệu tự dựng, không đọc dữ liệu thật)."""
from __future__ import annotations

import pandas as pd
import pytest

from vnrank.evaluation.trial_portfolios import (
    build_staggered_daily_series,
    expand_period_returns_to_daily,
    staggered_rebalance_schedules,
)


def _calendar(n: int) -> list[pd.Timestamp]:
    return list(pd.bdate_range("2024-01-01", periods=n))


# --- staggered_rebalance_schedules -------------------------------------------------------------


def test_staggered_rebalance_schedules_returns_h_schedules_each_offset_by_one():
    cal = _calendar(20)
    schedules = staggered_rebalance_schedules(cal, h=5)
    assert len(schedules) == 5
    for offset, sched in enumerate(schedules):
        assert sched[0] == cal[offset]
        assert sched == cal[offset::5]


def test_staggered_rebalance_schedules_cover_every_session_across_offsets():
    cal = _calendar(23)
    schedules = staggered_rebalance_schedules(cal, h=5)
    covered = sorted(set().union(*[set(s) for s in schedules]))
    assert covered == sorted(cal)


# --- expand_period_returns_to_daily ------------------------------------------------------------


def test_expand_period_returns_to_daily_compounds_back_to_period_return():
    cal = _calendar(10)
    r_net_df = pd.DataFrame({"date": [cal[0], cal[5]], "r_net": [0.05, -0.02]})
    daily = expand_period_returns_to_daily(r_net_df, cal)
    assert len(daily) == 10
    first_period = daily.loc[cal[0]:cal[4]]
    growth = (1.0 + first_period).prod()
    assert growth == pytest.approx(1.05)
    second_period = daily.loc[cal[5]:cal[9]]
    assert (1.0 + second_period).prod() == pytest.approx(0.98)


def test_expand_period_returns_to_daily_constant_rate_within_period():
    cal = _calendar(5)
    r_net_df = pd.DataFrame({"date": [cal[0]], "r_net": [0.1]})
    daily = expand_period_returns_to_daily(r_net_df, cal)
    assert daily.nunique() == 1
    assert daily.iloc[0] == pytest.approx((1.1) ** (1 / 5) - 1)


def test_expand_period_returns_to_daily_ignores_period_start_outside_calendar():
    cal = _calendar(5)
    r_net_df = pd.DataFrame({"date": [cal[0], pd.Timestamp("2099-01-01")], "r_net": [0.1, 0.5]})
    daily = expand_period_returns_to_daily(r_net_df, cal)
    assert len(daily) == 5


# --- build_staggered_daily_series --------------------------------------------------------------


def test_build_staggered_daily_series_averages_across_offsets():
    cal = _calendar(20)

    def build_r_net_fn(rebalance_dates: list) -> pd.DataFrame:
        # loi suat ky co dinh 0,02 cho moi ky bat ke lich nao - chuoi ngay ket qua phai la mot
        # hang so duy nhat (moi lich so le deu cho cung mot ty le ngay nhu nhau).
        return pd.DataFrame({"date": rebalance_dates, "r_net": [0.02] * len(rebalance_dates)})

    daily = build_staggered_daily_series(build_r_net_fn, cal, h=5)
    assert len(daily) == 20
    # 4 phien cuoi la vung bien (ky cuoi cua cac lich lech bi cat ngan boi do dai lich tu dung
    # thu nghiem, 20 khong chia het dep cho ca 5 do lech) - chi kiem phan noi dia khong bi anh
    # huong, noi ca 5 do lech deu cho cung mot ty le ngay.
    assert daily.iloc[:16].to_numpy() == pytest.approx((1.02) ** (1 / 5) - 1, abs=1e-9)


def test_build_staggered_daily_series_empty_when_too_few_rebalance_dates():
    cal = _calendar(3)

    def build_r_net_fn(rebalance_dates: list) -> pd.DataFrame:
        return pd.DataFrame({"date": rebalance_dates, "r_net": [0.0] * len(rebalance_dates)})

    daily = build_staggered_daily_series(build_r_net_fn, cal, h=5)
    assert daily.empty
