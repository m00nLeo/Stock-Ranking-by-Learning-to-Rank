"""Test cho split_protocol.py (bước 3.7). Dữ liệu tổng hợp nhỏ, không gọi mạng thật."""
from __future__ import annotations

import pandas as pd
import pytest

from vnrank.splits.split_protocol import (
    assign_split,
    build_refit_schedule,
    determine_test_end,
    purge_by_realization,
    refit_training_window,
)


def test_assign_split_boundaries():
    assert assign_split("2022-12-31") is None  # trước mẫu có nhãn
    assert assign_split("2023-01-01") == "train"
    assert assign_split("2024-06-30") == "train"
    assert assign_split("2024-07-01") == "validation"
    assert assign_split("2025-03-31") == "validation"
    assert assign_split("2025-04-01") == "test"


def test_assign_split_open_ended_test_without_test_end():
    assert assign_split("2026-11-03") == "test"  # chưa truyền test_end -> test mở, không chặn trên


def test_assign_split_respects_test_end():
    assert assign_split("2026-09-30", test_end=pd.Timestamp("2026-09-29")) is None
    assert assign_split("2026-09-29", test_end=pd.Timestamp("2026-09-29")) == "test"


def test_determine_test_end_uses_max_date_with_horizon():
    labels = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-20", "2026-09-25", "2026-09-29", "2026-09-29"]),
            "h": [21, 21, 21, 5],
        }
    )
    assert determine_test_end(labels, horizon=21) == pd.Timestamp("2026-09-29")


def test_purge_by_realization_keeps_only_realized_before_tau():
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "t1": pd.to_datetime(["2024-06-01", "2024-06-30", "2024-07-01", None]),
        }
    )
    out = purge_by_realization(df, tau="2024-06-30", realization_col="t1")
    assert set(out["symbol"]) == {"A", "B"}


def test_refit_training_window_expands_and_purges():
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "date": pd.to_datetime(["2022-12-01", "2023-01-01", "2024-07-01", "2025-05-01"]),
            "t1": pd.to_datetime(["2022-12-10", "2023-01-10", "2024-07-10", "2025-05-10"]),
        }
    )
    out = refit_training_window(df, "t1", train_start="2023-01-01", refit_date="2024-08-01")
    # A bị loại vì date < train_start (trước mẫu có nhãn); D bị loại vì t1 sau refit_date
    assert set(out["symbol"]) == {"B", "C"}


def test_build_refit_schedule_every_quarter():
    calendar = list(pd.date_range("2025-04-01", "2026-09-30", freq="D"))
    schedule = build_refit_schedule("2025-04-01", "2026-09-30", calendar, frequency=63)
    assert schedule[0] == pd.Timestamp("2025-04-01")
    assert len(schedule) >= 5
    # khoảng cách giữa các mốc đúng 63 phiên (lịch ở đây là ngày liên tục nên đúng bằng ngày)
    assert (schedule[1] - schedule[0]).days == 63
