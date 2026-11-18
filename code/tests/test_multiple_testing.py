"""Test cho multiple_testing.py (bước 4.1): hiệu chỉnh Holm step-down cho kiểm định đồng thời."""
from __future__ import annotations

from vnrank.evaluation.multiple_testing import holm_step_down


def test_holm_step_down_known_example():
    # p gốc không sắp xếp; 0.03 > 0.05/3 nên dừng ở đó, không bác bỏ 0.04 và 0.20
    assert holm_step_down([0.01, 0.04, 0.03, 0.20], alpha=0.05) == [
        True,
        False,
        False,
        False,
    ]


def test_holm_step_down_all_rejected():
    assert holm_step_down([0.001, 0.002, 0.0005], alpha=0.05) == [True, True, True]


def test_holm_step_down_none_rejected():
    assert holm_step_down([0.5, 0.4, 0.6], alpha=0.05) == [False, False, False]


def test_holm_step_down_single_p_value():
    assert holm_step_down([0.01], alpha=0.05) == [True]
    assert holm_step_down([0.10], alpha=0.05) == [False]