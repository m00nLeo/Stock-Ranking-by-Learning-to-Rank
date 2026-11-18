"""Test cho bảng kết quả `ket_qua()` của hệ thống minh họa (bước S8, kế hoạch D109, phát hiện
D113). Dữ liệu tổng hợp nhỏ tự dựng (3 mã, 6 ngày), không đọc dữ liệu/giá thật của dự án — hàm đọc
giá thật (`momentum_test_scores`) nhận kết quả như một tham số đã tính sẵn (xem docstring
`system_results.py`), đúng nguyên tắc cô lập của `test_artifact_loader.py`/`test_sensitivity_analysis.py`."""
from __future__ import annotations

import pandas as pd

from vnrank.evaluation.sensitivity_analysis import rank_ic_mean_under_filter
from vnrank.evaluation.system_results import (
    MUC_CHI_PHI_HOP_LE,
    build_system_results_table,
)
from vnrank.models.trend_integration import precision_at_k_for_selection, two_stage_select

DATES = pd.bdate_range("2024-01-01", periods=6)
SYMBOLS = ["A", "B", "C"]

CONFIG = {
    # K khong bao gio vuot top_M (dung nhu san xuat: top_M=20, K_sensitivity=[5,20]) de
    # precision_at_k_for_selection khong bi bo qua ngay nao (yeu cau len(selected) >= k).
    "trend_integration": {"K": 1, "K_sensitivity": [2], "top_M": 2},
    # Phai bao quanh C_RT_BASE=0,0060 (hang so co dinh cua "co so") de "thap"<"co so"<"cao" ve chi phi.
    "costs": {"c_rt_sensitivity": [0.0030, 0.0120]},
}


def _build_inputs():
    # L1: diem co dinh A > B > C moi ngay (thu tu khong doi) -> top_M=2 luon la {A, B}.
    l1_test = pd.DataFrame(
        [{"date": d, "symbol": s, "score": v} for d in DATES for s, v in [("A", 0.9), ("B", 0.5), ("C", 0.1)]]
    )
    # T2 p_diff: C > B > A moi ngay -> trong top_M={A,B}, B thang A.
    t2_a1 = pd.DataFrame(
        [{"date": d, "symbol": s, "p_diff": v} for d in DATES for s, v in [("C", 0.8), ("B", 0.3), ("A", -0.2)]]
    )
    # Momentum (B2): C > B > A moi ngay (khac L1) -> top-k momentum theo dung thu tu nay.
    momentum_scores = pd.DataFrame(
        [{"date": d, "symbol": s, "score": v} for d in DATES for s, v in [("C", 3.0), ("B", 2.0), ("A", 1.0)]]
    )

    r_by_symbol_day = {
        "A": [0.01, 0.02, -0.01, 0.015, 0.005, 0.03],
        "B": [0.00, -0.02, 0.01, -0.005, 0.02, -0.01],
        "C": [0.03, 0.01, 0.02, 0.025, -0.01, 0.015],
    }
    r_excess_by_symbol = {"A": 0.01, "B": -0.01, "C": 0.02}  # co dinh, khong doi theo ngay
    labels_h5 = pd.DataFrame(
        [
            {
                "date": DATES[i], "symbol": s, "h": 5,
                "r": r_by_symbol_day[s][i],
                "r_excess": r_excess_by_symbol[s],
            }
            for i in range(len(DATES)) for s in SYMBOLS
        ]
    )

    universe = pd.DataFrame([{"date": d, "symbol": s} for d in DATES for s in SYMBOLS])
    splits = pd.DataFrame({"date": DATES, "split": ["test"] * len(DATES)})
    rebalance_dates = list(DATES)

    return l1_test, t2_a1, labels_h5, universe, splits, momentum_scores, rebalance_dates


def _table():
    l1_test, t2_a1, labels_h5, universe, splits, momentum_scores, rebalance_dates = _build_inputs()
    return build_system_results_table(
        l1_test, t2_a1, labels_h5, universe, splits, momentum_scores, rebalance_dates, CONFIG
    ), (l1_test, t2_a1, labels_h5, universe, splits, momentum_scores, rebalance_dates)


# --- Hình dạng bảng: 3 cấu hình x 2 K x 3 mức chi phí = 18 dòng, đúng cột hợp đồng ---------------


def test_table_shape_and_columns():
    table, _ = _table()
    assert len(table) == 18
    assert list(table.columns) == [
        "h", "K", "muc_chi_phi", "cau_hinh", "rank_ic", "precision_at_k",
        "loi_suat_nam_hoa_rong", "sharpe_rong", "mdd", "turnover",
    ]
    assert set(table["h"].unique()) == {5}
    assert set(table["cau_hinh"].value_counts()) == {6}  # moi cau hinh dung 6 dong (2K x 3 chi phi)
    assert set(table["K"].unique()) == {1, 2}
    assert set(table["muc_chi_phi"].unique()) == set(MUC_CHI_PHI_HOP_LE)


# --- B0: khong chon K cu the -> rank_ic=0, precision_at_k/turnover khong doi theo K/chi phi ------


def test_b0_rank_ic_is_zero_and_constant():
    table, _ = _table()
    b0 = table[table["cau_hinh"] == "B0"]
    assert (b0["rank_ic"] == 0.0).all()


def test_b0_precision_at_k_matches_whole_universe_fraction_and_is_k_independent():
    table, _ = _table()
    b0 = table[table["cau_hinh"] == "B0"]
    assert b0["precision_at_k"].nunique() == 1
    # A, C co r_excess duong; B am -> 2/3 moi ngay, hang ngay nhu nhau -> trung binh = 2/3.
    assert abs(b0["precision_at_k"].iloc[0] - 2 / 3) < 1e-9


def test_b0_turnover_zero_and_cost_independent():
    table, _ = _table()
    b0 = table[table["cau_hinh"] == "B0"]
    assert (b0["turnover"] == 0.0).all()
    # Khong ap dung chi phi -> ba muc chi phi cho CUNG mot sharpe_rong/mdd/loi_suat_nam_hoa_rong.
    for k in b0["K"].unique():
        rows = b0[b0["K"] == k]
        assert rows["sharpe_rong"].nunique() == 1
        assert rows["mdd"].nunique() == 1
        assert rows["loi_suat_nam_hoa_rong"].nunique() == 1


# --- A1/B2: rank_ic khong doi theo K/chi phi (thuoc tinh cua xep hang toan U_t) ------------------


def test_a1_and_b2_rank_ic_constant_across_k_and_cost_and_matches_direct_call():
    table, inputs = _table()
    l1_test, _, labels_h5, _, splits, momentum_scores, _ = inputs

    a1 = table[table["cau_hinh"] == "A1"]
    assert a1["rank_ic"].nunique() == 1
    expected_a1 = rank_ic_mean_under_filter(l1_test[["date", "symbol", "score"]], labels_h5, splits)
    assert abs(a1["rank_ic"].iloc[0] - expected_a1) < 1e-12

    b2 = table[table["cau_hinh"] == "B2"]
    assert b2["rank_ic"].nunique() == 1
    expected_b2 = rank_ic_mean_under_filter(momentum_scores, labels_h5, splits)
    assert abs(b2["rank_ic"].iloc[0] - expected_b2) < 1e-12


# --- A1 precision@K: khop dung hai giai doan (top_M=2 roi xep theo p_diff) -----------------------


def test_a1_precision_at_k_matches_two_stage_select_directly():
    table, inputs = _table()
    l1_test, t2_a1, labels_h5, _, _, _, _ = inputs

    # top_M=2 -> ung vien luon {A, B} (C bi loai vi L1 xep thap nhat). Trong {A,B}, p_diff: B > A.
    # k=1 -> {B} (r_excess am) -> 0,0. k=2 -> {A,B} (ca hai, A+ B-) -> 1/2.
    for k, expected_precision in [(1, 0.0), (2, 0.5)]:
        selected = two_stage_select(
            l1_test[["date", "symbol", "score"]], t2_a1[["date", "symbol", "p_diff"]], "p_diff", 2, k
        )
        prec = precision_at_k_for_selection(selected, labels_h5, k)
        row = table[(table["cau_hinh"] == "A1") & (table["K"] == k) & (table["muc_chi_phi"] == "cơ sở")].iloc[0]
        assert abs(row["precision_at_k"] - float(prec.mean())) < 1e-12
        assert abs(row["precision_at_k"] - expected_precision) < 1e-9


def test_b2_precision_at_k_matches_hand_derived_values():
    table, _ = _table()
    # Momentum: C > B > A. k=1 -> {C} (r_excess duong) -> 1,0. k=2 -> {C,B} (C+ B-) -> 1/2.
    for k, expected in [(1, 1.0), (2, 0.5)]:
        row = table[(table["cau_hinh"] == "B2") & (table["K"] == k) & (table["muc_chi_phi"] == "cơ sở")].iloc[0]
        assert abs(row["precision_at_k"] - expected) < 1e-9


# --- Chi phi cao hon -> loi suat nam hoa rong thap hon hoac bang (khong the cao hon) -------------


def test_higher_cost_level_never_increases_annualized_return_for_a1_and_b2():
    table, _ = _table()
    for cau_hinh in ("A1", "B2"):
        for k in table["K"].unique():
            sub = table[(table["cau_hinh"] == cau_hinh) & (table["K"] == k)]
            thap = sub[sub["muc_chi_phi"] == "thấp"]["loi_suat_nam_hoa_rong"].iloc[0]
            co_so = sub[sub["muc_chi_phi"] == "cơ sở"]["loi_suat_nam_hoa_rong"].iloc[0]
            cao = sub[sub["muc_chi_phi"] == "cao"]["loi_suat_nam_hoa_rong"].iloc[0]
            assert thap >= co_so >= cao


# --- Khong co NaN trong bang (moi to hop deu co du lieu tren ca 6 ngay) --------------------------


def test_no_nan_in_table():
    table, _ = _table()
    numeric_cols = ["rank_ic", "precision_at_k", "loi_suat_nam_hoa_rong", "sharpe_rong", "mdd", "turnover"]
    assert not table[numeric_cols].isna().any().any()
