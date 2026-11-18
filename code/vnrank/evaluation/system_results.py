"""Bảng kết quả cho `ket_qua()` của hệ thống minh họa (bước S8, kế hoạch D109, phát hiện D113;
triển khai 17/11/2026). Đúng sáu thước đo của `system_architecture.md` mục 4.4: `rank_ic`,
`precision_at_k`, `loi_suat_nam_hoa_rong`, `sharpe_rong`, `mdd`, `turnover` — cho ba cấu hình `A1`,
`B0`, `B2` (baseline tín hiệu tốt nhất, xác nhận bằng dữ liệu thật ở D109), trên lưới
$K \\in \\{5, 10, 20\\}$ x mức chi phí $\\in \\{$thấp, cơ sở, cao$\\}$.

**Hai điểm quyết định ở D109, CHỐT bởi người dùng (17/11/2026, trước khi viết module này):**
1. **B0 không chọn $K$ mã cụ thể** (đồng trọng số toàn $U_t$) — `precision_at_k` và `rank_ic` của
   B0 dùng ĐÚNG MỘT nguyên tắc: giá trị KỲ VỌNG của việc không có kỹ năng xếp hạng/chọn lọc nào,
   nhất quán với cách A3 (chọn ngẫu nhiên) đã làm ở 4.13 (D103):
   - `precision_at_k`(B0) = tỷ lệ TOÀN BỘ $U_t$ mỗi ngày có `r_excess` > 0, trung bình trên test —
     đúng bằng kỳ vọng toán học của việc chọn ngẫu nhiên $K$ mã bất kỳ trong $U_t$ (không hoàn lại,
     kỳ vọng phân phối siêu bội = tỷ lệ tổng thể).
   - `rank_ic`(B0) = 0,0 — B0 không có điểm số nên không có thứ tự để tương quan; một mô hình
     không có kỹ năng xếp hạng có Rank IC kỳ vọng đúng bằng 0. (Suy luận thêm trong lúc triển khai,
     CÙNG tinh thần quyết định precision_at_k ở trên — không phải lựa chọn tùy ý; nêu rõ để người
     dùng/hội đồng có thể phản biện riêng nếu không đồng ý.)
   - B0 không áp dụng chi phí/turnover: tái dùng ĐÚNG `baselines.equal_weight_portfolio_returns`
     (D109 chỉ định rõ hàm này) — hàm này vốn không có tham số chi phí, nên `turnover`(B0) = 0,0 và
     `loi_suat_nam_hoa_rong`/`sharpe_rong`/`mdd` của B0 tính trên lợi suất GỘP (trước chi phí), GIỐNG
     NHAU ở cả ba mức `muc_chi_phi` (không phải lỗi, là hệ quả tất yếu của việc dùng đúng hàm đã chỉ
     định — nêu rõ để không hiểu nhầm là thiếu sót).
2. **A1 chỉ thật ở $h=5$** (D107, 4.13) — `ket_qua(h=10 hoặc 21, ...)` báo lỗi toàn bộ, giống hệt
   `xep_hang`/`xu_huong` (xem `read_functions.ket_qua`, sửa kèm D113-S8) — module này CHỈ dựng bảng
   cho $h=5$ (27 dòng: 3 cấu hình × 3 $K$ × 3 mức chi phí).

$B2$ = momentum cửa sổ 21 phiên, skip=0 — baseline tín hiệu tốt nhất xác nhận bằng dữ liệu thật ở
D109 (Rank IC validation cao nhất trong B2-B4 ở cả ba $h$, cấu hình đóng băng
`frozen_configurations_4_7.csv`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vnrank.costs.portfolio_backtest import (
    annualized_net_return,
    max_drawdown,
    preselected_portfolio_returns,
    sharpe_annualized,
    top_k_portfolio_returns,
)
from vnrank.evaluation.harness import icir, precision_at_k_by_day, rank_ic_series
from vnrank.evaluation.sensitivity_analysis import rank_ic_mean_under_filter
from vnrank.models.baselines import equal_weight_portfolio_returns, momentum_score
from vnrank.models.trend_integration import precision_at_k_for_selection, two_stage_select

A1_HORIZON = 5  # A1 chỉ xây dựng/đánh giá ở h=5 (4.13, D107) — xem docstring đầu file
MUC_CHI_PHI_HOP_LE: tuple[str, ...] = ("thấp", "cơ sở", "cao")
B2_MOMENTUM_WINDOW = 21  # frozen_configurations_4_7.csv, B2_momentum h=5: (21, 0)
B2_MOMENTUM_SKIP = 0


def momentum_test_scores() -> pd.DataFrame:
    """Điểm B2 (momentum, cấu hình đóng băng window=21/skip=0) cho mọi mã niêm yết, mọi ngày có
    giá — đọc giá THẬT từ `PRICE_CACHE_DIR` của dự án (không nhận `data_dir` vì cache giá nằm ở
    `data/raw/prices`, khác `data/processed`). Tái dùng `baselines.momentum_score` — không viết lại
    công thức. Việc hạn chế về $U_t$ mỗi ngày diễn ra tự nhiên khi ghép với `labels_h` (inner join),
    đúng cách 4.2 (`evaluate_on_test`) đã làm. TÁCH RIÊNG khỏi `build_system_results_table` (nhận
    kết quả hàm này như một tham số, giống `l1_test`/`t2_a1`) để hàm đó kiểm được bằng dữ liệu tổng
    hợp nhỏ, không phải đọc giá thật của dự án (cùng nguyên tắc `test_artifact_loader.py`)."""
    from vnrank.universe.build_universe import load_listed_stocks, load_price_panel

    stocks = load_listed_stocks()
    symbols = sorted(stocks["symbol"].unique())
    panel = load_price_panel(symbols)
    panel = panel[panel["symbol"].isin(symbols)]

    rows = []
    for symbol, group in panel.sort_values(["symbol", "date"]).groupby("symbol"):
        g = group.reset_index(drop=True)
        s = momentum_score(g["close"], B2_MOMENTUM_WINDOW, B2_MOMENTUM_SKIP)
        rows.append(pd.DataFrame({"date": g["date"], "symbol": symbol, "score": s.to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def _costs_for_level(muc_chi_phi: str, config: dict) -> tuple[float, float]:
    from vnrank.evaluation.sensitivity_analysis import C_BUY_BASE, C_SELL_BASE, scaled_costs

    if muc_chi_phi == "cơ sở":
        return C_BUY_BASE, C_SELL_BASE
    c_rt_low, c_rt_high = config["costs"]["c_rt_sensitivity"]
    return scaled_costs(c_rt_low if muc_chi_phi == "thấp" else c_rt_high)


def _b0_precision_at_k(labels_h5: pd.DataFrame, universe: pd.DataFrame, splits: pd.DataFrame) -> float:
    """precision@K cho B0 — quyết định người dùng (S8, 17/11/2026): tỷ lệ TOÀN BỘ $U_t$ mỗi ngày
    có `r_excess` > 0 (xem quyết định 1 ở docstring đầu file)."""
    merged = universe.merge(labels_h5[["date", "symbol", "r_excess"]], on=["date", "symbol"], how="inner")
    merged = merged.merge(splits, on="date", how="inner")
    test = merged[merged["split"] == "test"].dropna(subset=["r_excess"])
    if test.empty:
        return float("nan")
    by_day = test.groupby("date")["r_excess"].apply(lambda s: float((s > 0).mean()))
    return float(by_day.mean())


def _b0_returns_at_rebalance(
    labels_h5: pd.DataFrame, universe: pd.DataFrame, rebalance_dates: list
) -> list[float]:
    eq = equal_weight_portfolio_returns(labels_h5, universe, A1_HORIZON)
    return [float(eq[d]) for d in rebalance_dates if d in eq.index and eq[d] == eq[d]]


def build_system_results_table(
    l1_test: pd.DataFrame,
    t2_a1: pd.DataFrame,
    labels_h5: pd.DataFrame,
    universe: pd.DataFrame,
    splits: pd.DataFrame,
    momentum_scores: pd.DataFrame,
    rebalance_dates: list,
    config: dict,
) -> pd.DataFrame:
    """Dựng bảng 27 dòng (3 cấu hình × $K \\in \\{5,10,20\\}$ × mức chi phí) cho `ket_qua(h=5, ...)`.

    `l1_test`: date, symbol, score (điểm L1 test, $h=5$ — TÁI DÙNG từ nơi gọi, không tính lại).
    `t2_a1`: date, symbol, p_diff ($\\hat p^{+}-\\hat p^{-}$ của T2 test, $h=5$ — TÁI DÙNG tương tự).
    `labels_h5`: nhãn $h=5$ (cột `r`, `r_excess`). `universe`: `universe_daily.csv`. `momentum_scores`:
    date, symbol, score — điểm B2 (TÁI DÙNG, xem `momentum_test_scores`; nhận làm tham số thay vì tự
    đọc giá thật bên trong, để hàm này kiểm được bằng dữ liệu tổng hợp nhỏ)."""
    k_hop_le = sorted({config["trend_integration"]["K"], *config["trend_integration"]["K_sensitivity"]})
    top_m = config["trend_integration"]["top_M"]

    rank_ic_a1 = rank_ic_mean_under_filter(l1_test[["date", "symbol", "score"]], labels_h5, splits)
    rank_ic_b2 = rank_ic_mean_under_filter(momentum_scores, labels_h5, splits)
    rank_ic_b0 = 0.0  # xem quyết định 1, docstring đầu file

    prec_b0 = _b0_precision_at_k(labels_h5, universe, splits)
    b0_returns = _b0_returns_at_rebalance(labels_h5, universe, rebalance_dates)

    rows = []
    for k in k_hop_le:
        a1_selected = two_stage_select(
            l1_test[["date", "symbol", "score"]], t2_a1[["date", "symbol", "p_diff"]], "p_diff", top_m, k
        )
        prec_a1 = precision_at_k_for_selection(a1_selected, labels_h5, k)
        prec_a1_mean = float(prec_a1.mean()) if len(prec_a1) else float("nan")

        merged_b2 = momentum_scores.merge(labels_h5[["date", "symbol", "r_excess"]], on=["date", "symbol"], how="inner")
        merged_b2 = merged_b2.merge(splits, on="date", how="inner")
        test_b2 = merged_b2[merged_b2["split"] == "test"].dropna(subset=["r_excess"])
        prec_b2_series = precision_at_k_by_day(test_b2["date"], test_b2["score"].to_numpy(), test_b2["r_excess"].to_numpy(), k)
        prec_b2_mean = float(prec_b2_series.mean()) if len(prec_b2_series) else float("nan")

        for muc_chi_phi in MUC_CHI_PHI_HOP_LE:
            c_buy, c_sell = _costs_for_level(muc_chi_phi, config)

            ret_a1 = preselected_portfolio_returns(a1_selected, labels_h5, rebalance_dates, k, c_buy, c_sell)
            ret_b2 = top_k_portfolio_returns(momentum_scores, labels_h5, rebalance_dates, k, c_buy, c_sell)

            rows.append({
                "h": A1_HORIZON, "K": k, "muc_chi_phi": muc_chi_phi, "cau_hinh": "A1",
                "rank_ic": rank_ic_a1, "precision_at_k": prec_a1_mean,
                "loi_suat_nam_hoa_rong": annualized_net_return(ret_a1["r_net"].to_numpy(), A1_HORIZON),
                "sharpe_rong": sharpe_annualized(ret_a1["r_net"].to_numpy(), A1_HORIZON),
                "mdd": max_drawdown(ret_a1["r_net"].to_numpy()),
                "turnover": float((ret_a1["turnover_buy"] + ret_a1["turnover_sell"]).mean()) if len(ret_a1) else float("nan"),
            })
            rows.append({
                "h": A1_HORIZON, "K": k, "muc_chi_phi": muc_chi_phi, "cau_hinh": "B0",
                "rank_ic": rank_ic_b0, "precision_at_k": prec_b0,
                "loi_suat_nam_hoa_rong": annualized_net_return(b0_returns, A1_HORIZON),
                "sharpe_rong": sharpe_annualized(b0_returns, A1_HORIZON),
                "mdd": max_drawdown(b0_returns),
                "turnover": 0.0,
            })
            rows.append({
                "h": A1_HORIZON, "K": k, "muc_chi_phi": muc_chi_phi, "cau_hinh": "B2",
                "rank_ic": rank_ic_b2, "precision_at_k": prec_b2_mean,
                "loi_suat_nam_hoa_rong": annualized_net_return(ret_b2["r_net"].to_numpy(), A1_HORIZON),
                "sharpe_rong": sharpe_annualized(ret_b2["r_net"].to_numpy(), A1_HORIZON),
                "mdd": max_drawdown(ret_b2["r_net"].to_numpy()),
                "turnover": float((ret_b2["turnover_buy"] + ret_b2["turnover_sell"]).mean()) if len(ret_b2) else float("nan"),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import yaml

    from vnrank.costs.portfolio_backtest import build_rebalance_schedule
    from vnrank.models import ranking_lambdarank as l1
    from vnrank.models import trend_classifier as t2

    CODE_ROOT = Path(__file__).resolve().parents[2]
    CONFIG_PATH = CODE_ROOT / "config" / "config.yaml"
    DATA_DIR = CODE_ROOT / "data" / "processed"

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    features = pd.read_csv(DATA_DIR / "features_daily.csv", parse_dates=["date"])
    labels_all = pd.read_csv(DATA_DIR / "labels_daily.csv", parse_dates=["date", "exit_date_resolved", "t1"])
    splits = pd.read_csv(DATA_DIR / "splits_daily.csv", parse_dates=["date"])
    universe = pd.read_csv(DATA_DIR / "universe_daily.csv", parse_dates=["date"])
    frozen = pd.read_csv(DATA_DIR / "frozen_configurations_4_7.csv")

    import ast
    def _frozen_params(family: str, h: int) -> dict:
        row = frozen[(frozen["family"] == family) & (frozen["h"] == h)].iloc[0]
        return ast.literal_eval(row["chon_tren_validation"])

    train_start_date = splits.loc[splits["split"] == "train", "date"].min()
    test_start_date = splits.loc[splits["split"] == "test", "date"].min()
    test_end_date = splits.loc[splits["split"] == "test", "date"].max()
    calendar = sorted(splits["date"].unique())

    gbdt_common = cfg["model_grid"]["gbdt_common"]
    fraction = cfg["data_split"]["early_stopping_fraction_gbdt"]
    refit_frequency = cfg["data_split"]["refit_frequency_sessions"]
    seed = cfg["seed"]

    labels_h5 = labels_all[labels_all["h"] == A1_HORIZON]

    l1_params = _frozen_params("L1_lambdarank", A1_HORIZON)
    l1_labeled = l1.build_labeled_features(features, labels_h5)
    l1_scores = l1.walk_forward_test_scores(
        l1_labeled, l1_params, train_start_date, test_start_date, test_end_date, calendar, refit_frequency, gbdt_common, seed, fraction,
    )
    l1_test = l1_scores.merge(splits, on="date", how="inner")
    l1_test = l1_test[l1_test["split"] == "test"][["date", "symbol", "score"]].reset_index(drop=True)

    t2_params = _frozen_params("T2_lightgbm", A1_HORIZON)
    t2_labeled = t2.build_labeled_features(features, labels_h5)
    t2_scores = t2.walk_forward_test_scores(
        t2_labeled, t2_params, train_start_date, test_start_date, test_end_date, calendar, refit_frequency, gbdt_common, seed, fraction,
    )
    t2_test = t2_scores.merge(splits, on="date", how="inner")
    t2_test = t2_test[t2_test["split"] == "test"].reset_index(drop=True)
    t2_a1 = t2_test.copy()
    t2_a1["p_diff"] = t2_a1[1] - t2_a1[-1]
    t2_a1 = t2_a1[["date", "symbol", "p_diff"]]

    rebalance_dates = build_rebalance_schedule(test_start_date, test_end_date, calendar, h=A1_HORIZON)
    momentum_scores = momentum_test_scores()

    table = build_system_results_table(l1_test, t2_a1, labels_h5, universe, splits, momentum_scores, rebalance_dates, cfg)
    table.to_csv(DATA_DIR / "system_results_s8.csv", index=False)
    print(table.to_string(index=False))
    print(f"Da luu {len(table)} dong vao code/data/processed/system_results_s8.csv")
