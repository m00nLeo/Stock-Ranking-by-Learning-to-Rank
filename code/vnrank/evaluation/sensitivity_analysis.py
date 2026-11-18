"""Phân tích độ nhạy, RQ3 (bước 4.8). Đặc tả: `giao_thuc_danh_gia.md` mục 9 — theo quyết định của
người dùng (14/11/2026), dùng ĐÚNG bốn yếu tố ở mục 9 (Chi phí, Ngưỡng thanh khoản, K, Giai đoạn
thị trường), KHÔNG dùng "tần suất tái cân bằng" (câu tóm tắt ngắn, không khớp, ở charter mục 7/9).
Đổi MỘT yếu tố mỗi lần (one-factor-at-a-time) so với cấu hình chính đã đóng băng ($h=5$, $K=10$, L1
cho Rank IC/Sharpe, T2 cho macro-F1) — KHÔNG quét lưới đầy đủ, KHÔNG chọn lại mô hình theo kết quả
của ô nào, chỉ ĐÁNH GIÁ LẠI (không retrain) điểm L1/T2 đã có trên tập con/mức chi phí/K khác nhau.

Báo cáo (theo quyết định người dùng, 14/11/2026): ΔIC (H1), ΔSharpe ròng (H1b), ΔF1 (H3). **Δprecision
(H4) BỎ QUA TẠM THỜI** vì cần tích hợp xu hướng A0-A4 (chưa làm, để ở 4.13 — SAU 4.8 theo thứ tự
charter); sẽ bổ sung cột này sau khi 4.13 xong, không tính Holm/kết luận thiếu cột này ở đây.

Module mới `vnrank/costs/portfolio_backtest.py` (top-K, tái cân bằng mỗi $h$ phiên, chi phí theo
turnover) được xây NGAY TRONG bước này (theo quyết định người dùng) vì CHƯA từng có ở 4.2-4.7 —
mọi bước trước chỉ tính Rank IC/ICIR trên toàn universe, chưa dựng danh mục top-$K$ thật có chi phí.

Ngưỡng thanh khoản "chặt hơn" CHƯA chốt số cụ thể ở đâu trong hồ sơ (chỉ "Chốt ở 3.4" cho mức cơ
sở — D74). Theo quyết định người dùng (14/11/2026): dùng PHƯƠNG PHÁP giống D74 (bách phân vị của
`avg_value_l` trên train+validation) nhưng ở bách phân vị 50 (trung vị) thay vì 25-30 của mức cơ
sở — tính TẠI ĐÂY từ dữ liệu thật, không hardcode con số (`compute_strict_liquidity_threshold`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from vnrank.costs.portfolio_backtest import build_rebalance_schedule, sharpe_annualized, top_k_portfolio_returns
from vnrank.evaluation.harness import icir, macro_f1, rank_ic_series

CODE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = CODE_ROOT / "config" / "config.yaml"

H_MAIN = 5  # cấu hình chính (D54: H1-H3 chỉ kiểm định ở h=5)
K_BASE = 10  # ma_tran_baseline_ablation.md mục 5, D59
C_BUY_BASE = 0.0025
C_SELL_BASE = 0.0035
C_RT_BASE = 0.0060  # == C_BUY_BASE + C_SELL_BASE


def scaled_costs(c_rt_target: float) -> tuple[float, float]:
    """Suy ra `c_buy`/`c_sell` ở mức chi phí khứ hồi `c_rt_target` (độ nhạy 2.6, D61) bằng cách
    co giãn tỷ lệ TUYẾN TÍNH giữ nguyên tỷ trọng mua/bán của mức cơ sở (0,0025/0,0035, cộng đúng
    bằng `c_rt_base`=0,0060)."""
    factor = c_rt_target / C_RT_BASE
    return C_BUY_BASE * factor, C_SELL_BASE * factor


def compute_strict_liquidity_threshold(
    data_dir: Path, window_l: int = 21, lag_sessions: int = 0, percentile: float = 0.5
) -> tuple[float, pd.DataFrame]:
    """Ngưỡng thanh khoản "chặt hơn" (quyết định người dùng 14/11/2026): bách phân vị 50 của
    `avg_value_l` RAW (chưa chuẩn hóa chéo) trên train+validation, cùng phương pháp D74. Trả về
    (ngưỡng, bảng {date, symbol, avg_value_l} raw để lọc ứng viên theo ngày)."""
    from vnrank.universe.build_universe import compute_liquidity_features, load_listed_stocks, load_price_panel

    stocks = load_listed_stocks()
    symbols = sorted(stocks["symbol"].unique())
    panel = load_price_panel(symbols)
    panel = panel[panel["symbol"].isin(symbols)]
    feat = compute_liquidity_features(panel, window_l=window_l, lag_sessions=lag_sessions)

    splits = pd.read_csv(data_dir / "splits_daily.csv", parse_dates=["date"])
    train_val_dates = set(splits.loc[splits["split"].isin(["train", "validation"]), "date"])
    sub = feat[feat["date"].isin(train_val_dates)].dropna(subset=["avg_value_l"])
    threshold = float(sub["avg_value_l"].quantile(percentile))
    return threshold, feat[["date", "symbol", "avg_value_l"]]


def _test_rows(scored: pd.DataFrame, labels_h: pd.DataFrame, splits: pd.DataFrame, value_col: str) -> pd.DataFrame:
    merged = scored.merge(labels_h[["date", "symbol", value_col]], on=["date", "symbol"], how="inner")
    merged = merged.merge(splits, on="date", how="inner")
    return merged[merged["split"] == "test"]


def rank_ic_mean_under_filter(
    scored: pd.DataFrame, labels_h: pd.DataFrame, splits: pd.DataFrame,
    liquidity_table: pd.DataFrame | None = None, threshold: float | None = None,
) -> float:
    """Rank IC trung bình trên test, tùy chọn lọc thêm theo ngưỡng thanh khoản raw (không ảnh
    hưởng gì nếu `liquidity_table` là None — tức mức cơ sở, giữ nguyên $U_t$ đã có)."""
    test = _test_rows(scored, labels_h, splits, "r")
    if liquidity_table is not None:
        test = test.merge(liquidity_table, on=["date", "symbol"], how="left")
        test = test[test["avg_value_l"] >= threshold]
    ic = rank_ic_series(test["date"], test["score"].to_numpy(), test["r"].to_numpy())
    return icir(ic)["ic_mean"]


def macro_f1_under_filter(
    pred_scored: pd.DataFrame, labels_h: pd.DataFrame, splits: pd.DataFrame,
    liquidity_table: pd.DataFrame | None = None, threshold: float | None = None,
) -> float:
    """macro-F1 của T2 trên mẫu mã-ngày của test, tùy chọn lọc thêm theo ngưỡng thanh khoản raw."""
    test = _test_rows(pred_scored, labels_h, splits, "y")
    if liquidity_table is not None:
        test = test.merge(liquidity_table, on=["date", "symbol"], how="left")
        test = test[test["avg_value_l"] >= threshold]
    return macro_f1(test["y"].to_numpy(), test["pred"].to_numpy(), classes=(1, 0, -1))


def sharpe_for_config(
    scored: pd.DataFrame, labels_h: pd.DataFrame, rebalance_dates: list[pd.Timestamp],
    k: int, c_buy: float, c_sell: float, h: int,
    liquidity_table: pd.DataFrame | None = None, threshold: float | None = None,
) -> float:
    """Sharpe năm hóa của danh mục top-K với cấu hình (k, chi phí, lọc thanh khoản tùy chọn)."""
    candidates = scored
    if liquidity_table is not None:
        candidates = candidates.merge(liquidity_table, on=["date", "symbol"], how="left")
        candidates = candidates[candidates["avg_value_l"] >= threshold][["date", "symbol", "score"]]
    returns = top_k_portfolio_returns(candidates, labels_h, rebalance_dates, k, c_buy, c_sell)
    return sharpe_annualized(returns["r_net"].to_numpy(), h)


def market_state_rebalance_dates(
    rebalance_dates: list[pd.Timestamp], market_state: pd.DataFrame, regime: str
) -> list[pd.Timestamp]:
    """Lọc lịch tái cân bằng, chỉ giữ các ngày thuộc `regime` (tăng/giảm/đi ngang, 3.11) — dùng để
    tính Sharpe RIÊNG cho từng giai đoạn thị trường trên CÙNG lịch tái cân bằng gốc."""
    state_by_date = dict(zip(market_state["date"], market_state["state"]))
    return [d for d in rebalance_dates if state_by_date.get(d) == regime]


if __name__ == "__main__":
    from vnrank.models.ranking_lambdarank import build_labeled_features as build_labeled_l1
    from vnrank.models.ranking_lambdarank import walk_forward_test_scores as walk_forward_l1
    from vnrank.models.regression_topk import REALIZATION_COL as REALIZATION_COL_REG
    from vnrank.models.trend_classifier import REALIZATION_COL as REALIZATION_COL_TREND
    from vnrank.models.trend_classifier import build_labeled_features as build_labeled_t2
    from vnrank.models.trend_classifier import d0_walk_forward_scores
    from vnrank.models.trend_classifier import walk_forward_test_scores as walk_forward_t2

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = CODE_ROOT / "data" / "processed"
    features = pd.read_csv(data_dir / "features_daily.csv", parse_dates=["date"])
    labels_all = pd.read_csv(
        data_dir / "labels_daily.csv", parse_dates=["date", REALIZATION_COL_REG, REALIZATION_COL_TREND]
    )
    splits = pd.read_csv(data_dir / "splits_daily.csv", parse_dates=["date"])
    market_state = pd.read_csv(data_dir / "market_state_daily.csv", parse_dates=["date"])

    gbdt_common = cfg["model_grid"]["gbdt_common"]
    fraction = cfg["data_split"]["early_stopping_fraction_gbdt"]
    refit_frequency = cfg["data_split"]["refit_frequency_sessions"]
    seed = cfg["seed"]

    train_start_date = splits.loc[splits["split"] == "train", "date"].min()
    test_start_date = splits.loc[splits["split"] == "test", "date"].min()
    test_end_date = splits.loc[splits["split"] == "test", "date"].max()
    calendar = sorted(splits["date"].unique())

    labels_h5 = labels_all[labels_all["h"] == H_MAIN]
    l1_config = {"num_leaves": 15, "min_data_in_leaf": 200}  # da dong bang o 4.4 (h=5)
    t2_config = {"num_leaves": 15, "min_data_in_leaf": 200}  # da dong bang o 4.10 (h=5)

    l1_labeled = build_labeled_l1(features, labels_h5)
    l1_scores = walk_forward_l1(
        l1_labeled, l1_config, train_start_date, test_start_date, test_end_date, calendar, refit_frequency, gbdt_common, seed, fraction,
    )
    t2_labeled = build_labeled_t2(features, labels_h5)
    t2_scores = walk_forward_t2(
        t2_labeled, t2_config, train_start_date, test_start_date, test_end_date, calendar, refit_frequency, gbdt_common, seed, fraction,
    )

    rebalance_dates = build_rebalance_schedule(test_start_date, test_end_date, calendar, h=H_MAIN)

    ic_base = rank_ic_mean_under_filter(l1_scores, labels_h5, splits)
    f1_base = macro_f1_under_filter(t2_scores, labels_h5, splits)
    sharpe_base = sharpe_for_config(l1_scores, labels_h5, rebalance_dates, K_BASE, C_BUY_BASE, C_SELL_BASE, H_MAIN)
    print(f"CO SO (h=5, K={K_BASE}, chi phi co so, khong loc thanh khoan): IC={ic_base:.4f} F1={f1_base:.4f} Sharpe={sharpe_base:.4f}")

    rows = [{"yeu_to": "co_so", "muc": "co_so", "delta_ic": 0.0, "delta_sharpe": 0.0, "delta_f1": 0.0, "ic": ic_base, "sharpe": sharpe_base, "f1": f1_base}]

    # --- Yeu to 1: Chi phi (thap/co so/cao) -----------------------------------------------------
    c_rt_low, c_rt_high = cfg["costs"]["c_rt_sensitivity"]
    for ten_muc, c_rt in (("thap", c_rt_low), ("cao", c_rt_high)):
        c_buy, c_sell = scaled_costs(c_rt)
        sharpe = sharpe_for_config(l1_scores, labels_h5, rebalance_dates, K_BASE, c_buy, c_sell, H_MAIN)
        rows.append({"yeu_to": "chi_phi", "muc": ten_muc, "delta_ic": 0.0, "delta_sharpe": sharpe - sharpe_base, "delta_f1": 0.0, "ic": ic_base, "sharpe": sharpe, "f1": f1_base})

    # --- Yeu to 2: Nguong thanh khoan (chat hon) -------------------------------------------------
    threshold, liquidity_table = compute_strict_liquidity_threshold(data_dir)
    print(f"Nguong thanh khoan chat hon (bach phan vi 50 avg_value_l, train+validation): {threshold:.2f} nghin VND")
    ic_strict = rank_ic_mean_under_filter(l1_scores, labels_h5, splits, liquidity_table, threshold)
    f1_strict = macro_f1_under_filter(t2_scores, labels_h5, splits, liquidity_table, threshold)
    sharpe_strict = sharpe_for_config(l1_scores, labels_h5, rebalance_dates, K_BASE, C_BUY_BASE, C_SELL_BASE, H_MAIN, liquidity_table, threshold)
    rows.append({"yeu_to": "nguong_thanh_khoan", "muc": "chat_hon", "delta_ic": ic_strict - ic_base, "delta_sharpe": sharpe_strict - sharpe_base, "delta_f1": f1_strict - f1_base, "ic": ic_strict, "sharpe": sharpe_strict, "f1": f1_strict})

    # --- Yeu to 3: K (5, 20; co so la 10) -----------------------------------------------------
    for k in (5, 20):
        sharpe = sharpe_for_config(l1_scores, labels_h5, rebalance_dates, k, C_BUY_BASE, C_SELL_BASE, H_MAIN)
        rows.append({"yeu_to": "K", "muc": str(k), "delta_ic": 0.0, "delta_sharpe": sharpe - sharpe_base, "delta_f1": 0.0, "ic": ic_base, "sharpe": sharpe, "f1": f1_base})

    # --- Yeu to 4: Giai doan thi truong (tang/giam/di_ngang) ------------------------------------
    for regime in ("tang", "giam", "di_ngang"):
        regime_rebalance_dates = market_state_rebalance_dates(rebalance_dates, market_state, regime)
        regime_dates_set = set(market_state.loc[market_state["state"] == regime, "date"])
        l1_regime = l1_scores[l1_scores["date"].isin(regime_dates_set)]
        t2_regime = t2_scores[t2_scores["date"].isin(regime_dates_set)]
        ic_regime = rank_ic_mean_under_filter(l1_regime, labels_h5, splits) if len(l1_regime) else float("nan")
        f1_regime = macro_f1_under_filter(t2_regime, labels_h5, splits) if len(t2_regime) else float("nan")
        sharpe_regime = sharpe_for_config(l1_scores, labels_h5, regime_rebalance_dates, K_BASE, C_BUY_BASE, C_SELL_BASE, H_MAIN) if regime_rebalance_dates else float("nan")
        rows.append({"yeu_to": "giai_doan_thi_truong", "muc": regime, "delta_ic": ic_regime - ic_base, "delta_sharpe": sharpe_regime - sharpe_base if sharpe_regime == sharpe_regime else float("nan"), "delta_f1": f1_regime - f1_base, "ic": ic_regime, "sharpe": sharpe_regime, "f1": f1_regime})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(data_dir / "sensitivity_analysis_4_8.csv", index=False)
    print(out_df.to_string(index=False))
    print("Da luu bang do nhay vao code/data/processed/sensitivity_analysis_4_8.csv")
    print("LUU Y: cot delta_precision (H4) CHUA co, de bo sung sau 4.13 (quyet dinh nguoi dung 14/11/2026).")
