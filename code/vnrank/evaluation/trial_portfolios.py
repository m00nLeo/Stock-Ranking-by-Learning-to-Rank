"""Dựng lại danh mục VALIDATION cho từng cấu hình trong hai nhóm tìm kiếm của bước 4.14 (D116,
D118). Log 4.2/4.3/4.4/4.10 chỉ lưu Rank IC/macro-F1 validation (số vô hướng dùng để CHỌN cấu
hình), KHÔNG lưu chuỗi điểm/danh mục — phải chạy lại ĐÚNG quy trình đã đóng băng (cùng fit trên
TRAIN, cùng seed, cùng cấu hình lưới) để có chuỗi lợi suất danh mục trên validation, dùng cho:
- Sharpe (DSR, `giao_thuc_danh_gia.md` mục 7) và tương quan cặp (ước lượng N khi không độc lập).
- Chuỗi NGÀY so le (CSCV/PBO, bảng mục 2 `giao_thuc_danh_gia.md`: chạy song song h danh mục, mỗi
  danh mục lệch nhau một phiên, lợi suất ngày = trung bình h danh mục).

Hai nhóm tìm kiếm (quyết định người dùng 17/11/2026, xem `dsr_pbo.py`):
- "l1_pool" (N=21, h=5): B2 (5 ô)+B3 (2 ô)+B4 (2 ô)+R1 (3 ô)+R2 (3 ô)+R3 (3 ô)+L1 (3 ô).
- "a1_pool" (N=4, h=5): A0, A1, A2, A3 (tích hợp xu hướng, `trend_integration.py`).
T2 (4.10) không thuộc nhóm nào — chỉ dùng cấu hình ĐÃ ĐÓNG BĂNG (num_leaves=15,
min_data_in_leaf=200, giống `trend_integration.py`) để dựng tín hiệu xu hướng nuôi A1/A2 trên
validation, không tính Sharpe riêng cho T2.

K=10, chi phí "cơ sở" (`config.yaml` costs.c_buy/c_sell): DÙNG CHUNG cho mọi cấu hình trong cả hai
nhóm (kể cả B2-B4/R1-R3 vốn không tự nhiên có K) — [TỰ QUYẾT, ghi `decision_log.md`]: cần MỘT quy
ước K/chi phí thống nhất để Sharpe các thử nghiệm so sánh được với nhau; dùng đúng K/chi phí đã là
"cấu hình chính" của hệ thống (`trend_integration.K`, `system_results.py` mức "cơ sở") thay vì bịa
tham số mới.

Chuỗi NGÀY so le (PBO): vì không dựng lại giá hàng ngày trong kỳ nắm giữ, trải ĐỀU (lãi kép, không
phải cộng dồn tuyến tính) lợi suất mỗi kỳ h phiên ra các phiên của kỳ đó — [TỰ QUYẾT, ghi
`decision_log.md`]: giữ ĐÚNG lợi suất kỳ (gộp lại ra đúng số đã có), chỉ là một cách biểu diễn lại
ở độ phân giải ngày để CSCV có đủ số khối, không đổi nội dung đo lường.

A3 (chọn ngẫu nhiên): lặp lại `A3_N_REPLICATES` lần (giảm từ 10.000 của 4.13 — [TỰ QUYẾT]: ở đây
A3 chỉ là MỘT trong bốn thử nghiệm của nhóm a1_pool, cần Sharpe trung bình để đại diện, không cần độ
chính xác phân phối đầy đủ như kiểm định H4c ở 4.13), lấy Sharpe/chuỗi ngày trung bình qua các lần
lặp làm đại diện cho cấu hình A3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vnrank.costs.portfolio_backtest import preselected_portfolio_returns, top_k_portfolio_returns
from vnrank.features.feature_registry import fill_market_calendar_gaps
from vnrank.models.baselines import (
    AMIHUD_GRID,
    IMOM_GRID,
    MOMENTUM_GRID,
    _log_return,
    amihud_illiquidity,
    imom_score,
    momentum_score,
)
from vnrank.models.ranking_lambdarank import _fit_predict_lambdarank
from vnrank.models.ranking_lambdarank import build_candidate_configs as l1_candidate_configs
from vnrank.models.ranking_lambdarank import build_labeled_features as l1_build_labeled_features
from vnrank.models.regression_topk import REALIZATION_COL as REG_REALIZATION_COL
from vnrank.models.regression_topk import build_candidate_configs as reg_candidate_configs
from vnrank.models.regression_topk import build_labeled_features as reg_build_labeled_features
from vnrank.models.regression_topk import fit_predict as reg_fit_predict
from vnrank.models.trend_classifier import REALIZATION_COL as T2_REALIZATION_COL
from vnrank.models.trend_classifier import _fit_predict_t2
from vnrank.models.trend_classifier import build_labeled_features as t2_build_labeled_features
from vnrank.models.trend_integration import CLASSES_SIGN, build_labeled_features_sign, two_stage_select
from vnrank.splits.split_protocol import refit_training_window
from vnrank.universe.build_universe import PRICE_CACHE_DIR, load_price_panel

CODE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = CODE_ROOT / "config" / "config.yaml"
H = 5  # DSR/PBO (D116) chi tinh o h=5
K = 10  # trend_integration.K, dung chung cho ca hai nhom (xem docstring file)
T2_FROZEN_CONFIG = {"num_leaves": 15, "min_data_in_leaf": 200}  # dong bang 4.10/4.13
L1_FROZEN_CONFIG = {"num_leaves": 15, "min_data_in_leaf": 200}  # dong bang 4.4/4.13
A3_N_REPLICATES = 200


# --- Dữ liệu dùng chung -----------------------------------------------------------------------


def load_common(data_dir: Path = CODE_ROOT / "data" / "processed") -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    features = pd.read_csv(data_dir / "features_daily.csv", parse_dates=["date"])
    labels = pd.read_csv(
        data_dir / "labels_daily.csv", parse_dates=["date", REG_REALIZATION_COL, T2_REALIZATION_COL]
    )
    splits = pd.read_csv(data_dir / "splits_daily.csv", parse_dates=["date"])
    universe = pd.read_csv(data_dir / "universe_daily.csv", parse_dates=["date"])
    labels_h5 = labels[labels["h"] == H].copy()

    train_start_date = splits.loc[splits["split"] == "train", "date"].min()
    train_end_date = splits.loc[splits["split"] == "train", "date"].max()
    validation_dates = sorted(splits.loc[splits["split"] == "validation", "date"])

    return {
        "cfg": cfg,
        "features": features,
        "labels_h5": labels_h5,
        "splits": splits,
        "universe": universe,
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "validation_dates": validation_dates,
        "gbdt_common": cfg["model_grid"]["gbdt_common"],
        "fraction": cfg["data_split"]["early_stopping_fraction_gbdt"],
        "seed": cfg["seed"],
        "c_buy": cfg["costs"]["c_buy"],
        "c_sell": cfg["costs"]["c_sell"],
        "data_dir": data_dir,
    }


def _predict_df_for_validation(labeled: pd.DataFrame, validation_dates: list) -> pd.DataFrame:
    val_set = set(validation_dates)
    return labeled[labeled["date"].isin(val_set)]


# --- B2/B3/B4: điểm baseline trên validation (công thức, không cần fit) ------------------------


def build_baseline_scores_validation(common: dict) -> dict[str, pd.DataFrame]:
    """9 cấu hình (B2 x5, B3 x2, B4 x2) ở h=5, tái dựng ĐÚNG công thức/lưới của `baselines.py`
    (__main__), chỉ lọc lại còn ngày validation ở bước cuối."""
    universe = common["universe"]
    symbols = sorted(universe["symbol"].unique())
    panel = load_price_panel(symbols)
    panel = panel[panel["symbol"].isin(symbols)]

    market_raw = pd.read_csv(PRICE_CACHE_DIR / "VNINDEX.csv", parse_dates=["date"])
    calendar = sorted(panel["date"].unique())
    market = fill_market_calendar_gaps(market_raw, calendar)
    market_ret = _log_return(market["close"], 1)
    market_ret.index = market["date"]
    market_close = market.set_index("date")["close"]

    momentum_frames: dict = {}
    imom_frames: dict = {}
    amihud_frames: dict = {}
    for symbol, group in panel.sort_values(["symbol", "date"]).groupby("symbol"):
        g = group.reset_index(drop=True)
        dates = g["date"]
        stock_ret = _log_return(g["close"], 1)
        mkt_ret_aligned = pd.Series(dates.map(market_ret).to_numpy())
        mkt_close_aligned = pd.Series(dates.map(market_close).to_numpy())

        for window, skip in MOMENTUM_GRID:
            s = momentum_score(g["close"], window, skip)
            momentum_frames.setdefault((window, skip), {})[symbol] = pd.Series(s.to_numpy(), index=dates)
        for window, skip in IMOM_GRID:
            s = imom_score(g["close"], stock_ret, mkt_close_aligned, mkt_ret_aligned, window, skip)
            imom_frames.setdefault((window, skip), {})[symbol] = pd.Series(s.to_numpy(), index=dates)
        for window in AMIHUD_GRID:
            s = amihud_illiquidity(g["close"], g["volume"], window)
            amihud_frames.setdefault(window, {})[symbol] = pd.Series(s.to_numpy(), index=dates)

    def to_long(frames_by_config: dict, prefix: str) -> dict[str, pd.DataFrame]:
        out = {}
        for key, by_symbol in frames_by_config.items():
            rows = [
                pd.DataFrame({"date": s.index, "symbol": symbol, "score": s.to_numpy()})
                for symbol, s in by_symbol.items()
            ]
            scored = pd.concat(rows, ignore_index=True)
            scored = scored[scored["date"].isin(set(common["validation_dates"]))]
            out[f"{prefix}_{key}"] = scored.dropna(subset=["score"]).reset_index(drop=True)
        return out

    result: dict[str, pd.DataFrame] = {}
    result.update(to_long(momentum_frames, "B2"))
    result.update(to_long(imom_frames, "B3"))
    result.update(to_long(amihud_frames, "B4"))
    return result


# --- R1/R2/R3: fit một lần trên train, dự báo validation ----------------------------------------


def build_regression_scores_validation(common: dict) -> dict[str, pd.DataFrame]:
    """9 cấu hình (R1 x3, R2 x3, R3 x3) ở h=5: fit MỘT LẦN trên train (đúng `select_config_on_validation`
    của `regression_topk.py`), dự báo mọi ngày validation. Tái dùng `fit_predict` công khai, KHÔNG
    chọn lại trên validation (giữ cả 9 ô, không chỉ ô được chọn — dùng để tính N/Sharpe của CẢ nhóm)."""
    labeled = reg_build_labeled_features(common["features"], common["labels_h5"])
    train_window = refit_training_window(
        labeled, REG_REALIZATION_COL, common["train_start_date"], common["train_end_date"]
    )
    predict_df = _predict_df_for_validation(labeled, common["validation_dates"])

    candidates = reg_candidate_configs(common["cfg"]["model_grid"])
    out: dict[str, pd.DataFrame] = {}
    for model_family, configs in candidates.items():
        for key, params in configs.items():
            preds = reg_fit_predict(
                model_family, train_window, params, common["gbdt_common"], common["seed"],
                common["fraction"], predict_df,
            )
            out[f"{model_family}_{key}"] = pd.DataFrame(
                {"date": predict_df["date"].to_numpy(), "symbol": predict_df["symbol"].to_numpy(), "score": preds}
            )
    return out


# --- L1: fit một lần trên train, dự báo validation (3 cấu hình) ---------------------------------


def build_l1_scores_validation(common: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """3 cấu hình L1 ở h=5 (như `select_config_on_validation` của `ranking_lambdarank.py`). Trả
    về (dict 3 cấu hình, scored_df của cấu hình ĐÃ ĐÓNG BĂNG `L1_FROZEN_CONFIG` — tái dùng cho
    A0-A3 ở validation, không fit lại)."""
    labeled = l1_build_labeled_features(common["features"], common["labels_h5"])
    train_window = refit_training_window(
        labeled, REG_REALIZATION_COL, common["train_start_date"], common["train_end_date"]
    )
    predict_df = _predict_df_for_validation(labeled, common["validation_dates"])

    configs = l1_candidate_configs(common["cfg"]["model_grid"])
    out: dict[str, pd.DataFrame] = {}
    frozen_scored = None
    for key, params in configs.items():
        preds = _fit_predict_lambdarank(
            train_window, params, common["gbdt_common"], common["seed"], common["fraction"], predict_df
        )
        scored = pd.DataFrame(
            {"date": predict_df["date"].to_numpy(), "symbol": predict_df["symbol"].to_numpy(), "score": preds}
        )
        out[f"L1_{key}"] = scored
        if params == L1_FROZEN_CONFIG:
            frozen_scored = scored
    if frozen_scored is None:
        raise AssertionError("L1_FROZEN_CONFIG khong khop ô lưới nào — kiểm tra config.yaml")
    return out, frozen_scored


# --- T2/A2: cấu hình ĐÃ ĐÓNG BĂNG, dự báo validation (không chọn lại, không tính Sharpe riêng) ---


def build_t2_p_diff_validation(common: dict) -> pd.DataFrame:
    labeled = t2_build_labeled_features(common["features"], common["labels_h5"])
    train_window = refit_training_window(labeled, T2_REALIZATION_COL, common["train_start_date"], common["train_end_date"])
    predict_df = _predict_df_for_validation(labeled, common["validation_dates"])
    probs = _fit_predict_t2(train_window, T2_FROZEN_CONFIG, common["gbdt_common"], common["seed"], common["fraction"], predict_df)
    scored = pd.DataFrame({"date": predict_df["date"].to_numpy(), "symbol": predict_df["symbol"].to_numpy()})
    scored[1] = probs[:, 0]
    scored[-1] = probs[:, 2]
    scored["p_diff"] = scored[1] - scored[-1]
    return scored


def build_a2_p_diff_validation(common: dict) -> pd.DataFrame:
    labeled = build_labeled_features_sign(common["features"], common["labels_h5"])
    train_window = refit_training_window(labeled, REG_REALIZATION_COL, common["train_start_date"], common["train_end_date"])
    predict_df = _predict_df_for_validation(labeled, common["validation_dates"])
    probs = _fit_predict_t2(
        train_window, T2_FROZEN_CONFIG, common["gbdt_common"], common["seed"], common["fraction"], predict_df,
        CLASSES_SIGN, "y_sign",
    )
    scored = pd.DataFrame({"date": predict_df["date"].to_numpy(), "symbol": predict_df["symbol"].to_numpy()})
    scored[1] = probs[:, 0]
    scored[-1] = probs[:, 1]
    scored["p_diff"] = scored[1] - scored[-1]
    return scored


# --- A0-A3: lựa chọn trên validation (dùng L1/T2/A2 đã đóng băng ở trên) ------------------------


def build_a0_a3_selections_validation(common: dict, l1_frozen_scored: pd.DataFrame) -> dict[str, pd.DataFrame]:
    top_m = common["cfg"]["trend_integration"]["top_M"]
    k = common["cfg"]["trend_integration"]["K"]
    seed = common["seed"]

    t2_scores = build_t2_p_diff_validation(common)
    a2_scores = build_a2_p_diff_validation(common)

    a0_selected = (
        l1_frozen_scored.sort_values("date")
        .groupby("date", group_keys=False)
        .apply(lambda g: g.nlargest(min(k, len(g)), "score"))[["date", "symbol"]]
        .reset_index(drop=True)
    )
    a1_selected = two_stage_select(l1_frozen_scored, t2_scores[["date", "symbol", "p_diff"]], "p_diff", top_m, k)
    a2_selected = two_stage_select(l1_frozen_scored, a2_scores[["date", "symbol", "p_diff"]], "p_diff", top_m, k)

    rng = np.random.default_rng(seed)
    a3_selected_by_replicate = []
    for _ in range(A3_N_REPLICATES):
        rows = []
        for date, group in l1_frozen_scored.groupby("date"):
            top_m_group = group.nlargest(min(top_m, len(group)), "score")
            n = len(top_m_group)
            if n < k:
                continue
            chosen = top_m_group.sample(n=k, replace=False, random_state=rng.integers(0, 2**32 - 1))
            rows.append(chosen[["date", "symbol"]])
        a3_selected_by_replicate.append(pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "symbol"]))

    return {"A0": a0_selected, "A1": a1_selected, "A2": a2_selected, "A3_replicates": a3_selected_by_replicate}


# --- Lịch tái cân bằng so le + trải đều lợi suất kỳ ra ngày -------------------------------------


def staggered_rebalance_schedules(calendar: list, h: int) -> list[list]:
    """h lịch tái cân bằng lệch nhau 1 phiên (`giao_thuc_danh_gia.md` mục 2, hàng "Chuỗi ngày của
    h danh mục so le")."""
    cal_sorted = sorted(calendar)
    return [cal_sorted[o::h] for o in range(h)]


def expand_period_returns_to_daily(r_net_df: pd.DataFrame, calendar: list) -> pd.Series:
    """Trải ĐỀU (lãi kép) lợi suất mỗi kỳ ra các phiên của kỳ đó (xem docstring đầu file, mục
    "chuỗi NGÀY so le"). `r_net_df`: cột date (NGÀY BẮT ĐẦU kỳ), r_net."""
    cal_sorted = sorted(calendar)
    idx = {d: i for i, d in enumerate(cal_sorted)}
    period_starts = sorted(r_net_df["date"].tolist())
    r_by_date = dict(zip(r_net_df["date"], r_net_df["r_net"]))

    out = {}
    for j, t in enumerate(period_starts):
        if t not in idx:
            continue
        start_i = idx[t]
        end_i = idx[period_starts[j + 1]] if j + 1 < len(period_starts) and period_starts[j + 1] in idx else len(cal_sorted)
        n_sessions = max(end_i - start_i, 1)
        r_period = float(r_by_date[t])
        growth = 1.0 + r_period
        daily_rate = growth ** (1.0 / n_sessions) - 1.0 if growth > 0 else float("nan")
        for i in range(start_i, end_i):
            out[cal_sorted[i]] = daily_rate
    return pd.Series(out).sort_index()


def build_staggered_daily_series(build_r_net_fn, calendar: list, h: int) -> pd.Series:
    """Chuỗi ngày so le (D116/mục 2): trung bình `h` lịch tái cân bằng lệch nhau, mỗi lịch trải
    đều lợi suất kỳ ra ngày (`expand_period_returns_to_daily`). `build_r_net_fn(rebalance_dates)
    -> DataFrame[date, r_net]`."""
    schedules = staggered_rebalance_schedules(calendar, h)
    daily_series = []
    for rebalance_dates in schedules:
        if len(rebalance_dates) < 2:
            continue
        r_net_df = build_r_net_fn(rebalance_dates)
        if r_net_df.empty:
            continue
        daily_series.append(expand_period_returns_to_daily(r_net_df[["date", "r_net"]], calendar))
    if not daily_series:
        return pd.Series(dtype=float)
    combined = pd.concat(daily_series, axis=1)
    return combined.mean(axis=1, skipna=True)


# --- Lắp ráp: chuỗi lợi suất kỳ (Sharpe/N) + chuỗi ngày so le (CSCV) cho một cấu hình ------------


def top_k_trial(scored: pd.DataFrame, common: dict, rebalance_dates: list) -> pd.DataFrame:
    return top_k_portfolio_returns(scored, common["labels_h5"], rebalance_dates, K, common["c_buy"], common["c_sell"])


def preselected_trial(selected: pd.DataFrame, common: dict, rebalance_dates: list, k: int | None = None) -> pd.DataFrame:
    kk = k if k is not None else common["cfg"]["trend_integration"]["K"]
    return preselected_portfolio_returns(selected, common["labels_h5"], rebalance_dates, kk, common["c_buy"], common["c_sell"])
