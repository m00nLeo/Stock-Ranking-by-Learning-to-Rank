"""Baseline không dùng ML (bước 4.2): B0 (đồng trọng số), B1 (VN30), B2 (momentum nhiều cửa
sổ), B3 (IMOM), B4 (Amihud illiquidity). Đặc tả: `ma_tran_baseline_ablation.md` mục 2 (BẢN CHỐT,
D59); tham số lưới ở `config.yaml` mục `model_grid.baseline`.

B2, B3, B4 là ĐIỂM (không học), dùng Rank IC để đánh giá; ô lưới (cửa sổ, …) được CHỌN trên
validation theo Rank IC trung bình, rồi chỉ chạy một lần trên test (không chọn lại trên test,
đúng nguyên tắc của dự án). B0, B1 là mốc danh mục thụ động, không có điểm xếp hạng, chỉ báo mô tả.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vnrank.data.entrade_client import fetch_ohlcv
from vnrank.evaluation.harness import circular_block_bootstrap_mean_ci, icir, newey_west_test, rank_ic_series
from vnrank.features.feature_registry import rolling_beta
from vnrank.universe.build_universe import PRICE_CACHE_DIR

CODE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = CODE_ROOT / "config" / "config.yaml"
VN30_CACHE_PATH = PRICE_CACHE_DIR / "VN30.csv"

# --- B2: momentum nhiều cửa sổ -----------------------------------------------------------------

# (window, skip); loại ô (21, 21) vì suy biến (ln(C_{t-21}/C_{t-21}) luôn bằng 0, xác nhận với
# người dùng khi giao việc 4.2) khỏi lưới window=[21,63,126] x skip=[0,21] của config.yaml.
MOMENTUM_GRID: list[tuple[int, int]] = [(21, 0), (63, 0), (63, 21), (126, 0), (126, 21)]


def _log_return(close: pd.Series, window: int) -> pd.Series:
    return np.log(close / close.shift(window))


def momentum_score(close: pd.Series, window: int, skip: int) -> pd.Series:
    """s(w, skip) = ln(C_{t-skip} / C_{t-window}): quy ước kinh điển bỏ qua `skip` phiên gần nhất
    (Jegadeesh-Titman), xác nhận với người dùng khi giao việc 4.2. `skip=0` cho đúng kết quả như
    `ret_w` thô ở `feature_registry.py` (trước khi chuẩn hóa chéo)."""
    if skip == 0:
        return _log_return(close, window)
    return np.log(close.shift(skip) / close.shift(window))


# --- B3: IMOM (momentum riêng mã, phần dư sau khi trừ yếu tố thị trường) -----------------------

IMOM_ESTIMATION_WINDOW = 126  # beta cuộn, config.yaml model_grid.baseline.B3, xác nhận 4.2
IMOM_GRID: list[tuple[int, int]] = [(63, 21), (126, 21)]  # (window, skip); skip=21 cố định


def imom_score(
    stock_close: pd.Series,
    stock_ret: pd.Series,
    market_close: pd.Series,
    market_ret: pd.Series,
    window: int,
    skip: int,
    estimation_window: int = IMOM_ESTIMATION_WINDOW,
) -> pd.Series:
    """IMOM = tổng phần dư log-return hàng ngày so với thị trường trên $[t-window, t-skip]$, dùng
    MỘT beta cuộn ($estimation\\_window$ phiên, tái dùng `rolling_beta`) tại mỗi ngày $t$ cho toàn
    bộ tổng đó: $\\sum_d (r_{i,d} - \\beta_t \\cdot r_{m,d})$. Vì lợi suất log cộng dồn theo đường
    thẳng (telescoping), tổng này bằng ĐÚNG (không xấp xỉ):

        IMOM_t = ln(C_{t-skip}/C_{t-window}) - beta_t * ln(Mkt_{t-skip}/Mkt_{t-window})

    tức động lượng riêng của mã trừ beta nhân động lượng thị trường cùng cửa sổ."""
    beta_t = rolling_beta(stock_ret, market_ret, window=estimation_window)
    stock_mom = momentum_score(stock_close, window, skip)
    market_mom = momentum_score(market_close, window, skip)
    return stock_mom - beta_t * market_mom


# --- B4: Amihud illiquidity ----------------------------------------------------------------

AMIHUD_GRID: list[int] = [21, 63]  # cửa sổ, config.yaml model_grid.baseline.B4


def amihud_illiquidity(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """ILLIQ (Amihud 2002), quy ước kinh điển toàn cầu (không phải số liệu quy định thị trường
    VN, cùng cách RSI/MACD ở `feature_registry.py`): trung bình $|$lợi suất ngày$|$ / giá trị
    giao dịch ngày, trên `window` phiên gần nhất. Điểm CAO hơn = kém thanh khoản hơn — đúng chiều
    B4 (điểm cao cho mã kém thanh khoản, `ma_tran_baseline_ablation.md` mục 2)."""
    daily_ret = _log_return(close, 1)
    dollar_volume = close * volume
    daily_illiq = daily_ret.abs() / dollar_volume
    return daily_illiq.rolling(window, min_periods=window).mean()


# --- B0: mốc đồng trọng số toàn U_t ------------------------------------------------------------


def equal_weight_portfolio_returns(labels: pd.DataFrame, universe: pd.DataFrame, h: int) -> pd.Series:
    """B0: lợi suất danh mục đồng trọng số toàn $U_t$ mỗi ngày quyết định, kỳ hạn `h`, dùng đúng
    cột `r` (lợi suất thô, quy ước `exit_price/entry_price - 1`, `labeling.py`) đã có trong
    `labels`. Không có điểm xếp hạng nên không tính Rank IC. Trả về Series index theo ngày."""
    merged = universe.merge(labels[labels["h"] == h], on=["date", "symbol"], how="inner")
    return merged.groupby("date")["r"].mean()


# --- B1: chỉ số VN30 (chỉ tham chiếu) -----------------------------------------------------------


def load_vn30_prices(start: str, end: str, cache_path: Path = VN30_CACHE_PATH) -> pd.DataFrame:
    """Tải và cache giá VN30 (cùng API Entrade/DNSE dùng cho VN-Index, `symbol_type="index"`,
    xác nhận có dữ liệu khi giao việc 4.2). Trả về cột date, close."""
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        df = fetch_ohlcv("VN30", start, end, symbol_type="index")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)[["date", "close"]]


def vn30_index_returns(vn30_prices: pd.DataFrame, h: int) -> pd.Series:
    """B1: lợi suất kỳ `h` của chỉ số VN30, cùng quy ước vào/thoát của hệ thống (đóng cửa $t+1$
    đến đóng cửa $t+1+h$, `config.yaml` mục `execution`), lợi suất thô (không phải log, cùng quy
    ước `r` của `labeling.py`). Chỉ để tham chiếu, không cùng universe (D56). Trả về Series index
    theo ngày quyết định $t$ (NaN ở các ngày cuối chuỗi không đủ $h+1$ phiên sau)."""
    close = vn30_prices["close"]
    entry = close.shift(-1)
    exit_ = close.shift(-(1 + h))
    ret = exit_ / entry - 1.0
    return pd.Series(ret.to_numpy(), index=vn30_prices["date"])


# --- Chọn cấu hình lưới trên validation, đánh giá một lần trên test -----------------------------


def _mean_rank_ic_on_split(
    scored: pd.DataFrame, labels_h: pd.DataFrame, splits: pd.DataFrame, split_name: str
) -> float:
    """Rank IC trung bình của `scored` (cột date, symbol, score) trên các ngày thuộc `split_name`,
    so với `r` của `labels_h` (đã lọc đúng `h`). Trung bình bỏ qua NaN (ngày không đủ quan sát)."""
    merged = scored.merge(labels_h[["date", "symbol", "r"]], on=["date", "symbol"], how="inner")
    merged = merged.merge(splits, on="date", how="inner")
    merged = merged[merged["split"] == split_name]
    if merged.empty:
        return float("nan")
    ic = rank_ic_series(merged["date"], merged["score"].to_numpy(), merged["r"].to_numpy())
    return float(ic.mean()) if len(ic) else float("nan")


def select_best_config(
    scores_by_config: dict, labels_h: pd.DataFrame, splits: pd.DataFrame
) -> tuple:
    """Chọn cấu hình (khóa của `scores_by_config`) có Rank IC trung bình cao nhất trên validation.
    Không bao giờ chọn trên test (đúng nguyên tắc dự án). Trả về (khóa đã chọn, Rank IC validation
    của mọi cấu hình để ghi log)."""
    val_ic = {
        key: _mean_rank_ic_on_split(scored, labels_h, splits, "validation")
        for key, scored in scores_by_config.items()
    }
    best_key = max(val_ic, key=lambda k: (val_ic[k] if val_ic[k] == val_ic[k] else -float("inf")))
    return best_key, val_ic


def evaluate_on_test(
    scored: pd.DataFrame, labels_h: pd.DataFrame, splits: pd.DataFrame, h: int, config: dict
) -> dict:
    """Rank IC/ICIR, Newey-West và bootstrap khối tròn trên TEST cho một cấu hình đã chọn (một
    lần duy nhất, sau khi chọn trên validation). `config`: `config.yaml` đã nạp, lấy
    `bootstrap_n`/`bootstrap_block_len_factor_h` (D60, không hardcode)."""
    merged = scored.merge(labels_h[["date", "symbol", "r"]], on=["date", "symbol"], how="inner")
    merged = merged.merge(splits, on="date", how="inner")
    test = merged[merged["split"] == "test"]
    ic = rank_ic_series(test["date"], test["score"].to_numpy(), test["r"].to_numpy())
    ic_stats = icir(ic)
    nw = newey_west_test(ic.to_numpy(), lag=h - 1)
    block_len = config["hypotheses"]["bootstrap_block_len_factor_h"] * h
    n_boot = config["hypotheses"]["bootstrap_n"]
    boot = circular_block_bootstrap_mean_ci(ic.to_numpy(), block_len=block_len, n_boot=n_boot)
    return {"n_days_test": ic_stats["n"], **ic_stats, **nw, "boot_ci_low": boot["ci_low"]}


if __name__ == "__main__":
    from vnrank.features.feature_registry import fill_market_calendar_gaps
    from vnrank.universe.build_universe import load_listed_stocks, load_price_panel

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = CODE_ROOT / "data" / "processed"
    universe = pd.read_csv(data_dir / "universe_daily.csv", parse_dates=["date"])
    labels = pd.read_csv(data_dir / "labels_daily.csv", parse_dates=["date"])
    splits = pd.read_csv(data_dir / "splits_daily.csv", parse_dates=["date"])

    stocks = load_listed_stocks()
    symbols = sorted(universe["symbol"].unique())
    panel = load_price_panel(symbols)
    panel = panel[panel["symbol"].isin(symbols)]

    market_raw = pd.read_csv(PRICE_CACHE_DIR / "VNINDEX.csv", parse_dates=["date"])
    calendar = sorted(panel["date"].unique())
    market = fill_market_calendar_gaps(market_raw, calendar)
    market_ret = _log_return(market["close"], 1)
    market_ret.index = market["date"]
    market_close = market.set_index("date")["close"]

    momentum_frames, imom_frames, amihud_frames = {}, {}, {}
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

    def to_long(frames_by_config: dict) -> dict:
        out = {}
        for key, by_symbol in frames_by_config.items():
            rows = []
            for symbol, s in by_symbol.items():
                rows.append(pd.DataFrame({"date": s.index, "symbol": symbol, "score": s.to_numpy()}))
            out[key] = pd.concat(rows, ignore_index=True)
        return out

    momentum_long = to_long(momentum_frames)
    imom_long = to_long(imom_frames)
    amihud_long = to_long(amihud_frames)

    results = []
    grid_log = []  # mọi ô lưới, kể cả ô không được chọn (ma_tran_baseline_ablation.md mục 1, nguyên tắc 4)
    for h in cfg["horizons"]:
        labels_h = labels[labels["h"] == h]

        for name, scores_by_config in (("B2_momentum", momentum_long), ("B3_imom", imom_long), ("B4_amihud", amihud_long)):
            best_key, val_ic = select_best_config(scores_by_config, labels_h, splits)
            for key, ic_val in val_ic.items():
                grid_log.append(
                    {"baseline": name, "h": h, "config": str(key), "rank_ic_validation": ic_val, "chon": key == best_key}
                )
            test_stats = evaluate_on_test(scores_by_config[best_key], labels_h, splits, h, cfg)
            results.append(
                {"baseline": name, "h": h, "chon_tren_validation": str(best_key), "rank_ic_validation": val_ic[best_key], **test_stats}
            )

        b0 = equal_weight_portfolio_returns(labels_h, universe, h)
        b0_splits = b0.rename("r").to_frame().join(splits.set_index("date"), how="inner")
        b0_test = b0_splits[b0_splits["split"] == "test"]["r"]
        results.append(
            {
                "baseline": "B0_dong_trong_so",
                "h": h,
                "n_days_test": len(b0_test),
                "mean": float(b0_test.mean()),
                "sharpe_annualized": float(b0_test.mean() / b0_test.std(ddof=1) * (250 / h) ** 0.5),
            }
        )

    vn30 = load_vn30_prices(start="2022-01-01", end=str(pd.Timestamp.today().date()))
    for h in cfg["horizons"]:
        vn30_ret = vn30_index_returns(vn30, h)
        vn30_splits = vn30_ret.rename("r").to_frame().join(splits.set_index("date"), how="inner")
        vn30_test = vn30_splits[vn30_splits["split"] == "test"]["r"].dropna()
        results.append(
            {
                "baseline": "B1_vn30",
                "h": h,
                "n_days_test": len(vn30_test),
                "mean": float(vn30_test.mean()),
                "sharpe_annualized": float(vn30_test.mean() / vn30_test.std(ddof=1) * (250 / h) ** 0.5),
            }
        )

    out_df = pd.DataFrame(results)
    out_path = data_dir / "baseline_results_4_2.csv"
    out_df.to_csv(out_path, index=False)

    grid_df = pd.DataFrame(grid_log)
    grid_path = data_dir / "baseline_grid_search_log_4_2.csv"
    grid_df.to_csv(grid_path, index=False)

    print(out_df.to_string())
    print(grid_df.to_string())
    print("Da luu ket qua va log luoi tim kiem vao code/data/processed/")
