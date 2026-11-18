"""Sổ đăng ký feature (bước 3.5): tên, công thức, cửa sổ nhìn lại, độ trễ cho từng đặc trưng
dùng trong mô hình (xem `bai_toan_hinh_thuc.md` mục 3, charter.md mục 4: nhóm giá và lợi suất quá
khứ, SMA, RSI, MACD, khối lượng, thanh khoản, một số đặc trưng chéo mã đơn giản).

Quyết định tham số (D76, 06/11/2026):
- Độ trễ 0 phiên cho MỌI đặc trưng: dữ liệu OHLCV của phiên t đã là thông tin biết được khi đóng
  cửa phiên t (charter mục 5 điều 1), giống lý do chọn ell = 0 ở 3.4 (xem
  `code/docs/3_4_universe.md` mục 1.2). Không cần trễ thêm.
- Cửa sổ nhìn lại cho lợi suất quá khứ và tỷ lệ so với SMA dùng lại ĐÚNG ba cửa sổ đã chốt cho họ
  baseline momentum B2 (21, 63, 126 phiên, D59, `config.yaml` mục model_grid.baseline.B2), để
  không phát sinh một bộ cửa sổ mới không có căn cứ (cần và đủ, D63).
- RSI(14) và MACD(12, 26, 9) dùng đúng quy ước kinh điển toàn cầu của hai chỉ báo này (không phải
  số liệu quy định thị trường VN, nên không cần nguồn chính thức như charter mục 5 điều 7 yêu cầu
  cho phí, thuế, biên độ).
- Thanh khoản tái dùng `vnrank.universe.build_universe.compute_liquidity_features` (L = 21, ell = 0,
  đã chốt ở 3.4) thay vì tính lại, tránh trùng lặp logic.
- Đặc trưng chéo mã đơn giản: beta cuộn 63 phiên so với VN-Index (dự án chưa có dữ liệu cơ bản để
  làm đặc trưng chéo mã phức tạp hơn).
- Mọi đặc trưng cuối cùng đều qua chuẩn hóa thứ hạng chéo ngày về [-1, 1] (mục 3 `bai_toan_hinh_thuc.md`),
  nên phép biến đổi đơn điệu (ví dụ lấy log) trên một đặc trưng không làm thay đổi kết quả cuối; vì
  vậy module này không lấy log cho thanh khoản hay khối lượng.
"""

from __future__ import annotations

import pandas as pd

from vnrank.universe.build_universe import compute_liquidity_features

# --- Sổ đăng ký: tên, nhóm, công thức, cửa sổ, độ trễ (tài liệu, dùng để đối chiếu) ---
FEATURE_REGISTRY: list[dict] = [
    {"ten": "ret_21", "nhom": "loi_suat_qua_khu", "cong_thuc": "ln(C_t / C_{t-21})", "cua_so": 21, "do_tre": 0},
    {"ten": "ret_63", "nhom": "loi_suat_qua_khu", "cong_thuc": "ln(C_t / C_{t-63})", "cua_so": 63, "do_tre": 0},
    {"ten": "ret_126", "nhom": "loi_suat_qua_khu", "cong_thuc": "ln(C_t / C_{t-126})", "cua_so": 126, "do_tre": 0},
    {"ten": "sma_ratio_21", "nhom": "sma", "cong_thuc": "C_t / SMA_21(t) - 1", "cua_so": 21, "do_tre": 0},
    {"ten": "sma_ratio_63", "nhom": "sma", "cong_thuc": "C_t / SMA_63(t) - 1", "cua_so": 63, "do_tre": 0},
    {"ten": "sma_ratio_126", "nhom": "sma", "cong_thuc": "C_t / SMA_126(t) - 1", "cua_so": 126, "do_tre": 0},
    {"ten": "rsi_14", "nhom": "rsi", "cong_thuc": "RSI chuẩn Wilder, 14 phiên", "cua_so": 14, "do_tre": 0},
    {"ten": "macd_hist", "nhom": "macd", "cong_thuc": "EMA_12 - EMA_26 - EMA_9(EMA_12 - EMA_26)", "cua_so": "12,26,9", "do_tre": 0},
    {"ten": "volume_ratio_21", "nhom": "khoi_luong", "cong_thuc": "volume_t / mean(volume_{t-21..t-1}) - 1", "cua_so": 21, "do_tre": 0},
    {"ten": "avg_value_l", "nhom": "thanh_khoan", "cong_thuc": "giá trị giao dịch trung bình 21 phiên (tái dùng 3.4)", "cua_so": 21, "do_tre": 0},
    {"ten": "beta_63", "nhom": "cheo_ma", "cong_thuc": "beta cuộn 63 phiên so với lợi suất log VN-Index", "cua_so": 63, "do_tre": 0},
]

FEATURE_COLUMNS: list[str] = [item["ten"] for item in FEATURE_REGISTRY]


def _log_return(close: pd.Series, window: int) -> pd.Series:
    import numpy as np

    return np.log(close / close.shift(window))


def past_return(df: pd.DataFrame, window: int) -> pd.Series:
    """Lợi suất log tích lũy `window` phiên, kết thúc tại mỗi ngày (NaN nếu chưa đủ `window` phiên)."""
    return _log_return(df["close"], window)


def sma_ratio(df: pd.DataFrame, window: int) -> pd.Series:
    """Tỷ lệ giá đóng cửa so với đường trung bình động giản đơn `window` phiên, trừ 1."""
    sma = df["close"].rolling(window, min_periods=window).mean()
    return df["close"] / sma - 1


def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """RSI chuẩn Wilder: trung bình tăng và giảm làm mượt bằng EMA hệ số 1/window."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd_hist(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Histogram MACD chuẩn: EMA nhanh trừ EMA chậm, trừ tiếp đường tín hiệu (EMA của hiệu trên)."""
    ema_fast = df["close"].ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    return macd_line - signal_line


def volume_ratio(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Tỷ lệ khối lượng phiên hiện tại so với khối lượng trung bình `window` phiên LIỀN TRƯỚC đó
    (không gồm chính phiên hiện tại, để đo mức bất thường so với mức bình thường gần đây), trừ 1."""
    avg_vol = df["volume"].shift(1).rolling(window, min_periods=window).mean()
    return df["volume"] / avg_vol - 1


def rolling_beta(stock_ret: pd.Series, market_ret: pd.Series, window: int = 63) -> pd.Series:
    """Beta cuộn `window` phiên: Cov(r_mã, r_thị trường) / Var(r_thị trường), cùng chỉ số ngày."""
    cov = (stock_ret * market_ret).rolling(window, min_periods=window).mean() - (
        stock_ret.rolling(window, min_periods=window).mean()
        * market_ret.rolling(window, min_periods=window).mean()
    )
    var = (market_ret**2).rolling(window, min_periods=window).mean() - (
        market_ret.rolling(window, min_periods=window).mean() ** 2
    )
    return cov / var


def fill_market_calendar_gaps(market_df: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    """Dặm vào lịch giao dịch đầy đủ và lùi giá đóng cửa (forward-fill) cho các phiên VNINDEX bị
    thiếu do lỗi hụt dữ liệu của nguồn Entrade/DNSE (xác nhận ở 3.3: 07/04/2023, 12/08/2024 là phiên
    có thật, không phải thị trường nghỉ). Forward-fill chỉ dùng giá đã biết trong quá khứ, không rò
    rỉ thông tin tương lai."""
    out = market_df.set_index("date").reindex(calendar)
    out["close"] = out["close"].ffill()
    out.index.name = "date"
    return out.reset_index()


def compute_features(panel: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """Tính toàn bộ đặc trưng thô (chưa chuẩn hóa chéo) cho mỗi mã trong `panel`.

    `panel`: cột symbol, date, open, high, low, close, volume (ví dụ từ
    `vnrank.universe.build_universe.load_price_panel`).
    `market_df`: cột date, close của VN-Index, đã lấp khoảng trống lịch bằng `fill_market_calendar_gaps`.

    Trả về bảng dài: symbol, date, cộng các cột trong `FEATURE_COLUMNS`. Mọi đặc trưng chỉ dùng dữ
    liệu <= ngày đó (độ trễ 0, xem ghi chú đầu file).
    """
    market = market_df.sort_values("date").copy()
    market["mkt_ret"] = _log_return(market["close"], 1)
    market_ret = market.set_index("date")["mkt_ret"]

    liquidity = compute_liquidity_features(panel, window_l=21, lag_sessions=0)[
        ["symbol", "date", "avg_value_l"]
    ]

    frames = []
    for symbol, group in panel.sort_values(["symbol", "date"]).groupby("symbol"):
        g = group.reset_index(drop=True)
        stock_ret = _log_return(g["close"], 1)
        mkt_ret_aligned = g["date"].map(market_ret)
        out = pd.DataFrame({"symbol": symbol, "date": g["date"]})
        out["ret_21"] = past_return(g, 21)
        out["ret_63"] = past_return(g, 63)
        out["ret_126"] = past_return(g, 126)
        out["sma_ratio_21"] = sma_ratio(g, 21)
        out["sma_ratio_63"] = sma_ratio(g, 63)
        out["sma_ratio_126"] = sma_ratio(g, 126)
        out["rsi_14"] = rsi(g, 14)
        out["macd_hist"] = macd_hist(g)
        out["volume_ratio_21"] = volume_ratio(g, 21)
        out["beta_63"] = rolling_beta(stock_ret, mkt_ret_aligned, 63)
        frames.append(out)

    features = pd.concat(frames, ignore_index=True)
    features = features.merge(liquidity, on=["symbol", "date"], how="left")
    return features[["symbol", "date"] + FEATURE_COLUMNS]


def cross_sectional_normalize(
    features: pd.DataFrame, universe: pd.DataFrame, feature_cols: list[str] = FEATURE_COLUMNS
) -> pd.DataFrame:
    """Chuẩn hóa chéo theo từng ngày (mục 3 `bai_toan_hinh_thuc.md`): với mỗi ngày, giới hạn vào
    $U_t$ (bảng `universe`: cột date, symbol), thay giá trị thiếu bằng trung vị chéo mã của ngày đó,
    rồi đổi mỗi đặc trưng thành thứ hạng trong $U_t$ và đưa về [-1, 1]:
    $z = 2 (rank - 1) / (n - 1) - 1$, rank trung bình khi có giá trị trùng nhau.

    Nếu một ngày không có quan sát hợp lệ nào cho một đặc trưng (toàn bộ $U_t$ đều thiếu, ví dụ ngày
    VN-Index bị hụt dữ liệu chưa lấp được), đặc trưng đó được gán 0 (điểm giữa thang [-1, 1]) cho cả
    ngày, vì không có trung vị chéo mã nào để thay thế.
    """
    merged = universe[["date", "symbol"]].merge(features, on=["date", "symbol"], how="left")
    out = merged[["date", "symbol"]].copy()

    for col in feature_cols:
        grouped = merged.groupby("date")[col]
        median_by_day = grouped.transform("median")
        filled = merged[col].fillna(median_by_day)
        rank = filled.groupby(merged["date"]).rank(method="average")
        n = filled.groupby(merged["date"]).transform("count")
        normalized = (2 * (rank - 1) / (n - 1) - 1).where(n > 1, 0.0)
        out[col] = normalized.fillna(0.0)

    return out


if __name__ == "__main__":
    # Chạy cục bộ một lần để tính feature cho toàn bộ universe đã dựng ở 3.4, dùng dữ liệu đã cache.
    from pathlib import Path

    from vnrank.universe.build_universe import PRICE_CACHE_DIR, load_listed_stocks, load_price_panel

    code_root = Path(__file__).resolve().parents[2]
    universe = pd.read_csv(code_root / "data" / "processed" / "universe_daily.csv", parse_dates=["date"])

    stocks = load_listed_stocks()
    panel = load_price_panel(stocks["symbol"].tolist())
    panel = panel[panel["symbol"].isin(universe["symbol"].unique())]

    market_raw = pd.read_csv(PRICE_CACHE_DIR / "VNINDEX.csv", parse_dates=["date"])
    calendar = sorted(panel["date"].unique())
    market = fill_market_calendar_gaps(market_raw, calendar)

    raw_features = compute_features(panel, market)
    normalized = cross_sectional_normalize(raw_features, universe)

    out_path = code_root / "data" / "processed" / "features_daily.csv"
    normalized.to_csv(out_path, index=False)
    print(f"Da tinh feature cho {normalized['symbol'].nunique()} ma, {normalized['date'].nunique()} ngay")
    print(f"Luu vao {out_path}")
    print(normalized[FEATURE_COLUMNS].describe().T[["min", "max", "mean"]])
