"""Dựng universe theo từng thời điểm (bước 3.4).

Ba việc của bước này (xem `bai_toan_hinh_thuc.md` mục 2, `giao_thuc_chia_du_lieu_bias.md`
rủi ro #2, #3, #6, #12, #13):
1. Dựng $U_t$ theo từng ngày: gộp HOSE và HNX, lọc cổ phiếu giá thấp (penny) và kém thanh khoản,
   giữ N mã thanh khoản cao nhất, 50 <= N <= 150; N và các ngưỡng lọc CHỈ chọn từ phân phối dữ liệu
   trên train và validation (D62: train 01/2023-06/2024, validation 07/2024-03/2025), không chạm test.
2. Lát cắt báo cáo VN30 theo từng kỳ xét duyệt, dùng `load_vn30_membership` + `vn30_members_asof`
   (không dùng danh sách VN30 hiện tại cho quá khứ).
3. Ước lượng số mã bị thiếu do universe không giữ được mã đã hủy niêm yết (D71) — xem
   `estimate_delisted_undercount`.

Nguồn danh sách mã: `mã_niêm_yết_HOSE_HNX.xlsx` (người dùng cung cấp 04/11/2026, từ
finance.vietstock.vn/truy-xuat-du-lieu/tong-hop-doanh-nghiep.htm), loại các mã quỹ ETF/quỹ đầu tư
(tên có "Quỹ", mã dài 8 ký tự, không phải cổ phiếu phổ thông — 24 mã, còn lại 707 mã: 408 HOSE, 299 HNX).
Nguồn VN30 theo kỳ: `Thành_viên_VN30.csv` (người dùng cung cấp 04/11/2026, 8 kỳ 01/2023-07/2026,
30 mã/kỳ, mỗi kỳ có nguồn và liên kết: bản PDF công bố của HOSE hoặc công ty chứng khoán/báo tài
chính dẫn lại nguyên văn thông báo của HOSE). Bản gốc của hai file này nằm ở
`03_Giai_doan_2_Cu_the_hoa/8_He_thong_minh_hoa/` (ngoài `code/`, không lên GitHub); bản dùng cho
pipeline được sao chép vào `code/data/raw/universe_inputs/` để package tự chứa trong `code/`
(chạy được trên Colab chỉ với repo GitHub, không cần toàn bộ thư mục dự án).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from vnrank.data.entrade_client import fetch_ohlcv

CODE_ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_INPUT_DIR = CODE_ROOT / "data" / "raw" / "universe_inputs"
LISTING_XLSX = UNIVERSE_INPUT_DIR / "mã_niêm_yết_HOSE_HNX.xlsx"
VN30_CSV = UNIVERSE_INPUT_DIR / "Thành_viên_VN30.csv"
PRICE_CACHE_DIR = CODE_ROOT / "data" / "raw" / "prices"


# --- 1a. Danh sách mã niêm yết HOSE + HNX --------------------------------------------------


def load_listed_stocks(path: Path = LISTING_XLSX) -> pd.DataFrame:
    """Đọc danh sách mã từ file Vietstock, loại mã quỹ ETF/quỹ đầu tư.

    File có 5 dòng tiêu đề trang trí trước dòng cột thật (dòng chỉ số 5: STT, Mã CK, Tên công ty,
    Sàn CK). Mã quỹ có mã dài 8 ký tự (ví dụ FUEVFVND, E1VFVN30); cổ phiếu phổ thông luôn 3 ký tự
    trong danh sách này. Trả về DataFrame cột `symbol`, `exchange`.
    """
    raw = pd.read_excel(path, sheet_name="Sheet1", header=None)
    header_row_idx = raw.index[raw.iloc[:, 1].astype(str) == "Mã CK"][0]
    header_row = raw.iloc[header_row_idx].tolist()
    df = raw.iloc[header_row_idx + 1 :].copy()
    df.columns = header_row
    df = df.dropna(subset=["Mã CK"])
    df = df[df["Mã CK"].astype(str).str.len() == 3]
    df = df[["Mã CK", "Sàn CK"]].rename(columns={"Mã CK": "symbol", "Sàn CK": "exchange"})
    df = df[df["exchange"].isin(["HOSE", "HNX"])].reset_index(drop=True)
    return df


# --- 1b. Tải và cache giá OHLCV cho toàn bộ danh sách --------------------------------------


def fetch_price_cache(
    symbols: list[str],
    start: str,
    end: str,
    out_dir: Path = PRICE_CACHE_DIR,
    max_workers: int = 6,
) -> dict[str, str]:
    """Tải OHLCV cho từng mã, lưu CSV riêng ở `out_dir/<MA>.csv`.

    Bỏ qua mã đã có file (cho phép chạy lại/tiếp tục khi bị ngắt giữa chừng). Dùng
    ThreadPoolExecutor nhỏ để không gửi quá nhiều yêu cầu đồng thời tới máy chủ công khai
    Entrade/DNSE (không giả lập trình duyệt, không vượt rào chống bot, theo bài học Run 38).
    Trả về dict symbol -> trạng thái ("ok", "skip", hoặc "error: ...").
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}

    def _one(symbol: str) -> tuple[str, str]:
        out_path = out_dir / f"{symbol}.csv"
        if out_path.exists():
            return symbol, "skip"
        try:
            df = fetch_ohlcv(symbol, start, end, symbol_type="stock")
            df.to_csv(out_path, index=False)
            return symbol, "ok" if not df.empty else "ok_empty"
        except Exception as exc:  # noqa: BLE001 - ghi lỗi, không dừng cả batch
            return symbol, f"error: {exc}"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_one, s): s for s in symbols}
        for fut in as_completed(futures):
            symbol, result = fut.result()
            status[symbol] = result
    return status


def load_price_panel(symbols: list[str], cache_dir: Path = PRICE_CACHE_DIR) -> pd.DataFrame:
    """Gộp các file CSV đã cache thành một bảng dài: symbol, date, open, high, low, close, volume."""
    frames = []
    for symbol in symbols:
        path = cache_dir / f"{symbol}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["symbol"] = symbol
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


# --- 2. VN30 theo kỳ -------------------------------------------------------------------------


def load_vn30_membership(path: Path = VN30_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["announcement_date", "effective_date"])
    return df


def vn30_members_asof(vn30_df: pd.DataFrame, date: pd.Timestamp) -> set[str]:
    """Trả về tập 30 mã VN30 có hiệu lực tại `date` (kỳ gần nhất có effective_date <= date).

    Nếu `date` trước effective_date của kỳ đầu tiên (02/2023), trả về tập rỗng: dữ liệu VN30
    không có trước kỳ đó.
    """
    date = pd.Timestamp(date)
    eligible = vn30_df[vn30_df["effective_date"] <= date]
    if eligible.empty:
        return set()
    latest_period = eligible.sort_values("effective_date")["review_period"].iloc[-1]
    return set(vn30_df.loc[vn30_df["review_period"] == latest_period, "ticker"])


# --- 3. Ước lượng số mã thiếu do bỏ mã đã hủy niêm yết ---------------------------------------


def estimate_delisted_undercount(
    confirmed_delisted_count: int,
    plausible_liquid_fraction: tuple[float, float],
) -> dict[str, float]:
    """Ước lượng khoảng số mã có thể đã lọt vào $U_t$ nếu còn dữ liệu, dựa trên số mã hủy niêm
    yết đã xác nhận trên HOSE+HNX trong giai đoạn mẫu và một khoảng tỷ lệ mã đủ thanh khoản
    hợp lý (tự ước lượng, không có cách đo trực tiếp vì không còn dữ liệu giá của các mã này).

    Trả về {"so_ma_huy_niem_yet": ..., "uoc_luong_thap": ..., "uoc_luong_cao": ...}.
    """
    low_frac, high_frac = plausible_liquid_fraction
    return {
        "so_ma_huy_niem_yet": confirmed_delisted_count,
        "uoc_luong_thap": confirmed_delisted_count * low_frac,
        "uoc_luong_cao": confirmed_delisted_count * high_frac,
    }


# --- 1c. Dựng U_t theo ngày -------------------------------------------------------------------


def compute_liquidity_features(
    panel: pd.DataFrame, window_l: int, lag_sessions: int
) -> pd.DataFrame:
    """Thêm các đặc trưng thanh khoản dùng để lọc, tính riêng theo từng mã, chỉ dùng dữ liệu
    đến ngày hiện tại (không rò rỉ tương lai):
    - `traded_value`: close * volume của phiên đó.
    - `avg_value_l`: giá trị giao dịch trung bình `window_l` phiên gần nhất, đã dịch `lag_sessions`
      phiên (độ trễ ℓ) để tại ngày quyết định t chỉ dùng dữ liệu đến t - ℓ.
    - `active_sessions_l`: số phiên có volume > 0 trong `window_l` phiên gần nhất, cùng độ trễ.
    - `lagged_close`: giá đóng cửa tại t - ℓ, dùng cho lọc giá thấp (penny).
    """
    out = panel.sort_values(["symbol", "date"]).copy()
    out["traded_value"] = out["close"] * out["volume"]
    grouped = out.groupby("symbol", group_keys=False)
    out["avg_value_l"] = grouped["traded_value"].apply(
        lambda s: s.rolling(window_l, min_periods=window_l).mean().shift(lag_sessions)
    )
    out["active_sessions_l"] = grouped["volume"].apply(
        lambda s: (s > 0).rolling(window_l, min_periods=window_l).sum().shift(lag_sessions)
    )
    out["lagged_close"] = grouped["close"].apply(lambda s: s.shift(lag_sessions))
    return out


def build_daily_universe(
    features: pd.DataFrame,
    listed: pd.DataFrame,
    p_min: float,
    v_min: float,
    n_min: int,
    n_size: int,
) -> pd.DataFrame:
    """Dựng $U_t$ cho từng ngày theo đúng ba bước của `bai_toan_hinh_thuc.md` mục 2:
    gộp hai sàn (đã gộp sẵn ở `features`/`listed`), lọc penny và kém thanh khoản, rồi giữ
    `n_size` mã có `avg_value_l` cao nhất trong số còn lại.

    Trả về bảng dài: date, symbol (các mã thuộc $U_t$ của ngày đó).
    """
    df = features.dropna(subset=["avg_value_l", "active_sessions_l", "lagged_close"]).copy()
    eligible = df[
        (df["lagged_close"] >= p_min)
        & (df["avg_value_l"] >= v_min)
        & (df["active_sessions_l"] >= n_min)
    ].copy()
    eligible["_rank_in_day"] = eligible.groupby("date")["avg_value_l"].rank(
        method="first", ascending=False
    )
    universe = eligible[eligible["_rank_in_day"] <= n_size]
    return universe[["date", "symbol"]].reset_index(drop=True)


if __name__ == "__main__":
    # Chạy cục bộ một lần để tải và cache giá cho toàn bộ danh sách (bước 3.4). Có thể dừng và
    # chạy lại: các mã đã có file trong data/raw/prices/ sẽ được bỏ qua.
    stocks = load_listed_stocks()
    print(f"Tong so ma (da loai quy ETF): {len(stocks)}")
    print(stocks["exchange"].value_counts().to_string())
    t0 = time.time()
    result = fetch_price_cache(stocks["symbol"].tolist(), start="2022-01-01", end="2026-11-04")
    ok = sum(1 for v in result.values() if v == "ok")
    skip = sum(1 for v in result.values() if v == "skip")
    empty = sum(1 for v in result.values() if v == "ok_empty")
    errors = {k: v for k, v in result.items() if v.startswith("error")}
    print(f"ok={ok} skip={skip} empty={empty} errors={len(errors)} thoi_gian={time.time() - t0:.0f}s")
    if errors:
        print("Loi:", errors)
