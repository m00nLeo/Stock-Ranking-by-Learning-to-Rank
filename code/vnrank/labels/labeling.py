"""Tạo nhãn 5/10/21 ngày (bước 3.6; gộp luôn nội dung nhãn xu hướng triple-barrier của 3.10 trong
cùng phiên vì là một khối công việc tính nhãn, xem quyết định D78).

Mọi công thức đã chốt ở Giai đoạn 2, bước này chỉ lập trình, không quyết định tham số mới:
- Nhãn hồi quy, mức liên quan: `bai_toan_hinh_thuc.md` mục 4.1-4.3 (D55, D59).
- Nhãn xu hướng triple-barrier: `dinh_nghia_xu_huong.md` (D63): $k=1$, $\\hat\\sigma$ EWMA span 20 của
  lợi suất log ngày, $b = \\max(k\\hat\\sigma\\sqrt{h}, 0{,}60\\%)$, chỉ xét chạm rào bằng giá đóng cửa từ
  phiên $t+3$ (T+2, D61) đến $t+1+h$.
- Quy tắc khóa trần/sàn CHO NHÃN HỒI QUY (charter mục 7 dòng 3.6, D66): khóa trần ở phiên vào lệnh
  ($t+1$) = không mua được, bỏ mẫu; khóa sàn ở phiên thoát lệnh dự kiến ($t+1+h$) = chưa bán được, "giữ
  đến phiên bán được" tức dời ngày thoát tới phiên giao dịch thật gần nhất sau đó không bị khóa sàn.
- Nhãn xu hướng (bảng ở `dinh_nghia_xu_huong.md` mục 4) KHÔNG áp quy tắc dời ngày này: khóa trần/sàn giữa
  cửa sổ chỉ gắn cờ `khoa_tran_san`, không đổi cách xét chạm rào, vì nhãn xu hướng mô tả đường đi giá chứ
  không mô phỏng giao dịch thật (khả năng khớp lệnh để backtest xử lý riêng, không phải việc của bước này).
- Phiên không có giao dịch (tạm ngừng) giữa cửa sổ: không xét chạm ngày đó (coi như "chưa có gì xảy ra",
  cài đặt bằng cách lấy giá đóng cửa gần nhất — forward-fill — cho chuỗi dùng để xét chạm rào, giá không đổi
  nên không thể tạo chạm rào mới ở đúng ngày đó, tương đương "không xét"); nếu đến tận $t+1+h$ vẫn chưa có
  giao dịch thật, rào thời gian lấy theo giá đóng cửa gần nhất, gắn cờ `tam_ngung` (dinh_nghia_xu_huong.md
  mục 4).
- Mã đã hủy niêm yết giữa cửa sổ không xảy ra trong dữ liệu hiện có (universe chỉ còn mã đang niêm yết,
  D71), nên trường hợp đó trong `dinh_nghia_xu_huong.md` mục 4 không cần cài đặt ở đây.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vnrank.data.cleaning import detect_limit_lock

HORIZONS: tuple[int, ...] = (5, 10, 21)
TOUCH_START_OFFSET = 3  # D61: T+2, chỉ xét chạm rào từ phiên t+3
SIGMA_SPAN = 20  # D63
K_FIXED = 1  # D63
C_RT_BASE = 0.0060  # D61
RELEVANCE_QUANTILES = 5  # D59
MAX_SELL_EXTENSION_SESSIONS = 60  # giới hạn tìm phiên bán được, tránh tìm vô hạn nếu mã kẹt sàn quá lâu


def build_trading_calendar(panel: pd.DataFrame) -> list[pd.Timestamp]:
    """Lịch giao dịch tham chiếu: hợp nhất ngày có dữ liệu của mọi mã trong `panel`."""
    return sorted(panel["date"].unique())


def ewma_sigma(log_returns: pd.Series, span: int = SIGMA_SPAN) -> pd.Series:
    """Độ lệch chuẩn EWMA của lợi suất log ngày, chỉ dùng dữ liệu <= ngày đó."""
    return log_returns.ewm(span=span, min_periods=span).std()


def _resolve_sellable_exit(
    lock_by_date: dict, trade_dates: list[pd.Timestamp], start_pos: int
) -> tuple[pd.Timestamp | None, bool, int]:
    """Tìm phiên giao dịch thật đầu tiên từ vị trí `start_pos` (trong `trade_dates`) trở đi mà
    KHÔNG bị khóa sàn. `lock_by_date` là dict ngày -> hướng khóa ("tran"/"san"/None).
    Trả về (ngày thoát thật, đã_dời_ngày, vị trí tìm thấy). Trả về (None, True, -1) nếu không tìm
    được trong `MAX_SELL_EXTENSION_SESSIONS` phiên kể từ `start_pos`.
    """
    n = len(trade_dates)
    limit = min(start_pos + MAX_SELL_EXTENSION_SESSIONS, n)
    for pos in range(start_pos, limit):
        date = trade_dates[pos]
        if lock_by_date.get(date) != "san":
            return date, pos > start_pos, pos
    return None, True, -1


def compute_symbol_labels(
    symbol_df: pd.DataFrame, exchange: str, calendar: list[pd.Timestamp], horizons: tuple[int, ...] = HORIZONS
) -> pd.DataFrame:
    """Tính nhãn hồi quy và nhãn xu hướng cho một mã, mọi ngày quyết định $t$ có đủ dữ liệu tương lai.

    `symbol_df`: cột date, open, high, low, close, volume của một mã (đã sort theo ngày).
    `exchange`: "HOSE" hoặc "HNX", để chọn đúng biên độ cho `detect_limit_lock`.
    """
    df = symbol_df.sort_values("date").reset_index(drop=True)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    sigma = ewma_sigma(log_ret).to_numpy()

    locked = detect_limit_lock(df, san=exchange)
    lock_by_date: dict = dict(zip(locked["date"], locked["huong_khoa"]))
    close_by_date: dict = dict(zip(df["date"], df["close"]))

    ffilled_values = df.set_index("date")["close"].reindex(calendar).ffill().to_numpy()
    trade_dates = df["date"].tolist()
    trade_date_set = set(trade_dates)

    date_to_idx = {d: i for i, d in enumerate(calendar)}
    n_cal = len(calendar)

    # Vị trí (trong trade_dates) của phiên giao dịch thật đầu tiên >= mỗi ngày trong `calendar`,
    # tính một lần bằng searchsorted thay vì bisect lặp lại cho từng (t, h).
    cal_ns = np.array(calendar, dtype="datetime64[ns]")
    trade_ns = np.array(trade_dates, dtype="datetime64[ns]")
    trade_pos_for_cal_idx = np.searchsorted(trade_ns, cal_ns, side="left")

    rows: list[dict] = []
    for t in df["date"]:
        idx_t = date_to_idx.get(t)
        if idx_t is None:
            continue
        idx_entry = idx_t + 1
        if idx_entry >= n_cal:
            continue
        entry_date = calendar[idx_entry]
        if entry_date not in trade_date_set:
            continue  # phiên vào không có giao dịch: không tạo nhãn cho (i, t)
        if lock_by_date.get(entry_date) == "tran":
            continue  # khóa trần ở phiên vào: coi như không mua được
        entry_price = ffilled_values[idx_entry]
        sig = sigma[idx_t] if idx_t < len(sigma) else np.nan

        for h in horizons:
            idx_exit = idx_t + 1 + h
            if idx_exit >= n_cal:
                continue
            exit_date_nominal = calendar[idx_exit]

            # --- Nhãn xu hướng triple-barrier (dinh_nghia_xu_huong.md) ---
            y, t1, b, san_chi_phi, khoa_tran_san, tam_ngung_xu_huong = None, None, None, None, False, False
            if not np.isnan(sig):
                b = max(K_FIXED * sig * np.sqrt(h), C_RT_BASE)
                san_chi_phi = bool(K_FIXED * sig * np.sqrt(h) < C_RT_BASE)
                window_start = idx_t + TOUCH_START_OFFSET
                window_end = idx_exit + 1
                if window_end > window_start:
                    r_window = ffilled_values[window_start:window_end] / entry_price - 1.0
                    hit_up = r_window >= b
                    hit_down = r_window <= -b
                    pos_up = int(np.argmax(hit_up)) if hit_up.any() else None
                    pos_down = int(np.argmax(hit_down)) if hit_down.any() else None
                    if pos_up is not None and (pos_down is None or pos_up < pos_down):
                        y, t1 = 1, calendar[window_start + pos_up]
                    elif pos_down is not None and (pos_up is None or pos_down < pos_up):
                        y, t1 = -1, calendar[window_start + pos_down]
                    else:
                        y, t1 = 0, exit_date_nominal
                    khoa_tran_san = any(
                        lock_by_date.get(calendar[i]) in ("tran", "san") for i in range(window_start, window_end)
                    )
                    tam_ngung_xu_huong = exit_date_nominal not in trade_date_set

            # --- Nhãn hồi quy (bai_toan_hinh_thuc.md mục 4.1-4.3), có dời ngày thoát khi khóa sàn ---
            start_pos = int(trade_pos_for_cal_idx[idx_exit])
            resolved_exit, exit_extended, _ = _resolve_sellable_exit(lock_by_date, trade_dates, start_pos)
            if resolved_exit is None:
                r = None
            else:
                exit_price = close_by_date[resolved_exit]
                r = exit_price / entry_price - 1.0

            rows.append(
                {
                    "symbol": None,  # gán ở compute_raw_labels
                    "date": t,
                    "h": h,
                    "r": r,
                    "exit_date_resolved": resolved_exit,
                    "exit_extended_khoa_san": exit_extended if resolved_exit is not None else True,
                    "y": y,
                    "t1": t1,
                    "b": b,
                    "san_chi_phi": san_chi_phi,
                    "khoa_tran_san": khoa_tran_san,
                    "tam_ngung": tam_ngung_xu_huong,
                }
            )

    return pd.DataFrame(rows)


def compute_raw_labels(
    panel: pd.DataFrame, stocks: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS
) -> pd.DataFrame:
    """Tính nhãn thô (chưa trừ trung bình chéo, chưa chia phân vị) cho mọi mã trong `panel`."""
    calendar = build_trading_calendar(panel)
    exchange_by_symbol = dict(zip(stocks["symbol"], stocks["exchange"]))

    frames = []
    for symbol, group in panel.groupby("symbol"):
        exchange = exchange_by_symbol.get(symbol)
        if exchange is None:
            continue
        out = compute_symbol_labels(group, exchange, calendar, horizons)
        if out.empty:
            continue
        out["symbol"] = symbol
        # "Sàn giao dịch tại t" (dinh_nghia_xu_huong.md mục 7): dùng sàn hiện hành từ danh sách niêm
        # yết (mã_niêm_yết_HOSE_HNX.xlsx), KHÔNG theo dõi lịch sử chuyển sàn (dự án chưa có dữ liệu đó,
        # trường hợp mã chuyển sàn trong cửa sổ là hiếm, ghi nhận giới hạn này).
        out["exchange"] = exchange
        frames.append(out)

    if not frames:
        return pd.DataFrame(
            columns=[
                "symbol", "date", "h", "r", "exit_date_resolved", "exit_extended_khoa_san", "exchange",
                "y", "t1", "b", "san_chi_phi", "khoa_tran_san", "tam_ngung",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def add_excess_and_relevance(
    raw_labels: pd.DataFrame, universe: pd.DataFrame, quantiles: int = RELEVANCE_QUANTILES
) -> pd.DataFrame:
    """Giới hạn vào $U_t$, tính lợi suất vượt trung bình universe và mức liên quan theo phân vị.

    `universe`: cột date, symbol (bảng $U_t$ từ 3.4). Chỉ các dòng (date, symbol, h) có `r` hợp lệ
    (không None, tức có bán được) mới được dùng để tính trung bình chéo và phân vị của ngày đó.
    """
    merged = universe[["date", "symbol"]].merge(raw_labels, on=["date", "symbol"], how="left")
    valid = merged.dropna(subset=["r"]).copy()

    group = valid.groupby(["date", "h"])["r"]
    valid["r_excess"] = valid["r"] - group.transform("mean")

    def _quantile_bucket(s: pd.Series) -> pd.Series:
        n = len(s)
        if n < quantiles:
            return pd.Series([0] * n, index=s.index)
        rank = s.rank(method="first")
        return ((rank - 1) * quantiles // n).astype(int).clip(upper=quantiles - 1)

    valid["relevance"] = valid.groupby(["date", "h"])["r_excess"].transform(_quantile_bucket)

    merged = merged.merge(
        valid[["date", "symbol", "h", "r_excess", "relevance"]], on=["date", "symbol", "h"], how="left"
    )
    return merged
