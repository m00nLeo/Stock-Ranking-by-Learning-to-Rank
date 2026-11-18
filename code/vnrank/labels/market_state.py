"""Trạng thái xu hướng thị trường theo ngày (bước 3.11, D63 mục 6): luật SMA50/SMA200 trên
giá đóng cửa VN-Index, chỉ dùng dữ liệu <= t. Chỉ để chia kết quả RQ3, không làm đặc trưng
hay chọn mô hình.
"""

from __future__ import annotations

import pandas as pd


def compute_market_state(
    vnindex_df: pd.DataFrame, sma_short: int = 50, sma_long: int = 200
) -> pd.DataFrame:
    """Phân loại trạng thái thị trường cho từng ngày có đủ cả hai SMA theo luật D63 mục 6.

    `vnindex_df`: cột date, open, high, low, close, volume của VN-Index.
    Trả về DataFrame hai cột date, state ("tang"/"giam"/"di_ngang"; None cho ngày chưa đủ lịch sử).
    """
    df = vnindex_df.sort_values("date").reset_index(drop=True)
    close = df["close"]
    sma_s = close.rolling(window=sma_short, min_periods=sma_short).mean()
    sma_l = close.rolling(window=sma_long, min_periods=sma_long).mean()

    state = pd.Series(None, index=df.index, dtype=object)
    has_both = sma_s.notna() & sma_l.notna()
    tang = has_both & (close > sma_l) & (sma_s > sma_l)
    giam = has_both & (close < sma_l) & (sma_s < sma_l)
    state[tang] = "tang"
    state[giam] = "giam"
    state[has_both & ~(tang | giam)] = "di_ngang"

    return pd.DataFrame({"date": df["date"], "state": state})


if __name__ == "__main__":
    # Chạy cục bộ một lần để tính trạng thái thị trường theo ngày từ VN-Index đã cache (3.2, D66).
    from pathlib import Path

    code_root = Path(__file__).resolve().parents[2]
    vnindex = pd.read_csv(code_root / "data" / "raw" / "prices" / "VNINDEX.csv", parse_dates=["date"])
    out = compute_market_state(vnindex)
    out_path = code_root / "data" / "processed" / "market_state_daily.csv"
    out.to_csv(out_path, index=False)
    print(f"Đã ghi {len(out)} dòng vào {out_path}")