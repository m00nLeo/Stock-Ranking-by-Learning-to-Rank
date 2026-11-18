"""Phân tích theo nhóm ngành (mở rộng hệ thống minh họa, D91, D92): cho một ngày `as_of`, với mỗi
mã trong `sector_mapping`, tính độ ổn định (nghịch đảo độ lệch chuẩn lợi suất log gần đây) và xu
hướng gần nhất, rồi xếp hạng TRONG TỪNG NGÀNH — trả lời đúng yêu cầu "mã nào ổn định/cao nhất kèm
xu hướng, lọc theo ngành". Chỉ dùng dữ liệu `<= as_of` (point-in-time, cùng nguyên tắc toàn dự
án).

**Dữ liệu ngành (D92):** người dùng cung cấp 11/11/2026, 11 file Excel Vietstock
"Cổ phiếu theo Ngành <tên ngành>.xlsx" (chuẩn GICS 11 ngành, cùng khuôn mẫu file danh sách mã đã
dùng ở 3.4), sao chép vào `code/data/raw/sector_inputs/`. `build_sector_mapping_from_vietstock`
đọc và gộp 11 file này, loại 24 mã quỹ ETF (cùng cách `load_listed_stocks` ở `build_universe.py`),
còn lại khớp TUYỆT ĐỐI 707 mã trong danh sách niêm yết đã dùng xuyên suốt dự án (707 = 731 - 24,
408 HOSE + 299 HNX cả hai nguồn) — đã xác nhận bằng đối chiếu trực tiếp, không phải trùng hợp.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
SECTOR_INPUT_DIR = CODE_ROOT / "data" / "raw" / "sector_inputs"


def load_sector_mapping(path: str) -> pd.DataFrame:
    """Đọc file ánh xạ mã - ngành đã xử lý sẵn (CSV cột `symbol`, `sector`)."""
    return pd.read_csv(path)


def build_sector_mapping_from_vietstock(input_dir: Path = SECTOR_INPUT_DIR) -> pd.DataFrame:
    """Đọc toàn bộ file `Cổ phiếu theo Ngành *.xlsx` (định dạng xuất Vietstock, cùng khuôn mẫu
    file danh sách mã của 3.4: vài dòng trang trí, dòng tiêu đề thật có cột "Mã CK") trong
    `input_dir`, gộp thành một bảng `symbol`, `exchange`, `sector`. Tên ngành lấy từ tên file (bỏ
    tiền tố "Cổ phiếu theo Ngành "). Loại các mã quỹ ETF/quỹ đầu tư (mã dài khác 3 ký tự, cùng quy
    ước `load_listed_stocks`)."""
    frames = []
    for path in sorted(Path(input_dir).glob("Cổ phiếu theo Ngành *.xlsx")):
        raw = pd.read_excel(path, sheet_name="Sheet1", header=None)
        header_row_idx = raw.index[raw.iloc[:, 1].astype(str) == "Mã CK"][0]
        header_row = raw.iloc[header_row_idx].tolist()
        df = raw.iloc[header_row_idx + 1 :].copy()
        df.columns = header_row
        df = df.dropna(subset=["Mã CK"])
        df = df[df["Mã CK"].astype(str).str.len() == 3]
        sector = path.stem.replace("Cổ phiếu theo Ngành ", "")
        frames.append(
            df[["Mã CK", "Sàn CK"]].rename(columns={"Mã CK": "symbol", "Sàn CK": "exchange"}).assign(sector=sector)
        )
    return pd.concat(frames, ignore_index=True).sort_values("symbol").reset_index(drop=True)


if __name__ == "__main__":
    mapping = build_sector_mapping_from_vietstock()
    out_path = CODE_ROOT / "data" / "processed" / "sector_mapping.csv"
    mapping.to_csv(out_path, index=False)
    print(f"Đã ghi {len(mapping)} mã vào {out_path}")
    print(mapping["sector"].value_counts())


def _log_return(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def stability_and_trend_by_symbol(
    panel: pd.DataFrame, as_of: pd.Timestamp, lookback_sessions: int = 63
) -> pd.DataFrame:
    """Với mỗi mã trong `panel` (cột `symbol`, `date`, `close`), chỉ dùng dữ liệu `date <= as_of`,
    tính trên `lookback_sessions` phiên gần nhất tính đến `as_of`:
    - `stability_score`: 1 / độ lệch chuẩn lợi suất log (càng cao càng ổn định; NaN nếu độ lệch
      chuẩn bằng 0 hoặc chưa đủ phiên).
    - `trend`: dấu lợi suất log tích lũy trong cùng cửa sổ ("tang", "giam", "di_ngang" nếu gần 0).

    Trả về DataFrame cột `symbol`, `stability_score`, `trend`, `cumulative_return`.
    """
    as_of = pd.Timestamp(as_of)
    rows = []
    for symbol, group in panel[panel["date"] <= as_of].groupby("symbol"):
        g = group.sort_values("date").tail(lookback_sessions)
        if len(g) < lookback_sessions:
            continue
        rets = _log_return(g["close"]).dropna()
        std = rets.std(ddof=1)
        cum_ret = float(np.log(g["close"].iloc[-1] / g["close"].iloc[0]))
        trend = "tang" if cum_ret > 0.01 else "giam" if cum_ret < -0.01 else "di_ngang"
        rows.append(
            {
                "symbol": symbol,
                "stability_score": float(1.0 / std) if std and std > 0 else float("nan"),
                "trend": trend,
                "cumulative_return": cum_ret,
            }
        )
    return pd.DataFrame(rows)


def sector_ranking(
    panel: pd.DataFrame,
    sector_mapping: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_sessions: int = 63,
) -> pd.DataFrame:
    """Gộp `stability_and_trend_by_symbol` với `sector_mapping`, xếp hạng độ ổn định TRONG TỪNG
    NGÀNH (hạng 1 = ổn định nhất ngành đó). Trả về cột `symbol`, `sector`, `stability_score`,
    `trend`, `cumulative_return`, `rank_in_sector`. Mã không có trong `sector_mapping` hoặc chưa
    đủ lịch sử bị bỏ qua."""
    stats = stability_and_trend_by_symbol(panel, as_of, lookback_sessions)
    merged = sector_mapping.merge(stats, on="symbol", how="inner").dropna(subset=["stability_score"])
    merged["rank_in_sector"] = merged.groupby("sector")["stability_score"].rank(
        ascending=False, method="first"
    )
    return merged.sort_values(["sector", "rank_in_sector"]).reset_index(drop=True)


def filter_by_sector(ranking: pd.DataFrame, sector: str) -> pd.DataFrame:
    """Lọc bảng đã xếp hạng (`sector_ranking`) còn đúng một ngành, giữ nguyên thứ tự ổn định."""
    return ranking[ranking["sector"] == sector].reset_index(drop=True)
