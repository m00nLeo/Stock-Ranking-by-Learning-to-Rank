"""Làm sạch dữ liệu giá từ entrade_client (bước 3.3).

Phát hiện khi viết module này (xem code/docs/3_3_lam_sach_du_lieu.md):
- Giá trả về bởi API Entrade/DNSE là giá ĐÃ ĐIỀU CHỈNH (adjusted), không phải giá thô.
  Xác nhận bằng dữ liệu HPG quanh ngày GDKHQ cổ tức cổ phiếu 10% (25/05/2026, xác nhận
  từ thông báo điều chỉnh chứng quyền và tin tức tài chính): không có khoảng trống giá
  khoảng -9% đáng lẽ phải có nếu là giá thô.
- Chuỗi ngày giao dịch của VNINDEX qua API này có thiếu một vài phiên thực tế có giao dịch
  (ví dụ 07/04/2023, 12/08/2024, xác nhận VN-Index vẫn có điểm đóng cửa qua tin tức ngày đó),
  trong khi mã cổ phiếu riêng lẻ (VNM) không thiếu các phiên này. Vì vậy không dùng riêng
  VNINDEX làm lịch giao dịch chuẩn; hàm `trading_calendar_from_symbols` hợp nhất lịch từ
  nhiều mã để giảm rủi ro thiếu phiên.
- Không có nguồn giá thô (D71). Giải quyết bằng `detect_limit_lock`: phát hiện khóa trần,
  sàn qua tính chất OHLC suy biến (open=high=low=close), không suy ngược giá thô tuyệt
  đối (D73). Đã kiểm chứng bằng dữ liệu thật: mã L10 (HOSE) có chuỗi 6 phiên liên tiếp
  (20-27/10/2026) O=H=L=C, mỗi phiên tăng đúng khoảng +7% so với phiên trước, khớp biên
  độ HOSE, đúng một đợt "tím trần" thật.
"""

from __future__ import annotations

import pandas as pd

from vnrank.data.entrade_client import SymbolType, fetch_ohlcv


def validate_ohlc_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột `ohlc_hop_le`: True nếu high >= {open, close, low}, low <= {open, close},
    và mọi giá > 0. Không xóa dòng; chỉ đánh dấu để bước sau quyết định xử lý."""
    out = df.copy()
    gia_duong = (out[["open", "high", "low", "close"]] > 0).all(axis=1)
    high_dung = (out["high"] >= out[["open", "close", "low"]].max(axis=1))
    low_dung = (out["low"] <= out[["open", "close", "high"]].min(axis=1))
    out["ohlc_hop_le"] = gia_duong & high_dung & low_dung
    return out


def flag_zero_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột `khoi_luong_bang_khong`: True nếu volume == 0 (nghi tạm ngừng hoặc
    không có giao dịch khớp lệnh trong phiên). Không xóa dòng (charter mục 5 điều 7
    và giao_thuc_chia_du_lieu_bias.md rủi ro #7: có cờ, không tự loại)."""
    out = df.copy()
    out["khoi_luong_bang_khong"] = out["volume"] == 0
    return out


def trading_calendar_from_symbols(
    symbols: list[tuple[str, SymbolType]], start: str, end: str
) -> list[str]:
    """Lịch giao dịch tham chiếu: hợp nhất ngày có dữ liệu của nhiều mã/chỉ số.

    Không dùng riêng VNINDEX vì chuỗi của nó qua API này có thể thiếu vài phiên thực tế
    (xem ghi chú đầu file). Hợp nhất với một vài mã thanh khoản cao để giảm rủi ro.
    """
    all_dates: set[str] = set()
    for symbol, symbol_type in symbols:
        df = fetch_ohlcv(symbol, start, end, symbol_type=symbol_type)
        all_dates.update(df["date"].tolist())
    return sorted(all_dates)


def flag_missing_calendar_days(dates_present: list[str], calendar: list[str]) -> list[str]:
    """Trả về các ngày có trong lịch tham chiếu nhưng mã này không có dữ liệu
    (nghi tạm ngừng giao dịch hoặc mã chưa niêm yết tại thời điểm đó)."""
    present = set(dates_present)
    return sorted(d for d in calendar if d not in present)


# Biên độ giá thông thường (D58, mo_hinh_chi_phi_vn.md); không gồm biên độ ngày đầu
# niêm yết hoặc sau tạm ngừng >= 25 phiên (20%/30%), truyền band khác qua tham số nếu cần.
BIEN_DO_THONG_THUONG = {"HOSE": 0.07, "HNX": 0.10}


def detect_limit_lock(df: pd.DataFrame, san: str, band: float | None = None, dung_sai: float = 0.005) -> pd.DataFrame:
    """Phát hiện phiên khóa trần hoặc sàn mà KHÔNG cần giá thô tuyệt đối.

    Logic (xem quyết định D73, chưa suy ngược giá thô, chỉ dùng tính chất tương đối của
    chính dữ liệu đã có, kể cả khi đã bị điều chỉnh split/cổ tức theo D71):
    một phiên bị khóa trần hoặc sàn thì không có giao dịch ở mức giá nào khác ngoài mức
    đó, nên open = high = low = close (OHLC suy biến). Tính chất suy biến này không đổi
    khi điều chỉnh, vì điều chỉnh nhân đều cả bốn giá trong cùng phiên theo cùng hệ số.
    Chỉ coi là khóa khi mức thay đổi so với đóng cửa phiên trước gần đúng bằng biên độ
    của sàn (trong `dung_sai`), để tránh nhầm với mã quá ít thanh khoản chỉ có một lệnh
    khớp ngẫu nhiên trong ngày.

    Tham số `band` cho phép truyền biên độ khác (ví dụ 0,20 hoặc 0,30 cho ngày đầu niêm
    yết hoặc sau tạm ngừng từ 25 phiên trở lên, D58); mặc định dùng biên độ thông thường
    theo `san`.
    """
    if band is None:
        if san not in BIEN_DO_THONG_THUONG:
            raise ValueError(f"san phải là 'HOSE' hoặc 'HNX', nhận '{san}'")
        band = BIEN_DO_THONG_THUONG[san]

    out = df.sort_values("date").reset_index(drop=True).copy()
    dong_cua_truoc = out["close"].shift(1)
    suy_bien = (out["open"] == out["high"]) & (out["high"] == out["low"]) & (out["low"] == out["close"])
    thay_doi = (out["close"] - dong_cua_truoc) / dong_cua_truoc
    khop_bien_do = (thay_doi.abs() - band).abs() <= dung_sai

    out["khoa_bien_do"] = suy_bien & khop_bien_do & dong_cua_truoc.notna()
    out["huong_khoa"] = None
    out.loc[out["khoa_bien_do"] & (thay_doi > 0), "huong_khoa"] = "tran"
    out.loc[out["khoa_bien_do"] & (thay_doi < 0), "huong_khoa"] = "san"
    return out
