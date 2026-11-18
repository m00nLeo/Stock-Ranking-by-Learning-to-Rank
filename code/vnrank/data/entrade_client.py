"""Client tự viết gọi thẳng API công khai Entrade/DNSE cho giá lịch sử OHLCV.

Thay cho vnstock: vnstock và phụ thuộc bắt buộc vnai bị PyPI cách ly từ
24/09/2026 (toàn bộ bản, kể cả 3.x cũ), không cài đặt được; vnai còn không có
mã nguồn mở để đọc trước (bước 3.2, D71). Module này không phụ thuộc gì ngoài
`requests`, chỉ đọc dữ liệu công khai, không ghi hay thay đổi gì trên máy chủ.

Giới hạn đã biết (xem config/config.yaml mục data_source và
03_Giai_doan_2_Cu_the_hoa/6_Chia_du_lieu_bias/giao_thuc_chia_du_lieu_bias.md
rủi ro #3, #6):
- Không trả dữ liệu cho mã đã hủy niêm yết (universe chỉ còn mã đang niêm yết).
- Không có danh sách thành viên VN30 theo từng kỳ; cần tổng hợp thủ công ở 3.4.
- Chưa xác nhận giá trả về là giá thô hay đã điều chỉnh sau chia tách, cổ tức;
  cần kiểm ở 3.3 bằng một ngày có sự kiện doanh nghiệp đã xác minh từ nguồn
  chính thức, không suy đoán.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import pandas as pd
import requests

BASE_URL = "https://services.entrade.com.vn/chart-api/v2/ohlcs"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

SymbolType = Literal["stock", "index", "derivative"]


def _to_unix(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def fetch_ohlcv(symbol: str, start: str, end: str, symbol_type: SymbolType = "stock") -> pd.DataFrame:
    """Lấy giá OHLCV theo ngày cho một mã hoặc chỉ số.

    symbol_type phải khớp đúng loại mã: "stock" cho cổ phiếu, "index" cho
    VNINDEX/HNXINDEX, "derivative" cho hợp đồng tương lai. Dùng sai loại trả
    về lỗi HTTP 400 ("invalid symbol"), KHÔNG phải không có dữ liệu.

    Trả về DataFrame rỗng nếu máy chủ báo không tìm thấy mã (ví dụ mã đã hủy
    niêm yết); gọi hàm raise_for_status của response gốc nếu cần phân biệt
    lỗi mạng với "không có dữ liệu".

    Phát hiện ở 3.5 (D77): máy chủ thỉnh thoảng trả HAI phiên cho cùng một ngày lịch
    VN (ví dụ 27/12/2022 có một bản ghi mốc 00:00 UTC lẫn một bản ghi mốc 02:00 UTC,
    tức 07:00 và 09:00 giờ VN; toàn bộ các ngày khác trong chuỗi đều chỉ có đúng mốc
    02:00 UTC). Xảy ra ở 3 ngày (27/12/2022, 07/04/2023, 10/04/2023), ảnh hưởng hơn
    500 mã. Giữ bản ghi có timestamp gốc LỚN HƠN cho mỗi ngày trùng (khớp quy ước
    02:00 UTC dùng xuyên suốt phần còn lại của chuỗi), loại bản ghi mốc giờ lệch.
    """
    url = f"{BASE_URL}/{symbol_type}"
    params = {
        "resolution": "1D",
        "symbol": symbol,
        "from": _to_unix(start),
        "to": _to_unix(end),
    }
    response = requests.get(url, headers=_HEADERS, params=params, timeout=15)
    if response.status_code == 400:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    response.raise_for_status()
    payload = response.json()
    if not payload or not payload.get("t"):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    raw = pd.DataFrame(
        {
            "timestamp": payload["t"],
            "date": pd.to_datetime(payload["t"], unit="s", utc=True)
            .tz_convert("Asia/Ho_Chi_Minh")
            .strftime("%Y-%m-%d"),
            "open": payload["o"],
            "high": payload["h"],
            "low": payload["l"],
            "close": payload["c"],
            "volume": payload["v"],
        }
    )
    deduped = raw.sort_values("timestamp").drop_duplicates(subset="date", keep="last")
    return deduped.drop(columns="timestamp").sort_values("date").reset_index(drop=True)
