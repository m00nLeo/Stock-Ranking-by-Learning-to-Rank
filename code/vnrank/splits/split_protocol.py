"""Chia train/validation/test, purge theo $t^1$, embargo (bước 3.7).

Mốc chia (D62; `test_end` xác định cụ thể ở đây từ dữ liệu thật, D79, thay giá trị null trong
`config.yaml`): train 2023-01 .. 2024-06, validation 2024-07 .. 2025-03, test từ 2025-04 đến phiên
quyết định cuối có nhãn $h=21$ đủ.

Purge theo $t^1$ (`giao_thuc_chia_du_lieu_bias.md` mục 3): mẫu $(i,t)$ chỉ dùng để huấn luyện mô hình
dùng từ mốc $\\tau$ nếu nhãn đã hiện thực trước hoặc tại $\\tau$.
- **Nhãn xu hướng:** dùng thẳng cột `t1` đã có ở `labels_daily.csv` (3.6): thời điểm chạm rào, hoặc
  $t+1+h$ nếu không chạm.
- **Nhãn hồi quy/mức liên quan:** đặc tả gốc ghi $t^1 = t+1+h$ (D62, trước khi có quy tắc khóa sàn
  D66). [SỬA 3.7, D79] Dùng cột `exit_date_resolved` (đã có ở 3.6) thay cho $t+1+h$ thuần túy: quy tắc
  "giữ đến phiên bán được" khi khóa sàn (D66) có thể đẩy thời điểm biết giá thoát thật ra SAU $t+1+h$;
  dùng $t+1+h$ cho các mẫu này sẽ đưa thông tin chưa biết vào huấn luyện. Ảnh hưởng rất nhỏ (0,10% mẫu,
  `3_6_nhan.md`) nhưng là lựa chọn nhất quán với D66, tránh rò rỉ nhẹ.

Embargo = 0 (walk-forward chỉ đi tới, charter mục 5 điều 2); 22 phiên chỉ áp khi dùng thủ tục nhìn
ngược (CSCV cho PBO), chưa cần cài đặt ở bước này (để 4.x khi tính PBO).
"""

from __future__ import annotations

import pandas as pd

TRAIN_START = "2023-01"
TRAIN_END = "2024-06"
VALIDATION_START = "2024-07"
VALIDATION_END = "2025-03"
TEST_START = "2025-04"
REFIT_FREQUENCY_SESSIONS = 63  # D62: refit mỗi quý trong test
EMBARGO_LOOKBACK_PROCEDURE_SESSIONS = 22  # D62: h_max + 1, chỉ dùng cho thủ tục nhìn ngược (PBO)


def _month_start(ym: str) -> pd.Timestamp:
    return pd.Timestamp(ym + "-01")


def _month_end_exclusive(ym: str) -> pd.Timestamp:
    """Ngày đầu tiên của tháng KẾ TIẾP `ym` — mốc trên loại trừ cho đoạn kết thúc ở cuối `ym`."""
    start = pd.Timestamp(ym + "-01")
    return start + pd.DateOffset(months=1)


def assign_split(
    date: pd.Timestamp,
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
    validation_start: str = VALIDATION_START,
    validation_end: str = VALIDATION_END,
    test_start: str = TEST_START,
    test_end: pd.Timestamp | None = None,
) -> str | None:
    """Gán nhãn "train"/"validation"/"test" cho một ngày quyết định, theo mốc tháng đã chốt.
    Trả về None nếu ngày nằm ngoài cả ba đoạn (ví dụ thuộc đoạn lịch sử đặc trưng trước 01/2023)."""
    date = pd.Timestamp(date)
    if _month_start(train_start) <= date < _month_end_exclusive(train_end):
        return "train"
    if _month_start(validation_start) <= date < _month_end_exclusive(validation_end):
        return "validation"
    if date >= _month_start(test_start):
        if test_end is None or date <= pd.Timestamp(test_end):
            return "test"
    return None


def determine_test_end(labels_daily: pd.DataFrame, horizon: int = 21) -> pd.Timestamp:
    """Phiên quyết định cuối cùng có nhãn $h$ (mặc định 21) đủ: ngày lớn nhất có dòng ứng với
    `h == horizon` trong `labels_daily` (bảng chỉ chứa dòng mà đủ dữ liệu tương lai để tính nhãn)."""
    rows = labels_daily[labels_daily["h"] == horizon]
    return rows["date"].max()


def purge_by_realization(labels_df: pd.DataFrame, tau: pd.Timestamp, realization_col: str) -> pd.DataFrame:
    """Giữ lại các dòng có `realization_col` (t1 hoặc exit_date_resolved) <= `tau`."""
    tau = pd.Timestamp(tau)
    valid = labels_df[realization_col].notna()
    return labels_df[valid & (labels_df[realization_col] <= tau)]


def build_refit_schedule(
    test_start: pd.Timestamp, test_end: pd.Timestamp, calendar: list[pd.Timestamp], frequency: int = REFIT_FREQUENCY_SESSIONS
) -> list[pd.Timestamp]:
    """Lịch refit trong test: cứ `frequency` phiên kể từ đầu test (đã sort tăng dần), không vượt `test_end`."""
    test_start, test_end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    test_dates = [d for d in calendar if test_start <= d <= test_end]
    return test_dates[::frequency]


def refit_training_window(
    labels_df: pd.DataFrame, realization_col: str, train_start: pd.Timestamp, refit_date: pd.Timestamp
) -> pd.DataFrame:
    """Cửa sổ huấn luyện mở rộng cho một lần refit: mọi mẫu từ `train_start` có nhãn đã hiện thực
    trước `refit_date` (purge theo `realization_col`), bất kể mẫu đó gốc ở train, validation hay phần
    test đã qua (D62 mục 2, cửa sổ mở rộng)."""
    train_start = pd.Timestamp(train_start)
    in_window = labels_df["date"] >= train_start
    return purge_by_realization(labels_df[in_window], refit_date, realization_col)
