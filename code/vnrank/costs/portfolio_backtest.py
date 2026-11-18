"""Mô phỏng danh mục top-K (bước 4.8, mô-đun mới dùng chung cho phân tích độ nhạy Sharpe ròng và
sau này cho H1b/H2b/H4b). CHƯA từng có ở bất kỳ bước nào 4.2-4.7 (mọi bước trước chỉ tính Rank
IC/ICIR trên toàn universe, chưa dựng danh mục top-K thật có chi phí).

Đặc tả: `bai_toan_hinh_thuc.md` mục 6 (Danh mục). Tại mỗi ngày tái cân bằng (kỳ KHÔNG chồng lấn,
mỗi $h$ phiên), chọn top-$K$ mã của $U_t$ theo điểm, trọng số bằng nhau $w = 1/K$, chỉ mua không bán
khống. Lợi suất ròng của kỳ:
$$R^{\\mathrm{net}}_{p,t} = \\sum_i w_i\\, r^{(h)}_{i,t} - C_t,\\quad
C_t = c^{\\mathrm{buy}}\\cdot\\mathrm{TO}^{\\mathrm{buy}}_t + c^{\\mathrm{sell}}\\cdot\\mathrm{TO}^{\\mathrm{sell}}_t,$$
với $\\mathrm{TO}^{\\mathrm{buy}}_t$/$\\mathrm{TO}^{\\mathrm{sell}}_t$ là tỷ lệ danh mục (trên $K$)
mua mới/bán ra so với kỳ trước (kỳ đầu tiên: mua mới 100%, bán 0%).

Dùng THẲNG cột `r` đã có trong `labels_daily.csv` (lợi suất thô, đã xử lý đúng quy ước vào/thoát
lệnh và khóa trần/sàn, 3.6) làm lợi suất nắm giữ mỗi mã — KHÔNG tính lại giá. Mã bị khóa trần ở
phiên vào lệnh không có dòng `r` (đã loại ở 3.6, "coi như không mua được", `config.yaml`
`execution.ceiling_at_entry`) — các mã này tự động bị loại khỏi ứng viên top-$K$ vì không có `r`
để ghép.

Sharpe năm hóa (`giao_thuc_danh_gia.md` mục 2): trên chuỗi lợi suất ròng theo kỳ KHÔNG chồng lấn
$h$ phiên, hệ số $\\sqrt{250/h}$.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vnrank.splits.split_protocol import build_refit_schedule


def build_rebalance_schedule(
    test_start: pd.Timestamp, test_end: pd.Timestamp, calendar: list[pd.Timestamp], h: int
) -> list[pd.Timestamp]:
    """Lịch tái cân bằng danh mục: cứ `h` phiên một lần kể từ đầu test, kỳ KHÔNG chồng lấn — tái
    dùng nguyên văn `build_refit_schedule` (cùng công thức cắt lịch, khác ý nghĩa tham số
    `frequency`: ở đây là độ dài kỳ nắm giữ `h`, không phải tần suất refit mô hình)."""
    return build_refit_schedule(test_start, test_end, calendar, frequency=h)


def select_top_k(candidates: pd.DataFrame, k: int) -> list[str]:
    """`candidates`: DataFrame của MỘT ngày, cột `symbol`/`score` (đã lọc chỉ giữ mã có `r` hợp lệ
    — mua được). Trả về danh sách mã top-`k` theo điểm giảm dần. Nếu có ít hơn `k` ứng viên, lấy
    hết số ứng viên hiện có (không suy ra mã khống)."""
    return candidates.nlargest(k, "score")["symbol"].tolist()


def _returns_from_selection_by_date(
    selected_by_date: dict,
    returns_by_date_symbol: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    denom_turnover: int,
    c_buy: float,
    c_sell: float,
) -> pd.DataFrame:
    """Lõi dùng chung: cho một tập mã ĐÃ CHỌN mỗi ngày tái cân bằng (`selected_by_date`: ngày ->
    set mã), tính lợi suất ròng/turnover theo công thức mục đầu file. `denom_turnover`: mẫu số chia
    turnover (K cho danh mục top-K; với B0 không áp dụng — xem `preselected_portfolio_returns`).
    Tách riêng khỏi `top_k_portfolio_returns`/`preselected_portfolio_returns` để CHỈ MỘT nơi tính
    công thức turnover/chi phí (bước S8, D109 mục 5 — "không viết lại logic đã có")."""
    prev_set: set[str] | None = None
    rows = []
    for t in rebalance_dates:
        selected_set = selected_by_date.get(t)
        if not selected_set:
            continue
        day_returns = returns_by_date_symbol[
            (returns_by_date_symbol["date"] == t) & (returns_by_date_symbol["symbol"].isin(selected_set))
        ]
        if day_returns.empty:
            continue
        gross = day_returns["r"].mean()

        if prev_set is None:
            to_buy, to_sell = 1.0, 0.0
        else:
            to_buy = len(selected_set - prev_set) / denom_turnover
            to_sell = len(prev_set - selected_set) / denom_turnover

        cost = c_buy * to_buy + c_sell * to_sell
        rows.append(
            {"date": t, "r_net": float(gross - cost), "turnover_buy": to_buy, "turnover_sell": to_sell, "n_held": len(selected_set)}
        )
        prev_set = selected_set

    return pd.DataFrame(rows)


def top_k_portfolio_returns(
    scored: pd.DataFrame,
    labels_h: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    k: int,
    c_buy: float,
    c_sell: float,
) -> pd.DataFrame:
    """Chuỗi lợi suất ròng theo từng kỳ tái cân bằng (không chồng lấn). `scored`: date, symbol,
    score (điểm ngoài mẫu đã có sẵn, ví dụ test scores của L1). `labels_h`: date, symbol, r (đúng
    $h$ của kỳ nắm giữ). Trả về DataFrame {date, r_net, turnover_buy, turnover_sell, n_held}."""
    merged = scored.merge(labels_h[["date", "symbol", "r"]], on=["date", "symbol"], how="inner")
    merged = merged.dropna(subset=["r"])

    selected_by_date = {}
    for t in rebalance_dates:
        candidates = merged[merged["date"] == t]
        if candidates.empty:
            continue
        selected_by_date[t] = set(select_top_k(candidates, k))

    return _returns_from_selection_by_date(selected_by_date, merged[["date", "symbol", "r"]], rebalance_dates, k, c_buy, c_sell)


def preselected_portfolio_returns(
    selected: pd.DataFrame,
    labels_h: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    k: int,
    c_buy: float,
    c_sell: float,
) -> pd.DataFrame:
    """Như `top_k_portfolio_returns`, nhưng danh sách mã mỗi ngày tái cân bằng đã được CHỌN SẴN từ
    nơi khác (vd A1: `trend_integration.two_stage_select`, chọn hai giai đoạn — không phải top-K
    đơn thuần theo MỘT điểm số, nên không dùng được `select_top_k`). `selected`: date, symbol (đã
    lọc đúng mỗi ngày tối đa `k` mã, như `two_stage_select` trả về). `k`: K DANH NGHĨA của cấu hình
    (mẫu số chia turnover — dùng `k`, không dùng số mã THỰC NHẬN được mỗi ngày, để turnover nhất
    quán với `top_k_portfolio_returns` khi số mã thực nhận đôi khi ít hơn `k`, giống `giao_thuc_danh_gia.md`
    mục 2 "turnover mỗi kỳ = phần mua mới + phần bán ra" trên K mục tiêu)."""
    merged = selected.merge(labels_h[["date", "symbol", "r"]], on=["date", "symbol"], how="inner")
    merged = merged.dropna(subset=["r"])

    selected_by_date = {t: set(g["symbol"]) for t, g in merged.groupby("date")}
    return _returns_from_selection_by_date(selected_by_date, merged[["date", "symbol", "r"]], rebalance_dates, k, c_buy, c_sell)


def sharpe_annualized(r_net: np.ndarray, h: int) -> float:
    """Sharpe năm hóa trên chuỗi lợi suất ròng theo kỳ không chồng lấn $h$ phiên
    (`giao_thuc_danh_gia.md` mục 2): $\\mathrm{mean}/\\mathrm{std} \\times \\sqrt{250/h}$."""
    r_net = np.asarray(r_net, dtype=float)
    if len(r_net) < 2:
        return float("nan")
    std = r_net.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(r_net.mean() / std * np.sqrt(250 / h))


def annualized_net_return(r_net: np.ndarray, h: int) -> float:
    """Lợi suất ròng năm hóa (bước S8, TỰ TÍNH — `giao_thuc_danh_gia.md` chỉ định nghĩa lợi suất
    ròng từng kỳ và Sharpe năm hóa, không có công thức năm hóa riêng cho lợi suất): gộp lãi kép
    toàn bộ chuỗi kỳ không chồng lấn rồi quy đổi về một năm bằng CÙNG quy ước 250 phiên/năm đã dùng
    cho Sharpe năm hóa (hệ số mũ $250/(h \\times n)$, $n$ = số kỳ)."""
    r_net = np.asarray(r_net, dtype=float)
    n = len(r_net)
    if n == 0:
        return float("nan")
    growth = float(np.prod(1.0 + r_net))
    if growth <= 0:
        return float("nan")  # tổng lỗ >= 100% gộp lãi kép, năm hóa không xác định (không gặp trên dữ liệu thật)
    return float(growth ** (250.0 / (h * n)) - 1.0)


def max_drawdown(r_net: np.ndarray) -> float:
    """Mức sụt giảm tối đa (MDD), bước S8 — `giao_thuc_danh_gia.md` mục 2: "Trên chuỗi giá trị
    danh mục". Dựng đường vốn lũy kế từ `r_net` theo kỳ không chồng lấn (vốn ban đầu = 1), lấy sụt
    giảm lớn nhất so với đỉnh gần nhất trước đó. Trả về số KHÔNG ÂM (độ lớn sụt giảm; vd 0,15 nghĩa
    là đã mất 15% từ đỉnh, không phải -0,15)."""
    r_net = np.asarray(r_net, dtype=float)
    if len(r_net) == 0:
        return float("nan")
    wealth = np.cumprod(1.0 + r_net)
    running_peak = np.maximum.accumulate(wealth)
    drawdown = (running_peak - wealth) / running_peak
    return float(drawdown.max())
