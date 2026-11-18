"""Lõi bộ đánh giá (bước 4.1), theo `giao_thuc_danh_gia.md` (BẢN CHỐT, D60). Dùng để kiểm định bốn
giả thuyết chính H1-H4 và báo các thước đo mô tả. Chỉ numpy/pandas, không phụ thuộc scipy/statsmodels
(cùng cách 3.8/D80, 3.13/D83).

Phạm vi bước này (4.1): Rank IC/ICIR (mục 1, 2), sai số chuẩn Newey-West và bootstrap khối tròn cho
kiểm định một chuỗi d_t (mục 3), precision@K/macro-F1/balanced accuracy/ma trận nhầm lẫn (mục 2),
Brier đa lớp/BSS/ECE chỉ mô tả (mục 6). Hiệu chỉnh Holm ở `multiple_testing.py` (viết song song bởi
Trường hợp 3). KHÔNG làm DSR, PBO (mục 7): công thức DSR và thủ tục CSCV của PBO giao thức ghi rõ
"cần chép từ bản gốc", hồ sơ hiện chưa có — để lại cho khi có nguồn, không tự suy diễn công thức phổ
biến thay thế.

Độ trễ Newey-West: giao thức ghi "quy tắc chọn độ trễ tự động cần chép từ bản gốc" (chưa có trong hồ
sơ). `newey_west_test` vì vậy KHÔNG tự chọn độ trễ; người gọi truyền `lag` (tối thiểu bắt buộc là
h-1 do chồng lấn nhãn, theo mục 3), đây là sàn tối thiểu theo giao thức, không phải quy tắc tự động
đầy đủ.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = CODE_ROOT / "config" / "config.yaml"


def _default_seed() -> int:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["seed"]


# --- Mục 1, 2: Rank IC theo ngày và ICIR -----------------------------------------------------


def rank_ic_series(dates: pd.Series, scores: np.ndarray, true_values: np.ndarray) -> pd.Series:
    """Tương quan hạng Spearman giữa `scores` và `true_values`, tính riêng từng ngày quyết định
    (mục 1 giao_thuc_danh_gia.md). Trả về Series index theo ngày (sắp tăng dần), bỏ qua ngày có dưới
    3 quan sát hoặc một trong hai cột không đổi giá trị (tương quan hạng không xác định)."""
    df = pd.DataFrame({"date": dates, "score": scores, "true": true_values})
    out = {}
    for date, group in df.groupby("date"):
        if len(group) < 3 or group["true"].nunique() < 2 or group["score"].nunique() < 2:
            continue
        ic = group["score"].rank().corr(group["true"].rank())
        if not np.isnan(ic):
            out[date] = ic
    return pd.Series(out).sort_index()


def icir(ic_series: pd.Series) -> dict:
    """ICIR = trung bình / độ lệch chuẩn của chuỗi Rank IC (mục 2). Trả về
    {"ic_mean", "ic_std", "icir", "n"}."""
    arr = ic_series.to_numpy()
    n = len(arr)
    if n < 2:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "icir": float("nan"), "n": n}
    mean, std = float(arr.mean()), float(arr.std(ddof=1))
    return {"ic_mean": mean, "ic_std": std, "icir": mean / std if std > 0 else float("nan"), "n": n}


def icir_nonoverlapping(ic_series: pd.Series, h: int) -> dict:
    """Bản ICIR theo kỳ không chồng lấn (mục 2): lấy mẫu chuỗi Rank IC theo ngày cứ cách `h` ngày
    một lần (bắt đầu từ ngày đầu tiên), vì nhãn h-ngày của các ngày liên tiếp chồng lấn h-1 phiên nên
    IC theo ngày tự tương quan; lấy mẫu cách `h` phiên loại bỏ phần lớn chồng lấn đó."""
    return icir(ic_series.iloc[::h])


# --- Mục 2: precision@K theo ngày ------------------------------------------------------------


def precision_at_k_by_day(
    dates: pd.Series, scores: np.ndarray, returns: np.ndarray, k: int
) -> pd.Series:
    """precision@K theo từng ngày quyết định: tỷ lệ mã trong top-K theo `scores` có `returns` > 0
    (mục 2). Trả về Series index theo ngày; ngày có dưới `k` mã bị bỏ qua."""
    df = pd.DataFrame({"date": dates, "score": scores, "ret": returns})
    out = {}
    for date, group in df.groupby("date"):
        if len(group) < k:
            continue
        top_k = group.nlargest(k, "score")
        out[date] = float((top_k["ret"] > 0).mean())
    return pd.Series(out).sort_index()


# --- Mục 3: sai số chuẩn Newey-West, kiểm định một phía --------------------------------------


def newey_west_se(d: np.ndarray, lag: int) -> float:
    """Sai số chuẩn Newey-West (nhân Bartlett) cho trung bình chuỗi `d` (mục 3). `lag`: độ trễ tối
    đa đưa vào hiệu chỉnh tự tương quan (người gọi truyền, xem ghi chú đầu file)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    dev = d - d.mean()
    gamma0 = float((dev @ dev) / n)
    long_run = gamma0
    for lag_l in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - lag_l / (lag + 1)
        gamma_l = float((dev[lag_l:] @ dev[:-lag_l]) / n)
        long_run += 2.0 * weight * gamma_l
    long_run = max(long_run, 0.0)  # tránh phương sai âm do số hạng tự tương quan ở mẫu nhỏ
    return math.sqrt(long_run / n)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def newey_west_test(d: np.ndarray, lag: int, margin: float = 0.0) -> dict:
    """Kiểm định một phía H0: E[d] <= -margin, H1: E[d] > -margin (mục 3; margin=0 cho H1 dạng
    thường, margin=delta cho H2/H4 dạng không kém hơn). Thống kê t = (mean + margin) / se_NW,
    p-giá trị theo phân phối chuẩn (mục 3: "p-giá trị theo phân phối chuẩn"). Trả về
    {"mean", "se", "t_stat", "p_value", "lag", "n"}."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    mean = float(d.mean())
    se = newey_west_se(d, lag)
    t_stat = (mean + margin) / se if se > 0 else float("inf") if (mean + margin) > 0 else float("-inf")
    p_value = 1.0 - _norm_cdf(t_stat)
    return {"mean": mean, "se": se, "t_stat": t_stat, "p_value": p_value, "lag": lag, "n": n}


# --- Mục 3: bootstrap khối tròn ---------------------------------------------------------------


def circular_block_bootstrap_mean_ci(
    d: np.ndarray,
    block_len: int,
    n_boot: int,
    margin: float = 0.0,
    confidence: float = 0.95,
    seed: int | None = None,
) -> dict:
    """Bootstrap khối tròn (circular block bootstrap, mục 3) cho trung bình chuỗi `d + margin`: lấy
    lại ngẫu nhiên các khối độ dài `block_len` liên tiếp (vòng quanh cuối chuỗi về đầu chuỗi), nối
    lại đủ độ dài gốc, lặp `n_boot` lần. Trả về khoảng tin cậy MỘT PHÍA mức `confidence` (cận dưới là
    phân vị (1 - confidence) của phân phối trung bình boot) cùng {"mean", "ci_low", "n_boot",
    "block_len"}. `n_boot` do người gọi truyền (không hardcode số lần lặp cố định trong hàm)."""
    d = np.asarray(d, dtype=float) + margin
    n = len(d)
    rng = np.random.default_rng(seed if seed is not None else _default_seed())
    n_blocks = math.ceil(n / block_len)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        pieces = [d[np.arange(s, s + block_len) % n] for s in starts]
        resampled = np.concatenate(pieces)[:n]
        boot_means[b] = resampled.mean()
    ci_low = float(np.quantile(boot_means, 1.0 - confidence))
    return {
        "mean": float(d.mean()),
        "ci_low": ci_low,
        "n_boot": n_boot,
        "block_len": block_len,
    }


# --- Mục 2: precision@K tổng hợp, macro-F1, balanced accuracy, ma trận nhầm lẫn --------------


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple) -> float:
    """Trung bình recall từng lớp (không theo tỷ trọng số mẫu). Lớp không có mẫu thật bị bỏ qua."""
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            continue
        recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple) -> pd.DataFrame:
    """Ma trận nhầm lẫn, hàng là nhãn thật, cột là nhãn dự báo, cùng thứ tự `classes`."""
    mat = np.zeros((len(classes), len(classes)), dtype=int)
    index = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            mat[index[t], index[p]] += 1
    return pd.DataFrame(mat, index=list(classes), columns=list(classes))


def f1_per_class(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple) -> dict:
    """F1 từng lớp (mục 2). Lớp không có mẫu thật nào bị bỏ qua (cùng quy ước `balanced_accuracy`);
    lớp có mẫu thật nhưng mô hình không bao giờ đoán trúng vẫn được tính, cho F1 = 0."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores: dict = {}
    for c in classes:
        true_mask = y_true == c
        if true_mask.sum() == 0:
            continue
        pred_mask = y_pred == c
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores[c] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return scores


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple) -> float:
    """Macro-F1 (mục 2, H3): trung bình không trọng số của `f1_per_class`, bỏ qua lớp không có mẫu
    thật (cùng quy ước `balanced_accuracy`)."""
    scores = f1_per_class(y_true, y_pred, classes)
    return float(np.mean(list(scores.values()))) if scores else float("nan")


# --- Mục 6: Brier đa lớp, BSS, ECE (chỉ mô tả, không kiểm định) ------------------------------


def brier_multiclass(y_true: np.ndarray, probs: pd.DataFrame, classes: tuple) -> float:
    """Brier đa lớp: trung bình trên mẫu của tổng bình phương sai số giữa xác suất dự báo và chỉ
    báo one-hot của lớp thật. `probs`: DataFrame cột đúng `classes`, mỗi hàng cộng lại bằng 1."""
    onehot = np.column_stack([(np.asarray(y_true) == c).astype(float) for c in classes])
    p = probs[list(classes)].to_numpy()
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def brier_skill_score(brier_model: float, brier_baseline: float) -> float:
    """BSS = 1 - Brier(mô hình) / Brier(mốc so sánh) (mục 6; mốc so sánh D1 = tần suất lớp)."""
    return float(1.0 - brier_model / brier_baseline) if brier_baseline > 0 else float("nan")


def expected_calibration_error(
    y_true: np.ndarray, probs: pd.DataFrame, classes: tuple, n_bins: int = 10
) -> float:
    """ECE theo 10 nhóm CÙNG SỐ MẪU (mục 6, bins theo phân vị, không theo khoảng đều): với mỗi mẫu,
    lấy xác suất và nhãn của lớp có xác suất dự báo cao nhất; chia thành `n_bins` nhóm theo phân vị
    của xác suất đó; ECE = tổng theo nhóm của |tỷ lệ đúng thật - xác suất trung bình dự báo|, trọng
    số theo cỡ nhóm."""
    probs_arr = probs[list(classes)].to_numpy()
    pred_idx = np.argmax(probs_arr, axis=1)
    confidence = probs_arr[np.arange(len(probs_arr)), pred_idx]
    pred_class = np.array(classes)[pred_idx]
    correct = (pred_class == np.asarray(y_true)).astype(float)

    order = np.argsort(confidence)
    n = len(confidence)
    bin_edges = np.linspace(0, n, n_bins + 1).astype(int)
    ece = 0.0
    for b in range(n_bins):
        idx = order[bin_edges[b] : bin_edges[b + 1]]
        if len(idx) == 0:
            continue
        ece += len(idx) / n * abs(correct[idx].mean() - confidence[idx].mean())
    return float(ece)
