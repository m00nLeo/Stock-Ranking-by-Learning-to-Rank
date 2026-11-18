"""Kiểm tra rò rỉ (bước 3.8): xáo trộn nhãn, nhãn ngẫu nhiên phải cho Rank IC gần 0. Bước 3.13 bổ sung
kiểm tra tương tự cho NHÃN XU HƯỚNG (phân loại ba lớp): xáo trộn nhãn đường đi, AUC và macro-F1 phải về
mức ngẫu nhiên.

Bài học từ D82 (rà soát bước 3.8): số lần lặp ít (10) khiến tỷ số thống kê từng kiểm tra không ổn định,
dễ cho kết luận phóng đại. `run_label_leakage_check_repeated` và `run_trend_leakage_check_repeated` vì
vậy đều trả về toàn bộ mảng các lần lặp (không chỉ trung bình), để bên gọi tự đánh giá độ ổn định (ví dụ
chạy lại với số lần lặp khác và so sánh) thay vì tin vào một con số t-thống kê duy nhất.

Mô hình dùng ở đây CHỈ để kiểm tra rò rỉ, KHÔNG phải mô hình chính thức của Giai đoạn 4 (charter mục
7 dòng 3.8, 4.x). Chọn Ridge (đã có trong `requirements.txt`) vì đơn giản, nhanh, không cần tinh chỉnh
siêu tham số — tinh chỉnh và lựa chọn mô hình thật để ở 4.x.

Quy trình (rủi ro #1 "nhìn trước trong đặc trưng" và nguyên tắc point-in-time, charter mục 5):
1. Huấn luyện Ridge trên tập TRAIN (đã purge theo `exit_date_resolved` <= cuối train) với nhãn bị XÁO
   TRỘN (phá vỡ quan hệ thật giữa đặc trưng và nhãn) hoặc nhãn NGẪU NHIÊN thuần túy.
2. Đánh giá Rank IC (tương quan hạng Spearman theo từng ngày, lấy trung bình) trên tập VALIDATION (đã
   purge theo `exit_date_resolved` <= đầu test) với nhãn THẬT — KHÔNG đụng vào test (test chỉ chạy một
   lần, charter mục 5 điều 3).
3. Nếu không có rò rỉ, Rank IC trung bình phải gần 0 (mô hình học từ nhiễu không thể dự báo nhãn thật).
   IC lệch khỏi 0 rõ rệt gợi ý rò rỉ (ví dụ trùng lặp dòng giữa train và validation, hoặc một đặc trưng
   vô tình đóng vai trò định danh duy nhất cho dòng dữ liệu).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Hồi quy Ridge dạng đóng: beta = (X^T X + alpha I)^-1 X^T y, trên đặc trưng đã chuẩn hóa sẵn
    (feature_registry đã đưa về [-1, 1], không cần chuẩn hóa lại). Thêm cột hằng số (intercept)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    n_features = X1.shape[1]
    penalty = alpha * np.eye(n_features)
    penalty[0, 0] = 0.0  # không phạt hệ số chặn
    beta = np.linalg.solve(X1.T @ X1 + penalty, X1.T @ y)
    return beta


def ridge_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    return X1 @ beta


def rank_ic_by_day(dates: pd.Series, scores: np.ndarray, true_values: np.ndarray) -> dict:
    """Tương quan hạng Spearman giữa `scores` và `true_values`, tính riêng từng ngày rồi lấy trung bình.
    Trả về {"ic_mean", "ic_std", "n_days"}. Ngày có dưới 3 quan sát hoặc true_values không đổi bị bỏ qua."""
    df = pd.DataFrame({"date": dates, "score": scores, "true": true_values})
    ics = []
    for _, group in df.groupby("date"):
        if len(group) < 3 or group["true"].nunique() < 2 or group["score"].nunique() < 2:
            continue
        # Spearman = Pearson trên thứ hạng; tính thủ công để không phụ thuộc scipy.
        ic = group["score"].rank().corr(group["true"].rank())
        if not np.isnan(ic):
            ics.append(ic)
    if not ics:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "n_days": 0}
    arr = np.array(ics)
    return {"ic_mean": float(arr.mean()), "ic_std": float(arr.std(ddof=1)), "n_days": len(arr)}


def run_label_leakage_check(
    X_train: np.ndarray,
    y_train_for_fit: np.ndarray,
    val_dates: pd.Series,
    X_val: np.ndarray,
    y_val_true: np.ndarray,
    alpha: float = 1.0,
) -> dict:
    """Huấn luyện Ridge trên (`X_train`, `y_train_for_fit`) — nhãn đã bị xáo trộn hoặc thay ngẫu nhiên
    bên ngoài hàm này — rồi đo Rank IC trên (`X_val`, `y_val_true`, nhãn THẬT). Trả về kết quả
    `rank_ic_by_day`."""
    beta = ridge_fit(X_train, y_train_for_fit, alpha=alpha)
    scores = ridge_predict(X_val, beta)
    return rank_ic_by_day(val_dates, scores, y_val_true)


def run_label_leakage_check_repeated(
    X_train: np.ndarray,
    make_y_train_for_fit,
    val_dates: pd.Series,
    X_val: np.ndarray,
    y_val_true: np.ndarray,
    alpha: float = 1.0,
    n_repeats: int = 10,
) -> dict:
    """Lặp lại `run_label_leakage_check` nhiều lần, mỗi lần gọi `make_y_train_for_fit()` để lấy một
    bản xáo trộn/ngẫu nhiên MỚI của nhãn huấn luyện, rồi gộp các Rank IC lại.

    Một lần xáo trộn/ngẫu nhiên đơn lẻ có thể tình cờ cho hệ số hồi quy trùng hướng với tín hiệu thật
    (xác suất không nhỏ khi số đặc trưng ít), gây IC lệch khỏi 0 dù không có rò rỉ thật. Lặp lại nhiều
    lần với các bản xáo trộn/ngẫu nhiên ĐỘC LẬP rồi lấy trung bình loại bỏ hiệu ứng trùng hợp này, cho
    ước lượng IC "không rò rỉ" đáng tin cậy hơn một lần chạy đơn.
    """
    all_ics: list[float] = []
    for _ in range(n_repeats):
        y_fit = make_y_train_for_fit()
        out = run_label_leakage_check(X_train, y_fit, val_dates, X_val, y_val_true, alpha=alpha)
        if out["n_days"] > 0:
            all_ics.append(out["ic_mean"])
    if not all_ics:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "n_repeats": 0, "all_ics": []}
    arr = np.array(all_ics)
    return {
        "ic_mean": float(arr.mean()),
        "ic_std": float(arr.std(ddof=1)),
        "n_repeats": len(arr),
        "all_ics": all_ics,  # giữ toàn bộ để bên gọi tự đánh giá độ ổn định (bài học D82)
    }


def auc_one_vs_rest(scores: np.ndarray, y_binary: np.ndarray) -> float:
    """AUC qua thống kê Mann-Whitney U (hạng trung bình khi có trùng), không cần scipy/sklearn.
    `y_binary`: mảng 0/1. Trả về NaN nếu chỉ có một lớp (không đủ để tính AUC)."""
    n_pos = int(y_binary.sum())
    n_neg = len(y_binary) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank().to_numpy()
    sum_ranks_pos = ranks[y_binary == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple) -> float:
    """Macro-F1: trung bình F1 từng lớp, không theo tỷ trọng. Lớp không xuất hiện trong dự báo lẫn
    thực tế được bỏ qua khỏi trung bình (không chia 0/0)."""
    f1s = []
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else float("nan")


def ovr_ridge_classifier_fit(X: np.ndarray, y: np.ndarray, classes: tuple, alpha: float = 1.0) -> dict:
    """Huấn luyện một Ridge riêng cho mỗi lớp (one-vs-rest: nhãn 1 nếu thuộc lớp đó, 0 nếu không),
    CHỈ để kiểm tra rò rỉ (không phải mô hình phân loại chính thức). Trả về dict lớp -> beta."""
    return {c: ridge_fit(X, (y == c).astype(float), alpha=alpha) for c in classes}


def ovr_ridge_classifier_predict(X: np.ndarray, betas: dict, classes: tuple) -> tuple[np.ndarray, dict]:
    """Dự báo lớp bằng cách lấy lớp có điểm one-vs-rest cao nhất. Trả về (nhãn dự báo, dict lớp -> điểm)."""
    scores = {c: ridge_predict(X, betas[c]) for c in classes}
    stacked = np.column_stack([scores[c] for c in classes])
    pred = np.array(classes)[np.argmax(stacked, axis=1)]
    return pred, scores


def run_trend_leakage_check(
    X_train: np.ndarray,
    y_train_for_fit: np.ndarray,
    X_val: np.ndarray,
    y_val_true: np.ndarray,
    classes: tuple = (-1, 0, 1),
    alpha: float = 1.0,
) -> dict:
    """Tương tự `run_label_leakage_check` nhưng cho nhãn xu hướng (phân loại 3 lớp): huấn luyện
    Ridge one-vs-rest trên (`X_train`, `y_train_for_fit` — đã xáo trộn/ngẫu nhiên bên ngoài hàm này),
    đo AUC trung bình (macro, qua điểm one-vs-rest) và macro-F1 (qua lớp dự báo) trên
    (`X_val`, `y_val_true`, nhãn THẬT)."""
    betas = ovr_ridge_classifier_fit(X_train, y_train_for_fit, classes, alpha=alpha)
    pred, scores = ovr_ridge_classifier_predict(X_val, betas, classes)
    aucs = [auc_one_vs_rest(scores[c], (y_val_true == c).astype(int)) for c in classes]
    aucs = [a for a in aucs if not np.isnan(a)]
    return {
        "auc_macro": float(np.mean(aucs)) if aucs else float("nan"),
        "macro_f1": macro_f1(y_val_true, pred, classes),
    }


def run_trend_leakage_check_repeated(
    X_train: np.ndarray,
    make_y_train_for_fit,
    X_val: np.ndarray,
    y_val_true: np.ndarray,
    classes: tuple = (-1, 0, 1),
    alpha: float = 1.0,
    n_repeats: int = 10,
) -> dict:
    """Lặp lại `run_trend_leakage_check` nhiều lần với bản xáo trộn/ngẫu nhiên độc lập mỗi lần (cùng
    lý do với `run_label_leakage_check_repeated`: một lần đơn lẻ có thể tình cờ lệch)."""
    aucs, f1s = [], []
    for _ in range(n_repeats):
        y_fit = make_y_train_for_fit()
        out = run_trend_leakage_check(X_train, y_fit, X_val, y_val_true, classes=classes, alpha=alpha)
        if not np.isnan(out["auc_macro"]):
            aucs.append(out["auc_macro"])
        if not np.isnan(out["macro_f1"]):
            f1s.append(out["macro_f1"])
    auc_arr, f1_arr = np.array(aucs), np.array(f1s)
    return {
        "auc_mean": float(auc_arr.mean()) if len(auc_arr) else float("nan"),
        "auc_std": float(auc_arr.std(ddof=1)) if len(auc_arr) > 1 else float("nan"),
        "f1_mean": float(f1_arr.mean()) if len(f1_arr) else float("nan"),
        "f1_std": float(f1_arr.std(ddof=1)) if len(f1_arr) > 1 else float("nan"),
        "n_repeats": len(auc_arr),
        "all_aucs": aucs,
        "all_f1s": f1s,
    }
