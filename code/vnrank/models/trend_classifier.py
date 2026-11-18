"""Bộ phân loại xu hướng T2 (bước 4.10): LightGBM đa lớp trên nhãn triple-barrier `y` (ba lớp:
1 = tăng, 0 = đi ngang, -1 = giảm; `labels_daily.csv`, 3.6/3.10, KHÔNG tính lại), MÔ HÌNH của H3.
Đặc tả: `ma_tran_baseline_ablation.md` mục 4.1 (D59); lưới T2 CHUNG với R2/L1
(`config.yaml` mục `model_grid.trend_classifier_T2_configs` = 3, cùng dạng lưới
`model_grid.lightgbm_configs`). Cùng `feature_registry`, cùng universe, cùng walk-forward hai giai
đoạn như R1-R3 (4.3)/L1 (4.4): chọn cấu hình trên VALIDATION bằng macro-F1 (thước đo chính của H3),
refit mỗi quý trong TEST theo cửa sổ mở rộng, chỉ chạy MỘT LẦN trên test.

Purge theo cột `t1` (thời điểm chạm rào hoặc $t+1+h$ nếu không chạm) — khác với nhãn hồi quy dùng
`exit_date_resolved` (`split_protocol.py`, đầu file): nhãn xu hướng KHÔNG áp quy tắc dời ngày khi
khóa sàn (`labeling.py`), nên thời điểm hiện thực đúng là `t1` thẳng từ `labels_daily.csv`.

Đối chứng (ma_tran_baseline_ablation.md mục 4.1):
- **D0** (đối chứng chính của H3): lớp đa số, ƯỚC LƯỢNG LẠI trên cửa sổ huấn luyện của TỪNG lần
  walk-forward (không phải một hằng số cố định toàn bộ dữ liệu).
- **D1** (đối chứng Brier của H4d, phụ, không kiểm định): mốc tần suất lớp trên cửa sổ huấn luyện.
- **D2** (phụ): dự báo ngẫu nhiên theo tỷ lệ lớp của cửa sổ huấn luyện, chỉ để đọc macro-F1/balanced
  accuracy dễ hơn (D0 có macro-F1 thấp một cách cơ học khi một lớp áp đảo, ghi chú mục 4.1).

Kiểm định H3 (`giao_thuc_danh_gia.md` mục 3, D60): T2 hơn D0 về macro-F1 ở MỖI $h \\in \\{5,10,21\\}$,
bootstrap khối theo NGÀY (độ dài khối $b=2h$, giữ nguyên mọi mã của những ngày được lấy lại, để giữ
tương quan chéo mã và chồng lấn nhãn), $p_h = (1 + \\#\\{\\Delta^*_h - \\Delta_h \\ge \\Delta_h\\})/(B+1)$
(công thức bootstrap đã dời tâm, chép nguyên văn từ giao thức — ghi chú gốc: "cách tính chính xác
chép từ tài liệu bootstrap khi viết chương Phương pháp", chưa có nguồn gốc đầy đủ, nhưng công thức cụ
thể ĐÃ CHỐT ở D60 nên dùng trực tiếp, không tự suy diễn công thức khác). Cách đếm trong Holm: kiểm
định GIAO-HỢP, $p_{H3} = \\max(p_5, p_{10}, p_{21})$ (D60 mục 12 câu 1) — H3 chỉ đạt khi CẢ BA horizon
cùng đạt.

KHÔNG làm hiệu chuẩn T3 (bỏ, D60 mục 12 câu 4) — chỉ báo Brier đa lớp, BSS (so D1), ECE như mô tả.
"""

from __future__ import annotations

import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from vnrank.evaluation.harness import (
    balanced_accuracy,
    brier_multiclass,
    brier_skill_score,
    confusion_matrix,
    expected_calibration_error,
    f1_per_class,
    macro_f1,
)
from vnrank.features.feature_registry import FEATURE_COLUMNS
from vnrank.models.regression_topk import _lightgbm_configs, _split_early_stopping
from vnrank.splits.split_protocol import build_refit_schedule, refit_training_window

CODE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = CODE_ROOT / "config" / "config.yaml"

CLASSES: tuple[int, ...] = (1, 0, -1)  # tăng, đi ngang, giảm — đúng thứ tự config.yaml trend_label.classes
REALIZATION_COL = "t1"  # purge theo t1 cho nhãn xu hướng (khác exit_date_resolved của nhãn hồi quy)


def build_candidate_configs(model_grid: dict) -> dict[str, dict]:
    """Lưới T2: CHUNG với R2/L1, `config.yaml` mục `model_grid.lightgbm_configs` (đóng băng 2.9)."""
    return _lightgbm_configs(model_grid["lightgbm_configs"])


def build_labeled_features(
    features: pd.DataFrame, labels_h: pd.DataFrame, label_col: str = "y", realization_col: str = REALIZATION_COL
) -> pd.DataFrame:
    """Ghép đặc trưng với nhãn `label_col` và cột purge `realization_col`, bỏ mẫu thiếu (đuôi dữ
    liệu chưa hiện thực nhãn, hoặc đầu chuỗi chưa đủ `sigma_span` phiên để ước lượng ngưỡng,
    `labeling.py`). Mặc định `y`/`t1` (T2, 4.10); A2 (4.13) dùng `y_sign`/`exit_date_resolved`."""
    merged = features.merge(labels_h[["date", "symbol", label_col, realization_col]], on=["date", "symbol"], how="inner")
    out = merged.dropna(subset=FEATURE_COLUMNS + [label_col, realization_col]).reset_index(drop=True)
    out[label_col] = out[label_col].astype(int)
    return out


# --- T2: LightGBM đa lớp -------------------------------------------------------------------


def _fit_predict_t2(
    window: pd.DataFrame, params: dict, gbdt_common: dict, seed: int, fraction: float, predict_df: pd.DataFrame,
    classes: tuple = CLASSES, label_col: str = "y",
) -> np.ndarray:
    fit_part, eval_part = _split_early_stopping(window, fraction)
    class_idx = {c: i for i, c in enumerate(classes)}
    train_set = lgb.Dataset(
        fit_part[FEATURE_COLUMNS].to_numpy(), label=fit_part[label_col].map(class_idx).to_numpy()
    )
    valid_set = lgb.Dataset(
        eval_part[FEATURE_COLUMNS].to_numpy(), label=eval_part[label_col].map(class_idx).to_numpy(), reference=train_set
    )
    lgb_params = {
        # "multiclass" cua LightGBM chay dung voi ca 2 lop (num_class=2), dung chung cho T2 (3 lop)
        # va A2 (4.13, 2 lop tren y_sign) - khong can nhanh rieng cho "binary".
        "objective": "multiclass",
        "num_class": len(classes),
        "metric": "multi_logloss",
        "learning_rate": gbdt_common["learning_rate"],
        "num_leaves": params["num_leaves"],
        "min_data_in_leaf": params["min_data_in_leaf"],
        "bagging_fraction": gbdt_common["subsample_row"],
        "bagging_freq": 1,
        "feature_fraction": gbdt_common["subsample_col"],
        "verbosity": -1,
        "seed": seed,
    }
    booster = lgb.train(
        lgb_params,
        train_set,
        num_boost_round=gbdt_common["max_trees"],
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(gbdt_common["early_stopping_rounds"], verbose=False)],
    )
    return booster.predict(predict_df[FEATURE_COLUMNS].to_numpy(), num_iteration=booster.best_iteration)


def _to_scored(predict_df: pd.DataFrame, probs: np.ndarray, classes: tuple = CLASSES) -> pd.DataFrame:
    """`probs`: cột theo thứ tự `classes`. Trả về date, symbol, một cột xác suất cho MỖI lớp (tên cột
    là chính giá trị lớp, để dùng thẳng với `harness.brier_multiclass`/`expected_calibration_error`),
    và `pred` = lớp có xác suất cao nhất."""
    out = pd.DataFrame({"date": predict_df["date"].to_numpy(), "symbol": predict_df["symbol"].to_numpy()})
    for i, c in enumerate(classes):
        out[c] = probs[:, i]
    out["pred"] = np.array(classes)[np.argmax(probs, axis=1)]
    return out


def select_config_on_validation(
    labeled: pd.DataFrame,
    configs: dict[str, dict],
    splits: pd.DataFrame,
    train_start_date: pd.Timestamp,
    train_end_date: pd.Timestamp,
    gbdt_common: dict,
    seed: int,
    fraction: float,
    classes: tuple = CLASSES,
    label_col: str = "y",
    realization_col: str = REALIZATION_COL,
) -> tuple[str, dict[str, float]]:
    """Fit mỗi cấu hình MỘT LẦN trên TRAIN (purge tới `train_end_date`), dự báo VALIDATION, chọn
    macro-F1 cao nhất (thước đo chính của H3). Không bao giờ chọn trên test. Mặc định T2
    (`classes`=(1,0,-1), `label_col`="y", `realization_col`="t1"); A2 (4.13) truyền
    `classes`=(1,-1), `label_col`="y_sign", `realization_col`="exit_date_resolved"."""
    train_window = refit_training_window(labeled, realization_col, train_start_date, train_end_date)
    validation_dates = set(splits.loc[splits["split"] == "validation", "date"])
    predict_df = labeled[labeled["date"].isin(validation_dates)]

    val_f1: dict[str, float] = {}
    for key, params in configs.items():
        probs = _fit_predict_t2(train_window, params, gbdt_common, seed, fraction, predict_df, classes, label_col)
        scored = _to_scored(predict_df, probs, classes)
        val_f1[key] = macro_f1(predict_df[label_col].to_numpy(), scored["pred"].to_numpy(), classes)
    best_key = max(val_f1, key=lambda k: (val_f1[k] if val_f1[k] == val_f1[k] else -float("inf")))
    return best_key, val_f1


def walk_forward_test_scores(
    labeled: pd.DataFrame,
    params: dict,
    train_start_date: pd.Timestamp,
    test_start_date: pd.Timestamp,
    test_end_date: pd.Timestamp,
    calendar: list[pd.Timestamp],
    refit_frequency: int,
    gbdt_common: dict,
    seed: int,
    fraction: float,
    classes: tuple = CLASSES,
    label_col: str = "y",
    realization_col: str = REALIZATION_COL,
) -> pd.DataFrame:
    """Điểm dự báo (xác suất + lớp) ngoài mẫu cho MỌI ngày TEST: refit mỗi `refit_frequency` phiên
    trên cửa sổ mở rộng (purge tới ngày refit), dự báo đến trước mốc refit kế tiếp."""
    refit_dates = build_refit_schedule(test_start_date, test_end_date, calendar, frequency=refit_frequency)
    segment_ends = list(refit_dates[1:]) + [None]

    all_scores = []
    for seg_start, seg_end_exclusive in zip(refit_dates, segment_ends):
        window = refit_training_window(labeled, realization_col, train_start_date, seg_start)
        if seg_end_exclusive is None:
            mask = (labeled["date"] >= seg_start) & (labeled["date"] <= test_end_date)
        else:
            mask = (labeled["date"] >= seg_start) & (labeled["date"] < seg_end_exclusive)
        predict_df = labeled[mask]
        if predict_df.empty:
            continue
        probs = _fit_predict_t2(window, params, gbdt_common, seed, fraction, predict_df, classes, label_col)
        all_scores.append(_to_scored(predict_df, probs, classes))
    return pd.concat(all_scores, ignore_index=True)


# --- D0, D1, D2: đối chứng không dùng đặc trưng, chỉ dựa trên nhãn của cửa sổ huấn luyện --------


def majority_class(y_values: np.ndarray, classes: tuple = CLASSES) -> int:
    """D0: lớp đa số trong `y_values` (nhãn của cửa sổ huấn luyện walk-forward hiện tại). Đồng hạng
    (hầu như không xảy ra với hàng nghìn mẫu mỗi cửa sổ): lấy lớp đứng trước trong thứ tự `CLASSES`
    (tăng, đi ngang, giảm) — quy ước tất định, ghi lại để minh bạch, không ảnh hưởng số liệu thật vì
    chưa từng xảy ra đồng hạng tuyệt đối trên dữ liệu thật (mỗi cửa sổ có hàng nghìn mẫu mã-ngày)."""
    counts = {c: int(np.sum(y_values == c)) for c in classes}
    max_count = max(counts.values())
    for c in classes:
        if counts[c] == max_count:
            return c
    raise AssertionError("khong toi day: it nhat mot lop dat max_count")


def class_frequencies(y_values: np.ndarray, classes: tuple = CLASSES) -> dict[int, float]:
    """D1: tỷ lệ từng lớp trong `y_values` (nhãn của cửa sổ huấn luyện walk-forward hiện tại)."""
    n = len(y_values)
    return {c: float(np.sum(y_values == c)) / n for c in classes}


def _walk_forward_label_only_scores(
    labeled: pd.DataFrame,
    train_start_date: pd.Timestamp,
    test_start_date: pd.Timestamp,
    test_end_date: pd.Timestamp,
    calendar: list[pd.Timestamp],
    refit_frequency: int,
    predict_fn,
) -> pd.DataFrame:
    """Khung walk-forward DÙNG CHUNG cho D0/D1/D2: cùng lịch refit như T2, nhưng `predict_fn` chỉ
    nhận mảng `y` của cửa sổ huấn luyện (KHÔNG dùng đặc trưng) và trả về DataFrame điểm cho đoạn dự
    báo tương ứng."""
    refit_dates = build_refit_schedule(test_start_date, test_end_date, calendar, frequency=refit_frequency)
    segment_ends = list(refit_dates[1:]) + [None]

    all_scores = []
    for seg_start, seg_end_exclusive in zip(refit_dates, segment_ends):
        window = refit_training_window(labeled, REALIZATION_COL, train_start_date, seg_start)
        if seg_end_exclusive is None:
            mask = (labeled["date"] >= seg_start) & (labeled["date"] <= test_end_date)
        else:
            mask = (labeled["date"] >= seg_start) & (labeled["date"] < seg_end_exclusive)
        predict_df = labeled[mask]
        if predict_df.empty:
            continue
        all_scores.append(predict_fn(window["y"].to_numpy(), predict_df))
    return pd.concat(all_scores, ignore_index=True)


def d0_walk_forward_scores(
    labeled: pd.DataFrame, train_start_date, test_start_date, test_end_date, calendar, refit_frequency
) -> pd.DataFrame:
    def predict_fn(y_window: np.ndarray, predict_df: pd.DataFrame) -> pd.DataFrame:
        pred = majority_class(y_window)
        return pd.DataFrame(
            {"date": predict_df["date"].to_numpy(), "symbol": predict_df["symbol"].to_numpy(), "pred": pred}
        )

    return _walk_forward_label_only_scores(
        labeled, train_start_date, test_start_date, test_end_date, calendar, refit_frequency, predict_fn
    )


def d1_walk_forward_scores(
    labeled: pd.DataFrame, train_start_date, test_start_date, test_end_date, calendar, refit_frequency
) -> pd.DataFrame:
    def predict_fn(y_window: np.ndarray, predict_df: pd.DataFrame) -> pd.DataFrame:
        freq = class_frequencies(y_window)
        out = pd.DataFrame({"date": predict_df["date"].to_numpy(), "symbol": predict_df["symbol"].to_numpy()})
        for c in CLASSES:
            out[c] = freq[c]
        out["pred"] = majority_class(y_window)  # tham khảo; D1 chủ yếu phục vụ Brier (H4d), không phải macro-F1
        return out

    return _walk_forward_label_only_scores(
        labeled, train_start_date, test_start_date, test_end_date, calendar, refit_frequency, predict_fn
    )


def d2_walk_forward_scores(
    labeled: pd.DataFrame, train_start_date, test_start_date, test_end_date, calendar, refit_frequency, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def predict_fn(y_window: np.ndarray, predict_df: pd.DataFrame) -> pd.DataFrame:
        freq = class_frequencies(y_window)
        probs_vec = [freq[c] for c in CLASSES]
        pred = rng.choice(np.array(CLASSES), size=len(predict_df), p=probs_vec)
        return pd.DataFrame({"date": predict_df["date"].to_numpy(), "symbol": predict_df["symbol"].to_numpy(), "pred": pred})

    return _walk_forward_label_only_scores(
        labeled, train_start_date, test_start_date, test_end_date, calendar, refit_frequency, predict_fn
    )


# --- Đánh giá trên mẫu mã-ngày của test ----------------------------------------------------


def evaluate_classification(scored: pd.DataFrame, labels_h: pd.DataFrame, splits: pd.DataFrame, classes: tuple = CLASSES) -> dict:
    """macro-F1, balanced accuracy, F1 từng lớp, ma trận nhầm lẫn trên MẪU MÃ-NGÀY của test (không
    phải chuỗi ngày quyết định — khác Rank IC của R1-R3/L1/F0-F2)."""
    merged = scored.merge(labels_h[["date", "symbol", "y"]], on=["date", "symbol"], how="inner")
    merged = merged.merge(splits, on="date", how="inner")
    test = merged[merged["split"] == "test"]
    y_true = test["y"].to_numpy()
    y_pred = test["pred"].to_numpy()
    return {
        "n": len(test),
        "macro_f1": macro_f1(y_true, y_pred, classes),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, classes),
        "f1_per_class": f1_per_class(y_true, y_pred, classes),
        "confusion_matrix": confusion_matrix(y_true, y_pred, classes),
    }


def brier_bss_ece(t2_scored: pd.DataFrame, d1_scored: pd.DataFrame, labels_h: pd.DataFrame, splits: pd.DataFrame, classes: tuple = CLASSES) -> dict:
    """Brier đa lớp, BSS (so D1), ECE (mục 6 giao_thuc_danh_gia.md, D60: chỉ mô tả, KHÔNG kiểm định,
    vì đã bỏ hiệu chuẩn T3)."""
    def _eval_frame(scored):
        merged = scored.merge(labels_h[["date", "symbol", "y"]], on=["date", "symbol"], how="inner")
        merged = merged.merge(splits, on="date", how="inner")
        return merged[merged["split"] == "test"]

    t2_eval = _eval_frame(t2_scored)
    d1_eval = _eval_frame(d1_scored)
    joined = t2_eval.merge(d1_eval, on=["date", "symbol"], suffixes=("_t2", "_d1"))

    y_true = joined["y_t2"].to_numpy()
    probs_t2 = joined[[f"{c}_t2" for c in classes]].copy()
    probs_t2.columns = list(classes)
    probs_d1 = joined[[f"{c}_d1" for c in classes]].copy()
    probs_d1.columns = list(classes)

    brier_t2 = brier_multiclass(y_true, probs_t2, classes)
    brier_d1 = brier_multiclass(y_true, probs_d1, classes)
    return {
        "n": len(joined),
        "brier_t2": brier_t2,
        "brier_d1": brier_d1,
        "bss_t2_vs_d1": brier_skill_score(brier_t2, brier_d1),
        "ece_t2": expected_calibration_error(y_true, probs_t2, classes),
    }


# --- H3: T2 hơn D0 về macro-F1, bootstrap khối theo ngày -----------------------------------


def _macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 từ một ma trận nhầm lẫn k x k (hàng = thật, cột = dự báo) — CÙNG công thức
    `harness.macro_f1`/`f1_per_class` (lớp không có mẫu thật, tức tổng hàng = 0, bị bỏ qua), viết lại
    ở dạng ma trận để tính nhanh trong vòng lặp bootstrap (tránh quét lại toàn bộ mảng mỗi lần lặp)."""
    k = cm.shape[0]
    scores = []
    for c in range(k):
        row_sum = cm[c].sum()
        if row_sum == 0:
            continue
        tp = cm[c, c]
        col_sum = cm[:, c].sum()
        fp = col_sum - tp
        fn = row_sum - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def block_bootstrap_h3(
    dates: np.ndarray, y_true: np.ndarray, pred_t2: np.ndarray, pred_d0: np.ndarray,
    block_len: int, n_boot: int, seed: int, classes: tuple = CLASSES,
) -> dict:
    """H3 (`giao_thuc_danh_gia.md` mục 3, D60): bootstrap khối theo NGÀY cho
    $\\Delta_h = \\mathrm{macroF1}(T2) - \\mathrm{macroF1}(D0)$ trên mẫu mã-ngày của test. Mỗi lần lấy
    lại các khối ngày liên tiếp (vòng quanh, độ dài `block_len` = $2h$), giữ nguyên mọi mã của những
    ngày đó (giữ tương quan chéo mã và chồng lấn nhãn). $p = (1 + \\#\\{\\Delta^*-\\Delta \\ge
    \\Delta\\})/(B+1)$ (công thức bootstrap đã dời tâm, chép nguyên văn từ giao thức, D60).

    Tính nhanh bằng ma trận nhầm lẫn THEO NGÀY (mỗi ngày một ma trận $k\\times k$ cố định): mỗi lần
    bootstrap chỉ cần CỘNG lại các ma trận của những ngày được chọn (`mats[day_idx_seq].sum(axis=0)`)
    thay vì quét lại toàn bộ mẫu mã-ngày — kết quả giống hệt cách tính trực tiếp trên mẫu đã gộp (xem
    `test_trend_classifier.py`), chỉ nhanh hơn nhiều cho `n_boot` lớn."""
    unique_dates = np.array(sorted(pd.unique(dates)))
    n_days = len(unique_dates)
    k = len(classes)
    class_idx = {c: i for i, c in enumerate(classes)}
    y_true_idx = np.array([class_idx[v] for v in y_true])
    pred_t2_idx = np.array([class_idx[v] for v in pred_t2])
    pred_d0_idx = np.array([class_idx[v] for v in pred_d0])

    date_codes = pd.factorize(dates, sort=True)[0]
    combo_t2 = date_codes * (k * k) + y_true_idx * k + pred_t2_idx
    combo_d0 = date_codes * (k * k) + y_true_idx * k + pred_d0_idx
    mats_t2 = np.bincount(combo_t2, minlength=n_days * k * k).reshape(n_days, k, k)
    mats_d0 = np.bincount(combo_d0, minlength=n_days * k * k).reshape(n_days, k, k)

    delta_obs = _macro_f1_from_confusion(mats_t2.sum(axis=0)) - _macro_f1_from_confusion(mats_d0.sum(axis=0))

    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n_days / block_len)
    deltas_boot = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n_days, size=n_blocks)
        day_idx_seq = np.concatenate([np.arange(s, s + block_len) % n_days for s in starts])[:n_days]
        cm_t2 = mats_t2[day_idx_seq].sum(axis=0)
        cm_d0 = mats_d0[day_idx_seq].sum(axis=0)
        deltas_boot[b] = _macro_f1_from_confusion(cm_t2) - _macro_f1_from_confusion(cm_d0)

    p_value = (1 + np.sum((deltas_boot - delta_obs) >= delta_obs)) / (n_boot + 1)
    return {"delta_obs": float(delta_obs), "p_value": float(p_value), "n_boot": n_boot, "block_len": block_len, "n_days": n_days}


def h3_test(t2_scored: pd.DataFrame, d0_scored: pd.DataFrame, labels_h: pd.DataFrame, splits: pd.DataFrame, h: int, n_boot: int, seed: int) -> dict:
    """H3 cho một $h$: ghép điểm T2 và D0 trên cùng mẫu mã-ngày của test, bootstrap khối theo ngày
    độ dài $b = 2h$ (D60, cùng hệ số `bootstrap_block_len_factor_h` dùng cho H1/H2)."""
    def _eval_frame(scored):
        merged = scored.merge(labels_h[["date", "symbol", "y"]], on=["date", "symbol"], how="inner")
        merged = merged.merge(splits, on="date", how="inner")
        return merged[merged["split"] == "test"]

    t2_eval = _eval_frame(t2_scored)[["date", "symbol", "y", "pred"]].rename(columns={"pred": "pred_t2"})
    d0_eval = _eval_frame(d0_scored)[["date", "symbol", "pred"]].rename(columns={"pred": "pred_d0"})
    joined = t2_eval.merge(d0_eval, on=["date", "symbol"], how="inner")

    out = block_bootstrap_h3(
        joined["date"].to_numpy(), joined["y"].to_numpy(), joined["pred_t2"].to_numpy(), joined["pred_d0"].to_numpy(),
        block_len=2 * h, n_boot=n_boot, seed=seed,
    )
    return {"h": h, **out}


if __name__ == "__main__":
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = CODE_ROOT / "data" / "processed"
    features = pd.read_csv(data_dir / "features_daily.csv", parse_dates=["date"])
    labels = pd.read_csv(data_dir / "labels_daily.csv", parse_dates=["date", REALIZATION_COL])
    splits = pd.read_csv(data_dir / "splits_daily.csv", parse_dates=["date"])

    train_start_date = splits.loc[splits["split"] == "train", "date"].min()
    train_end_date = splits.loc[splits["split"] == "train", "date"].max()
    test_start_date = splits.loc[splits["split"] == "test", "date"].min()
    test_end_date = splits.loc[splits["split"] == "test", "date"].max()
    calendar = sorted(splits["date"].unique())

    model_grid = cfg["model_grid"]
    gbdt_common = model_grid["gbdt_common"]
    fraction = cfg["data_split"]["early_stopping_fraction_gbdt"]
    refit_frequency = cfg["data_split"]["refit_frequency_sessions"]
    seed = cfg["seed"]
    n_boot = cfg["hypotheses"]["bootstrap_n"]
    configs = build_candidate_configs(model_grid)

    results = []
    grid_log = []
    h3_rows = []

    for h in cfg["horizons"]:
        labels_h = labels[labels["h"] == h]
        labeled = build_labeled_features(features, labels_h)

        best_key, val_f1 = select_config_on_validation(
            labeled, configs, splits, train_start_date, train_end_date, gbdt_common, seed, fraction,
        )
        for key, f1_val in val_f1.items():
            grid_log.append({"model": "T2_lightgbm", "h": h, "config": str(configs[key]), "macro_f1_validation": f1_val, "chon": key == best_key})

        t2_scores = walk_forward_test_scores(
            labeled, configs[best_key], train_start_date, test_start_date, test_end_date, calendar, refit_frequency, gbdt_common, seed, fraction,
        )
        d0_scores = d0_walk_forward_scores(labeled, train_start_date, test_start_date, test_end_date, calendar, refit_frequency)
        d1_scores = d1_walk_forward_scores(labeled, train_start_date, test_start_date, test_end_date, calendar, refit_frequency)
        d2_scores = d2_walk_forward_scores(labeled, train_start_date, test_start_date, test_end_date, calendar, refit_frequency, seed)

        for name, scored in (("T2_lightgbm", t2_scores), ("D0_lop_da_so", d0_scores), ("D1_tan_suat_lop", d1_scores), ("D2_ngau_nhien", d2_scores)):
            stats = evaluate_classification(scored, labels_h, splits)
            results.append({
                "model": name, "h": h,
                "chon_tren_validation": str(configs[best_key]) if name == "T2_lightgbm" else "",
                "macro_f1_validation": val_f1[best_key] if name == "T2_lightgbm" else float("nan"),
                "n_mau_ma_ngay_test": stats["n"], "macro_f1": stats["macro_f1"], "balanced_accuracy": stats["balanced_accuracy"],
            })
            print(f"h={h} {name}: macro_f1={stats['macro_f1']:.4f} balanced_acc={stats['balanced_accuracy']:.4f} n={stats['n']}")

        bss_stats = brier_bss_ece(t2_scores, d1_scores, labels_h, splits)
        results.append({"model": "H4d_brier_bss_ece", "h": h, **{k: v for k, v in bss_stats.items() if k != "n"}, "n_mau_ma_ngay_test": bss_stats["n"]})
        print(f"h={h} H4d (mo ta, khong kiem dinh): {bss_stats}")

        h3_stats = h3_test(t2_scores, d0_scores, labels_h, splits, h, n_boot, seed)
        h3_rows.append(h3_stats)
        print(f"h={h} H3 (T2 so D0, bootstrap khoi theo ngay): {h3_stats}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(data_dir / "trend_classifier_results_4_10.csv", index=False)

    grid_df = pd.DataFrame(grid_log)
    grid_df.to_csv(data_dir / "trend_classifier_grid_search_log_4_10.csv", index=False)

    h3_df = pd.DataFrame(h3_rows)
    p_h3 = float(h3_df["p_value"].max())
    h3_df.to_csv(data_dir / "h3_test_4_10.csv", index=False)
    print(f"p_H3 (giao-hop, max ba h): {p_h3}")

    print(out_df.to_string())
    print("Da luu ket qua, log luoi tim kiem va kiem dinh H3 vao code/data/processed/")
