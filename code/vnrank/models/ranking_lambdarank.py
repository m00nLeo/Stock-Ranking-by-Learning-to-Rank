"""Xếp hạng L1 (bước 4.4): LightGBM LambdaRank, MÔ HÌNH CHÍNH của H1. Đặc tả:
`ma_tran_baseline_ablation.md` mục 2 (dòng L1, D59); lưới chung với R2 (đóng băng 2.9):
`config.yaml` mục `model_grid.lightgbm_configs`/`gbdt_common`. Cùng `feature_registry`, cùng
universe, cùng walk-forward/purge/refit như R1-R3 (4.3, `regression_topk.py`).

Mỗi ngày quyết định là MỘT truy vấn (query/group) cho LambdaRank; các mã trong U_t ngày đó là
item cần xếp hạng. Nhãn: cột `relevance` (Q=5 mức, `labels_daily.csv`, 3.6) — không tính lại.
Cùng walk-forward hai giai đoạn như 4.3: chọn cấu hình trên VALIDATION bằng Rank IC trung bình
(dùng `r` thô để đánh giá thứ tự, không dùng `relevance` để đánh giá — tái dùng nguyên văn
`select_best_config`/`evaluate_on_test` của `baselines.py`), refit mỗi quý trong TEST theo cửa sổ
mở rộng, CHỈ chạy một lần trên test cho cấu hình đã chọn.

Kiểm định H1 (`giao_thuc_danh_gia.md` mục 3, D60): so Rank IC của L1 với R2 (đối chứng chính) bằng
Newey-West, CHỈ ở h=5. Để có chuỗi IC theo ngày của R2 dùng cho d_t, script này tính lại R2 bằng
đúng quy trình đã đóng băng ở `regression_topk.py` (cùng seed, cùng cấu hình grid, tất định) —
không chọn lại cấu hình trên test, chỉ tái tạo đúng số liệu đã có ở 4.3 để ghép theo ngày.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from vnrank.features.feature_registry import FEATURE_COLUMNS
from vnrank.models.baselines import evaluate_on_test, select_best_config
from vnrank.models.regression_topk import (
    REALIZATION_COL,
    _lightgbm_configs,
    select_config_on_validation as _select_config_on_validation_r2,
    walk_forward_test_scores as _walk_forward_test_scores_r2,
)
from vnrank.evaluation.harness import newey_west_test, rank_ic_series
from vnrank.splits.split_protocol import build_refit_schedule, refit_training_window

CODE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = CODE_ROOT / "config" / "config.yaml"

MODEL_FAMILY = "L1_lightgbm_lambdarank"


def build_candidate_configs(model_grid: dict) -> dict[str, dict]:
    """Lưới L1: CHUNG với R2, `config.yaml` mục `model_grid.lightgbm_configs` (đóng băng 2.9)."""
    return _lightgbm_configs(model_grid["lightgbm_configs"])


def build_labeled_features(features: pd.DataFrame, labels_h: pd.DataFrame) -> pd.DataFrame:
    """Ghép đặc trưng với `relevance` (nhãn LambdaRank), `r` (để đánh giá Rank IC/precision, không
    dùng để huấn luyện) và cột purge `exit_date_resolved`, bỏ mẫu thiếu."""
    merged = features.merge(
        labels_h[["date", "symbol", "relevance", "r", REALIZATION_COL]],
        on=["date", "symbol"],
        how="inner",
    )
    return merged.dropna(subset=FEATURE_COLUMNS + ["relevance", REALIZATION_COL]).reset_index(drop=True)


def _split_early_stopping(window: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Như `regression_topk._split_early_stopping`: `fraction` phần CUỐI (theo ngày) làm tập dừng
    sớm, phần đầu để fit. Lặp lại ở đây (thay vì import riêng hàm private) vì sau khi chia còn cần
    sắp lại theo ngày để group LambdaRank liền khối (xem `_fit_predict_lambdarank`)."""
    dates = np.sort(window["date"].unique())
    cutoff_idx = min(max(1, int(round(len(dates) * (1 - fraction)))), len(dates) - 1)
    cutoff_date = dates[cutoff_idx]
    fit_part = window[window["date"] < cutoff_date]
    eval_part = window[window["date"] >= cutoff_date]
    return fit_part, eval_part


def _groups_by_date(df_sorted_by_date: pd.DataFrame) -> list[int]:
    """Kích thước mỗi truy vấn (số mã trong U_t của từng ngày), đúng thứ tự hàng của DataFrame đã
    sắp theo ngày — LightGBM yêu cầu các hàng cùng group liền khối."""
    return df_sorted_by_date.groupby("date", sort=True).size().tolist()


def _train_lambdarank_booster(window: pd.DataFrame, params: dict, gbdt_common: dict, seed: int, fraction: float) -> lgb.Booster:
    fit_part, eval_part = _split_early_stopping(window, fraction)
    fit_part = fit_part.sort_values("date").reset_index(drop=True)
    eval_part = eval_part.sort_values("date").reset_index(drop=True)

    train_set = lgb.Dataset(
        fit_part[FEATURE_COLUMNS].to_numpy(),
        label=fit_part["relevance"].to_numpy(),
        group=_groups_by_date(fit_part),
    )
    valid_set = lgb.Dataset(
        eval_part[FEATURE_COLUMNS].to_numpy(),
        label=eval_part["relevance"].to_numpy(),
        group=_groups_by_date(eval_part),
        reference=train_set,
    )
    lgb_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": gbdt_common["learning_rate"],
        "num_leaves": params["num_leaves"],
        "min_data_in_leaf": params["min_data_in_leaf"],
        "bagging_fraction": gbdt_common["subsample_row"],
        "bagging_freq": 1,
        "feature_fraction": gbdt_common["subsample_col"],
        "verbosity": -1,
        "seed": seed,
    }
    return lgb.train(
        lgb_params,
        train_set,
        num_boost_round=gbdt_common["max_trees"],
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(gbdt_common["early_stopping_rounds"], verbose=False)],
    )


def _fit_predict_lambdarank(
    window: pd.DataFrame, params: dict, gbdt_common: dict, seed: int, fraction: float, predict_df: pd.DataFrame
) -> np.ndarray:
    booster = _train_lambdarank_booster(window, params, gbdt_common, seed, fraction)
    return booster.predict(predict_df[FEATURE_COLUMNS].to_numpy(), num_iteration=booster.best_iteration)


def _to_scored(predict_df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"date": predict_df["date"].to_numpy(), "symbol": predict_df["symbol"].to_numpy(), "score": preds})


def select_config_on_validation(
    labeled: pd.DataFrame,
    configs: dict[str, dict],
    labels_h: pd.DataFrame,
    splits: pd.DataFrame,
    train_start_date: pd.Timestamp,
    train_end_date: pd.Timestamp,
    gbdt_common: dict,
    seed: int,
    fraction: float,
) -> tuple[str, dict[str, float]]:
    """Fit mỗi cấu hình MỘT LẦN trên TRAIN (purge tới `train_end_date`), dự báo VALIDATION, chọn
    Rank IC trung bình validation cao nhất (dùng `r` thô để đánh giá, không dùng `relevance`),
    không bao giờ chọn trên test."""
    train_window = refit_training_window(labeled, REALIZATION_COL, train_start_date, train_end_date)
    validation_dates = set(splits.loc[splits["split"] == "validation", "date"])
    predict_df = labeled[labeled["date"].isin(validation_dates)]

    scores_by_config = {}
    for key, params in configs.items():
        preds = _fit_predict_lambdarank(train_window, params, gbdt_common, seed, fraction, predict_df)
        scores_by_config[key] = _to_scored(predict_df, preds)
    return select_best_config(scores_by_config, labels_h, splits)


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
) -> pd.DataFrame:
    """Điểm dự báo ngoài mẫu cho MỌI ngày TEST: refit mỗi `refit_frequency` phiên (D62) trên cửa sổ
    mở rộng (purge tới ngày refit), dự báo đến trước mốc refit kế tiếp."""
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
        preds = _fit_predict_lambdarank(window, params, gbdt_common, seed, fraction, predict_df)
        all_scores.append(_to_scored(predict_df, preds))
    return pd.concat(all_scores, ignore_index=True)


def h1_test(l1_scores: pd.DataFrame, r2_scores: pd.DataFrame, labels_h: pd.DataFrame, splits: pd.DataFrame, h: int) -> dict:
    """H1 (`giao_thuc_danh_gia.md` mục 3, D60): d_t = IC^L1_t - IC^R2_t trên ngày quyết định TEST,
    Newey-West (độ trễ h-1), H0: E[d] <= 0, H1: E[d] > 0. CHỈ gọi ở h=5 (D54)."""
    def _test_ic_series(scored: pd.DataFrame) -> pd.Series:
        merged = scored.merge(labels_h[["date", "symbol", "r"]], on=["date", "symbol"], how="inner")
        merged = merged.merge(splits, on="date", how="inner")
        test = merged[merged["split"] == "test"]
        return rank_ic_series(test["date"], test["score"].to_numpy(), test["r"].to_numpy())

    ic_l1 = _test_ic_series(l1_scores)
    ic_r2 = _test_ic_series(r2_scores)
    joined = pd.concat({"L1": ic_l1, "R2": ic_r2}, axis=1).dropna()
    d = (joined["L1"] - joined["R2"]).to_numpy()
    return {"n_days": len(d), **newey_west_test(d, lag=h - 1)}


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
    configs = build_candidate_configs(model_grid)
    r2_configs = _lightgbm_configs(model_grid["lightgbm_configs"])

    results = []
    grid_log = []
    l1_test_scores_by_h = {}
    r2_test_scores_by_h = {}

    for h in cfg["horizons"]:
        labels_h = labels[labels["h"] == h]
        labeled = build_labeled_features(features, labels_h)

        best_key, val_ic = select_config_on_validation(
            labeled, configs, labels_h, splits, train_start_date, train_end_date, gbdt_common, seed, fraction,
        )
        for key, ic_val in val_ic.items():
            grid_log.append({"model": "L1_lambdarank", "h": h, "config": str(configs[key]), "rank_ic_validation": ic_val, "chon": key == best_key})

        test_scores = walk_forward_test_scores(
            labeled, configs[best_key], train_start_date, test_start_date, test_end_date, calendar, refit_frequency, gbdt_common, seed, fraction,
        )
        l1_test_scores_by_h[h] = test_scores
        test_stats = evaluate_on_test(test_scores, labels_h, splits, h, cfg)
        results.append({"model": "L1_lambdarank", "h": h, "chon_tren_validation": str(configs[best_key]), "rank_ic_validation": val_ic[best_key], **test_stats})
        print(f"h={h} L1_lambdarank: chon {configs[best_key]} (val IC={val_ic[best_key]:.4f}), test {test_stats}")

        # R2 (đối chứng chính H1): tái tạo đúng quy trình đã đóng băng ở 4.3 để có chuỗi IC theo
        # ngày trên test (4.3 chỉ lưu thống kê tổng hợp, không lưu điểm theo ngày).
        r2_labeled = features.merge(
            labels_h[["date", "symbol", "r_excess", REALIZATION_COL]], on=["date", "symbol"], how="inner"
        ).dropna(subset=FEATURE_COLUMNS + ["r_excess", REALIZATION_COL]).reset_index(drop=True)
        r2_best_key, r2_val_ic = _select_config_on_validation_r2(
            "R2_lightgbm", r2_labeled, r2_configs, labels_h, splits, train_start_date, train_end_date, gbdt_common, seed, fraction,
        )
        r2_test_scores = _walk_forward_test_scores_r2(
            "R2_lightgbm", r2_labeled, r2_configs[r2_best_key], train_start_date, test_start_date, test_end_date, calendar, refit_frequency, gbdt_common, seed, fraction,
        )
        r2_test_scores_by_h[h] = r2_test_scores

    out_df = pd.DataFrame(results)
    out_path = data_dir / "ranking_lambdarank_results_4_4.csv"
    out_df.to_csv(out_path, index=False)

    grid_df = pd.DataFrame(grid_log)
    grid_path = data_dir / "ranking_lambdarank_grid_search_log_4_4.csv"
    grid_df.to_csv(grid_path, index=False)

    # H1 (D54): CHỈ ở h=5.
    h = 5
    labels_h5 = labels[labels["h"] == h]
    h1_stats = h1_test(l1_test_scores_by_h[h], r2_test_scores_by_h[h], labels_h5, splits, h)
    h1_df = pd.DataFrame([{"hypothesis": "H1", "h": h, **h1_stats}])
    h1_path = data_dir / "h1_test_4_4.csv"
    h1_df.to_csv(h1_path, index=False)

    print(out_df.to_string())
    print("H1 (L1 so R2, h=5, Newey-West):", h1_stats)
    print("Da luu ket qua, log luoi tim kiem va kiem dinh H1 vao code/data/processed/")
