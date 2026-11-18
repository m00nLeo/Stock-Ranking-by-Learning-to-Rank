"""Hồi quy rồi lấy Top-K (bước 4.3): R1 (Ridge), R2 (LightGBM hồi quy, ĐỐI CHỨNG CHÍNH của H1),
R3 (XGBoost hồi quy, kiểm tra R2 không phải do thư viện) trên lợi suất vượt trung bình chéo
`r_excess` (`labels_daily.csv`, 3.6). Đặc tả: `ma_tran_baseline_ablation.md` mục 2 (R1-R3, D59);
lưới R1 (alpha), R3 (max_depth), cấu hình LightGBM dùng chung với L1: `config.yaml` mục
`model_grid.regression_topk`/`gbdt_common`/`lightgbm_configs`. Cùng `feature_registry` (11 đặc
trưng, 3.5), cùng universe, cùng lịch walk-forward với B2-B4 (4.2, nguyên tắc 1 mục 1 của tài liệu
trên).

Walk-forward hai giai đoạn:
1. **Chọn cấu hình trên VALIDATION:** fit MỘT LẦN trên cửa sổ TRAIN (purge theo `exit_date_resolved`,
   `split_protocol.py`, đúng quy ước nhãn hồi quy/mức liên quan), dự báo điểm cho mọi ngày VALIDATION
   (không refit trong validation), chọn cấu hình Rank IC trung bình validation cao nhất
   (`select_best_config`, tái dùng từ `baselines.py`, cùng cách 4.2).
2. **Test một lần duy nhất cho cấu hình đã chọn:** refit mỗi quý (63 phiên, D62) trong TEST theo cửa
   sổ mở rộng (`build_refit_schedule`/`refit_training_window`), nối điểm dự báo ngoài mẫu của mọi
   đoạn, rồi `evaluate_on_test` (tái dùng từ `baselines.py`) MỘT LẦN — không chọn lại trên test.

GBDT (R2, R3) dừng sớm trên `early_stopping_fraction_gbdt` (= 0,20, D62) phần CUỐI theo thời gian của
mỗi cửa sổ huấn luyện (không phải ngẫu nhiên, để không rò rỉ thứ tự thời gian vào tập dừng sớm).

Môi trường (giao việc 4.3): scikit-learn/lightgbm/xgboost cài được cục bộ nhưng chỉ bản MỚI HƠN bản
đã chốt cho Colab trong `requirements.txt` (D67: scikit-learn 1.5.2/lightgbm 4.5.0/xgboost 2.1.1,
không có wheel cho Python 3.14 mặc định của máy) (D95). Colab thật chạy Python ~3.12, nên D96 cài
thêm Python 3.12.10 cục bộ và dựng venv riêng `code/.venv` với ĐÚNG bản đã chốt — số liệu CHÍNH THỨC
của bước này chạy bằng `code/.venv` (xem `decision_log.md` D95, D96; `code/docs/4_3_baseline_regression_topk.md`
mục 2 cho bảng đối chiếu hai phiên bản).
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.linear_model import Ridge

from vnrank.features.feature_registry import FEATURE_COLUMNS
from vnrank.models.baselines import evaluate_on_test, select_best_config
from vnrank.splits.split_protocol import build_refit_schedule, refit_training_window

CODE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = CODE_ROOT / "config" / "config.yaml"

REALIZATION_COL = "exit_date_resolved"  # nhãn hồi quy/mức liên quan, split_protocol.py mục purge


def _ridge_configs(alphas: list[float]) -> dict[str, dict]:
    return {f"alpha_{a}": {"alpha": a} for a in alphas}


def _lightgbm_configs(grid: list[dict]) -> dict[str, dict]:
    return {f"nl{c['num_leaves']}_md{c['min_data_in_leaf']}": c for c in grid}


def _xgboost_configs(max_depths: list[int]) -> dict[str, dict]:
    return {f"depth_{d}": {"max_depth": d} for d in max_depths}


def build_candidate_configs(model_grid: dict) -> dict[str, dict[str, dict]]:
    """Lưới cấu hình R1-R3 từ `config.yaml` mục `model_grid` (mỗi họ tối đa 3 cấu hình, 2.3 mục 6)."""
    return {
        "R1_ridge": _ridge_configs(model_grid["regression_topk"]["R1_ridge"]["alpha"]),
        "R2_lightgbm": _lightgbm_configs(model_grid["lightgbm_configs"]),
        "R3_xgboost": _xgboost_configs(model_grid["regression_topk"]["R3_xgboost"]["max_depth"]),
    }


def build_labeled_features(features: pd.DataFrame, labels_h: pd.DataFrame) -> pd.DataFrame:
    """Ghép đặc trưng (`features_daily.csv`, đã chuẩn hóa chéo) với `r_excess` và cột purge
    `exit_date_resolved` của đúng `h`, bỏ mẫu thiếu (đuôi dữ liệu chưa hiện thực nhãn, 3.6)."""
    merged = features.merge(
        labels_h[["date", "symbol", "r_excess", REALIZATION_COL]], on=["date", "symbol"], how="inner"
    )
    return merged.dropna(subset=FEATURE_COLUMNS + ["r_excess", REALIZATION_COL]).reset_index(drop=True)


def _split_early_stopping(window: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chia cửa sổ huấn luyện GBDT theo thời gian: `fraction` phần CUỐI (theo ngày) làm tập dừng
    sớm (`early_stopping_fraction_gbdt`, D62), phần đầu còn lại để fit. Không xáo trộn thứ tự ngày."""
    dates = np.sort(window["date"].unique())
    cutoff_idx = min(max(1, int(round(len(dates) * (1 - fraction)))), len(dates) - 1)
    cutoff_date = dates[cutoff_idx]
    fit_part = window[window["date"] < cutoff_date]
    eval_part = window[window["date"] >= cutoff_date]
    return fit_part, eval_part


def _fit_predict_ridge(window: pd.DataFrame, params: dict, predict_df: pd.DataFrame) -> np.ndarray:
    model = Ridge(alpha=params["alpha"])
    model.fit(window[FEATURE_COLUMNS].to_numpy(), window["r_excess"].to_numpy())
    return model.predict(predict_df[FEATURE_COLUMNS].to_numpy())


def _fit_predict_lightgbm(
    window: pd.DataFrame, params: dict, gbdt_common: dict, seed: int, fraction: float, predict_df: pd.DataFrame
) -> np.ndarray:
    fit_part, eval_part = _split_early_stopping(window, fraction)
    train_set = lgb.Dataset(fit_part[FEATURE_COLUMNS].to_numpy(), label=fit_part["r_excess"].to_numpy())
    valid_set = lgb.Dataset(
        eval_part[FEATURE_COLUMNS].to_numpy(), label=eval_part["r_excess"].to_numpy(), reference=train_set
    )
    lgb_params = {
        "objective": "regression",
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


def _fit_predict_xgboost(
    window: pd.DataFrame, params: dict, gbdt_common: dict, seed: int, fraction: float, predict_df: pd.DataFrame
) -> np.ndarray:
    fit_part, eval_part = _split_early_stopping(window, fraction)
    dtrain = xgb.DMatrix(fit_part[FEATURE_COLUMNS].to_numpy(), label=fit_part["r_excess"].to_numpy())
    dvalid = xgb.DMatrix(eval_part[FEATURE_COLUMNS].to_numpy(), label=eval_part["r_excess"].to_numpy())
    xgb_params = {
        "objective": "reg:squarederror",
        "max_depth": params["max_depth"],
        "eta": gbdt_common["learning_rate"],
        "subsample": gbdt_common["subsample_row"],
        "colsample_bytree": gbdt_common["subsample_col"],
        "seed": seed,
        "verbosity": 0,
    }
    booster = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=gbdt_common["max_trees"],
        evals=[(dvalid, "validation")],
        early_stopping_rounds=gbdt_common["early_stopping_rounds"],
        verbose_eval=False,
    )
    dpredict = xgb.DMatrix(predict_df[FEATURE_COLUMNS].to_numpy())
    return booster.predict(dpredict, iteration_range=(0, booster.best_iteration + 1))


def fit_predict(
    model_family: str, window: pd.DataFrame, params: dict, gbdt_common: dict, seed: int, fraction: float, predict_df: pd.DataFrame
) -> np.ndarray:
    """Fit một cấu hình trên `window` (đã purge), dự báo điểm cho `predict_df`. `model_family` trong
    {"R1_ridge", "R2_lightgbm", "R3_xgboost"}."""
    if model_family == "R1_ridge":
        return _fit_predict_ridge(window, params, predict_df)
    if model_family == "R2_lightgbm":
        return _fit_predict_lightgbm(window, params, gbdt_common, seed, fraction, predict_df)
    if model_family == "R3_xgboost":
        return _fit_predict_xgboost(window, params, gbdt_common, seed, fraction, predict_df)
    raise ValueError(f"model_family không hợp lệ: {model_family}")


def _to_scored(predict_df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"date": predict_df["date"].to_numpy(), "symbol": predict_df["symbol"].to_numpy(), "score": preds})


def select_config_on_validation(
    model_family: str,
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
    """Fit mỗi cấu hình MỘT LẦN trên cửa sổ TRAIN (purge tới `train_end_date`), dự báo cho mọi ngày
    VALIDATION, chọn Rank IC trung bình validation cao nhất (không refit trong validation, không bao
    giờ chọn trên test)."""
    train_window = refit_training_window(labeled, REALIZATION_COL, train_start_date, train_end_date)
    validation_dates = set(splits.loc[splits["split"] == "validation", "date"])
    predict_df = labeled[labeled["date"].isin(validation_dates)]

    scores_by_config = {}
    for key, params in configs.items():
        preds = fit_predict(model_family, train_window, params, gbdt_common, seed, fraction, predict_df)
        scores_by_config[key] = _to_scored(predict_df, preds)
    return select_best_config(scores_by_config, labels_h, splits)


def walk_forward_test_scores(
    model_family: str,
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
    """Điểm dự báo ngoài mẫu cho MỌI ngày TEST, bằng cấu hình đã chọn: refit mỗi `refit_frequency`
    phiên (D62) trên cửa sổ mở rộng (purge tới ngày refit), dự báo đến trước mốc refit kế tiếp."""
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
        preds = fit_predict(model_family, window, params, gbdt_common, seed, fraction, predict_df)
        all_scores.append(_to_scored(predict_df, preds))
    return pd.concat(all_scores, ignore_index=True)


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
    candidates = build_candidate_configs(model_grid)

    results = []
    grid_log = []  # mọi cấu hình đã thử, kể cả cấu hình không được chọn (nguyên tắc 4, mục 1)
    for h in cfg["horizons"]:
        labels_h = labels[labels["h"] == h]
        labeled = build_labeled_features(features, labels_h)

        for model_family, configs in candidates.items():
            best_key, val_ic = select_config_on_validation(
                model_family, labeled, configs, labels_h, splits,
                train_start_date, train_end_date, gbdt_common, seed, fraction,
            )
            for key, ic_val in val_ic.items():
                grid_log.append(
                    {"model": model_family, "h": h, "config": str(configs[key]), "rank_ic_validation": ic_val, "chon": key == best_key}
                )

            test_scores = walk_forward_test_scores(
                model_family, labeled, configs[best_key],
                train_start_date, test_start_date, test_end_date, calendar,
                refit_frequency, gbdt_common, seed, fraction,
            )
            test_stats = evaluate_on_test(test_scores, labels_h, splits, h, cfg)
            results.append(
                {
                    "model": model_family, "h": h,
                    "chon_tren_validation": str(configs[best_key]),
                    "rank_ic_validation": val_ic[best_key],
                    **test_stats,
                }
            )
            print(f"h={h} {model_family}: chon {configs[best_key]} (val IC={val_ic[best_key]:.4f}), test {test_stats}")

    out_df = pd.DataFrame(results)
    out_path = data_dir / "regression_topk_results_4_3.csv"
    out_df.to_csv(out_path, index=False)

    grid_df = pd.DataFrame(grid_log)
    grid_path = data_dir / "regression_topk_grid_search_log_4_3.csv"
    grid_df.to_csv(grid_path, index=False)

    print(out_df.to_string())
    print("Da luu ket qua va log luoi tim kiem vao code/data/processed/")
