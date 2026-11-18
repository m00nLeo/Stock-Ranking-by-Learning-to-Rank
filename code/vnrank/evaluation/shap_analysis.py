"""Kiểm tra độ vững (bước 4.9), phần SHAP: giải thích đóng góp đặc trưng của L1 (LightGBM
LambdaRank, mô hình chính của H1), cấu hình đã đóng băng ở 4.7 (`frozen_configurations_4_7.csv`),
CHỈ ở h=5 (cùng quy ước dùng xuyên suốt H1/4.8/4.13).

Dùng `booster.predict(pred_contrib=True)` của chính LightGBM thay vì cài thêm thư viện `shap`:
LightGBM cài sẵn thuật toán Tree SHAP (Lundberg et al.) cho giá trị CHÍNH XÁC (không xấp xỉ), kết
quả giống hệt `shap.TreeExplainer` cho mô hình cây — tránh thêm một phụ thuộc mới không cần thiết
(bài học từ xung đột gradio/huggingface_hub ở D104).

Hai phần charter mục 7 nhắc tới ở 4.9 ngoài SHAP — "theo giai đoạn thị trường" (đã làm ở 4.8,
`sensitivity_analysis_4_8.csv`) và "ý nghĩa thống kê" (đã làm xuyên suốt H1-H4) — KHÔNG lặp lại ở
đây. DSR/PBO (còn lại của 4.9) vẫn chờ công thức gốc, để ở 4.14 (D86, D104: lớp chữ PDF mất ký
hiệu, giao_thuc_danh_gia.md mục 7).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vnrank.features.feature_registry import FEATURE_COLUMNS
from vnrank.models.ranking_lambdarank import _train_lambdarank_booster
from vnrank.splits.split_protocol import build_refit_schedule, refit_training_window

CODE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = CODE_ROOT / "config" / "config.yaml"

REALIZATION_COL = "exit_date_resolved"


def walk_forward_shap_contributions(
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
    """Đóng góp SHAP (Tree SHAP, chính xác) cho MỌI ngày TEST, cùng lịch refit mỗi
    `refit_frequency` phiên như `ranking_lambdarank.walk_forward_test_scores` (mỗi đoạn dùng lại
    đúng booster huấn luyện trên cửa sổ mở rộng của đoạn đó).

    Trả về bảng dài: date, symbol, cột `shap_<feature>` cho từng đặc trưng, cột `shap_base` (giá trị
    nền/bias), cột `score` (dự báo thô = tổng mọi cột shap_* cộng bias, tính CỘNG LẠI để đối chiếu
    với phép cộng Tree SHAP, không gọi lại `predict` thường)."""
    refit_dates = build_refit_schedule(test_start_date, test_end_date, calendar, frequency=refit_frequency)
    segment_ends = list(refit_dates[1:]) + [None]

    all_rows = []
    for seg_start, seg_end_exclusive in zip(refit_dates, segment_ends):
        window = refit_training_window(labeled, REALIZATION_COL, train_start_date, seg_start)
        if seg_end_exclusive is None:
            mask = (labeled["date"] >= seg_start) & (labeled["date"] <= test_end_date)
        else:
            mask = (labeled["date"] >= seg_start) & (labeled["date"] < seg_end_exclusive)
        predict_df = labeled[mask]
        if predict_df.empty:
            continue

        booster = _train_lambdarank_booster(window, params, gbdt_common, seed, fraction)
        contrib = booster.predict(
            predict_df[FEATURE_COLUMNS].to_numpy(), pred_contrib=True, num_iteration=booster.best_iteration
        )
        seg = pd.DataFrame(contrib, columns=[f"shap_{c}" for c in FEATURE_COLUMNS] + ["shap_base"])
        seg.insert(0, "symbol", predict_df["symbol"].to_numpy())
        seg.insert(0, "date", predict_df["date"].to_numpy())
        seg["score"] = seg[[f"shap_{c}" for c in FEATURE_COLUMNS] + ["shap_base"]].sum(axis=1)
        all_rows.append(seg)
    return pd.concat(all_rows, ignore_index=True)


def global_shap_importance(contributions: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    """Tầm quan trọng toàn cục: trung bình |SHAP| và SHAP có dấu của từng đặc trưng trên mọi hàng
    ngoài mẫu (mọi ngày TEST, mọi mã), xếp giảm dần theo |SHAP| trung bình."""
    shap_cols = [f"shap_{c}" for c in feature_columns]
    mean_abs = contributions[shap_cols].abs().mean()
    mean_signed = contributions[shap_cols].mean()
    out = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_abs_shap": mean_abs.to_numpy(),
            "mean_shap": mean_signed.to_numpy(),
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    total = out["mean_abs_shap"].sum()
    out["ty_trong"] = out["mean_abs_shap"] / total if total > 0 else 0.0
    out["rank"] = np.arange(1, len(out) + 1)
    return out


if __name__ == "__main__":
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = CODE_ROOT / "data" / "processed"
    features = pd.read_csv(data_dir / "features_daily.csv", parse_dates=["date"])
    labels = pd.read_csv(data_dir / "labels_daily.csv", parse_dates=["date", REALIZATION_COL])
    splits = pd.read_csv(data_dir / "splits_daily.csv", parse_dates=["date"])
    frozen = pd.read_csv(data_dir / "frozen_configurations_4_7.csv")

    H_MAIN = 5
    row = frozen[(frozen["family"] == "L1_lambdarank") & (frozen["h"] == H_MAIN)].iloc[0]
    params = ast.literal_eval(row["chon_tren_validation"])

    train_start_date = splits.loc[splits["split"] == "train", "date"].min()
    test_start_date = splits.loc[splits["split"] == "test", "date"].min()
    test_end_date = splits.loc[splits["split"] == "test", "date"].max()
    calendar = sorted(splits["date"].unique())

    gbdt_common = cfg["model_grid"]["gbdt_common"]
    fraction = cfg["data_split"]["early_stopping_fraction_gbdt"]
    refit_frequency = cfg["data_split"]["refit_frequency_sessions"]
    seed = cfg["seed"]

    labels_h = labels[labels["h"] == H_MAIN]
    labeled = features.merge(
        labels_h[["date", "symbol", "relevance", REALIZATION_COL]], on=["date", "symbol"], how="inner"
    ).dropna(subset=FEATURE_COLUMNS + ["relevance", REALIZATION_COL]).reset_index(drop=True)

    contributions = walk_forward_shap_contributions(
        labeled, params, train_start_date, test_start_date, test_end_date, calendar, refit_frequency, gbdt_common, seed, fraction,
    )
    importance = global_shap_importance(contributions)

    out_path = data_dir / "shap_importance_4_9.csv"
    importance.to_csv(out_path, index=False)

    print(f"L1 (h={H_MAIN}), cau hinh dong bang: {params}")
    print(importance.to_string(index=False))
    print(f"Da luu {out_path}")
