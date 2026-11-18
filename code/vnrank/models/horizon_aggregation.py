"""Gộp đa horizon (bước 4.5). Đặc tả: `ma_tran_baseline_ablation.md` mục 3 (D59); công thức gộp
`bai_toan_hinh_thuc.md` mục 5.

- **F0 (MÔ HÌNH CHÍNH của H2):** gộp cố định điểm của **L1** (mô hình chính H1, D59 mục 9 câu 4) ở
  ba horizon $h=5,10,21$: $S_{i,t} = \\frac{1}{3}\\sum_h \\bar s^{(h)}_{i,t}$, với $\\bar s^{(h)}$ là
  điểm chuẩn hóa thứ hạng trong $U_t$ về $[-1,1]$ — CÙNG công thức
  `feature_registry.cross_sectional_normalize`: $z = 2(\\mathrm{rank}-1)/(n-1) - 1$ (rank trung bình
  khi trùng, `n<=1` thì gán 0). Dùng cấu hình L1 ĐÃ ĐÓNG BĂNG ở 4.4 cho từng $h$ (không chọn lại).
- **F1 (ĐỐI CHỨNG của H2):** "tốt nhất theo horizon" — với mỗi $h$, mô hình (trong R1-R3, L1) có Rank
  IC validation cao nhất tại $h$ đó (đã có ở 4.3/4.4, không chọn lại). Khi đánh giá ở $h=5$ dùng cấu
  hình tốt nhất cho $h=5$.
- **F2 (phụ):** từng horizon riêng lẻ của L1 ($h=5,10,21$), MỖI điểm đánh giá trên nhãn $h=5$ — cho
  thấy horizon dài có mang thông tin cho kỳ 5 phiên hay không.

**Kiểm định H2** (`giao_thuc_danh_gia.md` mục 3, D60): F0 không kém F1 về Rank IC, $h=5$.
$d_t = \\mathrm{IC}^{F0}_t - \\mathrm{IC}^{F1}_t$, $H_0: \\mathbb{E}[d_t] \\le -\\delta_{IC}$,
$\\delta_{IC} = 0{,}2 \\times$ Rank IC validation của F1 — TÍNH TỪ SỐ LIỆU VALIDATION ĐÃ ĐÓNG BĂNG Ở
4.3/4.4, TRƯỚC KHI XEM KẾT QUẢ TEST (D60; ghi vào `decision_log.md` trước khi mở test). Nếu Rank IC
validation của F1 $\\le 0$: H2 "không kiểm định được", $p = 1$ (D60), vẫn vào Holm.

Cấu hình ĐÃ ĐÓNG BĂNG dùng ở đây (không tự chọn lại): đọc từ
`regression_topk_results_4_3.csv`/`ranking_lambdarank_results_4_4.csv` (cột `chon_tren_validation`,
`rank_ic_validation`), tái tạo đúng quy trình walk-forward đã đóng băng để có chuỗi điểm theo ngày
(hai file CSV đó chỉ lưu thống kê tổng hợp, không lưu điểm theo ngày) — tất định, cùng seed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vnrank.evaluation.harness import icir, newey_west_test, rank_ic_series
from vnrank.models import ranking_lambdarank as l1
from vnrank.models import regression_topk as reg
from vnrank.models.regression_topk import REALIZATION_COL

CODE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = CODE_ROOT / "config" / "config.yaml"

REG_FAMILIES = ("R1_ridge", "R2_lightgbm", "R3_xgboost")
L1_FAMILY = "L1_lambdarank"


# --- Đọc lại cấu hình ĐÃ ĐÓNG BĂNG ở 4.3/4.4 (không chọn lại) ----------------------------------


def load_frozen_selection(data_dir: Path) -> pd.DataFrame:
    """Gộp `regression_topk_results_4_3.csv` (R1-R3) và `ranking_lambdarank_results_4_4.csv` (L1)
    thành một bảng {model, h, config (dict đã parse), rank_ic_validation}."""
    reg_df = pd.read_csv(data_dir / "regression_topk_results_4_3.csv")
    l1_df = pd.read_csv(data_dir / "ranking_lambdarank_results_4_4.csv")
    combined = pd.concat([reg_df, l1_df], ignore_index=True)
    combined["config"] = combined["chon_tren_validation"].apply(ast.literal_eval)
    return combined[["model", "h", "config", "rank_ic_validation"]]


def best_family_at_h(selection: pd.DataFrame, h: int) -> pd.Series:
    """F1: mô hình (trong R1-R3, L1) có Rank IC validation cao nhất tại `h` — không chọn lại, chỉ
    đọc lại số liệu đã đóng băng."""
    rows = selection[selection["h"] == h]
    return rows.loc[rows["rank_ic_validation"].idxmax()]


# --- Tái tạo chuỗi điểm theo ngày trên TEST cho một cấu hình đã đóng băng ----------------------


def recompute_family_test_scores(
    family: str, h: int, config: dict, cfg: dict, features: pd.DataFrame, labels: pd.DataFrame, splits: pd.DataFrame
) -> pd.DataFrame:
    """Tái tạo điểm theo ngày trên TEST cho `family` ở `h`, dùng ĐÚNG `config` đã chọn trên
    validation ở 4.3/4.4 (tất định, cùng seed, KHÔNG chọn lại trên test)."""
    labels_h = labels[labels["h"] == h]
    gbdt_common = cfg["model_grid"]["gbdt_common"]
    fraction = cfg["data_split"]["early_stopping_fraction_gbdt"]
    refit_frequency = cfg["data_split"]["refit_frequency_sessions"]
    seed = cfg["seed"]
    train_start_date = splits.loc[splits["split"] == "train", "date"].min()
    test_start_date = splits.loc[splits["split"] == "test", "date"].min()
    test_end_date = splits.loc[splits["split"] == "test", "date"].max()
    calendar = sorted(splits["date"].unique())

    if family == L1_FAMILY:
        labeled = l1.build_labeled_features(features, labels_h)
        return l1.walk_forward_test_scores(
            labeled, config, train_start_date, test_start_date, test_end_date, calendar, refit_frequency, gbdt_common, seed, fraction,
        )
    if family in REG_FAMILIES:
        labeled = reg.build_labeled_features(features, labels_h)
        return reg.walk_forward_test_scores(
            family, labeled, config, train_start_date, test_start_date, test_end_date, calendar, refit_frequency, gbdt_common, seed, fraction,
        )
    raise ValueError(f"family không hợp lệ: {family}")


# --- F0: chuẩn hóa thứ hạng rồi gộp trung bình ba horizon của L1 -------------------------------


def normalize_rank_score(scored: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa `score` thành thứ hạng trong $[-1,1]$ theo từng ngày (CÙNG công thức
    `feature_registry.cross_sectional_normalize`): $z = 2(\\mathrm{rank}-1)/(n-1) - 1$, rank trung
    bình khi trùng giá trị, $n \\le 1$ thì gán 0. Trả về bản sao với cột `score` đã thay bằng $z$."""
    out = scored.copy()
    rank = out.groupby("date")["score"].rank(method="average")
    n = out.groupby("date")["score"].transform("count")
    normalized = (2 * (rank - 1) / (n - 1) - 1).where(n > 1, 0.0)
    out["score"] = normalized
    return out


def build_f0(l1_scores_by_h: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """F0 (mục 5 `bai_toan_hinh_thuc.md`): trung bình (các ngày và mã CHUNG cho cả ba horizon) của
    điểm L1 đã chuẩn hóa thứ hạng ở $h=5,10,21$."""
    normalized = [normalize_rank_score(df) for df in l1_scores_by_h.values()]
    merged = normalized[0][["date", "symbol", "score"]].rename(columns={"score": "score_0"})
    for i, df in enumerate(normalized[1:], start=1):
        merged = merged.merge(
            df[["date", "symbol", "score"]].rename(columns={"score": f"score_{i}"}), on=["date", "symbol"], how="inner"
        )
    score_cols = [c for c in merged.columns if c.startswith("score_")]
    merged["score"] = merged[score_cols].mean(axis=1)
    return merged[["date", "symbol", "score"]]


# --- Kiểm định H2: d_t = IC^F0_t - IC^F1_t, Newey-West với biên delta_IC -----------------------


def _test_ic_series(scored: pd.DataFrame, labels_h5: pd.DataFrame, splits: pd.DataFrame) -> pd.Series:
    merged = scored.merge(labels_h5[["date", "symbol", "r"]], on=["date", "symbol"], how="inner")
    merged = merged.merge(splits, on="date", how="inner")
    test = merged[merged["split"] == "test"]
    return rank_ic_series(test["date"], test["score"].to_numpy(), test["r"].to_numpy())


def h2_test(
    f0_scores: pd.DataFrame, f1_scores: pd.DataFrame, labels_h5: pd.DataFrame, splits: pd.DataFrame, delta_ic: float
) -> dict:
    """H2 (`giao_thuc_danh_gia.md` mục 3, D60): $d_t = \\mathrm{IC}^{F0}_t - \\mathrm{IC}^{F1}_t$,
    $H_0: \\mathbb{E}[d_t] \\le -\\delta_{IC}$, Newey-West (lag=4, $h-1$ ở $h=5$). Nếu
    `delta_ic` là NaN (Rank IC validation của F1 $\\le 0$): trả về "không kiểm định được", $p=1$."""
    if not (delta_ic == delta_ic) or delta_ic <= 0:
        return {"testable": False, "p_value": 1.0, "delta_ic": delta_ic}
    ic_f0 = _test_ic_series(f0_scores, labels_h5, splits)
    ic_f1 = _test_ic_series(f1_scores, labels_h5, splits)
    joined = pd.concat({"F0": ic_f0, "F1": ic_f1}, axis=1).dropna()
    d = (joined["F0"] - joined["F1"]).to_numpy()
    return {"testable": True, "n_days": len(d), "delta_ic": delta_ic, **newey_west_test(d, lag=4, margin=delta_ic)}


if __name__ == "__main__":
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = CODE_ROOT / "data" / "processed"
    features = pd.read_csv(data_dir / "features_daily.csv", parse_dates=["date"])
    labels = pd.read_csv(data_dir / "labels_daily.csv", parse_dates=["date", REALIZATION_COL])
    splits = pd.read_csv(data_dir / "splits_daily.csv", parse_dates=["date"])

    selection = load_frozen_selection(data_dir)

    # F0: L1 ở cả ba horizon, cấu hình đã đóng băng ở 4.4 (D59 mục 9 câu 4: gộp điểm của L1).
    l1_scores_by_h = {}
    for h in cfg["horizons"]:
        l1_row = selection[(selection["model"] == L1_FAMILY) & (selection["h"] == h)].iloc[0]
        l1_scores_by_h[h] = recompute_family_test_scores(L1_FAMILY, h, l1_row["config"], cfg, features, labels, splits)
    f0_scores = build_f0(l1_scores_by_h)

    # F1: mô hình tốt nhất theo validation TẠI h=5 (đọc lại, không chọn lại).
    h = 5
    f1_row = best_family_at_h(selection, h)
    f1_scores = recompute_family_test_scores(f1_row["model"], h, f1_row["config"], cfg, features, labels, splits)

    labels_h5 = labels[labels["h"] == h]

    # delta_IC TÍNH TRƯỚC KHI XEM KẾT QUẢ TEST, từ Rank IC validation của F1 (D60).
    f1_val_ic = f1_row["rank_ic_validation"]
    delta_ic = 0.2 * f1_val_ic if f1_val_ic == f1_val_ic and f1_val_ic > 0 else float("nan")
    print(f"F1 (tốt nhất theo horizon tại h=5): {f1_row['model']} {f1_row['config']}, "
          f"Rank IC validation={f1_val_ic:.4f}, delta_IC={delta_ic}")

    h2_stats = h2_test(f0_scores, f1_scores, labels_h5, splits, delta_ic)
    print("H2 (F0 so F1, h=5, Newey-West, biên delta_IC):", h2_stats)

    # F0: thống kê mô tả trên test (Rank IC/ICIR, h=5).
    ic_f0 = _test_ic_series(f0_scores, labels_h5, splits)
    f0_stats = {"model": "F0", "h": h, **icir(ic_f0)}
    ic_f1 = _test_ic_series(f1_scores, labels_h5, splits)
    f1_stats = {"model": f"F1_{f1_row['model']}", "h": h, **icir(ic_f1)}

    # F2 (phụ): từng horizon riêng lẻ của L1, đánh giá trên nhãn h=5.
    f2_rows = []
    for h_f2, scores in l1_scores_by_h.items():
        ic_f2 = _test_ic_series(scores, labels_h5, splits)
        f2_rows.append({"model": f"F2_L1_h{h_f2}", "h_danh_gia": 5, "h_mo_hinh": h_f2, **icir(ic_f2)})

    out_df = pd.DataFrame([f0_stats, f1_stats] + f2_rows)
    out_path = data_dir / "horizon_aggregation_results_4_5.csv"
    out_df.to_csv(out_path, index=False)

    h2_df = pd.DataFrame([{"hypothesis": "H2", "h": h, "f1_model": f1_row["model"], **h2_stats}])
    h2_path = data_dir / "h2_test_4_5.csv"
    h2_df.to_csv(h2_path, index=False)

    print(out_df.to_string())
    print("Da luu ket qua F0/F1/F2 va kiem dinh H2 vao code/data/processed/")
