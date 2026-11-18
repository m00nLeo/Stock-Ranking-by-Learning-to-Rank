"""Tích hợp xu hướng vào top-K (bước 4.13): ablation A0-A4, kiểm định H4. Đặc tả:
`ma_tran_baseline_ablation.md` mục 4.2 (cách tích hợp, D59 chọn (c′)), mục 4.3 (bảng A0-A4);
`giao_thuc_danh_gia.md` mục H4, mục 5 (H4c).

- **A0** (đối chứng H4): top-$K$ của $U_t$ theo điểm $s^{\\mathrm{rank},(5)}$ của L1 (4.4), KHÔNG
  dùng xu hướng.
- **A1** (mô hình chính H4, cách (c′) "chọn lại mềm"): trong top-$M$=20 của L1, chọn $K$=10 mã có
  $\\hat p^{+} - \\hat p^{-}$ cao nhất theo T2 (4.10, cấu hình đã đóng băng, không chọn lại).
- **A2** (phụ, H4c): như A1 nhưng điểm xu hướng lấy từ một bộ phân loại CÙNG CẤU HÌNH với T2 (không
  chọn lại) nhưng học trên nhãn đối chứng $y^{\\mathrm{sign},(h)}_{i,t} = \\operatorname{sign}(r^{(h)}_{i,t})$
  (hai lớp, `dinh_nghia_xu_huong.md` mục 4: lợi suất đúng bằng 0 xếp lớp $-1$; purge theo
  `exit_date_resolved` vì nhãn này suy từ $r$, không phải triple-barrier).
- **A3** (đối chứng): chọn ngẫu nhiên $K$ mã trong top-$M$ của L1, lặp lại nhiều lần (quyết định
  người dùng 14-15/11/2026: $B=10\\,000$ = `hypotheses.bootstrap_n`, cùng số lần dùng cho bootstrap
  ở nơi khác của dự án), báo phân phối precision@K trung bình theo ngày.
- **A4** (phụ, khám phá, cách (b) "xu hướng làm đặc trưng của L1"): CHƯA LÀM ở đây — đặc tả tự ghi
  "báo nếu đủ dữ liệu ngoài mẫu" (điều kiện, không bắt buộc); cần chuỗi $\\hat{\\mathbf p}$ ngoài mẫu
  cho CẢ giai đoạn train của L1 (mất một đoạn dữ liệu đầu, dữ liệu chỉ 3-4 năm) — để lại cho khi
  thực sự cần, không phải một phần của H4 chính thức.

Quy ước khi top-$M$ có ít hơn $M$ mã hợp lệ trong ngày (quyết định người dùng 14-15/11/2026): lấy
hết số mã hiện có làm top-$M$, rồi chọn $K$ trong số đó như bình thường; nếu sau đó vẫn thiếu $K$
mã, ngày đó bị bỏ qua khi tính precision@K (cùng cách `harness.precision_at_k_by_day` đã làm).

Kiểm định H4 (`giao_thuc_danh_gia.md` mục 3, D60): $d_t = \\mathrm{prec}^{A1}_t - \\mathrm{prec}^{A0}_t$,
$H_0: \\mathbb{E}[d_t] \\le -\\delta$, $\\delta = 0{,}02$ (D60 mục 12 câu 3), Newey-West lag $h-1=4$ ở
$h=5$ — TÁI DÙNG `harness.newey_west_test`, không viết lại công thức.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vnrank.evaluation.harness import newey_west_test, precision_at_k_by_day
from vnrank.features.feature_registry import FEATURE_COLUMNS
from vnrank.models import trend_classifier as t2
from vnrank.models.regression_topk import REALIZATION_COL as REALIZATION_COL_REG
from vnrank.models.ranking_lambdarank import build_labeled_features as build_labeled_l1
from vnrank.models.ranking_lambdarank import walk_forward_test_scores as walk_forward_l1

CODE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = CODE_ROOT / "config" / "config.yaml"

CLASSES_SIGN: tuple[int, ...] = (1, -1)  # y_sign (A2, H4c): tăng/giảm, không có đi ngang
DELTA_H4 = 0.02  # biên H4, D60 mục 12 câu 3 (chỉ ghi trong văn bản hypotheses.H4 của config.yaml)


def build_labeled_features_sign(features: pd.DataFrame, labels_h: pd.DataFrame) -> pd.DataFrame:
    """Nhãn đối chứng $y^{\\mathrm{sign}}$ cho A2: suy từ `r` (nhãn hồi quy, KHÔNG phải nhãn triple-
    barrier `y`), purge theo `exit_date_resolved` (cùng cột purge của nhãn hồi quy, vì $y^{sign}$
    suy trực tiếp từ $r$). $r \\le 0$ (gồm đúng bằng 0) xếp lớp $-1$ (`dinh_nghia_xu_huong.md`)."""
    merged = features.merge(labels_h[["date", "symbol", "r", REALIZATION_COL_REG]], on=["date", "symbol"], how="inner")
    out = merged.dropna(subset=FEATURE_COLUMNS + ["r", REALIZATION_COL_REG]).reset_index(drop=True)
    out["y_sign"] = np.where(out["r"] > 0, 1, -1).astype(int)
    return out


def two_stage_select(
    l1_scores: pd.DataFrame, rerank_scores: pd.DataFrame, rerank_col, m: int, k: int
) -> pd.DataFrame:
    """Cách (c′): trong top-$M$ của L1 (theo `score`), chọn $K$ mã có `rerank_col` cao nhất. Nếu
    một ngày có ít hơn $M$ ứng viên, lấy hết số hiện có làm top-$M$ (quyết định người dùng)."""
    merged = l1_scores.merge(rerank_scores, on=["date", "symbol"], how="inner")
    rows = []
    for date, group in merged.groupby("date"):
        top_m = group.nlargest(min(m, len(group)), "score")
        top_k = top_m.nlargest(min(k, len(top_m)), rerank_col)
        rows.append(top_k[["date", "symbol"]])
    if not rows:
        return pd.DataFrame(columns=["date", "symbol"])
    return pd.concat(rows, ignore_index=True)


def precision_at_k_for_selection(selected: pd.DataFrame, labels_h: pd.DataFrame, k: int) -> pd.Series:
    """precision@K (lợi suất vượt dương) cho một danh sách mã đã chọn sẵn mỗi ngày (A1/A2/A3) —
    khác `harness.precision_at_k_by_day` (tự chọn top-K theo điểm); ở đây danh sách đã chọn rồi."""
    merged = selected.merge(labels_h[["date", "symbol", "r_excess"]], on=["date", "symbol"], how="inner")
    merged = merged.dropna(subset=["r_excess"])
    out = {}
    for date, group in merged.groupby("date"):
        if len(group) < k:
            continue
        out[date] = float((group["r_excess"] > 0).mean())
    return pd.Series(out).sort_index()


def a3_precision_distribution(
    l1_scores: pd.DataFrame, labels_h: pd.DataFrame, m: int, k: int, n_repeat: int, seed: int
) -> np.ndarray:
    """A3: chọn ngẫu nhiên $K$ mã trong top-$M$ của L1, lặp `n_repeat` lần. Trả về mảng precision@K
    TRUNG BÌNH THEO NGÀY của mỗi lần lặp (phân phối A3, để báo cạnh A1).

    Số mã có $\\tilde r^{(5)} > 0$ trong $K$ mã chọn ngẫu nhiên KHÔNG hoàn lại từ $n$ ứng viên (trong
    đó `n_good` mã có lợi suất vượt dương) đúng là phân phối SIÊU BỘI (hypergeometric) — dùng thẳng
    `rng.hypergeometric` thay vì lấy mẫu hàng DataFrame lặp lại từng ngày × `n_repeat` lần (chậm,
    hàng triệu lượt gọi pandas cho `n_repeat`=10.000 × ~375 ngày); kết quả tương đương CHÍNH XÁC
    (không phải xấp xỉ) với cách lấy mẫu trực tiếp, chỉ nhanh hơn nhiều bậc."""
    merged = l1_scores.merge(labels_h[["date", "symbol", "r_excess"]], on=["date", "symbol"], how="inner")
    merged = merged.dropna(subset=["r_excess"])

    rng = np.random.default_rng(seed)
    day_fracs = []
    for _, group in merged.groupby("date"):
        top_m = group.nlargest(min(m, len(group)), "score")
        n = len(top_m)
        if n < k:
            continue
        n_good = int((top_m["r_excess"] > 0).sum())
        n_bad = n - n_good
        draws = rng.hypergeometric(n_good, n_bad, k, size=n_repeat)
        day_fracs.append(draws / k)

    if not day_fracs:
        return np.full(n_repeat, float("nan"))
    return np.mean(np.vstack(day_fracs), axis=0)


def h4_test(prec_a1: pd.Series, prec_a0: pd.Series, delta: float, h: int) -> dict:
    """H4 (`giao_thuc_danh_gia.md` mục 3, D60): $d_t = \\mathrm{prec}^{A1}_t - \\mathrm{prec}^{A0}_t$,
    Newey-West (lag $h-1$), biên `delta`. TÁI DÙNG `harness.newey_west_test`."""
    joined = pd.concat({"A1": prec_a1, "A0": prec_a0}, axis=1).dropna()
    d = (joined["A1"] - joined["A0"]).to_numpy()
    return {"n_days": len(d), **newey_west_test(d, lag=h - 1, margin=delta)}


def h4c_test(prec_a1: pd.Series, prec_a2: pd.Series, h: int) -> dict:
    """H4c (phụ, `giao_thuc_danh_gia.md` mục 5): chênh precision@K của A1 so A2, KHÔNG biên, HAI
    PHÍA, không hiệu chỉnh."""
    joined = pd.concat({"A1": prec_a1, "A2": prec_a2}, axis=1).dropna()
    d = (joined["A1"] - joined["A2"]).to_numpy()
    one_sided = newey_west_test(d, lag=h - 1, margin=0.0)
    p_two_sided = 2 * min(one_sided["p_value"], 1 - one_sided["p_value"])
    return {"n_days": len(d), **one_sided, "p_value_two_sided": p_two_sided}


if __name__ == "__main__":
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = CODE_ROOT / "data" / "processed"
    features = pd.read_csv(data_dir / "features_daily.csv", parse_dates=["date"])
    labels_all = pd.read_csv(data_dir / "labels_daily.csv", parse_dates=["date", REALIZATION_COL_REG, t2.REALIZATION_COL])
    splits = pd.read_csv(data_dir / "splits_daily.csv", parse_dates=["date"])

    h = 5  # H4 chỉ kiểm định ở h=5 (D54), cùng H1-H3
    labels_h5 = labels_all[labels_all["h"] == h]

    gbdt_common = cfg["model_grid"]["gbdt_common"]
    fraction = cfg["data_split"]["early_stopping_fraction_gbdt"]
    refit_frequency = cfg["data_split"]["refit_frequency_sessions"]
    seed = cfg["seed"]
    n_boot = cfg["hypotheses"]["bootstrap_n"]
    top_m = cfg["trend_integration"]["top_M"]
    k = cfg["trend_integration"]["K"]

    train_start_date = splits.loc[splits["split"] == "train", "date"].min()
    test_start_date = splits.loc[splits["split"] == "test", "date"].min()
    test_end_date = splits.loc[splits["split"] == "test", "date"].max()
    calendar = sorted(splits["date"].unique())

    # L1 (4.4, cau hinh da dong bang h=5) - TAI DUNG, khong chon lai.
    l1_config = {"num_leaves": 15, "min_data_in_leaf": 200}
    l1_labeled = build_labeled_l1(features, labels_h5)
    l1_scores = walk_forward_l1(
        l1_labeled, l1_config, train_start_date, test_start_date, test_end_date, calendar, refit_frequency, gbdt_common, seed, fraction,
    )
    l1_test = l1_scores.merge(splits, on="date", how="inner")
    l1_test = l1_test[l1_test["split"] == "test"][["date", "symbol", "score"]]

    # A0: top-K truc tiep tu diem L1.
    l1_test_with_r = l1_test.merge(labels_h5[["date", "symbol", "r_excess"]], on=["date", "symbol"], how="inner")
    prec_a0 = precision_at_k_by_day(l1_test_with_r["date"], l1_test_with_r["score"].to_numpy(), l1_test_with_r["r_excess"].to_numpy(), k)

    # T2 (4.10, cau hinh da dong bang h=5) - TAI DUNG, khong chon lai.
    t2_config = {"num_leaves": 15, "min_data_in_leaf": 200}
    t2_labeled = t2.build_labeled_features(features, labels_h5)
    t2_scores = t2.walk_forward_test_scores(
        t2_labeled, t2_config, train_start_date, test_start_date, test_end_date, calendar, refit_frequency, gbdt_common, seed, fraction,
    )
    t2_scores = t2_scores.copy()
    t2_scores["p_diff"] = t2_scores[1] - t2_scores[-1]

    # A1: chon lai mem (c') dung P(+)-P(-) cua T2.
    a1_selected = two_stage_select(l1_test[["date", "symbol", "score"]], t2_scores[["date", "symbol", "p_diff"]], "p_diff", top_m, k)
    prec_a1 = precision_at_k_for_selection(a1_selected, labels_h5, k)

    # A2 (H4c, phu): CUNG CAU HINH voi T2, hoc tren y_sign (hai lop), KHONG chon lai tren validation.
    a2_labeled = build_labeled_features_sign(features, labels_h5)
    a2_scores = t2.walk_forward_test_scores(
        a2_labeled, t2_config, train_start_date, test_start_date, test_end_date, calendar, refit_frequency, gbdt_common, seed, fraction,
        classes=CLASSES_SIGN, label_col="y_sign", realization_col=REALIZATION_COL_REG,
    )
    a2_scores = a2_scores.copy()
    a2_scores["p_diff"] = a2_scores[1] - a2_scores[-1]
    a2_selected = two_stage_select(l1_test[["date", "symbol", "score"]], a2_scores[["date", "symbol", "p_diff"]], "p_diff", top_m, k)
    prec_a2 = precision_at_k_for_selection(a2_selected, labels_h5, k)

    # A3: chon ngau nhien trong top-M, lap 10.000 lan (quyet dinh nguoi dung).
    prec_a3_dist = a3_precision_distribution(l1_test[["date", "symbol", "score"]], labels_h5, top_m, k, n_boot, seed)

    h4_stats = h4_test(prec_a1, prec_a0, DELTA_H4, h)
    h4c_stats = h4c_test(prec_a1, prec_a2, h)

    print(f"A0 precision@K trung binh: {prec_a0.mean():.4f} (n_ngay={len(prec_a0)})")
    print(f"A1 precision@K trung binh: {prec_a1.mean():.4f} (n_ngay={len(prec_a1)})")
    print(f"A2 precision@K trung binh: {prec_a2.mean():.4f} (n_ngay={len(prec_a2)})")
    print(f"A3 precision@K trung binh theo ngay, phan phoi {n_boot} lan lap: mean={np.nanmean(prec_a3_dist):.4f} sd={np.nanstd(prec_a3_dist):.4f}")
    print(f"H4 (A1 so A0, bien delta={DELTA_H4}, Newey-West): {h4_stats}")
    print(f"H4c (A1 so A2, phu, hai phia, khong hieu chinh): {h4c_stats}")

    out_df = pd.DataFrame(
        [
            {"ablation": "A0", "h": h, "precision_mean": float(prec_a0.mean()), "n_ngay": len(prec_a0)},
            {"ablation": "A1", "h": h, "precision_mean": float(prec_a1.mean()), "n_ngay": len(prec_a1)},
            {"ablation": "A2", "h": h, "precision_mean": float(prec_a2.mean()), "n_ngay": len(prec_a2)},
            {"ablation": "A3", "h": h, "precision_mean": float(np.nanmean(prec_a3_dist)), "precision_sd": float(np.nanstd(prec_a3_dist)), "n_lap": n_boot},
        ]
    )
    out_df.to_csv(data_dir / "trend_integration_results_4_13.csv", index=False)

    h4_df = pd.DataFrame([{"hypothesis": "H4", "h": h, "delta": DELTA_H4, **h4_stats}])
    h4_df.to_csv(data_dir / "h4_test_4_13.csv", index=False)

    h4c_df = pd.DataFrame([{"hypothesis": "H4c", "h": h, **h4c_stats}])
    h4c_df.to_csv(data_dir / "h4c_test_4_13.csv", index=False)

    # Cap nhat hypotheses_summary_4_7.csv voi p_H4 vua tinh, roi chay Holm tren ca 4 gia thuyet.
    from vnrank.evaluation.multiple_testing import holm_step_down

    hyp_df = pd.read_csv(data_dir / "hypotheses_summary_4_7.csv")
    hyp_df.loc[hyp_df["hypothesis"] == "H4", "p_value"] = h4_stats["p_value"]
    hyp_df.loc[hyp_df["hypothesis"] == "H4", "note"] = "A1 so A0, precision@K, bien 0.02, Newey-West (4.13, D103)"

    alpha = cfg["hypotheses"]["alpha"]
    rejected = holm_step_down(hyp_df["p_value"].tolist(), alpha)
    hyp_df["holm_rejected"] = rejected
    hyp_df.to_csv(data_dir / "hypotheses_summary_4_7.csv", index=False)

    print(hyp_df.to_string(index=False))
    print(f"Holm step-down (alpha={alpha}) tren 4 gia thuyet: {dict(zip(hyp_df['hypothesis'], rejected))}")
    print("Da luu ket qua A0-A3, kiem dinh H4/H4c va Holm vao code/data/processed/")
