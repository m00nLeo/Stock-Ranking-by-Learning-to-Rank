"""S3 (luồng S): nạp artefact THẬT cho `xep_hang`/`xu_huong`/`ket_qua` (bước S2, `read_functions.py`),
thay `tao_du_lieu_mau`/`nap_du_lieu_mau`. KHÔNG tính lại gì ngoài việc TÁI TẠO các chuỗi walk-forward
đã đóng băng (đúng cấu hình `frozen_configurations_4_7.csv`, không chọn lại) — cùng cách 4.4 (D97) đã
tái tạo chuỗi IC theo ngày của R2 để ghép với L1 cho kiểm định H1.

**Phạm vi S3 (quyết định người dùng, D107):** dựng `rank_scores` (L1, mọi $h$), `trend_scores` (T2,
mọi $h$), `topk_a1` (A1, CHỈ $h=5$ — 4.13 chỉ xây dựng và đánh giá A1 ở $h=5$, không có lựa chọn
top-K kiểu A1 nào ở $h=10, 21$).

**`test_results` (cho `ket_qua`, bước S8, kế hoạch D109, triển khai 17/11/2026):** dựng bằng
`vnrank.evaluation.system_results.build_system_results_table` — tái dùng CHÍNH `l1_test`/`t2_a1`
đã tính ở vòng lặp $h=5$ dưới đây (không tính lại walk-forward lần hai). Bảng 27 dòng (A1/B0/B2 ×
$K\\in\\{5,10,20\\}$ × mức chi phí), CHỈ ở $h=5$ (A1 chỉ thật ở $h=5$ — xem quyết định 2,
`system_results.py` đầu file; `ket_qua(h=10$ hoặc $21, ...)` báo lỗi toàn bộ, sửa kèm
`read_functions.ket_qua`, cùng tinh thần D107 đã làm cho `xep_hang`/`xu_huong`).

**Hệ quả cho `read_functions.py` (sửa kèm trong D107):** `_luoi_ngay`/`_ngay_gan_nhat` (lưới ngày
quyết định của CẢ `xep_hang` lẫn `xu_huong`) lấy từ `topk_a1`, vốn chỉ có ở $h=5$ với artefact thật —
`xep_hang`/`xu_huong` nay trả `{"loi": ...}` rõ ràng cho $h=10, 21$ thay vì lỗi hệ thống
(`IndexError`/`ValueError` khi lưới ngày rỗng), ĐÚNG như người dùng chọn, nhưng KHÁC `system_architecture.md`
mục 4.2/4.3 (ghi $h \\in \\{5, 10, 21\\}$ luôn hợp lệ) — giới hạn đã biết, cần cô/hội đồng chấp nhận.

**Mã băm cấu hình (`ma_bam`):** theo `system_architecture.md` mục 8.6, "không sinh mã băm mới" —
đọc từ `code/data/processed/system_artifact_manifest.json` (ghi MỘT LẦN bởi `write_manifest()`,
cùng quy ước SHA256 nội dung đã dùng ở `MANIFEST.md`/D81), không tính lại mỗi lần nạp.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from vnrank.costs.portfolio_backtest import build_rebalance_schedule
from vnrank.evaluation.system_results import build_system_results_table, momentum_test_scores
from vnrank.models import ranking_lambdarank as l1
from vnrank.models import trend_classifier as t2
from vnrank.models.trend_integration import two_stage_select
from vnrank.universe.build_universe import load_listed_stocks

CODE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = CODE_ROOT / "config" / "config.yaml"
DATA_DIR = CODE_ROOT / "data" / "processed"

DATASET_VERSION = "v20261107-ce5d2e5c"  # MANIFEST.md, D81 — mã băm nội dung 4 bảng dữ liệu đã đóng băng
A1_HORIZON = 5  # A1 chỉ được xây dựng/đánh giá ở h=5 (4.13) — xem docstring đầu file


def _frozen_params(frozen: pd.DataFrame, family: str, h: int) -> dict:
    row = frozen[(frozen["family"] == family) & (frozen["h"] == h)].iloc[0]
    return ast.literal_eval(row["chon_tren_validation"])


def write_manifest(config_path: Path = CONFIG_PATH, data_dir: Path = DATA_DIR) -> dict:
    """Tính MỘT LẦN mã băm cấu hình (SHA256 nối `DATASET_VERSION` + nội dung
    `frozen_configurations_4_7.csv` + `top_M`/`K` của `trend_integration`, cùng quy ước D81) và ghi
    `system_artifact_manifest.json`. Chạy lại chỉ khi cấu hình đóng băng thật sự đổi (xem decision_log
    D107) — `build_real_artifact` CHỈ ĐỌC, không gọi hàm này."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    frozen_bytes = (data_dir / "frozen_configurations_4_7.csv").read_bytes()
    trend_integration_str = f"top_M={cfg['trend_integration']['top_M']},K={cfg['trend_integration']['K']}"

    sha = hashlib.sha256()
    sha.update(DATASET_VERSION.encode("utf-8"))
    sha.update(frozen_bytes)
    sha.update(trend_integration_str.encode("utf-8"))
    full_hash = sha.hexdigest()

    den_ngay = str(pd.Timestamp(cfg["data_split"]["test_end"]).date())
    manifest = {
        "phien_ban": f"v{den_ngay.replace('-', '')}-{full_hash[:8]}",
        "ma_bam": full_hash,
        "tu_ngay": str(pd.Timestamp(cfg["data_split"]["test_start"]).date()),
        "den_ngay": den_ngay,
        "tep": ["frozen_configurations_4_7.csv", "MANIFEST.md (dataset, D81)", "config.yaml#trend_integration"],
    }
    with open(data_dir / "system_artifact_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def _load_manifest(data_dir: Path) -> dict:
    manifest_path = data_dir / "system_artifact_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} chưa có — chạy `write_manifest()` một lần trước (xem decision_log D107)"
        )
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def build_real_artifact(config_path: Path = CONFIG_PATH, data_dir: Path = DATA_DIR) -> dict:
    """Dựng artefact THẬT: `rank_scores`/`trend_scores` (L1/T2, mọi $h$, tái tạo walk-forward từ
    cấu hình đã đóng băng 4.7), `topk_a1` (A1, chỉ $h=5$, tái tạo đúng `trend_integration.py`
    4.13), và `test_results` (A1/B0/B2 × $K$ × chi phí, CHỈ $h=5$ — bước S8, xem docstring đầu
    file)."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    features = pd.read_csv(data_dir / "features_daily.csv", parse_dates=["date"])
    labels_all = pd.read_csv(data_dir / "labels_daily.csv", parse_dates=["date", "exit_date_resolved", "t1"])
    splits = pd.read_csv(data_dir / "splits_daily.csv", parse_dates=["date"])
    frozen = pd.read_csv(data_dir / "frozen_configurations_4_7.csv")
    universe = pd.read_csv(data_dir / "universe_daily.csv", parse_dates=["date"])

    train_start_date = splits.loc[splits["split"] == "train", "date"].min()
    test_start_date = splits.loc[splits["split"] == "test", "date"].min()
    test_end_date = splits.loc[splits["split"] == "test", "date"].max()
    calendar = sorted(splits["date"].unique())
    rebalance_dates = build_rebalance_schedule(test_start_date, test_end_date, calendar, h=A1_HORIZON)

    gbdt_common = cfg["model_grid"]["gbdt_common"]
    fraction = cfg["data_split"]["early_stopping_fraction_gbdt"]
    refit_frequency = cfg["data_split"]["refit_frequency_sessions"]
    seed = cfg["seed"]
    top_m = cfg["trend_integration"]["top_M"]
    k_a1 = cfg["trend_integration"]["K"]

    rank_frames, trend_frames = [], []
    topk_a1 = pd.DataFrame(columns=["t", "h", "hang", "ma", "san", "diem_xep_hang", "diem_xu_huong"])
    test_results = pd.DataFrame(
        columns=["h", "K", "muc_chi_phi", "cau_hinh", "rank_ic", "precision_at_k",
                 "loi_suat_nam_hoa_rong", "sharpe_rong", "mdd", "turnover"]
    )

    for h in cfg["horizons"]:
        labels_h = labels_all[labels_all["h"] == h]

        l1_params = _frozen_params(frozen, "L1_lambdarank", h)
        l1_labeled = l1.build_labeled_features(features, labels_h)
        l1_scores = l1.walk_forward_test_scores(
            l1_labeled, l1_params, train_start_date, test_start_date, test_end_date, calendar,
            refit_frequency, gbdt_common, seed, fraction,
        )
        l1_test = l1_scores.merge(splits, on="date", how="inner")
        l1_test = l1_test[l1_test["split"] == "test"][["date", "symbol", "score"]].reset_index(drop=True)
        rank_frames.append(pd.DataFrame({
            "ma": l1_test["symbol"],
            "t": l1_test["date"].dt.strftime("%Y-%m-%d"),
            "h": h,
            "diem_xep_hang": l1_test["score"],
        }))

        t2_params = _frozen_params(frozen, "T2_lightgbm", h)
        t2_labeled = t2.build_labeled_features(features, labels_h)
        t2_scores = t2.walk_forward_test_scores(
            t2_labeled, t2_params, train_start_date, test_start_date, test_end_date, calendar,
            refit_frequency, gbdt_common, seed, fraction,
        )
        t2_test = t2_scores.merge(splits, on="date", how="inner")
        t2_test = t2_test[t2_test["split"] == "test"].reset_index(drop=True)
        trend_frames.append(pd.DataFrame({
            "ma": t2_test["symbol"],
            "t": t2_test["date"].dt.strftime("%Y-%m-%d"),
            "h": h,
            "p_tang": t2_test[1],
            "p_ngang": t2_test[0],
            "p_giam": t2_test[-1],
        }))

        if h == A1_HORIZON:
            t2_a1 = t2_test.copy()
            t2_a1["p_diff"] = t2_a1[1] - t2_a1[-1]
            t2_a1 = t2_a1[["date", "symbol", "p_diff"]]
            selected = two_stage_select(
                l1_test[["date", "symbol", "score"]], t2_a1, "p_diff", top_m, k_a1,
            )
            enriched = selected.merge(l1_test, on=["date", "symbol"], how="inner").merge(
                t2_a1, on=["date", "symbol"], how="inner"
            )
            rows = []
            for date, group in enriched.groupby("date"):
                ranked = group.sort_values("p_diff", ascending=False).reset_index(drop=True)
                for pos, r in ranked.iterrows():
                    rows.append({
                        "t": date.strftime("%Y-%m-%d"), "h": h, "hang": pos + 1, "ma": r["symbol"],
                        "diem_xep_hang": float(r["score"]), "diem_xu_huong": float(r["p_diff"]),
                    })
            topk_a1 = pd.DataFrame(rows)

            # S8 (D109, D113): bảng kết quả cho ket_qua(), tái dùng l1_test/t2_a1 vừa tính ở trên.
            momentum_scores = momentum_test_scores()
            test_results = build_system_results_table(
                l1_test[["date", "symbol", "score"]], t2_a1, labels_h, universe, splits,
                momentum_scores, rebalance_dates, cfg,
            )

    rank_scores = pd.concat(rank_frames, ignore_index=True)
    trend_scores = pd.concat(trend_frames, ignore_index=True)

    listed = load_listed_stocks().rename(columns={"symbol": "ma", "exchange": "san"})
    all_ma = sorted(set(rank_scores["ma"]))
    symbols = listed[listed["ma"].isin(all_ma)].reset_index(drop=True)
    san_by_ma = dict(zip(symbols["ma"], symbols["san"]))
    if not topk_a1.empty:
        topk_a1["san"] = topk_a1["ma"].map(san_by_ma)
        topk_a1 = topk_a1[["t", "h", "hang", "ma", "san", "diem_xep_hang", "diem_xu_huong"]]

    manifest = _load_manifest(data_dir)
    return {
        "phien_ban": manifest["phien_ban"],
        "ma_bam": manifest["ma_bam"],
        "tu_ngay": manifest["tu_ngay"],
        "den_ngay": manifest["den_ngay"],
        "symbols": symbols,
        "rank_scores": rank_scores,
        "trend_scores": trend_scores,
        "topk_a1": topk_a1,
        "test_results": test_results,
    }


def nap_artefact_that() -> None:
    """Dựng artefact thật rồi nạp vào `read_functions._ARTEFACT` (thay `nap_du_lieu_mau`)."""
    from vnrank.system.read_functions import nap_artefact

    nap_artefact(build_real_artifact())


if __name__ == "__main__":
    manifest = write_manifest()
    print(f"Da ghi manifest: {manifest}")

    artefact = build_real_artifact()
    print(f"rank_scores: {len(artefact['rank_scores'])} dong, h={sorted(artefact['rank_scores']['h'].unique())}")
    print(f"trend_scores: {len(artefact['trend_scores'])} dong, h={sorted(artefact['trend_scores']['h'].unique())}")
    print(f"topk_a1: {len(artefact['topk_a1'])} dong, h={sorted(artefact['topk_a1']['h'].unique())}")
    print(f"test_results: {len(artefact['test_results'])} dong, h={sorted(artefact['test_results']['h'].unique())}")
    print(f"symbols: {len(artefact['symbols'])} ma")

    from vnrank.system.read_functions import ket_qua, nap_artefact, xep_hang, xu_huong

    nap_artefact(artefact)
    ngay_mau = artefact["topk_a1"]["t"].max()
    print("xep_hang (h=5, K=10, ngay cuoi):", xep_hang(ngay_mau, 5, 10))
    ma_mau = artefact["topk_a1"].iloc[0]["ma"]
    print(f"xu_huong ({ma_mau}, h=5, ngay cuoi):", xu_huong(ma_mau, ngay_mau, 5))
    print("xep_hang (h=10, phai loi ro rang):", xep_hang(ngay_mau, 10, 10))
    print("ket_qua (h=5, K=10, co so):", ket_qua(5, 10, "cơ sở"))
    print("ket_qua (h=10, phai loi ro rang):", ket_qua(10, 10, "cơ sở"))
