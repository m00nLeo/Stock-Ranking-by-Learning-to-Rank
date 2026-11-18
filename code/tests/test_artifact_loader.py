"""Test cho S3/S4 (D107) và S8 (D109, D113): nạp artefact THẬT cho xep_hang/xu_huong/ket_qua
(artifact_loader.py). Dữ liệu tổng hợp nhỏ tự dựng, hai mã có thật trong danh sách niêm yết (AAA,
AAM — để `load_listed_stocks` trả đúng `san`), không đọc dữ liệu thật của dự án — riêng điểm B2
(momentum) của `test_results` được giả lập qua `monkeypatch` (tránh đọc panel giá thật 707 mã,
chậm và phá cô lập test; logic B2 thật đã có test riêng ở `test_system_results.py`)."""
from __future__ import annotations

import json

import pandas as pd
import yaml

from vnrank.features.feature_registry import FEATURE_COLUMNS
from vnrank.system.artifact_loader import build_real_artifact, write_manifest

SYMBOLS = ["AAA", "AAM"]


def _gia_lap_momentum_scores(monkeypatch, dates) -> None:
    """Thay `artifact_loader.momentum_test_scores` bằng điểm cố định AAA > AAM mỗi ngày — đủ để
    `build_system_results_table` chạy hết, không cần đọc panel giá thật."""
    import vnrank.system.artifact_loader as al

    fake = pd.DataFrame(
        [{"date": d, "symbol": s, "score": v} for d in dates for s, v in [("AAA", 1.0), ("AAM", 0.0)]]
    )
    monkeypatch.setattr(al, "momentum_test_scores", lambda: fake)


def _write_config(tmp_path, horizons, test_start, test_end) -> "Path":
    cfg = {
        "seed": 42,
        "horizons": horizons,
        "data_split": {
            "test_start": test_start,
            "test_end": test_end,
            "refit_frequency_sessions": 5,
            "early_stopping_fraction_gbdt": 0.2,
        },
        "model_grid": {
            "gbdt_common": {
                "learning_rate": 0.1,
                "max_trees": 30,
                "early_stopping_rounds": 5,
                "subsample_row": 0.8,
                "subsample_col": 0.8,
            }
        },
        "trend_integration": {"top_M": 2, "K": 1, "K_sensitivity": [2]},
        "costs": {"c_rt_sensitivity": [0.0030, 0.0120]},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return config_path


def _zero_features_row(date, symbol) -> dict:
    row = {"date": date, "symbol": symbol}
    for col in FEATURE_COLUMNS:
        row[col] = 0.0
    return row


def _write_dataset(tmp_path, horizons, n_days: int = 40):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    train_end_idx = n_days // 2

    feat_rows = [_zero_features_row(d, s) for d in dates for s in SYMBOLS]
    features = pd.DataFrame(feat_rows)
    features.to_csv(tmp_path / "features_daily.csv", index=False)

    label_rows = []
    for h in horizons:
        for d in dates:
            for s in SYMBOLS:
                label_rows.append({
                    "date": d, "symbol": s, "h": h,
                    "relevance": 4 if s == "AAA" else 0,
                    "r": 0.02 if s == "AAA" else -0.02,
                    "r_excess": 0.01 if s == "AAA" else -0.01,
                    "exit_date_resolved": d + pd.Timedelta(days=1),
                    "y": 1 if s == "AAA" else -1,
                    "t1": d + pd.Timedelta(days=1),
                })
    labels = pd.DataFrame(label_rows)
    labels.to_csv(tmp_path / "labels_daily.csv", index=False)

    splits = pd.DataFrame({"date": dates})
    splits["split"] = ["train" if i < train_end_idx else "test" for i in range(n_days)]
    splits.to_csv(tmp_path / "splits_daily.csv", index=False)

    universe = pd.DataFrame([{"date": d, "symbol": s} for d in dates for s in SYMBOLS])
    universe.to_csv(tmp_path / "universe_daily.csv", index=False)

    frozen_rows = []
    for h in horizons:
        frozen_rows.append({
            "nhom": "ranking_lambdarank", "family": "L1_lambdarank", "h": h,
            "chon_tren_validation": "{'num_leaves': 15, 'min_data_in_leaf': 2}",
        })
        frozen_rows.append({
            "nhom": "trend_classifier", "family": "T2_lightgbm", "h": h,
            "chon_tren_validation": "{'num_leaves': 15, 'min_data_in_leaf': 2}",
        })
    pd.DataFrame(frozen_rows).to_csv(tmp_path / "frozen_configurations_4_7.csv", index=False)

    config_path = _write_config(tmp_path, horizons, str(dates[train_end_idx]), str(dates[-1]))
    return config_path


# --- write_manifest: mã băm ổn định, đọc được lại ------------------------------------------------


def test_write_manifest_is_deterministic_and_changes_with_frozen_content(tmp_path):
    config_path = _write_dataset(tmp_path, horizons=[5])

    manifest1 = write_manifest(config_path, tmp_path)
    assert len(manifest1["ma_bam"]) == 64  # SHA256 hex đầy đủ
    assert manifest1["phien_ban"].endswith(manifest1["ma_bam"][:8])

    manifest2 = write_manifest(config_path, tmp_path)
    assert manifest2 == manifest1  # tất định, không đổi khi nội dung không đổi

    with open(tmp_path / "system_artifact_manifest.json", encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk == manifest1

    # Đổi nội dung frozen_configurations -> mã băm phải đổi theo (không phải hằng số cố định).
    frozen = pd.read_csv(tmp_path / "frozen_configurations_4_7.csv")
    frozen.loc[0, "chon_tren_validation"] = "{'num_leaves': 31, 'min_data_in_leaf': 2}"
    frozen.to_csv(tmp_path / "frozen_configurations_4_7.csv", index=False)
    manifest3 = write_manifest(config_path, tmp_path)
    assert manifest3["ma_bam"] != manifest1["ma_bam"]


# --- build_real_artifact: đúng schema, A1 chỉ ở A1_HORIZON=5 -------------------------------------


def test_build_real_artifact_schema_and_a1_only_at_h5(tmp_path, monkeypatch):
    config_path = _write_dataset(tmp_path, horizons=[5, 10])
    write_manifest(config_path, tmp_path)
    _gia_lap_momentum_scores(monkeypatch, pd.bdate_range("2024-01-01", periods=40))

    artefact = build_real_artifact(config_path, tmp_path)

    assert set(artefact["rank_scores"]["h"].unique()) == {5, 10}
    assert list(artefact["rank_scores"].columns) == ["ma", "t", "h", "diem_xep_hang"]
    assert set(artefact["rank_scores"]["ma"].unique()) == set(SYMBOLS)

    assert set(artefact["trend_scores"]["h"].unique()) == {5, 10}
    assert list(artefact["trend_scores"].columns) == ["ma", "t", "h", "p_tang", "p_ngang", "p_giam"]
    probs_sum = artefact["trend_scores"][["p_tang", "p_ngang", "p_giam"]].sum(axis=1)
    import numpy as np
    np.testing.assert_allclose(probs_sum.to_numpy(), 1.0, atol=1e-6)

    topk = artefact["topk_a1"]
    assert set(topk["h"].unique()) == {5}  # CHỈ h=5, đúng A1_HORIZON (quyết định người dùng)
    assert list(topk.columns) == ["t", "h", "hang", "ma", "san", "diem_xep_hang", "diem_xu_huong"]
    assert (topk.groupby("t")["hang"].apply(lambda s: sorted(s) == list(range(1, len(s) + 1)))).all()
    assert topk["san"].isin(["HOSE", "HNX"]).all()

    assert set(artefact["symbols"]["ma"]) == set(SYMBOLS)
    assert artefact["symbols"].loc[artefact["symbols"]["ma"] == "AAA", "san"].iloc[0] == "HOSE"

    results = artefact["test_results"]
    assert not results.empty
    assert list(results.columns) == [
        "h", "K", "muc_chi_phi", "cau_hinh", "rank_ic", "precision_at_k",
        "loi_suat_nam_hoa_rong", "sharpe_rong", "mdd", "turnover",
    ]
    assert set(results["h"].unique()) == {5}  # CHỈ h=5, đúng A1_HORIZON (S8, D109/D113)
    assert set(results["cau_hinh"].unique()) == {"A1", "B0", "B2"}
    assert set(results["K"].unique()) == {1, 2}  # trend_integration.K=1, K_sensitivity=[2] (test config)
    assert set(results["muc_chi_phi"].unique()) == {"thấp", "cơ sở", "cao"}
    # Không kiểm "không NaN" ở đây: fixture chỉ 2 mã/nhãn hằng số nên Rank IC/Sharpe có thể NaN một
    # cách HỢP LỆ (dưới 3 quan sát/phương sai bằng 0, đúng chính `rank_ic_series`/`sharpe_annualized`
    # đã định nghĩa) — trường hợp này test đầy đủ hơn ở `test_system_results.py` (3 mã, có biến thiên).

    manifest_path = tmp_path / "system_artifact_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    assert artefact["ma_bam"] == manifest["ma_bam"]
    assert artefact["phien_ban"] == manifest["phien_ban"]
    assert artefact["tu_ngay"] == manifest["tu_ngay"]
    assert artefact["den_ngay"] == manifest["den_ngay"]
