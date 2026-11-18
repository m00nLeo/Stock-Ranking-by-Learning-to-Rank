"""Kết quả test cuối cùng (bước 4.7, "T7 Kiểm tra tổng thể", charter mục 7/9). KHÔNG chạy lại bất
kỳ test nào: mỗi họ mô hình (B0-B4 ở 4.2, R1-R3 ở 4.3, L1 ở 4.4, F0-F2 ở 4.5, T2/D0-D2 ở 4.10) đã
TỰ chạy test MỘT LẦN cho cấu hình đã chọn trên validation của riêng nó rồi, đúng nguyên tắc "test
chỉ chạy một lần" (charter mục 5 điều 3). 4.7 chỉ TỔNG HỢP các kết quả đã có thành một báo cáo cuối
cùng và ghi lại danh sách cấu hình ĐÃ ĐÓNG BĂNG cho từng họ (đọc lại từ cột `chon_tren_validation`
đã lưu, không chọn lại) — cùng phạm vi hẹp như quyết định của người dùng cho bước có đặc tả mở
tương tự ở 4.6 (D99): chỉ tổng hợp, không viết mô hình mới, không chạy thêm thử nghiệm nào.

Holm trên 4 giả thuyết (H1-H4) CHƯA chạy ở đây: còn thiếu $p_{H4}$ (tích hợp xu hướng A0-A4, để ở
4.13, `giao_thuc_danh_gia.md` mục 4 yêu cầu đủ cả 4 p-giá trị mới xếp hạng và áp dụng Holm step-down).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

CODE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = CODE_ROOT / "config" / "config.yaml"

# (nhóm, tên file, tên cột họ mô hình trong file đó) — khớp đúng cột đã ghi ở 4.2-4.5.
RANK_IC_SOURCES: list[tuple[str, str, str]] = [
    ("baseline", "baseline_results_4_2.csv", "baseline"),
    ("regression_topk", "regression_topk_results_4_3.csv", "model"),
    ("ranking_lambdarank", "ranking_lambdarank_results_4_4.csv", "model"),
    ("horizon_aggregation", "horizon_aggregation_results_4_5.csv", "model"),
]


def load_rank_ic_sources(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Đọc bốn bảng kết quả test đã có (B0-B4, R1-R3, L1, F0-F2) — cùng họ thước đo Rank IC/ICIR."""
    return {nhom: pd.read_csv(data_dir / filename) for nhom, filename, _ in RANK_IC_SOURCES}


def consolidate_rank_ic_results(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Gộp bốn bảng kết quả Rank IC (chuẩn hóa cột tên họ mô hình khác nhau giữa các file: `baseline`
    ở 4.2, `model` ở 4.3-4.5) thành một bảng thống nhất, thêm cột `nhom`. Giữ nguyên MỌI cột của mỗi
    nguồn (horizon_aggregation có ít cột hơn — ví dụ không có `chon_tren_validation`/`p_value` vì
    F0/F1/F2 không tự chọn cấu hình ở 4.5 mà đọc lại từ 4.3/4.4); cột thiếu ở nguồn nào thành NaN."""
    rows = []
    for nhom, df in frames.items():
        family_col = "baseline" if "baseline" in df.columns else "model"
        part = df.rename(columns={family_col: "family"}).copy()
        part.insert(0, "nhom", nhom)
        rows.append(part)
    return pd.concat(rows, ignore_index=True, sort=False)


def frozen_configuration_summary(rank_ic_df: pd.DataFrame, trend_df: pd.DataFrame) -> pd.DataFrame:
    """Danh sách cấu hình ĐÃ ĐÓNG BĂNG cho từng họ/h — đọc lại từ cột `chon_tren_validation` đã lưu ở
    4.2-4.5 (Rank IC) và 4.10 (T2, macro-F1), KHÔNG chọn lại. Bỏ các dòng không có lưới (B0, B1, F0,
    F1, F2 — F0/F1/F2 tự chúng không chọn cấu hình riêng ở 4.5, D0-D2 không dùng đặc trưng)."""
    rank_ic_frozen = (
        rank_ic_df.dropna(subset=["chon_tren_validation"])[["nhom", "family", "h", "chon_tren_validation", "rank_ic_validation"]]
        .rename(columns={"rank_ic_validation": "validation_metric"})
    )
    rank_ic_frozen["metric_name"] = "rank_ic"

    trend_frozen = (
        trend_df.dropna(subset=["chon_tren_validation"])[["model", "h", "chon_tren_validation", "macro_f1_validation"]]
        .rename(columns={"model": "family", "macro_f1_validation": "validation_metric"})
    )
    trend_frozen.insert(0, "nhom", "trend_classifier")
    trend_frozen["metric_name"] = "macro_f1"

    cols = ["nhom", "family", "h", "chon_tren_validation", "validation_metric", "metric_name"]
    return pd.concat([rank_ic_frozen[cols], trend_frozen[cols]], ignore_index=True)


def hypotheses_summary(h1_row: dict, h2_row: dict, h3_p_value: float) -> pd.DataFrame:
    """Tổng hợp p-giá trị của ba giả thuyết chính đã kiểm định (H1 ở 4.4/D97, H2 ở 4.5/D98, H3 ở
    4.10/D100). H4 (tích hợp xu hướng A0-A4) CHƯA kiểm định — để trống (NaN), chưa đủ 4 giá trị để
    chạy Holm (`giao_thuc_danh_gia.md` mục 4)."""
    rows = [
        {"hypothesis": "H1", "p_value": float(h1_row["p_value"]), "note": "L1 so R2, h=5, Newey-West (4.4, D97) — KHONG duoc ung ho"},
        {"hypothesis": "H2", "p_value": float(h2_row["p_value"]), "note": "F0 so F1, h=5, Newey-West bien delta_IC (4.5, D98) — KHONG duoc ung ho"},
        {"hypothesis": "H3", "p_value": float(h3_p_value), "note": "T2 so D0, giao-hop max(p5,p10,p21), bootstrap khoi theo ngay (4.10, D100) — KHONG duoc ung ho"},
        {"hypothesis": "H4", "p_value": float("nan"), "note": "Tich hop xu huong A0-A4 — CHUA chay, de o 4.13"},
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = CODE_ROOT / "data" / "processed"

    rank_ic_sources = load_rank_ic_sources(data_dir)
    rank_ic_df = consolidate_rank_ic_results(rank_ic_sources)
    rank_ic_df.to_csv(data_dir / "test_results_frozen_4_7.csv", index=False)

    trend_df = pd.read_csv(data_dir / "trend_classifier_results_4_10.csv")
    frozen_df = frozen_configuration_summary(rank_ic_df, trend_df)
    frozen_df.to_csv(data_dir / "frozen_configurations_4_7.csv", index=False)

    h1_row = pd.read_csv(data_dir / "h1_test_4_4.csv").iloc[0]
    h2_row = pd.read_csv(data_dir / "h2_test_4_5.csv").iloc[0]
    h3_df = pd.read_csv(data_dir / "h3_test_4_10.csv")
    h3_p_value = float(h3_df["p_value"].max())  # giao-hop, da tinh o 4.10 (D100), doc lai de ro rang
    hyp_df = hypotheses_summary(h1_row, h2_row, h3_p_value)
    hyp_df.to_csv(data_dir / "hypotheses_summary_4_7.csv", index=False)

    print(f"Gop {len(rank_ic_sources)} bang ket qua Rank IC thanh {len(rank_ic_df)} dong -> test_results_frozen_4_7.csv")
    print(f"Danh sach cau hinh da dong bang ({len(frozen_df)} dong) -> frozen_configurations_4_7.csv")
    print(frozen_df.to_string(index=False))
    print("Tong hop ba gia thuyet da kiem dinh (H4 cho 4.13) -> hypotheses_summary_4_7.csv")
    print(hyp_df.to_string(index=False))
    print("KHONG chay lai bat ky test nao o buoc nay - chi tong hop ket qua da co.")
