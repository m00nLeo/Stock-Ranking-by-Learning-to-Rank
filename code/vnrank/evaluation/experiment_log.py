"""Tổng hợp log thử nghiệm (bước 4.6). Đặc tả: charter.md dòng 4.6 ("tinh chỉnh siêu tham số chỉ
trên validation, ngân sách thử cố định và có log") — theo quyết định của người dùng (13/11/2026),
phần "tinh chỉnh" đã làm xuyên suốt 4.2-4.5 (mọi cấu hình chọn trên VALIDATION, lưới đã đóng băng ở
2.9, mọi ô kể cả ô không chọn đều ghi log); 4.6 chỉ còn việc GỘP các log lưới rời rạc đã có thành
MỘT log thử nghiệm thống nhất và đối chiếu số cấu hình thực tế ($N$) với ước lượng thô ở
`ma_tran_baseline_ablation.md` mục 6 ("khoảng 20 cấu hình cho mỗi $h$").

[SỬA D116/D118, 17/11/2026] Bổ sung T2 (4.10) và A0-A3 (4.13) để ghi chép ĐẦY ĐỦ mọi thử nghiệm
của dự án (`consolidate_frames`/`load_and_consolidate`/`summarize_by_h` GIỮ NGUYÊN, chỉ phục vụ
3 nguồn 4.2-4.4 cũ — không đổi hành vi, không phá test cũ). Hàm MỚI `load_and_consolidate_all`
sinh bảng mở rộng {buoc, family, h, config, metric_name, metric_value, chon} — tổng quát hóa vì
T2 đo bằng macro-F1 (không phải Rank IC) và A0-A3 (4.13) không phải lưới siêu tham số chọn trên
validation (không có cột "chon"/"config" gốc, chỉ có precision@K TRÊN TEST cho 4 biến thể
ablation) — ghi `metric_name="precision_test"` để phân biệt rõ, KHÔNG lẫn với các dòng
validation-selected khác. `chon=True` cho đúng cấu hình đã triển khai (A1).

LƯU Ý QUAN TRỌNG (D118): bảng mở rộng này CHỈ phục vụ đối chiếu N với ước lượng thô — KHÔNG phải
tập N dùng để tính DSR/PBO (`vnrank/evaluation/dsr_pbo.py`, `trial_portfolios.py`). T2 phục vụ một
bài toán chọn mô hình KHÁC (chọn bộ phân loại xu hướng cho H3), không cạnh tranh để trở thành cấu
hình xếp hạng chính như L1 — DSR/PBO của L1 chỉ dùng 21 cấu hình (B2-B4/R1-R3/L1, h=5); DSR/PBO
của A1 chỉ dùng 4 cấu hình (A0-A3, h=5). Quyết định người dùng 17/11/2026.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]

# (tên bước, tên file log, tên cột họ mô hình trong file đó) — khớp đúng cột đã ghi ở 4.2/4.3/4.4.
LOG_SOURCES: list[tuple[str, str, str]] = [
    ("4.2", "baseline_grid_search_log_4_2.csv", "baseline"),
    ("4.3", "regression_topk_grid_search_log_4_3.csv", "model"),
    ("4.4", "ranking_lambdarank_grid_search_log_4_4.csv", "model"),
]


def consolidate_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Gộp các log lưới rời rạc (mỗi file đã có cột `h`, `config`, `rank_ic_validation`, `chon`,
    và một cột tên họ mô hình khác nhau giữa các file — `baseline` ở 4.2, `model` ở 4.3/4.4) thành
    một bảng thống nhất {buoc, family, h, config, rank_ic_validation, chon}. `frames`: {tên bước:
    DataFrame}, mỗi DataFrame có đúng MỘT trong hai cột `baseline`/`model`."""
    rows = []
    for buoc, df in frames.items():
        family_col = "baseline" if "baseline" in df.columns else "model"
        part = df.rename(columns={family_col: "family"})
        part = part[["family", "h", "config", "rank_ic_validation", "chon"]].copy()
        part.insert(0, "buoc", buoc)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def load_and_consolidate(data_dir: Path = CODE_ROOT / "data" / "processed") -> pd.DataFrame:
    """Đọc ba log lưới đã có ở 4.2/4.3/4.4 từ `data_dir` và gộp bằng `consolidate_frames`."""
    frames = {buoc: pd.read_csv(data_dir / filename) for buoc, filename, _ in LOG_SOURCES}
    return consolidate_frames(frames)


def summarize_by_h(log_df: pd.DataFrame) -> pd.DataFrame:
    """Đếm số cấu hình đã thử (N) theo từng $h$, tách theo bước — để đối chiếu với ước lượng thô
    "khoảng 20 cấu hình cho mỗi $h$" (`ma_tran_baseline_ablation.md` mục 6)."""
    pivot = log_df.groupby(["h", "buoc"]).size().unstack(fill_value=0)
    pivot["tong"] = pivot.sum(axis=1)
    return pivot.reset_index()


def _rename_to_generic(df: pd.DataFrame, buoc: str, family_col: str, metric_col: str, metric_name: str) -> pd.DataFrame:
    part = df.rename(columns={family_col: "family", metric_col: "metric_value"})
    part = part[["family", "h", "config", "metric_value", "chon"]].copy()
    part.insert(0, "buoc", buoc)
    part.insert(4, "metric_name", metric_name)
    return part


def consolidate_frames_all(
    frames_423: dict[str, pd.DataFrame], trend_classifier_log: pd.DataFrame, trend_integration_results: pd.DataFrame
) -> pd.DataFrame:
    """Bảng mở rộng {buoc, family, h, config, metric_name, metric_value, chon} (D116/D118, xem
    docstring đầu file): gộp 4.2-4.4 (`frames_423`, như `consolidate_frames`) với T2 (4.10,
    `trend_classifier_log` — cột `model`/`macro_f1_validation`) và A0-A3 (4.13,
    `trend_integration_results` — cột `ablation`/`precision_mean`, KHÔNG có cột `config`/`chon`
    gốc nên tự gán: `config`=nhãn ablation, `metric_name`="precision_test" (TRÊN TEST, không phải
    validation — khác hẳn các dòng còn lại, không được lẫn vào N của DSR), `chon`=(ablation=="A1"))."""
    base_423 = consolidate_frames(frames_423)
    base_423.insert(4, "metric_name", "rank_ic")
    base_423 = base_423.rename(columns={"rank_ic_validation": "metric_value"})
    base_423 = base_423[["buoc", "family", "h", "config", "metric_name", "metric_value", "chon"]]

    t2_part = _rename_to_generic(trend_classifier_log, "4.10", "model", "macro_f1_validation", "macro_f1")

    a_part = trend_integration_results.copy()
    a_part["family"] = "trend_integration"
    a_part["config"] = a_part["ablation"]
    a_part["chon"] = a_part["ablation"] == "A1"
    a_part = a_part.rename(columns={"precision_mean": "metric_value"})
    a_part.insert(0, "buoc", "4.13")
    a_part.insert(4, "metric_name", "precision_test")
    a_part = a_part[["buoc", "family", "h", "config", "metric_name", "metric_value", "chon"]]

    return pd.concat([base_423, t2_part, a_part], ignore_index=True)


def load_and_consolidate_all(data_dir: Path = CODE_ROOT / "data" / "processed") -> pd.DataFrame:
    """Đọc đủ năm nguồn (4.2-4.4, 4.10, 4.13) từ `data_dir` và gộp bằng `consolidate_frames_all`."""
    frames_423 = {buoc: pd.read_csv(data_dir / filename) for buoc, filename, _ in LOG_SOURCES}
    trend_classifier_log = pd.read_csv(data_dir / "trend_classifier_grid_search_log_4_10.csv")
    trend_integration_results = pd.read_csv(data_dir / "trend_integration_results_4_13.csv")
    return consolidate_frames_all(frames_423, trend_classifier_log, trend_integration_results)


if __name__ == "__main__":
    log_df = load_and_consolidate_all()
    out_path = CODE_ROOT / "data" / "processed" / "experiment_log_4_6.csv"
    # LUU Y khi doc lai file nay: cot "buoc" la CHUOI ("4.10" khac "4.2"), truyen
    # dtype={"buoc": str} cho pd.read_csv - mac dinh pandas suy dien so thuc se lam "4.10" thanh
    # 4.1, trung voi "4.1" khong ton tai nhung lam mat phan biet voi cac buoc khac.
    log_df.to_csv(out_path, index=False)

    summary = summarize_by_h(log_df)
    print(summary.to_string(index=False))
    print(f"Tong so cau hinh da thu (N, ca ba h, da gop T2/A0-A3): {len(log_df)}")
    print("Uoc luong tho o ma_tran_baseline_ablation.md muc 6: ~20 cau hinh moi h "
          "(3 hoi quy x3 + L1 x3 + T2 x3 + vai o momentum).")
    print("Tap N dung cho DSR/PBO (D118): KHONG phai toan bo bang nay - xem "
          "vnrank/evaluation/dsr_pbo.py (L1: 21 cau hinh B2-B4/R1-R3/L1 o h=5; A1: 4 cau hinh A0-A3 o h=5).")
    print(f"Da luu log thu nghiem hop nhat (mo rong) vao {out_path}")
