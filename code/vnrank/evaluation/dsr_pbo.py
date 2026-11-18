"""DSR (Deflated Sharpe Ratio) và PBO (Probability of Backtest Overfitting) qua CSCV, bước 4.14.

Công thức chép nguyên văn từ đọc trực tiếp hai PDF gốc (`decision_log.md` D116, KHÔNG tự suy diễn
công thức):
- DSR: Bailey & López de Prado (2014), trang 7-9, 12-14.
- PBO/CSCV: Bailey, Borwein, López de Prado & Zhu (2017), trang 9-12.

Phạm vi (D116, quyết định người dùng 17/11/2026 về tập N): DSR/PBO tính RIÊNG cho hai nhóm tìm
kiếm khác nhau (N = số thử nghiệm trong CÙNG một lần tìm kiếm dẫn tới cấu hình được chọn):
- Nhóm chọn cỗ máy xếp hạng chính: L1 cạnh tranh với B2-B4/R1-R3, N=21 cấu hình ở h=5.
- Nhóm ablation tích hợp xu hướng: A1 cạnh tranh với A0/A2/A3, N=4 cấu hình ở h=5.
T2 (4.10) không thuộc nhóm nào (phục vụ bài toán chọn mô hình khác — chọn bộ phân loại xu hướng
cho H3), không tham gia phép tính ở đây; log 4.6 vẫn ghi T2 để đối chiếu N với ước lượng thô.

Đơn vị Sharpe (QUAN TRỌNG, xác nhận bằng ví dụ số D116 — N=100, V=0,5, T=1250, γ3=-3, γ4=10,
SR năm hóa=2,5 → SR0≈0,1132, DSR≈0,9004): `V[{SR_n}]` trong công thức SR0 là phương sai của Sharpe
NĂM HÓA giữa các thử nghiệm (đơn vị tự nhiên để so sánh chéo cấu hình); SR0 tính ra cũng ở đơn vị
năm hóa, phải CHIA cho hệ số năm hóa (căn(250/h)) mới cùng đơn vị với `sr_test` (Sharpe KHÔNG năm
hóa của cấu hình đã chọn) để dùng trong thống kê DSR. Khớp đúng ví dụ số khi làm theo cách này (xem
`tests/test_dsr_pbo.py::test_deflated_sharpe_ratio_matches_worked_example`).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def sharpe_raw(r: np.ndarray) -> float:
    """Sharpe KHÔNG năm hóa: trung bình / độ lệch chuẩn (ddof=1) của chuỗi lợi suất kỳ."""
    r = np.asarray(r, dtype=float)
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(r.mean() / sd)


def estimate_effective_n(trial_returns: dict[str, np.ndarray]) -> dict:
    """N ước lượng khi các thử nghiệm KHÔNG độc lập (D116, trang 14):
    N_hat = rho_hat + (1 - rho_hat) * M, rho_hat = tương quan trung bình theo cặp giữa chuỗi lợi
    suất của M cấu hình thật đã thử. `trial_returns`: {tên cấu hình: chuỗi lợi suất kỳ, CÙNG độ
    dài/cùng lịch rebalance — mọi cấu hình trong một nhóm tìm kiếm dùng chung lịch rebalance ở h
    cố định nên luôn thẳng hàng theo chỉ số)."""
    keys = list(trial_returns.keys())
    m = len(keys)
    if m < 2:
        return {"n_hat": float(m), "rho_hat": float("nan"), "m": m}
    mat = pd.DataFrame({k: pd.Series(np.asarray(trial_returns[k], dtype=float)) for k in keys})
    corr = mat.corr().to_numpy()
    pairs = list(combinations(range(m), 2))
    rhos = [corr[i, j] for i, j in pairs if np.isfinite(corr[i, j])]
    rho_hat = float(np.mean(rhos)) if rhos else 0.0
    n_hat = rho_hat + (1 - rho_hat) * m
    return {"n_hat": float(n_hat), "rho_hat": rho_hat, "m": m}


def deflated_sharpe_ratio(
    sr_test: float,
    n_trials: float,
    var_sr_trials_annualized: float,
    ann_factor: float,
    t_obs: int,
    skew: float,
    kurt: float,
) -> dict:
    """DSR (D116, trang 7-9, 12-13).

    `sr_test`: Sharpe KHÔNG năm hóa của cấu hình đã chọn, trên test.
    `n_trials`: N (số thử nghiệm độc lập, có thể là N_hat khi không độc lập — `estimate_effective_n`).
    `var_sr_trials_annualized`: phương sai Sharpe NĂM HÓA giữa các thử nghiệm (V[{SR_n}]).
    `ann_factor`: hệ số năm hóa dùng để quy SR0 (ra ở đơn vị năm hóa) về cùng đơn vị với `sr_test`
    (căn(250/h), `portfolio_backtest.sharpe_annualized`).
    `t_obs`: số quan sát (kỳ) trên test.
    `skew`/`kurt`: độ lệch/độ nhọn CHUẨN (kurt=3 ứng với phân phối chuẩn, KHÔNG phải excess
    kurtosis — công thức độ lệch chuẩn Sharpe của Mertens/Opdyke, khớp ví dụ số D116).
    """
    if n_trials is None or not np.isfinite(n_trials) or n_trials <= 1:
        sr0 = 0.0
    else:
        z_a = norm.ppf(1 - 1.0 / n_trials)
        z_b = norm.ppf(1 - 1.0 / (n_trials * np.e))
        sr0_annualized = np.sqrt(var_sr_trials_annualized) * (
            (1 - EULER_MASCHERONI) * z_a + EULER_MASCHERONI * z_b
        )
        sr0 = float(sr0_annualized) / ann_factor
    denom = np.sqrt(max(1 - skew * sr_test + (kurt - 1) / 4 * sr_test**2, 1e-12))
    z_stat = (sr_test - sr0) * np.sqrt(t_obs - 1) / denom
    dsr = float(norm.cdf(z_stat))
    return {
        "sr0": float(sr0),
        "dsr": dsr,
        "z_stat": float(z_stat),
        "n_trials": float(n_trials),
        "t_obs": int(t_obs),
        "sr_test": float(sr_test),
    }


def cscv_pbo(returns_matrix: pd.DataFrame, s: int) -> dict:
    """PBO qua CSCV (D116, trang 9-12). `returns_matrix`: T x N, chuỗi lợi suất ngày của N cấu
    hình trong MỘT nhóm tìm kiếm, cùng lịch (index = phiên, cột = tên cấu hình, không NaN — đã
    cắt đồng bộ trước khi gọi). `s`: số khối CHẴN (`config.yaml` `hypotheses.pbo_S`, D62).

    5 bước (D116): (1) ma trận T x N (đầu vào); (2) chia S khối hàng rời nhau bằng nhau (bỏ phần
    dư cuối nếu T không chia hết cho S); (3) mọi tổ hợp C(S, S/2) cách ghép S/2 khối làm tập huấn
    luyện; (4) mỗi tổ hợp: Sharpe IS/OOS mỗi cấu hình, n* = cấu hình Sharpe IS cao nhất, hạng
    tương đối OOS của n* (omega_bar), logit lambda; (5) PBO = tỷ lệ tổ hợp có lambda <= 0."""
    if s % 2 != 0:
        raise ValueError("S phải là số chẵn (D116)")
    mat = returns_matrix.to_numpy()
    t_obs, n = mat.shape
    block_size = t_obs // s
    used = block_size * s
    blocks = [mat[i * block_size : (i + 1) * block_size] for i in range(s)]
    half = s // 2

    lambdas = []
    for train_idx in combinations(range(s), half):
        test_idx = [i for i in range(s) if i not in train_idx]
        train_mat = np.vstack([blocks[i] for i in train_idx])
        test_mat = np.vstack([blocks[i] for i in test_idx])
        is_sharpe = np.array([sharpe_raw(train_mat[:, j]) for j in range(n)])
        best_idx = int(np.nanargmax(is_sharpe))
        oos_sharpe = np.array([sharpe_raw(test_mat[:, j]) for j in range(n)])
        rank_oos = pd.Series(oos_sharpe).rank(method="average")
        omega_bar = float(rank_oos.iloc[best_idx]) / (n + 1)
        if 0 < omega_bar < 1:
            lam = float(np.log(omega_bar / (1 - omega_bar)))
        else:
            lam = float("inf") if omega_bar >= 1 else float("-inf")
        lambdas.append(lam)

    lambdas_arr = np.array(lambdas)
    finite = lambdas_arr[np.isfinite(lambdas_arr)]
    pbo = float(np.mean(finite <= 0)) if len(finite) else float("nan")
    return {
        "pbo": pbo,
        "n_combos": len(lambdas_arr),
        "s": s,
        "block_size": block_size,
        "n_sessions_used": used,
        "n_sessions_total": t_obs,
    }
