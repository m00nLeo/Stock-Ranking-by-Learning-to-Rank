"""Hiệu chỉnh nhiều so sánh cho harness đánh giá (bước 4.1): kiểm định đồng thời 4 giả thuyết
chính H1-H4 với mức ý nghĩa chung alpha_FW = 0,05 theo thủ tục Holm step-down (mục 4 của
giao_thuc_danh_gia.md, chốt D60). Chỉ dùng list/sort thuần Python, không phụ thuộc scipy/statsmodels.
"""

from __future__ import annotations


def holm_step_down(p_values: list[float], alpha: float) -> list[bool]:
    """Bác bỏ H0 cho từng p-giá trị theo Holm step-down, trả kết quả đúng thứ tự ban đầu.

    Sắp xếp p tăng dần, giữ chỉ số gốc; đi từ p nhỏ nhất, giả thuyết thứ i (theo thứ tự đã sắp)
    bị bác bỏ nếu p_(j) <= alpha / (m - j + 1) đúng với MỌI j <= i. Hễ gặp một giả thuyết không
    vượt ngưỡng thì dừng, mọi giả thuyết từ đó trở đi đều không bị bác bỏ. True = bác bỏ H0.

    `alpha` không có giá trị mặc định: người gọi phải truyền từ `config.yaml` (`hypotheses.alpha`,
    D54/D60 chốt 0,05), theo đúng cách `newey_west_test` bắt buộc `lag` và
    `circular_block_bootstrap_mean_ci` bắt buộc `n_boot` ở `harness.py` — không hardcode hằng số
    đã chốt theo giao thức bên trong hàm.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    rejected = [False] * m
    for rank, i in enumerate(order, start=1):
        if p_values[i] > alpha / (m - rank + 1):
            break
        rejected[i] = True
    return rejected