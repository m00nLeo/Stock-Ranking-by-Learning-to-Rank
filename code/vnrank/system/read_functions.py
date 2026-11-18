"""Ba hàm đọc của hệ thống minh họa (S2): `xep_hang`, `xu_huong`, `ket_qua`.

Đây là lối duy nhất để lấy số liệu, dùng chung cho giao diện Gradio và chatbot. Ba hàm chỉ
**đọc** artefact đã đóng băng (không tính lại gì, không trả kết quả thực tế sau `t_as-of`),
đúng hợp đồng JSON ở `system_architecture.md` mục 4 (bản chốt D72):

- Mọi tham số được **kiểm trước khi đọc**, theo thứ tự tham số trong chữ ký; tham số sai thì trả
  đúng một dạng `{"loi": "<mô tả>"}` (tiếng Việt, nêu tham số sai và tập giá trị hợp lệ), không có
  mã lỗi số, không đoán giá trị.
- `ngay` là chuỗi `"YYYY-MM-DD"`; $h$ và $K$ là số nguyên; mã cổ phiếu là chuỗi.
- `ma_bam` trả về là **8 ký tự đầu** của mã băm cấu hình (system_architecture.md mục 8.6).

Hằng số đều đọc từ `config/config.yaml`, không lặp lại trong code: tập $h$ hợp lệ (`horizons`),
tập $K$ hợp lệ (`trend_integration.K` + `trend_integration.K_sensitivity`), mốc giai đoạn test
(`data_split.test_start`, `data_split.test_end`), seed (`seed`). Ba mức chi phí `muc_chi_phi`
(`"thấp"`, `"cơ sở"`, `"cao"`) là giá trị nguyên văn trong đặc tả mục 7 / system_architecture
mục 8.4; giá trị số của chúng nằm ở `costs.c_rt_base` và `costs.c_rt_sensitivity` (đường ống
nghiên cứu dùng khi ghi artefact — ba hàm đọc chỉ lọc theo nhãn chuỗi, không cần số chi phí).

Artefact được nạp qua `nap_artefact` (bởi `vnrank.system.artifact_loader` — S3/S4, D107; S8, D113
— hoặc `tao_du_lieu_mau`/`nap_du_lieu_mau` ở dưới cho dữ liệu mẫu giả) thành một dict chỉ đọc với
các khóa sau (ngày đều là chuỗi `"YYYY-MM-DD"`). `artifact_loader` dựng THẬT `rank_scores`/`trend_scores`
(mọi $h$), `topk_a1` (chỉ $h=5$) và `test_results` (A1/B0/B2 × $K$ × chi phí, CHỈ $h=5$ — `ket_qua`
báo lỗi rõ ràng cho $h=10, 21$, cùng tinh thần `xep_hang`/`xu_huong`, xem `artifact_loader.py` đầu
file):

- `phien_ban` (str), `ma_bam` (str, mã băm đầy đủ), `tu_ngay` (str), `den_ngay` (str);
- `symbols` (DataFrame: `ma`, `san`);
- `rank_scores` (DataFrame: `ma`, `t`, `h`, `diem_xep_hang`) — điểm $s^{\\mathrm{rank}}$ của A1;
- `trend_scores` (DataFrame: `ma`, `t`, `h`, `p_tang`, `p_ngang`, `p_giam`) — $\\hat{\\mathbf p}$ của T2;
- `topk_a1` (DataFrame: `t`, `h`, `hang`, `ma`, `san`, `diem_xep_hang`, `diem_xu_huong`);
- `test_results` (DataFrame: `h`, `K`, `muc_chi_phi`, `cau_hinh`, `rank_ic`, `precision_at_k`,
  `loi_suat_nam_hoa_rong`, `sharpe_rong`, `mdd`, `turnover`).

`tao_du_lieu_mau`/`nap_du_lieu_mau` sinh dữ liệu giả tất định cho demo S2, tạm thay artefact thật
(artefact thật chưa có ở bước S2).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = CODE_ROOT / "config" / "config.yaml"

_CACHED_CONFIG: dict | None = None


def _nap_cau_hinh(config_path: Path = CONFIG_PATH) -> dict:
    global _CACHED_CONFIG
    if _CACHED_CONFIG is None:
        with open(config_path, encoding="utf-8") as f:
            _CACHED_CONFIG = yaml.safe_load(f)
    return _CACHED_CONFIG


_CFG = _nap_cau_hinh()
_HORIZONS: list[int] = list(_CFG["horizons"])  # h hợp lệ, đúng thứ tự config.yaml
_K_HOP_LE: list[int] = sorted(
    {_CFG["trend_integration"]["K"], *_CFG["trend_integration"]["K_sensitivity"]}
)  # K hợp lệ {5, 10, 20}
_MUC_CHI_PHI: list[str] = ["thấp", "cơ sở", "cao"]  # nhãn chuỗi theo đặc tả mục 7 (không phải số)
TEST_START: str = pd.Timestamp(_CFG["data_split"]["test_start"]).strftime("%Y-%m-%d")
TEST_END: str = _CFG["data_split"]["test_end"]
SEED: int = int(_CFG["seed"])

# Thứ tự cấu hình trong bảng `ket_qua` (A1, B0, và baseline tín hiệu tốt nhất — ở mock lấy B2).
_THU_TU_CAU_HINH: list[str] = ["A1", "B0", "B2"]

# Artefact hiện hành (do `artifact_loader.py`/`nap_artefact` đặt vào).
_ARTEFACT: dict | None = None


def _loi(mo_ta: str) -> dict:
    return {"loi": mo_ta}


def _tap_hop(gia_tri: list[int]) -> str:
    """Chuỗi `{5, 10, 21}` cho thông báo lỗi, đúng định dạng hợp đồng mục 4."""
    return "{" + ", ".join(str(v) for v in gia_tri) + "}"


def _so_nguyen(x) -> int | None:
    """Ép `x` thành int nếu là số nguyên (int, float nguyên, chuỗi số); ngược lại trả None."""
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x) if x.is_integer() else None
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            return None
    return None


def _kiem_ngay_dinh_dang(ngay) -> dict | None:
    if not isinstance(ngay, str):
        return _loi("ngay phải có dạng YYYY-MM-DD")
    try:
        _dt.datetime.strptime(ngay, "%Y-%m-%d")
    except ValueError:
        return _loi("ngay phải có dạng YYYY-MM-DD")
    return None


def _kiem_ngay_trong_test(ngay: str) -> dict | None:
    if not (TEST_START <= ngay <= TEST_END):
        return _loi(f"ngay nằm ngoài giai đoạn test (từ {TEST_START} đến {TEST_END})")
    return None


def _kiem_h(h) -> tuple[dict | None, int | None]:
    h_int = _so_nguyen(h)
    if h_int not in _HORIZONS:
        return _loi(f"h phải thuộc {_tap_hop(_HORIZONS)}"), None
    return None, h_int


def _kiem_K(K) -> tuple[dict | None, int | None]:
    K_int = _so_nguyen(K)
    if K_int not in _K_HOP_LE:
        return _loi(f"K phải thuộc {_tap_hop(_K_HOP_LE)}"), None
    return None, K_int


def _kiem_muc_chi_phi(muc_chi_phi) -> dict | None:
    if muc_chi_phi not in _MUC_CHI_PHI:
        return _loi("muc_chi_phi phải là một trong: thấp, cơ sở, cao")
    return None


def nap_artefact(artefact: dict) -> None:
    """Đặt artefact hiện hành cho ba hàm đọc (thay vai trò `artifact_loader.py` ở S1)."""
    global _ARTEFACT
    _ARTEFACT = artefact


def _xoa_artefact() -> None:
    global _ARTEFACT
    _ARTEFACT = None


def _artefact_hien_hanh() -> dict | None:
    return _ARTEFACT


def _artefact_hoac_loi() -> dict:
    if _ARTEFACT is None:
        return _loi("chưa nạp artefact (gọi nap_artefact trước)")
    return _ARTEFACT


def _ma_bam_ngan(artefact: dict) -> str:
    return str(artefact["ma_bam"])[:8]


def _luoi_ngay(artefact: dict, h: int) -> list[str]:
    """Lưới ngày quyết định của chuỗi `h` (tăng dần), lấy từ `topk_a1`."""
    topk = artefact["topk_a1"]
    return sorted(str(d) for d in topk.loc[topk["h"] == h, "t"].unique())


def _ngay_gan_nhat(artefact: dict, h: int, ngay: str) -> str:
    """Ngày quyết định gần nhất $\\le$ `ngay` trên lưới của `h` (mục 4.1)."""
    return max(d for d in _luoi_ngay(artefact, h) if d <= ngay)


def xep_hang(ngay, h, K):
    """Top-$K$ của cấu hình A1 tại ngày quyết định gần nhất $\\le$ `ngay` (mục 4.2).

    Kiểm tham số theo thứ tự: `ngay` (định dạng, trong giai đoạn test), `h`, `K`; rồi kiểm
    `ngay` không trước ngày quyết định đầu tiên của chuỗi `h` (mục 8.5). Thành công trả JSON
    gồm `ngay_du_lieu`, `h`, `K`, `cau_hinh` (luôn `"A1"`), `ma_bam` và `danh_sach` đúng `K`
    phần tử theo `hang` tăng dần.
    """
    e = _kiem_ngay_dinh_dang(ngay)
    if e:
        return e
    e = _kiem_ngay_trong_test(ngay)
    if e:
        return e
    e, h = _kiem_h(h)
    if e:
        return e
    e, K = _kiem_K(K)
    if e:
        return e

    artefact = _artefact_hoac_loi()
    if "loi" in artefact:
        return artefact

    dates = _luoi_ngay(artefact, h)
    if not dates:
        return _loi(f"chưa có dữ liệu A1 cho h={h} (A1 chỉ được xây dựng và đánh giá ở h=5, bước 4.13)")
    if ngay < dates[0]:
        return _loi(f"ngay trước ngày quyết định đầu tiên của chuỗi h này ({dates[0]})")

    ngay_du_lieu = _ngay_gan_nhat(artefact, h, ngay)
    topk = artefact["topk_a1"]
    sub = topk[(topk["t"] == ngay_du_lieu) & (topk["h"] == h)].sort_values("hang").head(K)
    danh_sach = [
        {
            "hang": int(r["hang"]),
            "ma": str(r["ma"]),
            "san": str(r["san"]),
            "diem_xep_hang": float(r["diem_xep_hang"]),
            "diem_xu_huong": float(r["diem_xu_huong"]),
        }
        for _, r in sub.iterrows()
    ]
    return {
        "ngay_du_lieu": ngay_du_lieu,
        "h": h,
        "K": K,
        "cau_hinh": "A1",
        "ma_bam": _ma_bam_ngan(artefact),
        "danh_sach": danh_sach,
    }


def xu_huong(ma, ngay, h):
    """Xu hướng của một mã cổ phiếu tại ngày quyết định gần nhất $\\le$ `ngay` (mục 4.3).

    Kiểm tham số theo thứ tự: `ma` (chuỗi mã cổ phiếu), `ngay` (định dạng, trong giai đoạn
    test), `h`; rồi kiểm `ngay` không trước ngày quyết định đầu tiên của `h`, và `ma` thuộc
    $U_t$ tại `ngay_du_lieu`. Thành công trả `p_tang`, `p_ngang`, `p_giam` (xác suất thô),
    `diem_xu_huong` ($\\hat p^{+} - \\hat p^{-}$) và `hang` của mã theo $s^{\\mathrm{rank}}$ của A1.
    """
    if not isinstance(ma, str) or not ma.strip():
        return _loi("ma phải là chuỗi mã cổ phiếu")
    e = _kiem_ngay_dinh_dang(ngay)
    if e:
        return e
    e = _kiem_ngay_trong_test(ngay)
    if e:
        return e
    e, h = _kiem_h(h)
    if e:
        return e

    artefact = _artefact_hoac_loi()
    if "loi" in artefact:
        return artefact

    dates = _luoi_ngay(artefact, h)
    if not dates:
        return _loi(f"chưa có dữ liệu A1 cho h={h} (A1 chỉ được xây dựng và đánh giá ở h=5, bước 4.13)")
    if ngay < dates[0]:
        return _loi(f"ngay trước ngày quyết định đầu tiên của chuỗi h này ({dates[0]})")

    ngay_du_lieu = _ngay_gan_nhat(artefact, h, ngay)

    rank_scores = artefact["rank_scores"]
    u_t = rank_scores[(rank_scores["t"] == ngay_du_lieu) & (rank_scores["h"] == h)]
    if ma not in set(u_t["ma"]):
        return _loi(f"ma {ma} không thuộc tập mã tại ngày {ngay_du_lieu}")

    ranked = u_t.sort_values("diem_xep_hang", ascending=False).reset_index(drop=True)
    hang = int(ranked.index[ranked["ma"] == ma][0]) + 1

    trend_scores = artefact["trend_scores"]
    row = trend_scores[
        (trend_scores["t"] == ngay_du_lieu) & (trend_scores["h"] == h) & (trend_scores["ma"] == ma)
    ].iloc[0]
    p_tang = float(row["p_tang"])
    p_ngang = float(row["p_ngang"])
    p_giam = float(row["p_giam"])
    return {
        "ngay_du_lieu": ngay_du_lieu,
        "ma": ma,
        "h": h,
        "ma_bam": _ma_bam_ngan(artefact),
        "p_tang": p_tang,
        "p_ngang": p_ngang,
        "p_giam": p_giam,
        "diem_xu_huong": p_tang - p_giam,
        "hang": hang,
    }


def ket_qua(h, K, muc_chi_phi):
    """Bảng kết quả trên toàn giai đoạn test đã đóng băng (mục 4.4), không tính lại gì.

    Kiểm tham số theo thứ tự: `h`, `K`, `muc_chi_phi`. Thành công trả `tu_ngay`, `den_ngay`
    (đọc từ artefact), `ma_bam` và `bang` gồm 3 cấu hình với đúng sáu thước đo mục 7.
    """
    e, h = _kiem_h(h)
    if e:
        return e
    e, K = _kiem_K(K)
    if e:
        return e
    e = _kiem_muc_chi_phi(muc_chi_phi)
    if e:
        return e

    artefact = _artefact_hoac_loi()
    if "loi" in artefact:
        return artefact

    results = artefact["test_results"]
    if results.empty:
        return _loi("chưa có bảng kết quả thật cho ket_qua (S8 chưa làm); dùng nap_du_lieu_mau để xem demo giao diện")
    if h not in set(results["h"].unique()):
        return _loi(f"chưa có dữ liệu A1 cho h={h} (A1 chỉ được xây dựng và đánh giá ở h=5, bước 4.13)")

    sub = results[
        (results["h"] == h) & (results["K"] == K) & (results["muc_chi_phi"] == muc_chi_phi)
    ].copy()
    order = {c: i for i, c in enumerate(_THU_TU_CAU_HINH)}
    sub["_tt"] = sub["cau_hinh"].map(order).fillna(len(order))
    sub = sub.sort_values("_tt")

    bang = [
        {
            "cau_hinh": str(r["cau_hinh"]),
            "rank_ic": float(r["rank_ic"]),
            "precision_at_k": float(r["precision_at_k"]),
            "loi_suat_nam_hoa_rong": float(r["loi_suat_nam_hoa_rong"]),
            "sharpe_rong": float(r["sharpe_rong"]),
            "mdd": float(r["mdd"]),
            "turnover": float(r["turnover"]),
        }
        for _, r in sub.iterrows()
    ]
    return {
        "h": h,
        "K": K,
        "muc_chi_phi": muc_chi_phi,
        "tu_ngay": str(artefact["tu_ngay"]),
        "den_ngay": str(artefact["den_ngay"]),
        "ma_bam": _ma_bam_ngan(artefact),
        "bang": bang,
    }


def tao_du_lieu_mau(seed: int | None = None) -> dict:
    """Sinh artefact giả tất định cho demo S2 (tạm thay artefact thật, chưa có ở S2).

    Dùng `numpy.random.RandomState` với seed từ `config.yaml` (mặc định), nên chạy lại cho cùng
    dữ liệu. Lưới ngày quyết định là đầu tháng từ `test_start + 1 tháng` đến hết `test_end`.
    """
    if seed is None:
        seed = SEED
    rng = np.random.RandomState(seed)

    # 30 mã giả 3 ký tự, sàn xen kẽ HOSE/HNX.
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    codes = sorted({letters[i // 26] + letters[i % 26] + letters[(i * 7) % 26] for i in range(30)})
    sans = ["HOSE" if i % 2 == 0 else "HNX" for i in range(len(codes))]
    symbols = pd.DataFrame({"ma": codes, "san": sans})

    grid_start = pd.Timestamp(TEST_START) + pd.DateOffset(months=1)
    dates = pd.date_range(grid_start, periods=18, freq="MS")
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    rank_rows = []
    trend_rows = []
    for h in _HORIZONS:
        for t in date_strs:
            scores = rng.normal(0.0, 1.0, size=len(codes))
            probs = rng.dirichlet(np.ones(3), size=len(codes))
            for ma, s, p in zip(codes, scores, probs):
                rank_rows.append({"ma": ma, "t": t, "h": h, "diem_xep_hang": float(s)})
                trend_rows.append(
                    {
                        "ma": ma,
                        "t": t,
                        "h": h,
                        "p_tang": float(p[0]),
                        "p_ngang": float(p[1]),
                        "p_giam": float(p[2]),
                    }
                )
    rank_scores = pd.DataFrame(rank_rows)
    trend_scores = pd.DataFrame(trend_rows)

    topk_rows = []
    for h in _HORIZONS:
        for t in date_strs:
            sub = (
                rank_scores[(rank_scores["t"] == t) & (rank_scores["h"] == h)]
                .sort_values("diem_xep_hang", ascending=False)
                .head(20)
                .reset_index(drop=True)
            )
            for pos, (_, r) in enumerate(sub.iterrows(), start=1):
                ma = r["ma"]
                tr = trend_scores[
                    (trend_scores["t"] == t)
                    & (trend_scores["h"] == h)
                    & (trend_scores["ma"] == ma)
                ].iloc[0]
                san = symbols.loc[symbols["ma"] == ma, "san"].iloc[0]
                topk_rows.append(
                    {
                        "t": t,
                        "h": h,
                        "hang": pos,
                        "ma": ma,
                        "san": san,
                        "diem_xep_hang": float(r["diem_xep_hang"]),
                        "diem_xu_huong": float(tr["p_tang"] - tr["p_giam"]),
                    }
                )
    topk_a1 = pd.DataFrame(topk_rows)

    result_rows = []
    for h in _HORIZONS:
        for K in _K_HOP_LE:
            for muc_chi_phi in _MUC_CHI_PHI:
                for cau_hinh in _THU_TU_CAU_HINH:
                    result_rows.append(
                        {
                            "h": h,
                            "K": K,
                            "muc_chi_phi": muc_chi_phi,
                            "cau_hinh": cau_hinh,
                            "rank_ic": float(rng.normal(0.0, 0.1)),
                            "precision_at_k": float(rng.uniform(0.0, 1.0)),
                            "loi_suat_nam_hoa_rong": float(rng.normal(0.0, 0.5)),
                            "sharpe_rong": float(rng.normal(0.0, 1.0)),
                            "mdd": float(-abs(rng.normal(0.0, 0.2))),
                            "turnover": float(rng.uniform(0.0, 2.0)),
                        }
                    )
    test_results = pd.DataFrame(result_rows)

    ma_bam = "e9f3a7b2c4d10f8e"  # mã băm giả 16 ký tự hex; hàm đọc chỉ lấy 8 ký tự đầu.
    return {
        "phien_ban": f"v{TEST_END.replace('-', '')}-{ma_bam[:8]}",
        "ma_bam": ma_bam,
        "tu_ngay": TEST_START,
        "den_ngay": TEST_END,
        "symbols": symbols,
        "rank_scores": rank_scores,
        "trend_scores": trend_scores,
        "topk_a1": topk_a1,
        "test_results": test_results,
    }


def nap_du_lieu_mau(seed: int | None = None) -> dict:
    """Sinh dữ liệu mẫu và đặt làm artefact hiện hành (tiện cho demo Gradio)."""
    artefact = tao_du_lieu_mau(seed)
    nap_artefact(artefact)
    return artefact