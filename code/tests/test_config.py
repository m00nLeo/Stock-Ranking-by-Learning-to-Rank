"""Test khung bước 3.1: config.yaml nạp được và chứa đúng các hằng số đã chốt ở Giai đoạn 2."""
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_config_file_exists():
    assert CONFIG_PATH.is_file()


def test_seed_is_fixed_integer():
    cfg = load_config()
    assert isinstance(cfg["seed"], int)


def test_horizons_match_D54():
    cfg = load_config()
    assert cfg["horizons"] == [5, 10, 21]


def test_data_split_order_is_chronological():
    cfg = load_config()
    split = cfg["data_split"]
    order = [
        split["feature_history_start"],
        split["labeled_sample_start"],
        split["train_start"],
        split["train_end"],
        split["validation_start"],
        split["validation_end"],
        split["test_start"],
    ]
    assert order == sorted(order), "các mốc chia dữ liệu phải tăng dần theo thời gian (D44, D62)"


def test_no_parameter_selection_on_test_split():
    cfg = load_config()
    assert "test" not in cfg["universe"]["select_params_on"]


def test_cost_constants_match_D61():
    cfg = load_config()
    costs = cfg["costs"]
    assert costs["c_buy"] == 0.0025
    assert costs["c_sell"] == 0.0035
    assert costs["c_rt_base"] == 0.0060


def test_trend_label_barrier_matches_D63():
    cfg = load_config()
    label = cfg["trend_label"]
    assert label["k_fixed"] == 1
    assert label["sigma_span"] == 20
    assert label["classes"] == ["tăng", "đi ngang", "giảm"]


def test_hypothesis_family_matches_D54_D60():
    cfg = load_config()
    hyp = cfg["hypotheses"]
    assert hyp["family"] == ["H1", "H2", "H3", "H4"]
    assert hyp["correction"] == "Holm"
    assert hyp["alpha"] == 0.05


def test_system_runtime_is_colab_per_D64():
    cfg = load_config()
    assert cfg["system"]["runtime"] == "Google Colab"


def test_data_split_test_end_matches_D79():
    cfg = load_config()
    assert cfg["data_split"]["test_end"] == "2026-10-02"


def test_feature_registry_params_match_D76():
    cfg = load_config()
    features = cfg["features"]
    assert features["lookback_windows_sessions"] == [21, 63, 126]
    assert features["lag_sessions"] == 0
    assert features["rsi_window"] == 14
    assert features["macd"] == {"fast": 12, "slow": 26, "signal": 9}


def test_universe_params_match_D74():
    cfg = load_config()
    universe = cfg["universe"]
    assert universe["n_size"] == 150
    assert 50 <= universe["n_size"] <= universe["rank_universe_size_range"][1]
    assert universe["liquidity_window_sessions"] == 21
    assert universe["liquidity_lag_sessions"] == 0
    assert universe["price_min_thousand_vnd"] == 5.0
    assert universe["avg_value_min_thousand_vnd"] == 100000
