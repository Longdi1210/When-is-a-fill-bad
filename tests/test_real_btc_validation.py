import numpy as np
import pandas as pd

from fillbad.real_validation import (
    add_time_aggregates,
    apply_bins,
    build_side_dataset,
    local_block_shuffle,
    passive_markout_bps,
    side_component_frame,
    split_masks,
    train_quantile_bins,
)


def base_frame():
    ts = pd.to_datetime(
        [
            "2021-01-01 00:00:00+00:00",
            "2021-01-01 00:00:01+00:00",
            "2021-01-01 00:00:03+00:00",
            "2021-01-02 00:00:00+00:00",
        ],
        utc=True,
    )
    return pd.DataFrame(
        {
            "timestamp": ts,
            "spread": [1.0, 1.0, 1.0, 1.0],
            "bid_depth_top_15": [100.0, 100.0, 100.0, 100.0],
            "ask_depth_top_15": [200.0, 200.0, 200.0, 200.0],
            "imbalance_top_15": [-1 / 3, -1 / 3, -1 / 3, -1 / 3],
            "market_buy_pressure": [10.0, 20.0, 30.0, 40.0],
            "market_sell_pressure": [1.0, 2.0, 3.0, 4.0],
            "bid_cancellation_pressure": [5.0, 5.0, 5.0, 5.0],
            "ask_cancellation_pressure": [6.0, 6.0, 6.0, 6.0],
            "bid_limit_replenishment": [2.0, 2.0, 2.0, 2.0],
            "ask_limit_replenishment": [3.0, 3.0, 3.0, 3.0],
            "one_second_return_bps": [0.0, 1.0, -1.0, 0.5],
            "future_return_bps_1s": [1.0, np.nan, 2.0, -1.0],
        }
    )


def test_buy_and_sell_markout_sign_convention():
    returns = pd.Series([2.0, -3.0])
    assert passive_markout_bps(returns, "buy").tolist() == [2.0, -3.0]
    assert passive_markout_bps(returns, "sell").tolist() == [-2.0, 3.0]


def test_pressure_component_mapping_by_side():
    data = base_frame()
    buy = side_component_frame(data, "buy")
    sell = side_component_frame(data, "sell")
    assert buy["aggressive_market_pressure"].iloc[0] == 1.0
    assert buy["same_side_cancellation"].iloc[0] == 5.0
    assert buy["same_side_replenishment"].iloc[0] == 2.0
    assert buy["visible_depth"].iloc[0] == 100.0
    assert sell["aggressive_market_pressure"].iloc[0] == 10.0
    assert sell["same_side_cancellation"].iloc[0] == 6.0
    assert sell["same_side_replenishment"].iloc[0] == 3.0
    assert sell["visible_depth"].iloc[0] == 200.0


def test_formation_window_aggregation_is_time_aware_and_depth_normalized():
    buy = side_component_frame(base_frame(), "buy")
    aggregated = add_time_aggregates(buy, [1])
    assert aggregated["market_pressure_1s"].iloc[1] == 3.0
    assert aggregated["market_pressure_1s"].iloc[2] == 3.0
    assert aggregated["p3_depth_norm_1s"].iloc[0] == (1.0 + 5.0 - 2.0) / 100.0


def test_future_markout_gap_handling_is_preserved():
    data = base_frame()
    buy = build_side_dataset(data, "buy", [1], [1])
    assert buy["markout_bps_1s"].iloc[0] == 1.0
    assert pd.isna(buy["markout_bps_1s"].iloc[1])


def test_train_derived_quantile_boundaries_and_chronological_split():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"], utc=True),
            "x": [1.0, 2.0, 100.0, 200.0],
        }
    )
    split = pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "start_date_utc": ["2021-01-01", "2021-01-03", "2021-01-04"],
            "end_date_utc": ["2021-01-02", "2021-01-03", "2021-01-04"],
        }
    )
    masks = split_masks(frame, split)
    edges = train_quantile_bins(frame["x"], masks["train"], n_bins=2)
    bins = apply_bins(frame["x"], edges)
    assert masks["train"].sum() == 2
    assert bins.iloc[2] == 2


def test_deterministic_block_shuffle():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=10, freq="s", tz="UTC"),
            "x": np.arange(10.0),
        }
    )
    a = local_block_shuffle(frame, ["x"], block="1min", seed=7)
    b = local_block_shuffle(frame, ["x"], block="1min", seed=7)
    assert a["x"].tolist() == b["x"].tolist()
    assert sorted(a["x"].tolist()) == list(np.arange(10.0))
