import pandas as pd
import pytest

from fillbad.liquidity_features import depth_imbalance, visible_liquidity_fragility
from fillbad.real_btc import add_future_return_labels, build_canonical_table, classify_column, parse_timestamps


def fixture_frame():
    row = {
        "system_time": ["2021-01-01 00:00:00+00:00", "2021-01-01 00:00:01+00:00", "2021-01-01 00:00:03+00:00"],
        "midpoint": [100.0, 101.0, 102.0],
        "spread": [1.0, 1.0, 1.0],
        "buys": [0.0, 1.0, 0.0],
        "sells": [0.0, 0.0, 1.0],
    }
    for side, sign in [("bids", -1), ("asks", 1)]:
        for level in range(15):
            row[f"{side}_distance_{level}"] = [sign * 0.005 * (level + 1)] * 3
            row[f"{side}_notional_{level}"] = [100.0 + level] * 3
            row[f"{side}_cancel_notional_{level}"] = [1.0 if side == "asks" else 2.0] * 3
            row[f"{side}_limit_notional_{level}"] = [3.0 if side == "asks" else 4.0] * 3
            row[f"{side}_market_notional_{level}"] = [5.0 if side == "asks" else 6.0] * 3
    return pd.DataFrame(row)


def test_schema_detection_for_activity_columns():
    info = classify_column("asks_cancel_notional_7")
    assert info.group == "cancellation activity"
    assert info.side == "ask"
    assert info.level == 7
    assert info.event_type == "cancel"


def test_timestamp_parsing_is_utc():
    parsed = parse_timestamps(pd.Series(["2021-01-01 00:00:00+00:00"]))
    assert str(parsed.dt.tz) == "UTC"


def test_canonical_depth_and_price_features():
    canonical = build_canonical_table(fixture_frame())
    assert canonical["best_bid"].iloc[0] == pytest.approx(99.5)
    assert canonical["best_ask"].iloc[0] == pytest.approx(100.5)
    assert canonical["bid_depth_top_15"].iloc[0] == sum(100.0 + i for i in range(15))
    assert canonical["spread"].gt(0).all()


def test_imbalance_and_fragility_signs():
    imbalance = depth_imbalance(pd.Series([3.0]), pd.Series([1.0]))
    assert imbalance.iloc[0] == 0.5
    fragility = visible_liquidity_fragility(
        pd.Series([10.0]), pd.Series([1.0]), pd.Series([5.0]), pd.Series([2.0]), pd.Series([3.0]), pd.Series([4.0])
    )
    assert fragility.iloc[0] == 13.0


def test_future_labels_are_timestamp_aware_not_row_shift():
    canonical = build_canonical_table(fixture_frame())
    labeled = add_future_return_labels(canonical, [1])
    assert labeled["future_mid_change_1s"].iloc[0] == 1.0
    assert pd.isna(labeled["future_mid_change_1s"].iloc[1])
