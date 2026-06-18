import numpy as np
import pandas as pd

from fillbad.multilevel_flow import (
    cumulative_depth,
    potential_penetration_class,
    quote_survives,
    side_adjusted_markout_bps,
    weighted_level_flow,
)
from fillbad.shock_analysis import (
    classify_absorption,
    enforce_refractory,
    fit_shock_thresholds,
    stratified_absorption_null,
)


def test_cumulative_depth_uses_requested_visible_levels():
    depth = pd.DataFrame([[10, 20, 30, 40, 50], [1, 2, 3, 4, 5]])
    out = cumulative_depth(depth, levels=[1, 3, 5])
    assert out.loc[0, "depth_1"] == 10
    assert out.loc[0, "depth_3"] == 60
    assert out.loc[0, "depth_5"] == 150
    assert out.loc[1, "depth_3"] == 6


def test_weighted_level_flow_uniform_and_inverse_level():
    flow = pd.DataFrame([[10.0, 20.0, 30.0]])
    assert weighted_level_flow(flow, "uniform").iloc[0] == 20.0
    inverse = weighted_level_flow(flow, "inverse_level").iloc[0]
    weights = np.array([1.0, 0.5, 1.0 / 3.0])
    weights = weights / weights.sum()
    assert np.isclose(inverse, float(np.dot([10.0, 20.0, 30.0], weights)))


def test_weighted_level_flow_inverse_cumulative_depth_is_safe():
    flow = pd.DataFrame([[10.0, 20.0, 30.0], [5.0, 0.0, 5.0]])
    cum_depth = pd.DataFrame([[100.0, 200.0, 300.0], [10.0, 0.0, 20.0]])
    out = weighted_level_flow(flow, "inverse_cumulative_depth", cum_depth)
    assert np.isclose(out.iloc[0], 0.1)
    assert np.isfinite(out.iloc[1])


def test_potential_penetration_class_uses_lagged_depth_thresholds():
    shock = pd.Series([0.0, 5.0, 25.0, 55.0, 90.0, 140.0])
    depths = pd.DataFrame(
        {
            "lag_depth_1": [10] * 6,
            "lag_depth_3": [30] * 6,
            "lag_depth_5": [60] * 6,
            "lag_depth_10": [100] * 6,
            "lag_depth_15": [120] * 6,
        }
    )
    assert potential_penetration_class(shock, depths).tolist() == [
        "no_shock",
        "contained_touch",
        "threatens_near_book",
        "threatens_near_book",
        "threatens_medium_depth",
        "exceeds_visible_depth",
    ]


def test_side_adjusted_markout_is_favorable_positive_for_both_sides():
    base = pd.Series([100.0])
    assert side_adjusted_markout_bps(pd.Series([101.0]), base, "buy").iloc[0] > 0
    assert side_adjusted_markout_bps(pd.Series([99.0]), base, "sell").iloc[0] > 0
    assert side_adjusted_markout_bps(pd.Series([99.0]), base, "buy").iloc[0] < 0
    assert side_adjusted_markout_bps(pd.Series([101.0]), base, "sell").iloc[0] < 0


def test_quote_survival_buy_and_sell_conventions():
    pre = pd.Series([100.0, 100.0])
    assert quote_survives(pre, pd.Series([100.0, 99.0]), "buy").tolist() == [True, False]
    assert quote_survives(pre, pd.Series([100.0, 101.0]), "sell").tolist() == [True, False]


def test_enforce_refractory_keeps_clean_episode_spacing_by_side():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2021-01-01 00:00:00", "2021-01-01 00:00:10", "2021-01-01 00:00:40", "2021-01-01 00:00:12"],
                utc=True,
            ),
            "side": ["buy", "buy", "buy", "sell"],
        }
    )
    out = enforce_refractory(frame, seconds=30)
    assert out["timestamp"].dt.second.tolist() == [0, 12, 40]


def test_fit_shock_thresholds_uses_train_period_only():
    frame = pd.DataFrame({"score": [1.0, 2.0, 3.0, 100.0]})
    train_mask = pd.Series([True, True, True, False])
    assert fit_shock_thresholds(frame, train_mask, "score", 1.0) == 3.0


def test_classify_absorption_uses_train_fitted_score_and_states():
    episodes = pd.DataFrame(
        {
            "net_absorption_30s": [-2, -1, 0, 1, 2, 3],
            "depth_recovery_5_30s": [-2, -1, 0, 1, 2, 3],
            "quote_survives_30s": [0, 0, 1, 1, 1, 1],
        }
    )
    train_mask = pd.Series([True, True, True, True, False, False])
    out, params = classify_absorption(episodes, train_mask)
    assert {"weak_absorption", "partial_absorption", "strong_absorption"}.issubset(set(out["absorption_state"]))
    assert "score_thresholds" in params


def test_stratified_absorption_null_is_deterministic_and_preserves_strata():
    episodes = pd.DataFrame(
            {
                "date": ["2021-01-01"] * 4,
                "side": ["buy"] * 4,
                "penetration_class": ["a"] * 4,
            "absorption_state": ["weak"] * 4,
            "markout_60s": [1.0, 2.0, 3.0, 4.0],
            "shock_ratio_l5_10s": [0.1, 0.2, 0.3, 0.4],
        }
    )
    a = stratified_absorption_null(episodes, seed=7)
    b = stratified_absorption_null(episodes, seed=7)
    assert a["markout_60s"].tolist() == b["markout_60s"].tolist()
    assert sorted(a["markout_60s"].tolist()) == [1.0, 2.0, 3.0, 4.0]
