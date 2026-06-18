from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from fillbad.multilevel_flow import (
    DEPTH_LEVELS,
    cumulative_depth,
    depth_shape,
    level_columns,
    potential_penetration_class,
    quote_survives,
    side_adjusted_markout_bps,
    side_prefix,
    side_sign,
    weighted_level_flow,
)
from fillbad.shock_analysis import (
    OUTCOME_HORIZONS_AFTER_ABSORPTION_SECONDS,
    RESPONSE_TIMES_SECONDS,
    SHOCK_WINDOWS_SECONDS,
    STRICT_TOTAL_TIMES_SECONDS,
    classify_strict_absorption,
    clustered_difference_ci,
    chronological_masks,
    classify_absorption,
    effective_nonoverlap_count,
    enforce_refractory,
    empirical_p_value,
    fit_shock_thresholds,
    linear_projection,
    robust_location_summary,
    rolling_seconds,
    stratified_absorption_null,
    top_minus_bottom,
)


PARQUET_DIR = Path("data/processed/kaggle_btc")
EPISODES_PARQUET = Path("data/processed/real_btc_shock_episodes.parquet")
TABLE_DIR = Path("outputs/tables/main")
AUDIT_DIR = Path("outputs/tables/audit")
FIGURE_DIR = Path("outputs/figures/dynamic_lob_main")
APPENDIX_DIR = Path("outputs/figures/appendix/dynamic_lob")
NOTE = Path("archive/codex_intermediate/DYNAMIC_LOB_IMPLEMENTATION_NOTE.md")
SHOCK_WINDOW_SECONDS = 10
ABSORPTION_WINDOWS_SECONDS = [2, 5, 10]
MAIN_ABSORPTION_WINDOW_SECONDS = 5
NULL_SEEDS = list(range(200))


def read_level_data() -> pd.DataFrame:
    columns = ["system_time", "midpoint", "spread"]
    for prefix in ["bids", "asks"]:
        for field in ["distance", "notional", "market_notional", "cancel_notional", "limit_notional"]:
            columns.extend(level_columns(prefix, field))
    dataset = ds.dataset(PARQUET_DIR, format="parquet", partitioning="hive", exclude_invalid_files=True)
    df = dataset.to_table(columns=columns).to_pandas()
    df["timestamp"] = pd.to_datetime(df["system_time"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["best_bid"] = df["midpoint"] * (1 + df["bids_distance_0"])
    df["best_ask"] = df["midpoint"] * (1 + df["asks_distance_0"])
    return df


def split_table() -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / "real_btc_splits.csv")


def write_temporal_audit(absorption_window: int = MAIN_ABSORPTION_WINDOW_SECONDS) -> pd.DataFrame:
    rows = [
        {
            "variable": "shock_notional_10s",
            "measurement_start": "t-10s",
            "measurement_end": "t",
            "role": "shock intensity",
            "stage": "shock",
            "overlaps_outcome": False,
            "potential_leakage": False,
            "required_fix": "none",
        },
        {
            "variable": "lag_depth_1/3/5/10/15",
            "measurement_start": "t-1 observation",
            "measurement_end": "t-1 observation",
            "role": "pre-shock visible-depth proxy",
            "stage": "shock",
            "overlaps_outcome": False,
            "potential_leakage": False,
            "required_fix": "retain as pre-shock proxy",
        },
        {
            "variable": f"flow_absorption_{absorption_window}s",
            "measurement_start": "t",
            "measurement_end": f"t+{absorption_window}s",
            "role": "early limit-minus-cancel response",
            "stage": "absorption",
            "overlaps_outcome": False,
            "potential_leakage": False,
            "required_fix": "primary absorption component",
        },
        {
            "variable": f"depth_recovery_5_absorption_{absorption_window}s",
            "measurement_start": "t",
            "measurement_end": f"t+{absorption_window}s",
            "role": "early top-5 depth recovery",
            "stage": "absorption",
            "overlaps_outcome": False,
            "potential_leakage": False,
            "required_fix": "primary absorption component",
        },
        {
            "variable": f"spread_recovery_absorption_{absorption_window}s",
            "measurement_start": "t",
            "measurement_end": f"t+{absorption_window}s",
            "role": "early spread recovery",
            "stage": "absorption",
            "overlaps_outcome": False,
            "potential_leakage": False,
            "required_fix": "primary absorption component",
        },
        {
            "variable": "strict_absorption_score",
            "measurement_start": "t",
            "measurement_end": f"t+{absorption_window}s",
            "role": "train-scaled early absorption state",
            "stage": "absorption",
            "overlaps_outcome": False,
            "potential_leakage": False,
            "required_fix": "quote survival removed from score",
        },
        {
            "variable": "future_markout_after_absorption_*",
            "measurement_start": f"t+{absorption_window}s",
            "measurement_end": "t+10s / 30s / 60s / 120s / 300s",
            "role": "primary future side-adjusted markout",
            "stage": "outcome",
            "overlaps_outcome": True,
            "potential_leakage": False,
            "required_fix": "baseline moved to absorption-window end",
        },
        {
            "variable": "future_quote_survives_*",
            "measurement_start": f"t+{absorption_window}s",
            "measurement_end": "t+10s / 30s / 60s / 120s / 300s",
            "role": "primary conditional quote-survival outcome",
            "stage": "outcome",
            "overlaps_outcome": True,
            "potential_leakage": False,
            "required_fix": "not used in absorption score",
        },
        {
            "variable": "episode_markout_from_preshock_*",
            "measurement_start": "t-1 observation",
            "measurement_end": "t+10s / 30s / 60s / 120s / 300s",
            "role": "descriptive full episode path",
            "stage": "outcome",
            "overlaps_outcome": True,
            "potential_leakage": "descriptive only",
            "required_fix": "excluded from primary predictive claim",
        },
    ]
    audit = pd.DataFrame(rows)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_DIR / "temporal_identification_audit.csv", index=False)
    return audit


def side_frame(raw: pd.DataFrame, side: str) -> pd.DataFrame:
    prefix = side_prefix(side)
    depth = raw[level_columns(prefix, "notional")].astype(float)
    distance = raw[level_columns(prefix, "distance")].astype(float)
    market = raw[level_columns(prefix, "market_notional")].astype(float)
    cancel = raw[level_columns(prefix, "cancel_notional")].astype(float)
    limit = raw[level_columns(prefix, "limit_notional")].astype(float)
    cum = cumulative_depth(depth)
    cum_all = pd.concat([depth.iloc[:, : i + 1].sum(axis=1).rename(i) for i in range(15)], axis=1)
    shape = depth_shape(depth, distance)
    out = pd.DataFrame(
        {
            "timestamp": raw["timestamp"],
            "date": raw["date"],
            "side": side,
            "midpoint": raw["midpoint"].astype(float),
            "spread": raw["spread"].astype(float),
            "best_quote": raw["best_bid" if side == "buy" else "best_ask"].astype(float),
            "opposite_quote": raw["best_ask" if side == "buy" else "best_bid"].astype(float),
            "market_1": market.iloc[:, :1].sum(axis=1),
            "market_5": market.iloc[:, :5].sum(axis=1),
            "market_15": market.sum(axis=1),
            "cancel_15": cancel.sum(axis=1),
            "limit_15": limit.sum(axis=1),
            "weighted_market_uniform": weighted_level_flow(market, "uniform"),
            "weighted_market_inverse_level": weighted_level_flow(market, "inverse_level"),
            "weighted_market_inverse_depth": weighted_level_flow(market, "inverse_cumulative_depth", cum_all),
        }
    )
    for level in DEPTH_LEVELS:
        out[f"depth_{level}"] = cum[f"depth_{level}"]
        out[f"lag_depth_{level}"] = out[f"depth_{level}"].shift(1)
    for column in shape:
        out[column] = shape[column]
    out["pre_midpoint"] = out["midpoint"].shift(1)
    out["pre_best_quote"] = out["best_quote"].shift(1)
    out["pre_spread"] = out["spread"].shift(1)
    out["gap_from_prev_seconds"] = out["timestamp"].diff().dt.total_seconds()
    out["recent_volatility_30s"] = (10000 * np.log(out["midpoint"]).diff()).abs().rolling(30, min_periods=2).sum().fillna(0.0)
    out["book_imbalance_5"] = side_sign(side) * (raw[[f"bids_notional_{i}" for i in range(5)]].sum(axis=1) - raw[[f"asks_notional_{i}" for i in range(5)]].sum(axis=1)) / (
        raw[[f"bids_notional_{i}" for i in range(5)]].sum(axis=1) + raw[[f"asks_notional_{i}" for i in range(5)]].sum(axis=1)
    ).replace(0, np.nan)
    for window in SHOCK_WINDOWS_SECONDS:
        out[f"shock_notional_{window}s"] = rolling_seconds(out, "market_15", window)
        out[f"cancel_during_{window}s"] = rolling_seconds(out, "cancel_15", window)
        out[f"limit_during_{window}s"] = rolling_seconds(out, "limit_15", window)
        for level in DEPTH_LEVELS:
            out[f"shock_ratio_l{level}_{window}s"] = out[f"shock_notional_{window}s"] / out[f"lag_depth_{level}"].replace(0, np.nan)
    return out


def add_episode_responses(episodes: pd.DataFrame, side_data: pd.DataFrame, side: str, absorption_window: int = MAIN_ABSORPTION_WINDOW_SECONDS) -> pd.DataFrame:
    side_lookup = side_data[["timestamp", "midpoint", "best_quote", "spread", "depth_1", "depth_5", "depth_15", "limit_15", "cancel_15"]].sort_values("timestamp").copy()
    side_lookup["cum_limit_15"] = side_lookup["limit_15"].cumsum()
    side_lookup["cum_cancel_15"] = side_lookup["cancel_15"].cumsum()
    side_lookup["gap_bad"] = (side_data["gap_from_prev_seconds"].fillna(1.0) > 2.5).astype(int).to_numpy()
    side_lookup["cum_gap_bad"] = side_lookup["gap_bad"].cumsum()
    result = episodes.copy()
    tolerance = pd.Timedelta(milliseconds=750)

    def point_lookup(target_time: pd.Series, prefix: str) -> pd.DataFrame:
        lookup = pd.DataFrame({"episode_index": result.index, "target_timestamp": target_time})
        future_lookup = side_lookup.rename(columns={"timestamp": "observed_timestamp"})
        out = pd.merge_asof(
            lookup.sort_values("target_timestamp"),
            future_lookup,
            left_on="target_timestamp",
            right_on="observed_timestamp",
            direction="nearest",
            tolerance=tolerance,
        ).set_index("episode_index").reindex(result.index)
        gap = (out["observed_timestamp"] - lookup.set_index("episode_index").reindex(result.index)["target_timestamp"]).abs()
        invalid = gap > tolerance
        value_cols = [c for c in out.columns if c not in ["target_timestamp", "observed_timestamp"]]
        out.loc[invalid, value_cols] = np.nan
        return out.add_prefix(prefix)

    current = pd.merge_asof(
            pd.DataFrame({"episode_index": result.index, "timestamp": result["timestamp"]}).sort_values("timestamp"),
            side_lookup[["timestamp", "cum_limit_15", "cum_cancel_15", "cum_gap_bad"]],
            on="timestamp",
            direction="nearest",
            tolerance=tolerance,
    ).set_index("episode_index").reindex(result.index)

    for window in ABSORPTION_WINDOWS_SECONDS:
        absorption = point_lookup(result["timestamp"] + pd.to_timedelta(window, unit="s"), f"abs{window}_")
        gap_count = absorption[f"abs{window}_cum_gap_bad"] - current["cum_gap_bad"]
        valid_absorption = gap_count.fillna(1) == 0
        limit_after = absorption[f"abs{window}_cum_limit_15"] - current["cum_limit_15"]
        cancel_after = absorption[f"abs{window}_cum_cancel_15"] - current["cum_cancel_15"]
        result[f"limit_add_absorption_{window}s"] = limit_after.where(valid_absorption)
        result[f"cancel_absorption_{window}s"] = cancel_after.where(valid_absorption)
        result[f"flow_absorption_{window}s"] = ((limit_after - cancel_after) / result["shock_notional"].replace(0, np.nan)).where(valid_absorption)
        result[f"depth_recovery_1_absorption_{window}s"] = ((absorption[f"abs{window}_depth_1"] - result["depth_1"]) / result["depth_1"].replace(0, np.nan)).where(valid_absorption)
        result[f"depth_recovery_5_absorption_{window}s"] = ((absorption[f"abs{window}_depth_5"] - result["depth_5"]) / result["depth_5"].replace(0, np.nan)).where(valid_absorption)
        result[f"spread_recovery_absorption_{window}s"] = ((result["spread"] - absorption[f"abs{window}_spread"]) / result["spread"].replace(0, np.nan)).where(valid_absorption)
        result[f"survives_absorption_window_{window}s"] = quote_survives(result["pre_best_quote"], absorption[f"abs{window}_best_quote"], side).astype(float).where(valid_absorption)
        if window == absorption_window:
            result["absorption_midpoint"] = absorption[f"abs{window}_midpoint"].where(valid_absorption)
            result["absorption_best_quote"] = absorption[f"abs{window}_best_quote"].where(valid_absorption)
            result["absorption_end_timestamp"] = absorption[f"abs{window}_observed_timestamp"]
            result["selected_absorption_window"] = window

    for total_time in STRICT_TOTAL_TIMES_SECONDS:
        future = point_lookup(result["timestamp"] + pd.to_timedelta(total_time, unit="s"), f"future{total_time}_")
        gap_count = future[f"future{total_time}_cum_gap_bad"] - current["cum_gap_bad"]
        valid_future = gap_count.fillna(1) == 0
        result[f"episode_markout_from_preshock_{total_time}s"] = side_adjusted_markout_bps(future[f"future{total_time}_midpoint"], result["pre_midpoint"], side).where(valid_future)
        result[f"future_markout_after_absorption_{total_time}s"] = side_adjusted_markout_bps(future[f"future{total_time}_midpoint"], result["absorption_midpoint"], side).where(valid_future)
        total_survival = quote_survives(result["pre_best_quote"], future[f"future{total_time}_best_quote"], side).astype(float).where(valid_future)
        future_survival = quote_survives(result["absorption_best_quote"], future[f"future{total_time}_best_quote"], side).astype(float).where(valid_future)
        result[f"total_quote_survives_{total_time}s"] = total_survival
        result[f"future_quote_survives_{total_time}s"] = future_survival
        result[f"adverse_future_markout_{total_time}s"] = (result[f"future_markout_after_absorption_{total_time}s"] < 0).astype(float).where(valid_future)
        result[f"outcome_horizon_after_absorption_{total_time}s"] = total_time - absorption_window
        result[f"total_time_after_shock_{total_time}s"] = total_time
    return result


def detect_episodes(side_data: pd.DataFrame, split: pd.DataFrame, side: str, main_window: int = 10, threshold_quantile: float = 0.95) -> tuple[pd.DataFrame, dict]:
    masks = chronological_masks(side_data, split)
    score = f"shock_ratio_l5_{main_window}s"
    threshold = fit_shock_thresholds(side_data, masks["train"], score, threshold_quantile)
    candidates = side_data[(side_data[score] >= threshold) & (side_data["gap_from_prev_seconds"] <= 2.5)].copy()
    candidates = enforce_refractory(candidates, seconds=30)
    cols = [
        "timestamp",
        "date",
        "side",
        "midpoint",
        "pre_midpoint",
        "best_quote",
        "pre_best_quote",
        "spread",
        "pre_spread",
        "recent_volatility_30s",
        "book_imbalance_5",
        "near_touch_depth_share",
        "deep_book_depth_share",
        "depth_weighted_level",
        "depth_weighted_distance",
        "depth_slope_near_minus_deep",
        "depth_1",
        "depth_5",
        "depth_15",
        f"shock_notional_{main_window}s",
        f"cancel_during_{main_window}s",
        f"limit_during_{main_window}s",
    ]
    for level in DEPTH_LEVELS:
        cols.append(f"lag_depth_{level}")
        cols.append(f"shock_ratio_l{level}_{main_window}s")
    episodes = candidates[cols].rename(
        columns={
            f"shock_notional_{main_window}s": "shock_notional",
            f"cancel_during_{main_window}s": "cancel_during_shock",
            f"limit_during_{main_window}s": "limit_during_shock",
        }
    ).copy()
    episodes["shock_window"] = main_window
    episodes["shock_threshold_quantile"] = threshold_quantile
    episodes["shock_threshold_value"] = threshold
    episodes["penetration_class"] = potential_penetration_class(
        episodes["shock_notional"], episodes[[f"lag_depth_{level}" for level in DEPTH_LEVELS]]
    )
    episodes = add_episode_responses(episodes, side_data, side, MAIN_ABSORPTION_WINDOW_SECONDS)
    episodes = episodes.dropna(
        subset=[
            "future_markout_after_absorption_300s",
            f"flow_absorption_{MAIN_ABSORPTION_WINDOW_SECONDS}s",
            f"depth_recovery_5_absorption_{MAIN_ABSORPTION_WINDOW_SECONDS}s",
            "future_quote_survives_60s",
        ]
    )
    params = {"side": side, "window": main_window, "threshold_quantile": threshold_quantile, "threshold_value": threshold}
    return episodes, params


def event_study(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in STRICT_TOTAL_TIMES_SECONDS:
        for group_name, group in episodes.groupby(["side", "strict_absorption_state"], dropna=False):
            side, absorption = group_name
            rows.append(
                {
                    "side": side,
                    "absorption_state": absorption,
                    "total_time_after_shock": horizon,
                    "outcome_horizon_after_absorption": horizon - MAIN_ABSORPTION_WINDOW_SECONDS,
                    "mean_future_markout_bps": group[f"future_markout_after_absorption_{horizon}s"].mean(),
                    "mean_episode_markout_from_preshock_bps": group[f"episode_markout_from_preshock_{horizon}s"].mean(),
                    "future_quote_survival_rate": group[f"future_quote_survives_{horizon}s"].mean(),
                    "total_quote_survival_rate": group[f"total_quote_survives_{horizon}s"].mean(),
                    "adverse_frequency": group[f"adverse_future_markout_{horizon}s"].mean(),
                    "count": len(group),
                }
            )
    return pd.DataFrame(rows)


def local_projections(episodes: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    masks = chronological_masks(episodes, split)
    features = ["shock_ratio_l5_10s", "strict_absorption_score", "shock_absorption_interaction", "pre_spread", "lag_depth_5", "recent_volatility_30s"]
    rows = []
    for horizon in STRICT_TOTAL_TIMES_SECONDS:
        target = f"future_markout_after_absorption_{horizon}s"
        for side in ["buy", "sell"]:
            train = episodes[masks["train"] & (episodes["side"] == side)]
            test = episodes[masks["test"] & (episodes["side"] == side)]
            metrics = linear_projection(train, test, features, target)
            rows.append({"side": side, "total_time_after_shock": horizon, "outcome_horizon_after_absorption": horizon - MAIN_ABSORPTION_WINDOW_SECONDS, **metrics})
    return pd.DataFrame(rows)


def multilevel_comparison(episodes: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    masks = chronological_masks(episodes, split)
    rows = []
    signals = {
        "top_level_shock": "shock_ratio_l1_10s",
        "top5_penetration": "shock_ratio_l5_10s",
        "top15_penetration": "shock_ratio_l15_10s",
        "depth_shape_augmented": "dynamic_score",
    }
    for side in ["buy", "sell"]:
        for name, signal in signals.items():
            metrics = top_minus_bottom(episodes[episodes["side"] == side], signal, "future_markout_after_absorption_60s", masks["train"][episodes["side"] == side], masks["test"][episodes["side"] == side])
            rows.append({"side": side, "representation": name, **metrics})
    return pd.DataFrame(rows)


def dynamic_vs_static(episodes: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    masks = chronological_masks(episodes, split)
    rows = []
    signals = {
        "market_only_static": "shock_ratio_l5_10s",
        "static_p3_proxy": "static_p3_ratio",
        "top_level_dynamic": "shock_ratio_l1_10s",
        "multilevel_shock_absorption": "dynamic_score",
    }
    for side in ["buy", "sell"]:
        side_mask = episodes["side"] == side
        for name, signal in signals.items():
            metrics = top_minus_bottom(episodes[side_mask], signal, "future_markout_after_absorption_60s", masks["train"][side_mask], masks["test"][side_mask])
            rows.append({"side": side, "representation": name, **metrics})
    return pd.DataFrame(rows)


def null_results(episodes: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    masks = chronological_masks(episodes, split)
    rows = []
    real = dynamic_vs_static(episodes, split)
    real["sequence"] = "real"
    rows.append(real)
    for seed in [11, 17, 23, 31, 43]:
        shuffled = stratified_absorption_null(episodes, seed=seed)
        tmp = dynamic_vs_static(shuffled, split)
        tmp["sequence"] = f"stratified_null_{seed}"
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def add_null_strata(episodes: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    out = episodes.copy()
    masks = chronological_masks(out, split)
    train = masks["train"]
    for source, target in [
        ("shock_ratio_l5_10s", "shock_intensity_bin"),
        ("lag_depth_5", "pre_depth_bin"),
        ("pre_spread", "spread_regime"),
        ("recent_volatility_30s", "volatility_regime"),
    ]:
        edges = out.loc[train, source].replace([np.inf, -np.inf], np.nan).dropna().quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.array([-np.inf, np.inf])
        else:
            edges[0] = -np.inf
            edges[-1] = np.inf
        out[target] = pd.cut(out[source], bins=edges, labels=False, include_lowest=True).astype("Int64").astype(str)
    out["time_of_day_block"] = pd.to_datetime(out["timestamp"], utc=True).dt.hour.floordiv(4).astype(str)
    out["time_block"] = pd.to_datetime(out["timestamp"], utc=True).dt.floor("15min").astype(str)
    return out


def strict_group_results(episodes: pd.DataFrame, split: pd.DataFrame, n_bootstrap: int = 300) -> pd.DataFrame:
    masks = chronological_masks(episodes, split)
    test = episodes[masks["test"]].copy()
    rows = []
    for side in ["buy", "sell"]:
        side_test = test[test["side"] == side]
        for total_time in STRICT_TOTAL_TIMES_SECONDS:
            markout_col = f"future_markout_after_absorption_{total_time}s"
            survival_col = f"future_quote_survives_{total_time}s"
            adverse_col = f"adverse_future_markout_{total_time}s"
            markout = clustered_difference_ci(side_test, markout_col, seed=100 + total_time, n_bootstrap=n_bootstrap)
            survival = clustered_difference_ci(side_test, survival_col, seed=200 + total_time, n_bootstrap=n_bootstrap)
            adverse = clustered_difference_ci(side_test, adverse_col, seed=300 + total_time, n_bootstrap=n_bootstrap)
            weak = side_test[side_test["strict_absorption_state"] == "weak_absorption"]
            strong = side_test[side_test["strict_absorption_state"] == "strong_absorption"]
            rows.append(
                {
                    "side": side,
                    "total_time_after_shock": total_time,
                    "outcome_horizon_after_absorption": total_time - MAIN_ABSORPTION_WINDOW_SECONDS,
                    "weak_count": len(weak),
                    "strong_count": len(strong),
                    "weak_future_markout_mean": markout["low_mean"],
                    "strong_future_markout_mean": markout["high_mean"],
                    "strong_minus_weak_markout": markout["difference"],
                    "markout_ci_low": markout["ci_low"],
                    "markout_ci_high": markout["ci_high"],
                    "weak_future_quote_survival": survival["low_mean"],
                    "strong_future_quote_survival": survival["high_mean"],
                    "strong_minus_weak_quote_survival": survival["difference"],
                    "quote_survival_ci_low": survival["ci_low"],
                    "quote_survival_ci_high": survival["ci_high"],
                    "weak_adverse_frequency": adverse["low_mean"],
                    "strong_adverse_frequency": adverse["high_mean"],
                    "strong_minus_weak_adverse_frequency": adverse["difference"],
                    "adverse_ci_low": adverse["ci_low"],
                    "adverse_ci_high": adverse["ci_high"],
                }
            )
    return pd.DataFrame(rows)


def overlap_vs_nonoverlap(episodes: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    masks = chronological_masks(episodes, split)
    test = episodes[masks["test"]].copy()
    rows = []
    for side in ["buy", "sell"]:
        side_test = test[test["side"] == side]
        for total_time in STRICT_TOTAL_TIMES_SECONDS:
            for design, markout_col, survival_col in [
                ("descriptive_total_path", f"episode_markout_from_preshock_{total_time}s", f"total_quote_survives_{total_time}s"),
                ("strict_future_after_absorption", f"future_markout_after_absorption_{total_time}s", f"future_quote_survives_{total_time}s"),
            ]:
                markout = clustered_difference_ci(side_test, markout_col, seed=400 + total_time, n_bootstrap=300)
                survival = clustered_difference_ci(side_test, survival_col, seed=500 + total_time, n_bootstrap=300)
                weak = side_test[side_test["strict_absorption_state"] == "weak_absorption"]
                strong = side_test[side_test["strict_absorption_state"] == "strong_absorption"]
                rows.append(
                    {
                        "side": side,
                        "design": design,
                        "total_time_after_shock": total_time,
                        "weak_absorption_markout": markout["low_mean"],
                        "strong_absorption_markout": markout["high_mean"],
                        "strong_minus_weak_markout": markout["difference"],
                        "markout_ci_low": markout["ci_low"],
                        "markout_ci_high": markout["ci_high"],
                        "quote_survival_difference": survival["difference"],
                        "quote_survival_ci_low": survival["ci_low"],
                        "quote_survival_ci_high": survival["ci_high"],
                        "episode_count": int(len(weak) + len(strong)),
                    }
                )
    return pd.DataFrame(rows)


def interaction_sign_audit(episodes: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    masks = chronological_masks(episodes, split)
    features = ["shock_ratio_l5_10s", "strict_absorption_score", "shock_absorption_interaction", "pre_spread", "lag_depth_5", "recent_volatility_30s"]
    rows = []
    for side in ["buy", "sell"]:
        side_train = episodes[masks["train"] & (episodes["side"] == side)]
        for total_time in STRICT_TOTAL_TIMES_SECONDS:
            target = f"future_markout_after_absorption_{total_time}s"
            clean = side_train[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean) < 50:
                continue
            x = clean[features]
            y = clean[target].to_numpy()
            mean = x.mean()
            std = x.std(ddof=0).replace(0, 1.0)
            z = ((x - mean) / std).to_numpy()
            design = np.column_stack([np.ones(len(z)), z])
            coef = np.linalg.lstsq(design, y, rcond=None)[0]
            residual = y - design @ coef
            dof = max(len(y) - design.shape[1], 1)
            sigma2 = float(residual @ residual / dof)
            cov = sigma2 * np.linalg.pinv(design.T @ design)
            se = np.sqrt(np.diag(cov))
            shock_abs_corr = float(x["shock_ratio_l5_10s"].corr(x["strict_absorption_score"]))
            vif_notes = "interaction is conditional on standardized main effects; grouped paths are primary"
            grouped = strict_group_results(episodes, split, n_bootstrap=100)
            grouped_effect = grouped[(grouped["side"] == side) & (grouped["total_time_after_shock"] == total_time)]["strong_minus_weak_markout"].iloc[0]
            rows.append(
                {
                    "side": side,
                    "total_time_after_shock": total_time,
                    "shock_coefficient": coef[1],
                    "absorption_coefficient": coef[2],
                    "interaction_coefficient": coef[3],
                    "interaction_ci_low": coef[3] - 1.96 * se[3],
                    "interaction_ci_high": coef[3] + 1.96 * se[3],
                    "shock_absorption_correlation": shock_abs_corr,
                    "absorption_sign_definition": "higher strict_absorption_score means stronger early replenishment/depth/spread recovery",
                    "conditional_interpretation": "negative interaction means high shock intensity reduces the benefit of strong absorption",
                    "grouped_effect_consistent": bool(grouped_effect > 0),
                    "notes": vif_notes,
                }
            )
    return pd.DataFrame(rows)


def expanded_null_results(episodes: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    masks = chronological_masks(episodes, split)
    test = episodes[masks["test"]].reset_index(drop=True).copy()
    strata_cols = ["date", "side", "shock_intensity_bin", "pre_depth_bin", "spread_regime", "volatility_regime", "time_of_day_block"]
    group_positions = [idx.to_numpy() for _, idx in test.groupby(strata_cols).groups.items()]
    base_markout = test["future_markout_after_absorption_60s"].to_numpy(dtype=float)
    base_survival = test["future_quote_survives_60s"].to_numpy(dtype=float)
    state = test["strict_absorption_state"].to_numpy()
    side_values = test["side"].to_numpy()
    shock = test["shock_ratio_l5_10s"]
    static_bins = pd.Series(np.nan, index=test.index)
    for side in ["buy", "sell"]:
        mask = test["side"] == side
        edges = np.quantile(shock[mask].replace([np.inf, -np.inf], np.nan).dropna(), np.linspace(0, 1, 11))
        edges = np.unique(edges)
        if len(edges) >= 3:
            edges[0] = -np.inf
            edges[-1] = np.inf
            static_bins.loc[mask] = np.digitize(shock[mask], edges[1:-1], right=True) + 1
    static_bins = static_bins.to_numpy()

    def statistic_from_arrays(markout: np.ndarray, survival: np.ndarray, sequence: str) -> pd.DataFrame:
        rows = []
        for side in ["buy", "sell", "combined"]:
            side_mask = np.ones(len(test), dtype=bool) if side == "combined" else side_values == side
            if not side_mask.any():
                continue
            strong_mask = side_mask & (state == "strong_absorption")
            weak_mask = side_mask & (state == "weak_absorption")
            markout_diff = float(np.nanmean(markout[strong_mask]) - np.nanmean(markout[weak_mask]))
            survival_diff = float(np.nanmean(survival[strong_mask]) - np.nanmean(survival[weak_mask]))
            high_static = side_mask & (static_bins >= 9)
            low_static = side_mask & (static_bins <= 2)
            static_diff = float(np.nanmean(markout[high_static]) - np.nanmean(markout[low_static]))
            rows.append(
                {
                    "sequence": sequence,
                    "side": side,
                    "strong_minus_weak_markout": markout_diff,
                    "quote_survival_difference": survival_diff,
                    "dynamic_vs_static_improvement": markout_diff - static_diff,
                    "count": int(side_mask.sum()),
                }
            )
        return pd.DataFrame(rows)

    rows = [statistic_from_arrays(base_markout, base_survival, "real")]
    for seed in NULL_SEEDS:
        rng = np.random.default_rng(seed)
        markout = base_markout.copy()
        survival = base_survival.copy()
        for idx in group_positions:
            if len(idx) > 1:
                shuffled = rng.permutation(idx)
                markout[idx] = base_markout[shuffled]
                survival[idx] = base_survival[shuffled]
        rows.append(statistic_from_arrays(markout, survival, f"stratified_null_{seed}"))
    out = pd.concat(rows, ignore_index=True)
    p_rows = []
    for side in out["side"].unique():
        for metric in ["strong_minus_weak_markout", "quote_survival_difference", "dynamic_vs_static_improvement"]:
            real = out[(out["sequence"] == "real") & (out["side"] == side)][metric].iloc[0]
            null_values = out[(out["sequence"] != "real") & (out["side"] == side)][metric].to_numpy()
            p_rows.append(
                {
                    "sequence": "empirical_p_value",
                    "side": side,
                    "metric": metric,
                    "real_value": real,
                    "null_mean": float(np.nanmean(null_values)),
                    "empirical_p_two_sided": empirical_p_value(real, null_values, two_sided=True),
                    "one_sided_percentile": float((null_values <= real).mean()) if metric == "strong_minus_weak_markout" else float((null_values >= real).mean()),
                    "null_seed_count": len(null_values),
                }
            )
    pvals = pd.DataFrame(p_rows)
    return out, pvals


def episode_concentration_diagnostics(episodes: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    masks = chronological_masks(episodes, split)
    test = episodes[masks["test"]].copy()
    rows = []
    for side in ["buy", "sell"]:
        for state in ["weak_absorption", "strong_absorption"]:
            subset = test[(test["side"] == side) & (test["strict_absorption_state"] == state)]
            summary = robust_location_summary(subset["future_markout_after_absorption_60s"])
            ci = clustered_difference_ci(
                pd.concat(
                    [
                        subset.assign(tmp_group=state),
                        test[(test["side"] == side) & (test["strict_absorption_state"] != state)].assign(tmp_group="other"),
                    ]
                ),
                "future_markout_after_absorption_60s",
                group_col="tmp_group",
                high_label=state,
                low_label="other",
                n_bootstrap=200,
            )
            rows.append(
                {
                    "side": side,
                    "absorption_state": state,
                    "episode_count": len(subset),
                    "effective_nonoverlap_count_60s": effective_nonoverlap_count(subset["timestamp"], 60),
                    "fraction_2021_04_18": float((subset["date"] == "2021-04-18").mean()) if len(subset) else np.nan,
                    "penetration_classes": subset["penetration_class"].nunique(),
                    **summary,
                    "block_bootstrap_ci_low_vs_other": ci["ci_low"],
                    "block_bootstrap_ci_high_vs_other": ci["ci_high"],
                }
            )
    return pd.DataFrame(rows)


def regime_20210418(raw: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    daily_market = raw.groupby("date").agg(
        row_count=("timestamp", "size"),
        mean_spread=("spread", "mean"),
        volatility=("midpoint", lambda x: float((10000 * np.log(x).diff()).abs().sum())),
    ).reset_index()
    epi = episodes.groupby("date").agg(
        shock_episodes=("timestamp", "size"),
        mean_shock_ratio_l5=("shock_ratio_l5_10s", "mean"),
        weak_absorption_share=("strict_absorption_state", lambda x: float((x == "weak_absorption").mean())),
        future_quote_survival_60=("future_quote_survives_60s", "mean"),
        future_markout_60=("future_markout_after_absorption_60s", "mean"),
    ).reset_index()
    return daily_market.merge(epi, on="date", how="left").fillna(0.0)


def plot_figures(episodes: pd.DataFrame, strict_results: pd.DataFrame, overlap: pd.DataFrame, expanded_null: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for old in FIGURE_DIR.glob("*.png"):
        old.unlink()

    fig, ax = plt.subplots(figsize=(9, 2.7))
    segments = [(-10, 0, "shock formation\n[t-10s, t]", "#4c78a8"), (0, 5, "early absorption\n(t, t+5s]", "#72b7b2"), (5, 60, "strict future outcome\n(t+5s, t+60s]", "#f58518")]
    for start, end, label, color in segments:
        ax.barh([0], [end - start], left=[start], height=0.35, color=color)
        ax.text((start + end) / 2, 0, label, ha="center", va="center", color="white", fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(5, color="black", linewidth=0.8)
    ax.set_xlim(-11, 62)
    ax.set_yticks([])
    ax.set_xlabel("Seconds relative to shock end t")
    ax.set_title("Temporal Identification Design")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_temporal_identification.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for side, marker in [("buy", "o"), ("sell", "s")]:
        sub = strict_results[strict_results["side"] == side]
        x = sub["total_time_after_shock"]
        y = sub["strong_minus_weak_quote_survival"]
        ax.errorbar(x, y, yerr=[y - sub["quote_survival_ci_low"], sub["quote_survival_ci_high"] - y], marker=marker, capsize=3, label=side)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Future Quote Survival: Strong Minus Weak Early Absorption")
    ax.set_xlabel("Total seconds after shock")
    ax.set_ylabel("Quote-survival difference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_future_quote_survival.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for side, marker in [("buy", "o"), ("sell", "s")]:
        sub = strict_results[strict_results["side"] == side]
        x = sub["total_time_after_shock"]
        y = sub["strong_minus_weak_markout"]
        ax.errorbar(x, y, yerr=[y - sub["markout_ci_low"], sub["markout_ci_high"] - y], marker=marker, capsize=3, label=side)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Future Markout After Absorption: Strong Minus Weak")
    ax.set_xlabel("Total seconds after shock")
    ax.set_ylabel("Side-adjusted markout difference (bps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_future_markout_after_absorption.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sub = overlap[overlap["total_time_after_shock"] == 60].copy()
    labels = [f"{row.side}\n{row.design.replace('_', ' ')}" for row in sub.itertuples(index=False)]
    ax.bar(labels, sub["strong_minus_weak_markout"], color=["#4c78a8" if d == "descriptive_total_path" else "#f58518" for d in sub["design"]])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Overlap Versus Strict Non-Overlap Result")
    ax.set_ylabel("Strong minus weak markout at 60s (bps)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_overlap_vs_nonoverlap.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, side in zip(axes, ["buy", "sell"]):
        null = expanded_null[(expanded_null["sequence"] != "real") & (expanded_null["side"] == side)]["strong_minus_weak_markout"]
        real = expanded_null[(expanded_null["sequence"] == "real") & (expanded_null["side"] == side)]["strong_minus_weak_markout"].iloc[0]
        ax.hist(null, bins=28, color="#bab0ab", edgecolor="white")
        ax.axvline(real, color="#e45756", linewidth=2, label="real")
        ax.set_title(side)
        ax.set_xlabel("Strong minus weak 60s markout (bps)")
        ax.legend()
    axes[0].set_ylabel("Null seed count")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_expanded_stratified_null.png", dpi=170)
    plt.close(fig)


def main() -> None:
    start = time.perf_counter()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    write_temporal_audit(MAIN_ABSORPTION_WINDOW_SECONDS)
    raw = read_level_data()
    split = split_table()
    frames = {side: side_frame(raw, side) for side in ["buy", "sell"]}
    episode_parts = []
    params = []
    for side, frame in frames.items():
        episodes, param = detect_episodes(frame, split, side)
        episode_parts.append(episodes)
        params.append(param)
    episodes = pd.concat(episode_parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    masks = chronological_masks(episodes, split)
    episodes, absorption_params = classify_strict_absorption(episodes, masks["train"], MAIN_ABSORPTION_WINDOW_SECONDS)
    episodes = add_null_strata(episodes, split)
    episodes["shock_absorption_interaction"] = episodes["shock_ratio_l5_10s"] * episodes["strict_absorption_score"]
    episodes["static_p3_ratio"] = (episodes["shock_notional"] + episodes["cancel_during_shock"] - episodes["limit_during_shock"]) / episodes["lag_depth_5"].replace(0, np.nan)
    episodes["dynamic_score"] = episodes["shock_ratio_l5_10s"] - episodes["strict_absorption_score"] + episodes["depth_weighted_level"] / episodes["depth_weighted_level"].replace(0, np.nan).median()
    episodes.to_parquet(EPISODES_PARQUET, compression="zstd", index=False)
    masks = chronological_masks(episodes, split)

    manifest = pd.DataFrame(
        {
            "metric": ["episode_count", "train_count", "validation_count", "test_count", "selected_shock_window", "selected_absorption_window", "threshold_quantile", "null_seed_count"],
            "value": [
                len(episodes),
                int(masks["train"].sum()),
                int(masks["validation"].sum()),
                int(masks["test"].sum()),
                SHOCK_WINDOW_SECONDS,
                MAIN_ABSORPTION_WINDOW_SECONDS,
                0.95,
                len(NULL_SEEDS),
            ],
        }
    )
    manifest.to_csv(TABLE_DIR / "shock_episode_manifest.csv", index=False)
    pd.DataFrame(
        [
            {"candidate_absorption_window": window, "selected": window == MAIN_ABSORPTION_WINDOW_SECONDS, "selection_basis": "pre-specified preferred window; no test-set tuning"}
            for window in ABSORPTION_WINDOWS_SECONDS
        ]
    ).to_csv(TABLE_DIR / "absorption_window_selection.csv", index=False)
    pd.DataFrame(params).to_csv(TABLE_DIR / "penetration_summary.csv", index=False)
    absorption_summary = episodes.groupby(["side", "strict_absorption_state"]).agg(
        count=("timestamp", "size"),
        future_markout_60=("future_markout_after_absorption_60s", "mean"),
        future_survival_60=("future_quote_survives_60s", "mean"),
        flow_absorption_5=("flow_absorption_5s", "mean"),
    ).reset_index().rename(columns={"strict_absorption_state": "absorption_state"})
    absorption_summary.to_csv(TABLE_DIR / "absorption_state_summary.csv", index=False)
    event = event_study(episodes)
    quote_survival = event[["side", "absorption_state", "total_time_after_shock", "outcome_horizon_after_absorption", "future_quote_survival_rate", "total_quote_survival_rate", "count"]]
    quote_survival.to_csv(TABLE_DIR / "quote_survival_results.csv", index=False)
    absorption_summary.to_csv(TABLE_DIR / "depth_recovery_results.csv", index=False)
    event.to_csv(TABLE_DIR / "event_study_results.csv", index=False)
    projections = local_projections(episodes, split)
    projections.to_csv(TABLE_DIR / "local_projection_results.csv", index=False)
    multi = multilevel_comparison(episodes, split)
    multi.to_csv(TABLE_DIR / "multilevel_comparison.csv", index=False)
    dyn = null_results(episodes, split)
    dyn[dyn["sequence"] == "real"].to_csv(TABLE_DIR / "dynamic_vs_static.csv", index=False)
    regime = regime_20210418(raw, episodes)
    regime.to_csv(TABLE_DIR / "regime_20210418.csv", index=False)
    dyn.to_csv(TABLE_DIR / "shock_null_results.csv", index=False)
    strict = strict_group_results(episodes, split)
    strict.to_csv(TABLE_DIR / "strict_absorption_results.csv", index=False)
    overlap = overlap_vs_nonoverlap(episodes, split)
    overlap.to_csv(TABLE_DIR / "overlap_vs_nonoverlap_results.csv", index=False)
    interaction = interaction_sign_audit(episodes, split)
    interaction.to_csv(AUDIT_DIR / "interaction_sign_audit.csv", index=False)
    expanded_null, expanded_pvals = expanded_null_results(episodes, split)
    expanded_null.to_csv(TABLE_DIR / "expanded_shock_null_results.csv", index=False)
    expanded_pvals.to_csv(TABLE_DIR / "expanded_shock_null_pvalues.csv", index=False)
    concentration = episode_concentration_diagnostics(episodes, split)
    concentration.to_csv(TABLE_DIR / "episode_concentration_diagnostics.csv", index=False)
    plot_figures(episodes, strict, overlap, expanded_null)

    summary = {
        "episode_count": int(len(episodes)),
        "train_count": int(masks["train"].sum()),
        "validation_count": int(masks["validation"].sum()),
        "test_count": int(masks["test"].sum()),
        "thresholds": params,
        "absorption_params": absorption_params,
        "runtime_seconds": time.perf_counter() - start,
        "main_shock_window": SHOCK_WINDOW_SECONDS,
        "main_absorption_window": MAIN_ABSORPTION_WINDOW_SECONDS,
        "total_times_after_shock": STRICT_TOTAL_TIMES_SECONDS,
        "outcome_horizons_after_absorption": OUTCOME_HORIZONS_AFTER_ABSORPTION_SECONDS,
        "null_seed_count": len(NULL_SEEDS),
    }
    (TABLE_DIR / "dynamic_lob_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    NOTE.write_text(
        "# Dynamic LOB Implementation Note\n\n"
        "Existing baseline retained: static P1/P2/P3 proxy validation remains available.\n\n"
        "Strict temporal design added: shock formation [t-10s,t], early absorption (t,t+5s], and future outcomes (t+5s,t+H]. Absorption excludes quote-survival and markout outcomes.\n\n"
        "Dynamic objects added: multi-level depth, shock ratios, potential penetration classes, early post-shock absorption, conditional best-quote survival, future markout response paths, local projections, and expanded stratified shock null.\n\n"
        "Data-resolution limitation: one-second aggregates support potential penetration and best-quote survival proxies, not exact FIFO fills or intrasecond execution paths.\n"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
