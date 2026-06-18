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
    RESPONSE_TIMES_SECONDS,
    SHOCK_WINDOWS_SECONDS,
    chronological_masks,
    classify_absorption,
    enforce_refractory,
    fit_shock_thresholds,
    linear_projection,
    rolling_seconds,
    stratified_absorption_null,
    top_minus_bottom,
)


PARQUET_DIR = Path("data/processed/kaggle_btc")
EPISODES_PARQUET = Path("data/processed/real_btc_shock_episodes.parquet")
TABLE_DIR = Path("outputs/tables/main")
FIGURE_DIR = Path("outputs/figures/dynamic_lob_main")
APPENDIX_DIR = Path("outputs/figures/appendix/dynamic_lob")
NOTE = Path("archive/codex_intermediate/DYNAMIC_LOB_IMPLEMENTATION_NOTE.md")


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


def add_episode_responses(episodes: pd.DataFrame, side_data: pd.DataFrame, side: str) -> pd.DataFrame:
    side_lookup = side_data[["timestamp", "midpoint", "best_quote", "spread", "depth_1", "depth_5", "depth_15", "limit_15", "cancel_15"]].sort_values("timestamp").copy()
    side_lookup["cum_limit_15"] = side_lookup["limit_15"].cumsum()
    side_lookup["cum_cancel_15"] = side_lookup["cancel_15"].cumsum()
    result = episodes.copy()
    tolerance = pd.Timedelta(milliseconds=750)
    for horizon in RESPONSE_TIMES_SECONDS:
        target_time = result["timestamp"] + pd.to_timedelta(horizon, unit="s")
        lookup = pd.DataFrame({"episode_index": result.index, "timestamp": result["timestamp"], "target_timestamp": target_time})
        current = pd.merge_asof(
            lookup.sort_values("timestamp"),
            side_lookup[["timestamp", "cum_limit_15", "cum_cancel_15"]],
            on="timestamp",
            direction="nearest",
            tolerance=tolerance,
        ).set_index("episode_index").reindex(result.index)
        future_lookup = side_lookup.rename(columns={"timestamp": "observed_timestamp"})
        future = pd.merge_asof(
            lookup.sort_values("target_timestamp"),
            future_lookup,
            left_on="target_timestamp",
            right_on="observed_timestamp",
            direction="nearest",
            tolerance=tolerance,
        ).set_index("episode_index").reindex(result.index)
        future_gap = (future["observed_timestamp"] - lookup.set_index("episode_index").reindex(result.index)["target_timestamp"]).abs()
        future.loc[future_gap > tolerance, ["midpoint", "best_quote", "spread", "depth_1", "depth_5", "depth_15", "cum_limit_15", "cum_cancel_15"]] = np.nan
        result[f"markout_{horizon}s"] = side_adjusted_markout_bps(future["midpoint"], result["pre_midpoint"], side)
        result[f"spread_response_{horizon}s"] = future["spread"] - result["pre_spread"].to_numpy()
        survival = quote_survives(result["pre_best_quote"], future["best_quote"], side).astype(float)
        survival[future["best_quote"].isna()] = np.nan
        result[f"quote_survives_{horizon}s"] = survival
        for level in [1, 5, 15]:
            result[f"depth_recovery_{level}_{horizon}s"] = (future[f"depth_{level}"] - result[f"lag_depth_{level}"].to_numpy()) / result[f"lag_depth_{level}"].replace(0, np.nan).to_numpy()
        limit_after = future["cum_limit_15"] - current["cum_limit_15"]
        cancel_after = future["cum_cancel_15"] - current["cum_cancel_15"]
        result[f"post_limit_over_shock_{horizon}s"] = limit_after.to_numpy() / result["shock_notional"].replace(0, np.nan).to_numpy()
        result[f"post_cancel_over_shock_{horizon}s"] = cancel_after.to_numpy() / result["shock_notional"].replace(0, np.nan).to_numpy()
        result[f"net_absorption_{horizon}s"] = (limit_after.to_numpy() - cancel_after.to_numpy()) / result["shock_notional"].replace(0, np.nan).to_numpy()
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
    episodes = add_episode_responses(episodes, side_data, side)
    episodes = episodes.dropna(subset=["markout_300s", "depth_recovery_5_30s", "quote_survives_30s"])
    params = {"side": side, "window": main_window, "threshold_quantile": threshold_quantile, "threshold_value": threshold}
    return episodes, params


def event_study(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in RESPONSE_TIMES_SECONDS:
        for group_name, group in episodes.groupby(["side", "absorption_state"], dropna=False):
            side, absorption = group_name
            rows.append(
                {
                    "side": side,
                    "absorption_state": absorption,
                    "horizon": horizon,
                    "mean_markout_bps": group[f"markout_{horizon}s"].mean(),
                    "mean_depth_recovery_5": group[f"depth_recovery_5_{horizon}s"].mean(),
                    "mean_spread_response": group[f"spread_response_{horizon}s"].mean(),
                    "quote_survival_rate": group[f"quote_survives_{horizon}s"].mean(),
                    "count": len(group),
                }
            )
    return pd.DataFrame(rows)


def local_projections(episodes: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    masks = chronological_masks(episodes, split)
    features = ["shock_ratio_l5_10s", "absorption_score", "shock_absorption_interaction", "pre_spread", "lag_depth_5", "recent_volatility_30s"]
    rows = []
    for horizon in RESPONSE_TIMES_SECONDS:
        target = f"markout_{horizon}s"
        for side in ["buy", "sell"]:
            train = episodes[masks["train"] & (episodes["side"] == side)]
            test = episodes[masks["test"] & (episodes["side"] == side)]
            metrics = linear_projection(train, test, features, target)
            rows.append({"side": side, "horizon": horizon, **metrics})
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
            metrics = top_minus_bottom(episodes[episodes["side"] == side], signal, "markout_60s", masks["train"][episodes["side"] == side], masks["test"][episodes["side"] == side])
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
            metrics = top_minus_bottom(episodes[side_mask], signal, "markout_60s", masks["train"][side_mask], masks["test"][side_mask])
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


def regime_20210418(raw: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    daily_market = raw.groupby("date").agg(
        row_count=("timestamp", "size"),
        mean_spread=("spread", "mean"),
        volatility=("midpoint", lambda x: float((10000 * np.log(x).diff()).abs().sum())),
    ).reset_index()
    epi = episodes.groupby("date").agg(
        shock_episodes=("timestamp", "size"),
        mean_shock_ratio_l5=("shock_ratio_l5_10s", "mean"),
        weak_absorption_share=("absorption_state", lambda x: float((x == "weak_absorption").mean())),
        quote_survival_60=("quote_survives_60s", "mean"),
        markout_60=("markout_60s", "mean"),
    ).reset_index()
    return daily_market.merge(epi, on="date", how="left").fillna(0.0)


def plot_figures(episodes: pd.DataFrame, event: pd.DataFrame, multi: pd.DataFrame, dyn: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for old in FIGURE_DIR.glob("*.png"):
        old.unlink()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = episodes[["shock_ratio_l1_10s", "shock_ratio_l5_10s", "shock_ratio_l15_10s"]].clip(upper=5)
    ax.hist(data, bins=60, label=["level 1", "top 5", "top 15"], alpha=0.65)
    ax.set_title("Shock Penetration Structure")
    ax.set_xlabel("Shock notional / lagged depth ratio, clipped at 5")
    ax.set_ylabel("Episode count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_shock_penetration_structure.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for state in ["strong_absorption", "weak_absorption"]:
        subset = event[event["absorption_state"] == state].groupby("horizon", as_index=False).mean(numeric_only=True)
        axes[0].plot(subset["horizon"], subset["mean_depth_recovery_5"], marker="o", label=state)
        axes[1].plot(subset["horizon"], subset["mean_spread_response"], marker="o", label=state)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Depth Recovery")
    axes[1].set_title("Spread Response")
    for ax in axes:
        ax.set_xlabel("Seconds after shock")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_depth_spread_recovery.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for state in ["strong_absorption", "weak_absorption", "partial_absorption"]:
        subset = event[event["absorption_state"] == state].groupby("horizon", as_index=False).mean(numeric_only=True)
        ax.plot(subset["horizon"], subset["mean_markout_bps"], marker="o", label=state)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Post-Shock Markout Response Path")
    ax.set_xlabel("Seconds after shock")
    ax.set_ylabel("Side-adjusted markout (bps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_markout_impulse_response.png", dpi=170)
    plt.close(fig)

    table = episodes.pivot_table(index="absorption_state", columns="penetration_class", values="markout_60s", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    im = ax.imshow(table, aspect="auto", cmap="coolwarm")
    ax.set_title("Penetration x Absorption Response Map")
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    fig.colorbar(im, ax=ax, label="Mean 60s markout (bps)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_penetration_absorption_map.png", dpi=170)
    plt.close(fig)

    summary = dyn[dyn["sequence"] == "real"].groupby("representation", as_index=False)["rank_correlation"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.bar(summary["representation"], summary["rank_correlation"], color="#4c78a8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Dynamic Versus Static Representation")
    ax.set_ylabel("Mean test rank correlation")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_dynamic_vs_static.png", dpi=170)
    plt.close(fig)


def main() -> None:
    start = time.perf_counter()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
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
    episodes, absorption_params = classify_absorption(episodes, masks["train"])
    episodes["shock_absorption_interaction"] = episodes["shock_ratio_l5_10s"] * episodes["absorption_score"]
    episodes["static_p3_ratio"] = (episodes["shock_notional"] + episodes["cancel_during_shock"] - episodes["limit_during_shock"]) / episodes["lag_depth_5"].replace(0, np.nan)
    episodes["dynamic_score"] = episodes["shock_ratio_l5_10s"] - episodes["absorption_score"] + episodes["depth_weighted_level"] / episodes["depth_weighted_level"].replace(0, np.nan).median()
    episodes.to_parquet(EPISODES_PARQUET, compression="zstd", index=False)

    manifest = pd.DataFrame(
        {
            "metric": ["episode_count", "train_count", "validation_count", "test_count", "selected_window", "threshold_quantile"],
            "value": [
                len(episodes),
                int(masks["train"].sum()),
                int(masks["validation"].sum()),
                int(masks["test"].sum()),
                10,
                0.95,
            ],
        }
    )
    manifest.to_csv(TABLE_DIR / "shock_episode_manifest.csv", index=False)
    pd.DataFrame(params).to_csv(TABLE_DIR / "penetration_summary.csv", index=False)
    absorption_summary = episodes.groupby(["side", "absorption_state"]).agg(count=("timestamp", "size"), markout_60=("markout_60s", "mean"), survival_60=("quote_survives_60s", "mean")).reset_index()
    absorption_summary.to_csv(TABLE_DIR / "absorption_state_summary.csv", index=False)
    quote_survival = event_study(episodes)[["side", "absorption_state", "horizon", "quote_survival_rate", "count"]]
    quote_survival.to_csv(TABLE_DIR / "quote_survival_results.csv", index=False)
    event = event_study(episodes)
    event[["side", "absorption_state", "horizon", "mean_depth_recovery_5", "mean_spread_response", "count"]].to_csv(TABLE_DIR / "depth_recovery_results.csv", index=False)
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
    plot_figures(episodes, event, multi, dyn)

    summary = {
        "episode_count": int(len(episodes)),
        "thresholds": params,
        "absorption_params": absorption_params,
        "runtime_seconds": time.perf_counter() - start,
        "main_window": 10,
        "response_horizons": RESPONSE_TIMES_SECONDS,
    }
    (TABLE_DIR / "dynamic_lob_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    NOTE.write_text(
        "# Dynamic LOB Implementation Note\n\n"
        "Existing baseline retained: static P1/P2/P3 proxy validation remains available.\n\n"
        "Dynamic objects added: multi-level depth, shock ratios, potential penetration classes, post-shock absorption, best-quote survival, markout response paths, local projections, and stratified shock null.\n\n"
        "Data-resolution limitation: one-second aggregates support potential penetration and best-quote survival proxies, not exact FIFO fills or intrasecond execution paths.\n"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
