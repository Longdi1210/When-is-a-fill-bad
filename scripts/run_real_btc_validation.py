from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fillbad.real_btc import CANONICAL_PARQUET
from fillbad.real_validation import (
    FORMATION_WINDOWS_SECONDS,
    MARKOUT_HORIZONS_SECONDS,
    SIDES,
    apply_bins,
    build_side_dataset,
    linear_fit_predict,
    local_block_shuffle,
    mean_ci,
    rank_correlation,
    regression_metrics,
    split_masks,
    train_quantile_bins,
)


TABLE_DIR = Path("outputs/tables/main")
FIGURE_DIR = Path("outputs/figures/real_btc_main")
APPENDIX_FIGURE_DIR = Path("outputs/figures/appendix/real_btc")
REPORT = Path("archive/codex_intermediate/REAL_BTC_VALIDATION.md")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_parquet(CANONICAL_PARQUET)
    split = pd.read_csv(TABLE_DIR / "real_btc_splits.csv")
    return data, split


def pressure_quantile_results(side_frames: dict[str, pd.DataFrame], split: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side, frame in side_frames.items():
        masks = split_masks(frame, split)
        for window in FORMATION_WINDOWS_SECONDS:
            pressure_col = f"p3_depth_norm_{window}s"
            edges = train_quantile_bins(frame[pressure_col], masks["train"])
            quantile = apply_bins(frame[pressure_col], edges)
            for horizon in MARKOUT_HORIZONS_SECONDS:
                markout_col = f"markout_bps_{horizon}s"
                for split_name in ["validation", "test"]:
                    subset_mask = masks[split_name] & frame[markout_col].notna()
                    subset = pd.DataFrame(
                        {
                            "quantile": quantile.loc[subset_mask],
                            "markout": frame.loc[subset_mask, markout_col],
                            "adverse": frame.loc[subset_mask, f"adverse_{horizon}s"],
                        }
                    )
                    for q, group in subset.groupby("quantile"):
                        mean, lo, hi = mean_ci(group["markout"])
                        rows.append(
                            {
                                "split": split_name,
                                "side": side,
                                "formation_window": window,
                                "horizon": horizon,
                                "pressure_quantile": int(q),
                                "mean_markout_bps": mean,
                                "ci_low": lo,
                                "ci_high": hi,
                                "adverse_frequency": float(group["adverse"].mean()),
                                "observation_count": int(len(group)),
                            }
                        )
    return pd.DataFrame(rows)


def proxy_comparison(side_frames: dict[str, pd.DataFrame], split: pd.DataFrame) -> pd.DataFrame:
    rows = []
    proxies = {
        "P1_market": "p1_market_{window}s",
        "P2_market_cancel": "p2_market_cancel_{window}s",
        "P3_full_depth_norm": "p3_depth_norm_{window}s",
    }
    for side, frame in side_frames.items():
        masks = split_masks(frame, split)
        for window in FORMATION_WINDOWS_SECONDS:
            for proxy_name, template in proxies.items():
                proxy_col = template.format(window=window)
                edges = train_quantile_bins(frame[proxy_col], masks["train"])
                quantile = apply_bins(frame[proxy_col], edges)
                for horizon in MARKOUT_HORIZONS_SECONDS:
                    markout_col = f"markout_bps_{horizon}s"
                    for split_name in ["validation", "test"]:
                        m = masks[split_name] & frame[markout_col].notna()
                        low = frame.loc[m & (quantile <= 2), markout_col]
                        high = frame.loc[m & (quantile >= 9), markout_col]
                        rows.append(
                            {
                                "split": split_name,
                                "side": side,
                                "proxy": proxy_name,
                                "formation_window": window,
                                "horizon": horizon,
                                "low_pressure_mean_markout_bps": float(low.mean()),
                                "high_pressure_mean_markout_bps": float(high.mean()),
                                "high_minus_low_markout_bps": float(high.mean() - low.mean()),
                                "rank_correlation": rank_correlation(frame.loc[m, proxy_col], frame.loc[m, markout_col]),
                                "low_count": int(low.count()),
                                "high_count": int(high.count()),
                            }
                        )
    return pd.DataFrame(rows)


def nested_models(side_frames: dict[str, pd.DataFrame], split: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = {
        "M0_controls": lambda w: ["spread", f"visible_depth_{w}s", "book_imbalance", f"recent_volatility_{w}s"],
        "M1_market": lambda w: ["spread", f"visible_depth_{w}s", "book_imbalance", f"recent_volatility_{w}s", f"market_pressure_{w}s"],
        "M2_market_cancel": lambda w: [
            "spread",
            f"visible_depth_{w}s",
            "book_imbalance",
            f"recent_volatility_{w}s",
            f"market_pressure_{w}s",
            f"cancellation_{w}s",
        ],
        "M3_market_cancel_replenish": lambda w: [
            "spread",
            f"visible_depth_{w}s",
            "book_imbalance",
            f"recent_volatility_{w}s",
            f"market_pressure_{w}s",
            f"cancellation_{w}s",
            f"replenishment_{w}s",
        ],
        "M4_full_proxy": lambda w: ["spread", f"visible_depth_{w}s", "book_imbalance", f"recent_volatility_{w}s", f"p3_depth_norm_{w}s"],
    }
    for side, frame in side_frames.items():
        masks = split_masks(frame, split)
        for window in FORMATION_WINDOWS_SECONDS:
            for horizon in MARKOUT_HORIZONS_SECONDS:
                target = f"markout_bps_{horizon}s"
                train_base = masks["train"] & frame[target].notna()
                for model_name, columns_fn in specs.items():
                    columns = columns_fn(window)
                    train = frame.loc[train_base, columns + [target]].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(train) < 100:
                        continue
                    for split_name in ["validation", "test"]:
                        eval_base = masks[split_name] & frame[target].notna()
                        evaluate = frame.loc[eval_base, columns + [target]].replace([np.inf, -np.inf], np.nan).dropna()
                        if len(evaluate) < 100:
                            continue
                        pred, coef = linear_fit_predict(train[columns], train[target], evaluate[columns])
                        metrics = regression_metrics(evaluate[target], pred)
                        rows.append(
                            {
                                "split": split_name,
                                "side": side,
                                "formation_window": window,
                                "horizon": horizon,
                                "model": model_name,
                                **metrics,
                                "coefficient_last_feature": float(coef[-1]),
                                "last_feature": columns[-1],
                                "train_count": int(len(train)),
                                "eval_count": int(len(evaluate)),
                            }
                        )
    out = pd.DataFrame(rows)
    baseline = out[out["model"] == "M0_controls"][
        ["split", "side", "formation_window", "horizon", "r2"]
    ].rename(columns={"r2": "m0_r2"})
    out = out.merge(baseline, on=["split", "side", "formation_window", "horizon"], how="left")
    out["incremental_r2_vs_m0"] = out["r2"] - out["m0_r2"]
    return out


def select_primary_window(proxy_table: pd.DataFrame) -> tuple[int, int]:
    validation = proxy_table[
        (proxy_table["split"] == "validation") & (proxy_table["proxy"] == "P3_full_depth_norm")
    ].copy()
    grouped = validation.groupby(["formation_window", "horizon"], as_index=False)["high_minus_low_markout_bps"].mean()
    row = grouped.sort_values("high_minus_low_markout_bps").iloc[0]
    return int(row["formation_window"]), int(row["horizon"])


def daily_stability(side_frames: dict[str, pd.DataFrame], split: pd.DataFrame, window: int, horizon: int) -> pd.DataFrame:
    rows = []
    for side, frame in side_frames.items():
        masks = split_masks(frame, split)
        pressure = f"p3_depth_norm_{window}s"
        markout = f"markout_bps_{horizon}s"
        edges = train_quantile_bins(frame[pressure], masks["train"])
        quantile = apply_bins(frame[pressure], edges)
        test = frame.loc[masks["test"] & frame[markout].notna()].copy()
        test["quantile"] = quantile.loc[test.index]
        for date, group in test.groupby("date"):
            low = group.loc[group["quantile"] <= 2, markout]
            high = group.loc[group["quantile"] >= 9, markout]
            x = group[pressure].replace([np.inf, -np.inf], np.nan)
            y = group[markout].replace([np.inf, -np.inf], np.nan)
            valid = pd.DataFrame({"x": x, "y": y}).dropna()
            coef = np.nan
            if len(valid) > 10 and valid["x"].std(ddof=0) > 0:
                coef = float(np.polyfit(valid["x"], valid["y"], 1)[0])
            rows.append(
                {
                    "date": date,
                    "side": side,
                    "formation_window": window,
                    "response_horizon": horizon,
                    "high_minus_low_markout": float(high.mean() - low.mean()),
                    "rank_correlation": rank_correlation(group[pressure], group[markout]),
                    "coefficient": coef,
                    "observation_count": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def null_test(side_frames: dict[str, pd.DataFrame], split: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side, frame in side_frames.items():
        shuffled = local_block_shuffle(
            frame,
            [f"p3_depth_norm_{w}s" for w in FORMATION_WINDOWS_SECONDS],
            block="5min",
            seed=123 if side == "buy" else 456,
        )
        masks = split_masks(frame, split)
        for window in FORMATION_WINDOWS_SECONDS:
            pressure = f"p3_depth_norm_{window}s"
            real_edges = train_quantile_bins(frame[pressure], masks["train"])
            for horizon in MARKOUT_HORIZONS_SECONDS:
                markout = f"markout_bps_{horizon}s"
                for label, source in [("real", frame), ("local_block_shuffle", shuffled)]:
                    quantile = apply_bins(source[pressure], real_edges)
                    m = masks["test"] & source[markout].notna()
                    low = source.loc[m & (quantile <= 2), markout]
                    high = source.loc[m & (quantile >= 9), markout]
                    rows.append(
                        {
                            "sequence": label,
                            "side": side,
                            "formation_window": window,
                            "horizon": horizon,
                            "high_minus_low_markout_bps": float(high.mean() - low.mean()),
                            "rank_correlation": rank_correlation(source.loc[m, pressure], source.loc[m, markout]),
                            "low_count": int(low.count()),
                            "high_count": int(high.count()),
                        }
                    )
    return pd.DataFrame(rows)


def regime_labels(frame: pd.DataFrame, split: pd.DataFrame, window: int) -> dict[str, pd.Series]:
    masks = split_masks(frame, split)
    labels = {}
    specs = {
        "depth_tercile": f"visible_depth_{window}s",
        "volatility_tercile": f"recent_volatility_{window}s",
        "spread_regime": "spread",
    }
    for name, column in specs.items():
        train = frame.loc[masks["train"], column].replace([np.inf, -np.inf], np.nan).dropna()
        edges = np.unique(np.quantile(train, [0.0, 1 / 3, 2 / 3, 1.0]))
        if len(edges) < 4:
            labels[name] = pd.Series("middle", index=frame.index)
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        labels[name] = pd.cut(frame[column], bins=edges, labels=["low", "middle", "high"], include_lowest=True).astype(str)
    return labels


def contrast_for_signal(
    frame: pd.DataFrame,
    signal_col: str,
    markout_col: str,
    train_mask: pd.Series,
    eval_mask: pd.Series,
    edges: np.ndarray | None = None,
) -> dict[str, float | int]:
    if edges is None:
        edges = train_quantile_bins(frame[signal_col], train_mask)
    quantile = apply_bins(frame[signal_col], edges)
    valid = eval_mask & frame[markout_col].notna() & frame[signal_col].replace([np.inf, -np.inf], np.nan).notna()
    low = frame.loc[valid & (quantile <= 2), markout_col]
    high = frame.loc[valid & (quantile >= 9), markout_col]
    return {
        "low_mean_markout_bps": float(low.mean()),
        "high_mean_markout_bps": float(high.mean()),
        "high_minus_low_markout_bps": float(high.mean() - low.mean()),
        "rank_correlation": sampled_rank_correlation(frame.loc[valid, signal_col], frame.loc[valid, markout_col]),
        "observation_count": int(valid.sum()),
        "low_count": int(low.count()),
        "high_count": int(high.count()),
    }


def sampled_rank_correlation(x: pd.Series, y: pd.Series, max_rows: int = 50_000) -> float:
    xy = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(xy) < 3 or xy["x"].nunique() < 2 or xy["y"].nunique() < 2:
        return np.nan
    if len(xy) > max_rows:
        step = max(len(xy) // max_rows, 1)
        xy = xy.iloc[::step].head(max_rows)
    return float(xy["x"].rank().corr(xy["y"].rank()))


def mechanism_audit(side_frames: dict[str, pd.DataFrame], split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    loo_rows = []
    map_rows = []
    for side, frame in side_frames.items():
        masks = split_masks(frame, split)
        test_dates = sorted(frame.loc[masks["test"], "date"].unique())
        for window in FORMATION_WINDOWS_SECONDS:
            regimes = regime_labels(frame, split, window)
            frame[f"negative_replenishment_{window}s"] = -frame[f"replenishment_{window}s"]
            frame[f"inverse_depth_{window}s"] = 1.0 / frame[f"visible_depth_{window}s"].replace(0, np.nan)
            signals = {
                "market_flow": f"market_pressure_{window}s",
                "cancellation": f"cancellation_{window}s",
                "negative_replenishment": f"negative_replenishment_{window}s",
                "P2_market_cancel_raw": f"p2_market_cancel_{window}s",
                "P3_full_raw": f"p3_full_{window}s",
                "P3_depth_normalized": f"p3_depth_norm_{window}s",
                "inverse_depth_denominator": f"inverse_depth_{window}s",
            }
            signal_edges = {name: train_quantile_bins(frame[col], masks["train"]) for name, col in signals.items()}
            for horizon in MARKOUT_HORIZONS_SECONDS:
                markout = f"markout_bps_{horizon}s"
                subgroup_masks = [("all", "all", masks["test"])]
                if window == 10 and horizon == 60:
                    for date in test_dates:
                        subgroup_masks.append(("test_day", date, masks["test"] & (frame["date"] == date)))
                    for regime_name, label_series in regimes.items():
                        for label in ["low", "middle", "high"]:
                            subgroup_masks.append((regime_name, label, masks["test"] & (label_series == label)))

                for component, signal in signals.items():
                    for subgroup_type, subgroup, eval_mask in subgroup_masks:
                        if eval_mask.sum() < 100:
                            continue
                        metrics = contrast_for_signal(frame, signal, markout, masks["train"], eval_mask, signal_edges[component])
                        rows.append(
                            {
                                "side": side,
                                "formation_window": window,
                                "horizon": horizon,
                                "component": component,
                                "subgroup_type": subgroup_type,
                                "subgroup": subgroup,
                                **metrics,
                            }
                        )

                if window == 10 and horizon == 60:
                    for proxy_name, signal in {
                        "P2_market_cancel_raw": f"p2_market_cancel_{window}s",
                        "P3_depth_normalized": f"p3_depth_norm_{window}s",
                    }.items():
                        for omitted_date in test_dates:
                            eval_mask = masks["test"] & (frame["date"] != omitted_date)
                            metrics = contrast_for_signal(frame, signal, markout, masks["train"], eval_mask)
                            loo_rows.append(
                                {
                                    "side": side,
                                    "proxy": proxy_name,
                                    "omitted_test_day": omitted_date,
                                    "formation_window": window,
                                    "horizon": horizon,
                                    **metrics,
                                }
                            )

                    map_specs = [
                        ("pressure_x_depth", f"p3_depth_norm_{window}s", f"visible_depth_{window}s"),
                        ("pressure_x_volatility", f"p3_depth_norm_{window}s", f"recent_volatility_{window}s"),
                        ("cancellation_x_replenishment", f"cancellation_{window}s", f"replenishment_{window}s"),
                        ("market_flow_x_replenishment", f"market_pressure_{window}s", f"replenishment_{window}s"),
                    ]
                    for map_name, x_col, y_col in map_specs:
                        x_edges = train_quantile_bins(frame[x_col], masks["train"], n_bins=5)
                        y_edges = train_quantile_bins(frame[y_col], masks["train"], n_bins=5)
                        x_bin = apply_bins(frame[x_col], x_edges)
                        y_bin = apply_bins(frame[y_col], y_edges)
                        tmp = pd.DataFrame(
                            {
                                "x_bin": x_bin.loc[masks["test"]],
                                "y_bin": y_bin.loc[masks["test"]],
                                "markout": frame.loc[masks["test"], markout],
                            }
                        ).dropna()
                        grouped = tmp.groupby(["x_bin", "y_bin"]).agg(mean_markout_bps=("markout", "mean"), count=("markout", "size")).reset_index()
                        for item in grouped.itertuples(index=False):
                            map_rows.append(
                                {
                                    "map": map_name,
                                    "side": side,
                                    "formation_window": window,
                                    "horizon": horizon,
                                    "x_bin": int(item.x_bin),
                                    "y_bin": int(item.y_bin),
                                    "mean_markout_bps": float(item.mean_markout_bps),
                                    "count": int(item.count),
                                }
                            )
    return pd.DataFrame(rows), pd.DataFrame(loo_rows), pd.DataFrame(map_rows)


def synthetic_real_comparison(proxy_results: pd.DataFrame, nested: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    test_proxy = proxy_results[(proxy_results["split"] == "test") & (proxy_results["proxy"] == "P3_full_depth_norm")]
    best = test_proxy.sort_values("high_minus_low_markout_bps").iloc[0]
    m4 = nested[(nested["split"] == "test") & (nested["model"] == "M4_full_proxy")]
    m0 = nested[(nested["split"] == "test") & (nested["model"] == "M0_controls")]
    real_null_gap = null.pivot_table(
        index=["side", "formation_window", "horizon"], columns="sequence", values="high_minus_low_markout_bps"
    ).dropna()
    null_diff = float((real_null_gap["real"] - real_null_gap["local_block_shuffle"]).mean())
    return pd.DataFrame(
        [
            {
                "component": "execution likelihood / pressure",
                "synthetic exact-fill result": "Exact hypothetical fill labels and fill scores remain available.",
                "real BTC proxy result": "One-second execution pressure proxy is observable; exact fill probability is not.",
                "evidence boundary": "Real data has aggregated pressure, not passive-order outcomes.",
            },
            {
                "component": "post-fill or post-quote markout",
                "synthetic exact-fill result": "Signed post-fill markout is measured after simulated fills.",
                "real BTC proxy result": f"Most adverse tested high-minus-low proxy contrast: {best.high_minus_low_markout_bps:.4f} bps.",
                "evidence boundary": "Real result is post-quote mid-price response, not realized fill markout.",
            },
            {
                "component": "market-flow contribution",
                "synthetic exact-fill result": "Flow variables are controlled in the generator.",
                "real BTC proxy result": "Market-pressure-only proxy is benchmarked against richer proxies.",
                "evidence boundary": "One-second aggregation prevents order-level causality claims.",
            },
            {
                "component": "cancellation contribution",
                "synthetic exact-fill result": "Cancellations can be exact in replay.",
                "real BTC proxy result": "M2 versus M1 reports cancellation incremental value.",
                "evidence boundary": "Cancellation fields are aggregate notionals by side/level.",
            },
            {
                "component": "replenishment contribution",
                "synthetic exact-fill result": "Limit replenishment is simulated explicitly.",
                "real BTC proxy result": "M3 versus M2 reports replenishment incremental value.",
                "evidence boundary": "Limit fields are aggregate notionals, not individual orders.",
            },
            {
                "component": "time-scale dependence",
                "synthetic exact-fill result": "Event-window scans remain in the synthetic layer.",
                "real BTC proxy result": f"Mean real-minus-shuffle contrast across scale map: {null_diff:.4f} bps.",
                "evidence boundary": "Real analysis uses seconds, not event-time queue replay.",
            },
        ]
    )


def plot_pressure_vs_markout(side_frames: dict[str, pd.DataFrame], split: pd.DataFrame, window: int, horizon: int) -> None:
    rows = []
    for side, frame in side_frames.items():
        masks = split_masks(frame, split)
        for proxy, column in {
            "P1 market-only": f"p1_market_{window}s",
            "P3 full proxy": f"p3_depth_norm_{window}s",
        }.items():
            edges = train_quantile_bins(frame[column], masks["train"])
            quantile = apply_bins(frame[column], edges)
            markout = f"markout_bps_{horizon}s"
            test = pd.DataFrame({"quantile": quantile.loc[masks["test"]], "markout": frame.loc[masks["test"], markout]}).dropna()
            grouped = test.groupby("quantile", as_index=False).agg(mean=("markout", "mean"), n=("markout", "size"))
            grouped["proxy"] = proxy
            grouped["side"] = side
            rows.append(grouped)
    plot_data = pd.concat(rows).groupby(["proxy", "quantile"], as_index=False).agg(mean=("mean", "mean"), n=("n", "sum"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for proxy, group in plot_data.groupby("proxy"):
        ax.plot(group["quantile"], group["mean"], marker="o", label=proxy)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Pressure Quantiles and Future Markout, W={window}s, H={horizon}s")
    ax.set_xlabel("Train-defined pressure quantile")
    ax.set_ylabel("Test mean side-adjusted markout (bps)")
    ax.legend(title="Proxy")
    ax.text(0.01, -0.22, "Negative values are adverse for the passive trader.", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_pressure_vs_markout.png", dpi=170)
    plt.close(fig)


def component_increment_results(proxy_table: pd.DataFrame, nested: pd.DataFrame, mechanism: pd.DataFrame) -> pd.DataFrame:
    test_nested = nested[nested["split"] == "test"].pivot_table(
        index=["side", "formation_window", "horizon"], columns="model", values=["r2", "rank_correlation"]
    )
    rows = []
    for idx, row in test_nested.iterrows():
        side, window, horizon = idx
        rows.append(
            {
                "side": side,
                "formation_window": window,
                "horizon": horizon,
                "market_only_r2": row[("r2", "M1_market")],
                "plus_cancel_r2": row[("r2", "M2_market_cancel")],
                "plus_replenishment_r2": row[("r2", "M3_market_cancel_replenish")],
                "full_proxy_r2": row[("r2", "M4_full_proxy")],
                "cancel_increment_r2": row[("r2", "M2_market_cancel")] - row[("r2", "M1_market")],
                "replenishment_increment_r2": row[("r2", "M3_market_cancel_replenish")] - row[("r2", "M2_market_cancel")],
                "full_proxy_increment_r2_vs_controls": row[("r2", "M4_full_proxy")] - row[("r2", "M0_controls")],
                "market_only_rank_correlation": row[("rank_correlation", "M1_market")],
                "full_proxy_rank_correlation": row[("rank_correlation", "M4_full_proxy")],
            }
        )
    out = pd.DataFrame(rows)
    p3 = proxy_table[(proxy_table["split"] == "test") & (proxy_table["proxy"] == "P3_full_depth_norm")][
        ["side", "formation_window", "horizon", "high_minus_low_markout_bps"]
    ].rename(columns={"high_minus_low_markout_bps": "full_proxy_top_minus_bottom_markout_bps"})
    return out.merge(p3, on=["side", "formation_window", "horizon"], how="left")


def depth_conditioned_results(mechanism: pd.DataFrame) -> pd.DataFrame:
    return mechanism[
        (mechanism["component"] == "P3_depth_normalized") & (mechanism["subgroup_type"] == "depth_tercile")
    ].copy()


def formation_response_results(proxy_table: pd.DataFrame) -> pd.DataFrame:
    return proxy_table[(proxy_table["split"] == "test") & (proxy_table["proxy"] == "P3_full_depth_norm")].copy()


def plot_incremental_components(increments: pd.DataFrame, window: int, horizon: int) -> None:
    data = increments[(increments["formation_window"] == window) & (increments["horizon"] == horizon)]
    models = ["M0 controls", "M1 market", "M2 + cancel", "M3 + replenish", "M4 full proxy"]
    r2_values = [
        0.0,
        np.nan,
        data["cancel_increment_r2"].mean(),
        data["replenishment_increment_r2"].mean(),
        data["full_proxy_increment_r2_vs_controls"].mean(),
    ]
    rank_values = [
        np.nan,
        data["market_only_rank_correlation"].mean(),
        np.nan,
        np.nan,
        data["full_proxy_rank_correlation"].mean(),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].bar(models, r2_values, color="#4c78a8")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Incremental R2")
    axes[0].set_ylabel("Test incremental R2")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(models, rank_values, color="#72b7b2")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Rank Correlation")
    axes[1].set_ylabel("Test Spearman rank correlation")
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle(f"Incremental Component Value, W={window}s, H={horizon}s", y=0.99)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_incremental_components.png", dpi=170)
    plt.close(fig)


def plot_proxy_comparison(proxy_table: pd.DataFrame, window: int, horizon: int) -> None:
    data = proxy_table[
        (proxy_table["split"] == "test") & (proxy_table["formation_window"] == window) & (proxy_table["horizon"] == horizon)
    ]
    summary = data.groupby("proxy", as_index=False)["high_minus_low_markout_bps"].mean()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(summary["proxy"], summary["high_minus_low_markout_bps"], color=["#4c78a8", "#72b7b2", "#f58518"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Market-Only Versus Richer Execution Pressure, W={window}s, H={horizon}s")
    ax.set_xlabel("Proxy")
    ax.set_ylabel("High-minus-low markout (bps)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    APPENDIX_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(APPENDIX_FIGURE_DIR / "proxy_comparison.png", dpi=170)
    plt.close(fig)


def plot_pressure_depth_map(side_frames: dict[str, pd.DataFrame], split: pd.DataFrame, window: int, horizon: int) -> None:
    rows = []
    for side, frame in side_frames.items():
        masks = split_masks(frame, split)
        pressure = f"p3_depth_norm_{window}s"
        depth = f"visible_depth_{window}s"
        markout = f"markout_bps_{horizon}s"
        pe = train_quantile_bins(frame[pressure], masks["train"], n_bins=5)
        de = train_quantile_bins(frame[depth], masks["train"], n_bins=5)
        pbin = apply_bins(frame[pressure], pe)
        dbin = apply_bins(frame[depth], de)
        m = masks["test"] & frame[markout].notna()
        tmp = pd.DataFrame({"pressure_bin": pbin.loc[m], "depth_bin": dbin.loc[m], "markout": frame.loc[m, markout]})
        rows.append(tmp)
    table = pd.concat(rows).groupby(["depth_bin", "pressure_bin"]).agg(mean=("markout", "mean"), n=("markout", "size")).reset_index()
    matrix = table.pivot(index="depth_bin", columns="pressure_bin", values="mean")
    counts = table.pivot(index="depth_bin", columns="pressure_bin", values="n")
    matrix = matrix.mask(counts < 500)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(matrix.sort_index(ascending=False), aspect="auto", cmap="coolwarm")
    ax.set_title(f"Pressure x Visible-Depth Markout Map, W={window}s, H={horizon}s")
    ax.set_xlabel("Execution-pressure quintile")
    ax.set_ylabel("Visible-depth quintile")
    fig.colorbar(im, ax=ax, label="Mean side-adjusted markout (bps)")
    fig.tight_layout()
    fig.savefig(APPENDIX_FIGURE_DIR / "pressure_depth_map.png", dpi=170)
    plt.close(fig)


def plot_formation_response_map(proxy_table: pd.DataFrame) -> None:
    data = proxy_table[(proxy_table["split"] == "test") & (proxy_table["proxy"] == "P3_full_depth_norm")]
    table = data.groupby(["formation_window", "horizon"], as_index=False)["high_minus_low_markout_bps"].mean()
    matrix = table.pivot(index="horizon", columns="formation_window", values="high_minus_low_markout_bps")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_title("Formation Window x Response Horizon")
    ax.set_xlabel("Formation window (seconds)")
    ax.set_ylabel("Future markout horizon (seconds)")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    fig.colorbar(im, ax=ax, label="High-minus-low markout (bps)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_formation_response_map.png", dpi=170)
    plt.close(fig)


def plot_daily_and_null_stability(daily: pd.DataFrame, null: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for side, group in daily.groupby("side"):
        axes[0].plot(group["date"], group["high_minus_low_markout"], marker="o", label=side)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Daily Test Stability")
    axes[0].set_xlabel("Test day")
    axes[0].set_ylabel("High-minus-low markout (bps)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(title="Side")

    pivot = null.pivot_table(index=["side", "formation_window", "horizon"], columns="sequence", values="high_minus_low_markout_bps").dropna()
    pivot["real_minus_shuffle"] = pivot["real"] - pivot["local_block_shuffle"]
    axes[1].hist(pivot["real_minus_shuffle"], bins=18, color="#f58518", alpha=0.85)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Real Minus Local-Shuffled Null")
    axes[1].set_xlabel("Difference in high-minus-low markout (bps)")
    axes[1].set_ylabel("Cell count")
    fig.suptitle("Daily and Null Stability", y=0.99)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_daily_and_null_stability.png", dpi=170)
    plt.close(fig)


def plot_component_contribution(mechanism: pd.DataFrame) -> None:
    data = mechanism[
        (mechanism["formation_window"] == 10)
        & (mechanism["horizon"] == 60)
        & (mechanism["subgroup_type"] == "all")
        & (mechanism["component"].isin(["market_flow", "cancellation", "negative_replenishment", "P2_market_cancel_raw", "P3_full_raw", "P3_depth_normalized"]))
    ]
    pivot = data.pivot_table(index="component", columns="side", values="high_minus_low_markout_bps")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    pivot.loc[pivot.index].plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Component Contribution by Passive Side, W=10s, H=60s")
    ax.set_xlabel("Component")
    ax.set_ylabel("High-minus-low markout (bps)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(APPENDIX_FIGURE_DIR / "component_contribution_by_side.png", dpi=170)
    plt.close(fig)


def plot_sign_reversal_decomposition(mechanism: pd.DataFrame) -> None:
    data = mechanism[
        (mechanism["formation_window"] == 10)
        & (mechanism["horizon"] == 60)
        & (mechanism["subgroup_type"] == "all")
        & (mechanism["component"].isin(["P2_market_cancel_raw", "P3_full_raw", "P3_depth_normalized", "inverse_depth_denominator"]))
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for side, group in data.groupby("side"):
        ordered = group.set_index("component").loc[
            ["P2_market_cancel_raw", "P3_full_raw", "P3_depth_normalized", "inverse_depth_denominator"]
        ]
        ax.plot(ordered.index, ordered["high_minus_low_markout_bps"], marker="o", label=side)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("P2-to-P3 Sign-Reversal Decomposition")
    ax.set_xlabel("Signal used for train-derived high/low buckets")
    ax.set_ylabel("High-minus-low markout (bps)")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(title="Passive side")
    fig.tight_layout()
    fig.savefig(APPENDIX_FIGURE_DIR / "p2_to_p3_sign_reversal_decomposition.png", dpi=170)
    plt.close(fig)


def plot_leave_one_day_out(loo: pd.DataFrame) -> None:
    data = loo[(loo["formation_window"] == 10) & (loo["horizon"] == 60)]
    labels = sorted(data["omitted_test_day"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for (side, proxy), group in data.groupby(["side", "proxy"]):
        group = group.set_index("omitted_test_day").loc[labels]
        ax.plot(group.index, group["high_minus_low_markout_bps"], marker="o", label=f"{side} {proxy}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Leave-One-Test-Day-Out Stability, W=10s, H=60s")
    ax.set_xlabel("Omitted test day")
    ax.set_ylabel("High-minus-low markout (bps)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(APPENDIX_FIGURE_DIR / "leave_one_day_out_stability.png", dpi=170)
    plt.close(fig)


def plot_conditional_response_maps(maps: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, map_name in zip(axes.ravel(), ["pressure_x_depth", "pressure_x_volatility", "cancellation_x_replenishment", "market_flow_x_replenishment"]):
        data = maps[maps["map"] == map_name].groupby(["x_bin", "y_bin"]).agg(mean=("mean_markout_bps", "mean"), count=("count", "sum")).reset_index()
        matrix = data.pivot(index="y_bin", columns="x_bin", values="mean")
        counts = data.pivot(index="y_bin", columns="x_bin", values="count")
        matrix = matrix.mask(counts < 500)
        im = ax.imshow(matrix.sort_index(ascending=False), cmap="coolwarm", aspect="auto")
        ax.set_title(map_name.replace("_", " "))
        ax.set_xlabel("x quintile")
        ax.set_ylabel("y quintile")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Conditional Response Maps, W=10s, H=60s", y=0.99)
    fig.tight_layout()
    fig.savefig(APPENDIX_FIGURE_DIR / "conditional_response_maps.png", dpi=170)
    plt.close(fig)


def write_report(
    data: pd.DataFrame,
    split: pd.DataFrame,
    selected_window: int,
    selected_horizon: int,
    proxy_table: pd.DataFrame,
    nested: pd.DataFrame,
    null: pd.DataFrame,
    daily: pd.DataFrame,
    mechanism: pd.DataFrame,
    loo: pd.DataFrame,
    runtime: float,
) -> None:
    test_proxy = proxy_table[(proxy_table["split"] == "test") & (proxy_table["proxy"] == "P3_full_depth_norm")]
    selected = test_proxy[(test_proxy["formation_window"] == selected_window) & (test_proxy["horizon"] == selected_horizon)]
    selected_mean = selected["high_minus_low_markout_bps"].mean()
    m = nested[nested["split"] == "test"].pivot_table(
        index=["side", "formation_window", "horizon"], columns="model", values="r2"
    )
    m2_m1 = (m["M2_market_cancel"] - m["M1_market"]).mean()
    m3_m2 = (m["M3_market_cancel_replenish"] - m["M2_market_cancel"]).mean()
    m4_m0 = (m["M4_full_proxy"] - m["M0_controls"]).mean()
    null_pivot = null.pivot_table(index=["side", "formation_window", "horizon"], columns="sequence", values="high_minus_low_markout_bps")
    null_gap = (null_pivot["real"] - null_pivot["local_block_shuffle"]).mean()
    mech_scale = mechanism[
        (mechanism["formation_window"] == selected_window)
        & (mechanism["horizon"] == selected_horizon)
        & (mechanism["subgroup_type"] == "all")
    ]
    p2_mean = mech_scale[mech_scale["component"] == "P2_market_cancel_raw"]["high_minus_low_markout_bps"].mean()
    p3_raw_mean = mech_scale[mech_scale["component"] == "P3_full_raw"]["high_minus_low_markout_bps"].mean()
    p3_norm_mean = mech_scale[mech_scale["component"] == "P3_depth_normalized"]["high_minus_low_markout_bps"].mean()
    market_mean = mech_scale[mech_scale["component"] == "market_flow"]["high_minus_low_markout_bps"].mean()
    cancel_mean = mech_scale[mech_scale["component"] == "cancellation"]["high_minus_low_markout_bps"].mean()
    neg_replenish_mean = mech_scale[mech_scale["component"] == "negative_replenishment"]["high_minus_low_markout_bps"].mean()
    inverse_depth_mean = mech_scale[mech_scale["component"] == "inverse_depth_denominator"]["high_minus_low_markout_bps"].mean()
    buy_p3 = mech_scale[(mech_scale["component"] == "P3_depth_normalized") & (mech_scale["side"] == "buy")]["high_minus_low_markout_bps"].mean()
    sell_p3 = mech_scale[(mech_scale["component"] == "P3_depth_normalized") & (mech_scale["side"] == "sell")]["high_minus_low_markout_bps"].mean()
    omitted_0418 = loo[(loo["proxy"] == "P3_depth_normalized") & (loo["omitted_test_day"] == "2021-04-18")]["high_minus_low_markout_bps"].mean()
    split_lines = ["| split | start_date_utc | end_date_utc |", "|---|---|---|"]
    for row in split.itertuples(index=False):
        split_lines.append(f"| {row.split} | {row.start_date_utc} | {row.end_date_utc} |")
    split_md = "\n".join(split_lines)
    text = f"""# Real BTC Validation

## 1. Why real-data validation is needed

The synthetic layer tests exact hypothetical fills and post-fill markouts under controlled replay assumptions. The real Coinbase BTC layer checks whether analogous quote-consumption conditions appear in observed market states.

## 2. Data and evidence boundary

- Dataset: `data/processed/kaggle_btc_canonical.parquet`
- Rows: {len(data):,}
- Date range: {data['timestamp'].min()} to {data['timestamp'].max()}
- Frequency: one-second sampled market-state and interval-aggregate data
- Visible levels: 15
- Exact real passive fills: not observed
- FIFO queue position: not reconstructable

## 3. Passive-side execution-pressure proxy

For a passive buy at the bid, pressure uses aggressive sell activity, bid-side cancellations, bid-side limit replenishment, and bid depth. For a passive sell at the ask, the same construction uses aggressive buy activity, ask-side cancellations, ask-side limit replenishment, and ask depth.

P1 is market pressure only. P2 adds same-side cancellation. P3 adds cancellation and subtracts same-side replenishment. The main proxy is P3 divided by visible passive-side depth.

## 4. Future markout definition

Future markout is side-adjusted mid-price response. Positive is favorable to the passive quote; negative is adverse. Buy-side markout is future mid return; sell-side markout reverses the sign.

## 5. Chronological experimental design

{split_md}

Formation windows: {FORMATION_WINDOWS_SECONDS}

Response horizons: {MARKOUT_HORIZONS_SECONDS}

## 6. Main pressure-markout result

The validation-selected display scale is W={selected_window}s and H={selected_horizon}s. On the untouched test period, the average high-minus-low P3 depth-normalized pressure contrast is {selected_mean:.4f} bps. Negative values mean higher execution pressure is associated with worse passive-side future markout.

## 7. Cancellation and replenishment contribution

Mean test incremental R2, M2-M1 cancellation contribution: {m2_m1:.6f}.

Mean test incremental R2, M3-M2 replenishment contribution: {m3_m2:.6f}.

Mean test incremental R2, M4-M0 full-proxy contribution: {m4_m0:.6f}.

## 8. Time-scale result

The formation-window x response-horizon map is saved as `outputs/figures/real_btc_main/04_formation_response_map.png`. It should be interpreted as a proxy response map, not an exact fill map.

## 9. Null test

The local 5-minute block shuffle preserves broad local regimes while disrupting pressure/markout alignment. The mean real-minus-shuffled high-minus-low contrast is {null_gap:.4f} bps.

## 10. Daily stability

Daily stability is stored in `outputs/tables/main/real_btc_daily_stability.csv`. The table reports high-minus-low markout, rank correlation, a simple daily coefficient, and count by date and side.

## 11. Comparison with synthetic exact-fill experiment

See `outputs/tables/main/synthetic_real_comparison.csv`. The synthetic layer tests exact-fill mechanics; the real layer tests aggregated execution-pressure conditions.

## 12. Why does the proxy change sign?

At W={selected_window}s and H={selected_horizon}s, the mean test high-minus-low contrast changes from {p2_mean:.4f} bps for P2 to {p3_raw_mean:.4f} bps for raw P3 and {p3_norm_mean:.4f} bps for depth-normalized P3. This shows that subtracting same-side replenishment and then sorting by the depth-normalized proxy materially changes which states are classified as high pressure.

The component audit explains the sign reversal. Market flow alone has a mean contrast of {market_mean:.4f} bps and cancellation alone has a favorable mean contrast of {cancel_mean:.4f} bps, so P2 is positive. The negative-replenishment component has a mean contrast of {neg_replenish_mean:.4f} bps, which pulls raw P3 negative. The inverse-depth denominator contrast is {inverse_depth_mean:.4f} bps; low displayed depth contributes to extreme pressure values, but depth normalization attenuates rather than creates the raw P3 adverse contrast.

The adverse P3 result is concentrated more on passive buys ({buy_p3:.4f} bps) than passive sells ({sell_p3:.4f} bps). Leave-one-day-out stability also shows day dependence: when 2021-04-18 is removed, the average P3 contrast is {omitted_0418:.4f} bps. The mechanism audit therefore supports a regime-dependent proxy result rather than a uniform execution-pressure law.

The detailed mechanism table is `outputs/tables/main/real_btc_mechanism_audit.csv`. It separates market flow, cancellation, replenishment, raw P2, raw P3, depth-normalized P3, and inverse-depth denominator effects by side, test day, depth tercile, volatility tercile, and spread regime.

## 13. Supported findings

- Real Coinbase BTC data supports side-adjusted post-quote markout labels.
- Market, cancellation, limit replenishment, and visible depth are available as one-second aggregate proxies.
- The repository now links exact synthetic fills with real execution-pressure validation without claiming exact real fills.
- Higher pressure states are associated with more adverse passive-side markout in selected side/window/horizon regimes.

## 14. Partially supported findings

- Cancellation and replenishment add small average incremental R2 beyond market pressure, but the effect size is modest.
- The full depth-normalized proxy provides useful ordering in selected bins, but it does not consistently outperform controls across the full model grid.
- Daily stability is mixed; the selected pressure-markout contrast changes magnitude and sign across test days and sides.

## 15. Unsupported findings

- Exact passive fill probability is not measured in the real BTC layer.
- Exact FIFO queue position is not reconstructable.
- Any cancellation/replenishment contribution must be read through the one-second aggregate data boundary.
- The local shuffled null does not support a strong claim that the full proxy captures a stable sequence effect across all scales.

## 16. Limitations

The dataset is one-second aggregated, not order-level MBO. Hidden liquidity, latency, partial fills, and queue priority are not observed. The real-data result is a mechanism validation of post-quote price response, not a deployable trading claim.

Runtime: {runtime:.2f} seconds.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def main() -> None:
    start = time.perf_counter()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for old_png in FIGURE_DIR.glob("*.png"):
        old_png.unlink()
    data, split = load_inputs()
    side_frames = {side: build_side_dataset(data, side, FORMATION_WINDOWS_SECONDS, MARKOUT_HORIZONS_SECONDS) for side in SIDES}

    quantiles = pressure_quantile_results(side_frames, split)
    quantiles.to_csv(TABLE_DIR / "real_btc_pressure_quantiles.csv", index=False)
    quantiles.to_csv(TABLE_DIR / "pressure_quantile_results.csv", index=False)
    proxy_table = proxy_comparison(side_frames, split)
    proxy_table.to_csv(TABLE_DIR / "real_btc_proxy_results.csv", index=False)
    nested = nested_models(side_frames, split)
    nested.to_csv(TABLE_DIR / "real_btc_nested_models.csv", index=False)
    nested.to_csv(TABLE_DIR / "nested_model_results.csv", index=False)
    selected_window, selected_horizon = select_primary_window(proxy_table)
    daily = daily_stability(side_frames, split, selected_window, selected_horizon)
    daily.to_csv(TABLE_DIR / "real_btc_daily_stability.csv", index=False)
    daily.to_csv(TABLE_DIR / "daily_stability.csv", index=False)
    null = null_test(side_frames, split)
    null.to_csv(TABLE_DIR / "real_btc_null_test.csv", index=False)
    null.to_csv(TABLE_DIR / "local_null_results.csv", index=False)
    mechanism, loo, maps = mechanism_audit(side_frames, split)
    mechanism.to_csv(TABLE_DIR / "real_btc_mechanism_audit.csv", index=False)
    loo.to_csv(TABLE_DIR / "real_btc_leave_one_day_out.csv", index=False)
    maps.to_csv(TABLE_DIR / "real_btc_conditional_response_maps.csv", index=False)
    comparison = synthetic_real_comparison(proxy_table, nested, null)
    comparison.to_csv(TABLE_DIR / "synthetic_real_comparison.csv", index=False)
    comparison.to_csv(TABLE_DIR / "synthetic_real_bridge.csv", index=False)
    increments = component_increment_results(proxy_table, nested, mechanism)
    increments.to_csv(TABLE_DIR / "component_increment_results.csv", index=False)
    depth_conditioned_results(mechanism).to_csv(TABLE_DIR / "depth_conditioned_results.csv", index=False)
    formation_response_results(proxy_table).to_csv(TABLE_DIR / "formation_response_results.csv", index=False)

    plot_pressure_vs_markout(side_frames, split, selected_window, selected_horizon)
    plot_incremental_components(increments, selected_window, selected_horizon)
    plot_proxy_comparison(proxy_table, selected_window, selected_horizon)
    plot_pressure_depth_map(side_frames, split, selected_window, selected_horizon)
    plot_formation_response_map(proxy_table)
    plot_daily_and_null_stability(daily, null)
    plot_component_contribution(mechanism)
    plot_sign_reversal_decomposition(mechanism)
    plot_leave_one_day_out(loo)
    plot_conditional_response_maps(maps)

    runtime = time.perf_counter() - start
    write_report(data, split, selected_window, selected_horizon, proxy_table, nested, null, daily, mechanism, loo, runtime)
    summary = {
        "rows": int(len(data)),
        "date_range": [str(data["timestamp"].min()), str(data["timestamp"].max())],
        "selected_window": selected_window,
        "selected_horizon": selected_horizon,
        "runtime_seconds": runtime,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
