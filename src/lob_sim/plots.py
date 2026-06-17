from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_calibration(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], color="0.7", linewidth=1, label="perfect calibration")
    ax.errorbar(table["mean_pred_fill_prob"], table["realized_fill_rate"], yerr=table["realized_fill_rate_ci95"], marker="o", linewidth=1.5, label="deciles")
    ax.set_xlabel("Predicted fill probability")
    ax.set_ylabel("Realized fill rate")
    ax.set_title("Fill calibration - controlled synthetic experiment")
    ax.legend()
    _save(fig, path)


def plot_fill_rate_by_score(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(table["fill_score_bin"], table["realized_fill_rate"], color="#4c78a8")
    ax.errorbar(table["fill_score_bin"], table["realized_fill_rate"], yerr=table["realized_fill_rate_ci95"], fmt="none", color="black", capsize=3)
    ax.set_xlabel("Predicted fill-probability bin")
    ax.set_ylabel("Realized fill rate")
    ax.set_title("Execution likelihood by fill score - controlled synthetic experiment")
    _save(fig, path)


def plot_markout_by_score(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(table["fill_score_bin"], table["mean_signed_markout_cond_fill"], color="#e45756")
    ax.errorbar(table["fill_score_bin"], table["mean_signed_markout_cond_fill"], yerr=table["mean_signed_markout_ci95"], fmt="none", color="black", capsize=3)
    ax.axhline(0, color="0.5", linewidth=1)
    ax.set_xlabel("Predicted fill-probability bin")
    ax.set_ylabel("Signed post-fill markout (ticks)")
    ax.set_title("Execution quality by fill score - controlled synthetic experiment")
    _save(fig, path)


def plot_decile_metric(table: pd.DataFrame, y: str, yerr: str | None, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table["fill_prob_decile"], table[y], color="#4c78a8")
    if yerr and yerr in table:
        ax.errorbar(table["fill_prob_decile"], table[y], yerr=table[yerr], fmt="none", color="black", capsize=3, linewidth=1)
    ax.set_xlabel("Predicted fill-probability decile")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _save(fig, path)


def plot_frontier(table: pd.DataFrame, path: Path) -> None:
    table = table.dropna(subset=["realized_fill_rate", "mean_signed_markout_cond_fill", "expected_posting_value"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sizes = table["n"] / table["n"].max() * 180
    ax.scatter(table["realized_fill_rate"], table["mean_signed_markout_cond_fill"], s=sizes, c=table["expected_posting_value"], cmap="viridis")
    for _, row in table.iterrows():
        ax.text(row["realized_fill_rate"], row["mean_signed_markout_cond_fill"], str(int(row["fill_prob_decile"])), fontsize=8)
    ax.axhline(0, color="0.6", linewidth=1)
    ax.set_xlabel("Realized fill rate")
    ax.set_ylabel("Mean signed markout conditional on fill (ticks)")
    ax.set_title("Fill-toxicity frontier by fill-probability decile")
    _save(fig, path)


def plot_mechanism(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(table["regime"], table["mean_signed_markout"], color=["#f58518", "#54a24b"][: len(table)])
    ax.axhline(0, color="0.6", linewidth=1)
    ax.set_xlabel("Queue depletion regime")
    ax.set_ylabel("Mean signed markout conditional on fill (ticks)")
    ax.set_title("Trade-driven versus cancellation-driven depletion")
    _save(fig, path)


def plot_policy(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(table["policy"], table["expected_value_per_opportunity"], color="#4c78a8")
    ax.axhline(0, color="0.6", linewidth=1)
    ax.set_xlabel("Posting policy")
    ax.set_ylabel("Expected value per opportunity (ticks)")
    ax.set_title("Passive-order decision comparison")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, path)


def plot_regime_boundary(table: pd.DataFrame, path: Path) -> None:
    subset = table[table["boundary"] == "signed_flow"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(subset["regime"], subset["mean_signed_markout"], color="#e45756")
    ax.axhline(0, color="0.6", linewidth=1)
    ax.set_xlabel("Signed-flow regime")
    ax.set_ylabel("Mean signed markout conditional on fill (ticks)")
    ax.set_title("Regime boundary: weak versus persistent signed flow")
    _save(fig, path)


def plot_metric_by_window(table: pd.DataFrame, metric: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for model_name, group in table.groupby("model"):
        ax.plot(group["lookback_window"], group[metric], marker="o", label=model_name)
    ax.set_xscale("log")
    ax.set_xlabel("Lookback window W (events)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Incremental interaction value versus event-history scale")
    ax.legend()
    _save(fig, path)


def plot_surface(table: pd.DataFrame, value_label: str, title: str, path: Path) -> None:
    pivot = table.pivot(index="depletion_bin", columns="flow_bin", values="mean_value").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(pivot, aspect="auto", cmap="RdYlGn", interpolation="nearest")
    ax.set_xlabel("Local signed-flow persistence bin")
    ax.set_ylabel("Passive-side trade depletion bin")
    ax.set_title(title)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(i)) for i in pivot.index])
    fig.colorbar(image, ax=ax, label=value_label)
    _save(fig, path)


def plot_scale_map(table: pd.DataFrame, path: Path) -> None:
    pivot = table.pivot(index="markout_horizon", columns="lookback_window", values="high_low_markout_contrast")
    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(pivot, aspect="auto", cmap="RdBu", interpolation="nearest")
    ax.set_xlabel("Lookback window W (events)")
    ax.set_ylabel("Markout horizon H (events)")
    ax.set_title("Formation-window by response-horizon interaction map")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(i)) for i in pivot.index])
    fig.colorbar(image, ax=ax, label="High-minus-low interaction markout (ticks)")
    _save(fig, path)


def plot_real_vs_null(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for source, group in table.groupby("source"):
        ax.plot(group["lookback_window"], group["m3_minus_m2_auc"], marker="o", label=source)
    ax.axhline(0, color="0.6", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Lookback window W (events)")
    ax.set_ylabel("M3 - M2 fill ROC AUC")
    ax.set_title("Interaction diagnostic versus local-shuffled null")
    ax.legend()
    _save(fig, path)
