from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lob_sim.audit import run_audits
from lob_sim.config import ResearchConfig
from lob_sim.data import generate_event_log
from lob_sim.evaluation import interaction_contrast, response_surface
from lob_sim.evaluation import calibration_table, fill_score_bin_table
from lob_sim.interaction import add_flow_depletion_features, interaction_feature_columns
from lob_sim.labels import construct_passive_orders, label_orders
from lob_sim.models import (
    chronological_split,
    fill_metrics,
    fit_fill_model_with_features,
    fit_markout_model_with_features,
    markout_metrics,
    predict_fill_with_features,
    predict_markout_with_features,
)
from lob_sim.null_model import local_shuffle_trade_signs
from lob_sim.phd_layer import PhDProfile
from lob_sim.plots import plot_metric_by_window, plot_real_vs_null, plot_scale_map, plot_surface
from lob_sim.plots import plot_calibration, plot_fill_rate_by_score, plot_markout_by_score
from lob_sim.real_data import load_btc_events, normalize_btc_events


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_events(config: ResearchConfig, profile: PhDProfile) -> tuple[pd.DataFrame, str]:
    if config.data_mode == "real_btc":
        raw = load_btc_events(ROOT / config.btc_data_path)
        return normalize_btc_events(raw), "real_btc"
    return generate_event_log(config, profile), "synthetic_validation"


def fit_nested_fill(train: pd.DataFrame, test: pd.DataFrame, window: int) -> list[dict]:
    feature_sets = interaction_feature_columns(window)
    rows = []
    for model_name, features in zip(["M0_controls", "M1_flow", "M2_flow_depletion", "M3_interaction"], feature_sets):
        model = fit_fill_model_with_features(train, features)
        pred = predict_fill_with_features(model, test, features)
        rows.append({"lookback_window": window, "model": model_name, **fill_metrics(test["filled"], pred)})
    return rows


def fit_nested_markout(train: pd.DataFrame, test: pd.DataFrame, window: int, markout_col: str) -> list[dict]:
    feature_sets = interaction_feature_columns(window)
    rows = []
    train_filled = train[(train["filled"] == 1) & train[markout_col].notna()]
    test_filled = test[(test["filled"] == 1) & test[markout_col].notna()]
    for model_name, features in zip(["M0_controls", "M1_flow", "M2_flow_depletion", "M3_interaction"], feature_sets):
        if train_filled.empty or test_filled.empty:
            rows.append(
                {
                    "lookback_window": window,
                    "model": model_name,
                    "markout_horizon": int(markout_col.split("_")[1]),
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "mean_signed_error": float("nan"),
                    "rank_correlation": float("nan"),
                    "n": int(len(test_filled)),
                }
            )
            continue
        model = fit_markout_model_with_features(train_filled, markout_col, features)
        pred = predict_markout_with_features(model, test_filled, features)
        rows.append({"lookback_window": window, "model": model_name, "markout_horizon": int(markout_col.split("_")[1]), **markout_metrics(test_filled[markout_col], pred)})
    return rows


def summarize_increment(fill_results: pd.DataFrame, markout_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, group in fill_results.groupby("lookback_window"):
        m2 = group[group["model"] == "M2_flow_depletion"].iloc[0]
        m3 = group[group["model"] == "M3_interaction"].iloc[0]
        rows.append(
            {
                "lookback_window": window,
                "fill_m3_minus_m2_auc": m3["roc_auc"] - m2["roc_auc"],
                "fill_m3_minus_m2_brier": m3["brier_score"] - m2["brier_score"],
                "fill_m3_minus_m2_log_loss": m3["log_loss"] - m2["log_loss"],
            }
        )
    fill_increment = pd.DataFrame(rows)
    markout_rows = []
    for (window, horizon), group in markout_results.groupby(["lookback_window", "markout_horizon"]):
        m2 = group[group["model"] == "M2_flow_depletion"].iloc[0]
        m3 = group[group["model"] == "M3_interaction"].iloc[0]
        markout_rows.append(
            {
                "lookback_window": window,
                "markout_horizon": horizon,
                "markout_m3_minus_m2_rmse": m3["rmse"] - m2["rmse"],
                "markout_m3_minus_m2_rank_corr": m3["rank_correlation"] - m2["rank_correlation"],
            }
        )
    return fill_increment.merge(pd.DataFrame(markout_rows), on="lookback_window", how="outer")


def run_nested_pipeline(events: pd.DataFrame, config: ResearchConfig) -> dict:
    orders = construct_passive_orders(events, config)
    labeled = label_orders(orders, events, config)
    featured = add_flow_depletion_features(labeled, events, config.lookback_windows, config.flow_weighting)
    train, validation, test = chronological_split(featured, config.train_fraction, config.validation_fraction)
    fill_rows = []
    markout_rows = []
    for window in config.lookback_windows:
        fill_rows.extend(fit_nested_fill(train, test, window))
        for horizon in config.markout_horizons:
            markout_rows.extend(fit_nested_markout(train, test, window, f"markout_{horizon}"))
    return {
        "events": events,
        "orders": featured,
        "train": train,
        "validation": validation,
        "test": test,
        "fill_results": pd.DataFrame(fill_rows),
        "markout_results": pd.DataFrame(markout_rows),
    }


def main() -> None:
    config = ResearchConfig.from_yaml(ROOT / "config" / "default.yaml")
    profile = PhDProfile.from_json(ROOT / "configs" / "phd_profile.json")
    output_dir = ROOT / config.output_dir
    main_tables_dir = output_dir / "tables" / "main"
    appendix_tables_dir = output_dir / "tables" / "appendix"
    main_figures_dir = output_dir / "figures" / "main"
    appendix_figures_dir = output_dir / "figures" / "appendix"
    processed_dir = ROOT / "data" / "processed"
    events, data_source = load_events(config, profile)
    pipeline = run_nested_pipeline(events, config)

    fill_results = pipeline["fill_results"]
    markout_results = pipeline["markout_results"]
    increment = summarize_increment(fill_results, markout_results)
    best_window = int(fill_results[fill_results["model"] == "M3_interaction"].sort_values("roc_auc", ascending=False).iloc[0]["lookback_window"])
    test = pipeline["test"].copy()
    markout_col = f"markout_{config.markout_horizon_for_value}"
    fill_surface = response_surface(test, f"flow_persistence_{best_window}", f"trade_depletion_{best_window}", "filled")
    markout_surface = response_surface(test, f"flow_persistence_{best_window}", f"trade_depletion_{best_window}", markout_col, filled_only=True)
    m3_features = interaction_feature_columns(best_window)[-1]
    m3_fill_model = fit_fill_model_with_features(pipeline["train"], m3_features)
    test["pred_fill_prob"] = predict_fill_with_features(m3_fill_model, test, m3_features)
    score_bins = fill_score_bin_table(test, "pred_fill_prob", markout_col)
    calibration = calibration_table(test, "pred_fill_prob")
    scale_rows = []
    for window in config.lookback_windows:
        for horizon in config.markout_horizons:
            scale_rows.append({"lookback_window": window, "markout_horizon": horizon, **interaction_contrast(test, f"flow_depletion_interaction_{window}", f"markout_{horizon}")})
    scale_map = pd.DataFrame(scale_rows)

    shuffled_events = local_shuffle_trade_signs(events, config.shuffle_block_size, config.seed + 991)
    null_pipeline = run_nested_pipeline(shuffled_events, config)
    null_fill = null_pipeline["fill_results"]
    null_rows = []
    for source, results in [("sequence", fill_results), ("local_shuffled_null", null_fill)]:
        for window, group in results.groupby("lookback_window"):
            m2 = group[group["model"] == "M2_flow_depletion"].iloc[0]
            m3 = group[group["model"] == "M3_interaction"].iloc[0]
            null_rows.append({"source": source, "lookback_window": window, "m3_minus_m2_auc": m3["roc_auc"] - m2["roc_auc"]})
    null_comparison = pd.DataFrame(null_rows)

    dataset_summary = pd.DataFrame(
        [
            {
                "data_source": data_source,
                "instrument": config.instrument,
                "venue": config.venue,
                "events": len(events),
                "hypothetical_orders": len(pipeline["orders"]),
                "fills": int(pipeline["orders"]["filled"].sum()),
                "fill_rate": float(pipeline["orders"]["filled"].mean()),
                "real_btc_available": data_source == "real_btc",
            }
        ]
    )
    split_table = pd.DataFrame(
        [
            {"split": "train", "start_step": int(pipeline["train"]["step"].min()), "end_step": int(pipeline["train"]["step"].max()), "n": len(pipeline["train"])},
            {"split": "validation", "start_step": int(pipeline["validation"]["step"].min()), "end_step": int(pipeline["validation"]["step"].max()), "n": len(pipeline["validation"])},
            {"split": "test", "start_step": int(test["step"].min()), "end_step": int(test["step"].max()), "n": len(test)},
        ]
    )
    label_stats = pd.DataFrame(
        [
            {"label": "fill", "count": len(pipeline["orders"]), "mean": float(pipeline["orders"]["filled"].mean())},
            {"label": "adverse_fill", "count": int(pipeline["orders"]["adverse_fill"].notna().sum()), "mean": float(pipeline["orders"]["adverse_fill"].dropna().mean())},
        ]
    )
    markout_stats = pd.DataFrame(
        [
            {
                "markout_horizon": horizon,
                "count": int(pipeline["orders"][f"markout_{horizon}"].notna().sum()),
                "mean": float(pipeline["orders"][f"markout_{horizon}"].dropna().mean()),
            }
            for horizon in config.markout_horizons
        ]
    )
    audits = run_audits(pipeline["train"], pipeline["validation"], test, interaction_feature_columns(best_window)[-1], config.markout_horizons)

    for name, table in [
        ("dataset_summary.csv", dataset_summary),
        ("split_definitions.csv", split_table),
        ("fill_label_statistics.csv", label_stats),
        ("markout_label_statistics.csv", markout_stats),
        ("fill_score_bins.csv", score_bins),
        ("fill_calibration.csv", calibration),
        ("interaction_effect_summary.csv", increment),
        ("real_vs_null_comparison.csv", null_comparison),
        ("audit_checks.csv", audits),
    ]:
        write_table(table, main_tables_dir / name)

    for name, table in [
        ("nested_fill_model_results.csv", fill_results),
        ("nested_markout_model_results.csv", markout_results),
        ("fill_response_surface.csv", fill_surface),
        ("markout_response_surface.csv", markout_surface),
        ("formation_response_scale_map.csv", scale_map),
    ]:
        write_table(table, appendix_tables_dir / name)

    write_table(pipeline["events"], processed_dir / "synthetic_events.csv")
    write_table(pipeline["orders"], processed_dir / "passive_order_labels.csv")

    plot_calibration(calibration, main_figures_dir / "01_fill_calibration.png")
    plot_fill_rate_by_score(score_bins, main_figures_dir / "02_fill_rate_by_score.png")
    plot_markout_by_score(score_bins, main_figures_dir / "03_markout_by_fill_score.png")
    plot_real_vs_null(null_comparison, main_figures_dir / "04_mechanism_test.png")
    plot_metric_by_window(fill_results, "roc_auc", appendix_figures_dir / "interaction_value_vs_lookback.png")
    plot_surface(fill_surface, "P(fill)", "Synthetic validation: fill probability response surface", appendix_figures_dir / "fill_probability_surface.png")
    plot_surface(markout_surface, "Signed markout (ticks)", "Synthetic validation: post-fill markout response surface", appendix_figures_dir / "markout_surface.png")
    plot_scale_map(scale_map, appendix_figures_dir / "formation_response_scale_map.png")
    flow_dist = test[[f"flow_persistence_{w}" for w in config.lookback_windows]].describe().T.reset_index()
    flow_dist["lookback_window"] = flow_dist["index"].str.extract(r"(\d+)").astype(int)
    write_table(flow_dist, appendix_tables_dir / "flow_persistence_distribution.csv")
    plot_metric_by_window(
        flow_dist.rename(columns={"mean": "M3_interaction", "lookback_window": "lookback_window"}).assign(model="mean_flow_persistence"),
        "M3_interaction",
        appendix_figures_dir / "flow_persistence_across_scales.png",
    )

    summary = {
        "data_source": data_source,
        "best_window_by_m3_fill_auc": best_window,
        "test_m3_fill_auc": float(fill_results[(fill_results["lookback_window"] == best_window) & (fill_results["model"] == "M3_interaction")]["roc_auc"].iloc[0]),
        "test_m3_minus_m2_auc": float(null_comparison[(null_comparison["source"] == "sequence") & (null_comparison["lookback_window"] == best_window)]["m3_minus_m2_auc"].iloc[0]),
        "null_m3_minus_m2_auc": float(null_comparison[(null_comparison["source"] == "local_shuffled_null") & (null_comparison["lookback_window"] == best_window)]["m3_minus_m2_auc"].iloc[0]),
        "audit_passed": bool(audits["passed"].all()),
        "real_btc_results_available": data_source == "real_btc",
    }
    (output_dir / "main_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
