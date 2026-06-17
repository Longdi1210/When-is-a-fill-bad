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
from lob_sim.evaluation import (
    calibration_table,
    fill_toxicity_table,
    markout_decile_table,
    mechanism_table,
    regime_boundary_table,
)
from lob_sim.labels import construct_passive_orders, expected_posting_value, label_orders
from lob_sim.models import (
    FEATURE_COLUMNS,
    chronological_split,
    fill_metrics,
    fit_fill_model,
    fit_markout_model,
    markout_metrics,
    predict_fill,
    predict_markout,
)
from lob_sim.phd_layer import PhDProfile
from lob_sim.plots import (
    plot_calibration,
    plot_decile_metric,
    plot_frontier,
    plot_mechanism,
    plot_policy,
    plot_regime_boundary,
)
from lob_sim.policy import policy_comparison


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    config_path = ROOT / "config" / "default.yaml"
    config = ResearchConfig.from_yaml(config_path)
    output_dir = ROOT / config.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    profile = PhDProfile.from_json(ROOT / "configs" / "phd_profile.json")
    markout_col = f"markout_{config.markout_horizon_for_value}"

    events = generate_event_log(config, profile)
    orders = construct_passive_orders(events, config)
    labeled = label_orders(orders, events, config)
    train, validation, test = chronological_split(labeled, config.train_fraction, config.validation_fraction)

    fill_model = fit_fill_model(train, config.models.get("fill", "logistic"))
    markout_train = train[(train["filled"] == 1) & train[markout_col].notna()]
    markout_model = fit_markout_model(markout_train, markout_col, config.models.get("markout", "ridge"))

    for split_name, split in [("train", train), ("validation", validation), ("test", test)]:
        split["pred_fill_prob"] = predict_fill(fill_model, split)
        split["pred_markout"] = predict_markout(markout_model, split)
        split["pred_posting_value"] = [
            expected_posting_value(p, s, m, config.fees_ticks)
            for p, s, m in zip(split["pred_fill_prob"], split["spread"], split["pred_markout"])
        ]
        realized_if_filled = split["spread"] / 2.0 + split[markout_col].fillna(0.0) - config.fees_ticks
        split["realized_posting_value_if_filled"] = realized_if_filled
        split["realized_posting_value"] = split["filled"] * realized_if_filled
        if split_name == "train":
            train = split
        elif split_name == "validation":
            validation = split
        else:
            test = split

    fill_metric_rows = []
    markout_metric_rows = []
    for split_name, split in [("train", train), ("validation", validation), ("test", test)]:
        fill_metric_rows.append({"split": split_name, **fill_metrics(split["filled"], split["pred_fill_prob"])})
        filled = split[(split["filled"] == 1) & split[markout_col].notna()]
        markout_metric_rows.append({"split": split_name, **markout_metrics(filled[markout_col], filled["pred_markout"])})

    event_summary = pd.DataFrame(
        [
            {
                "data_type": "synthetic_lob",
                "events": len(events),
                "hypothetical_orders": len(labeled),
                "fills": int(labeled["filled"].sum()),
                "fill_rate": float(labeled["filled"].mean()),
                "fill_horizon_events": config.fill_horizon,
                "value_markout_horizon_events": config.markout_horizon_for_value,
                "queue_ahead_fraction": config.queue_ahead_fraction,
                "order_size": config.order_size,
            }
        ]
    )
    split_table = pd.DataFrame(
        [
            {"split": "train", "start_step": int(train["step"].min()), "end_step": int(train["step"].max()), "n": len(train)},
            {"split": "validation", "start_step": int(validation["step"].min()), "end_step": int(validation["step"].max()), "n": len(validation)},
            {"split": "test", "start_step": int(test["step"].min()), "end_step": int(test["step"].max()), "n": len(test)},
        ]
    )
    calibration = calibration_table(test)
    toxicity = fill_toxicity_table(test, markout_col, config.fees_ticks)
    markout_deciles = markout_decile_table(test, markout_col)
    mechanisms = mechanism_table(test, markout_col)
    regimes = regime_boundary_table(test, markout_col)
    policies = policy_comparison(test, markout_col)
    audits = run_audits(train, validation, test, FEATURE_COLUMNS, config.markout_horizons)

    sensitivity_rows = []
    for fraction in config.queue_ahead_sensitivity:
        scenario_config = ResearchConfig(**{**config.__dict__, "queue_ahead_fraction": fraction})
        scenario_orders = construct_passive_orders(events, scenario_config)
        scenario_labeled = label_orders(scenario_orders, events, scenario_config)
        filled = scenario_labeled[(scenario_labeled["filled"] == 1) & scenario_labeled[markout_col].notna()]
        sensitivity_rows.append(
            {
                "queue_ahead_fraction": fraction,
                "orders": len(scenario_labeled),
                "fill_rate": float(scenario_labeled["filled"].mean()),
                "mean_signed_markout_cond_fill": float(filled[markout_col].mean()) if len(filled) else None,
                "adverse_fill_rate_cond_fill": float(filled["adverse_fill"].mean()) if len(filled) else None,
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)

    write_table(events, output_dir / "synthetic_events.csv")
    write_table(labeled, output_dir / "passive_order_labels.csv")
    write_table(event_summary, tables_dir / "dataset_summary.csv")
    write_table(split_table, tables_dir / "split_definitions.csv")
    write_table(pd.DataFrame(fill_metric_rows), tables_dir / "fill_model_metrics.csv")
    write_table(pd.DataFrame(markout_metric_rows), tables_dir / "markout_model_metrics.csv")
    write_table(calibration, tables_dir / "fill_calibration_by_decile.csv")
    write_table(toxicity, tables_dir / "fill_toxicity_deciles.csv")
    write_table(markout_deciles, tables_dir / "markout_deciles.csv")
    write_table(mechanisms, tables_dir / "mechanism_decomposition.csv")
    write_table(regimes, tables_dir / "regime_boundary.csv")
    write_table(policies, tables_dir / "policy_comparison.csv")
    write_table(sensitivity, tables_dir / "queue_sensitivity.csv")
    write_table(audits, tables_dir / "audit_checks.csv")

    plot_calibration(calibration, figures_dir / "01_fill_model_calibration.png")
    plot_decile_metric(calibration, "realized_fill_rate", "realized_fill_rate_ci95", "Realized fill rate", "Fill rate by predicted fill-probability decile", figures_dir / "02_fill_rate_by_decile.png")
    plot_decile_metric(toxicity, "mean_signed_markout_cond_fill", "mean_signed_markout_ci95", "Mean signed markout (ticks)", "Conditional signed markout by fill-probability decile", figures_dir / "03_markout_by_fill_decile.png")
    plot_frontier(toxicity, figures_dir / "04_fill_toxicity_frontier.png")
    plot_decile_metric(toxicity, "expected_posting_value", None, "Expected posting value (ticks)", "Economic posting value by fill-probability decile", figures_dir / "05_value_by_fill_decile.png")
    plot_mechanism(mechanisms, figures_dir / "06_mechanism_trade_vs_cancel.png")
    plot_policy(policies, figures_dir / "07_policy_comparison.png")
    plot_regime_boundary(regimes, figures_dir / "08_regime_boundary.png")

    summary = {
        "dataset": event_summary.iloc[0].to_dict(),
        "fill_model_test": pd.DataFrame(fill_metric_rows).query("split == 'test'").iloc[0].to_dict(),
        "markout_model_test": pd.DataFrame(markout_metric_rows).query("split == 'test'").iloc[0].to_dict(),
        "best_policy_by_ev": policies.sort_values("expected_value_per_opportunity", ascending=False).iloc[0].to_dict(),
        "audit_passed": bool(audits["passed"].all()),
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

