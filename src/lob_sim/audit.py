from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS_NO_FUTURE = {
    "spread",
    "queue_ahead",
    "bid_depth",
    "ask_depth",
    "signed_queue_imbalance",
    "signed_microprice_deviation",
    "signed_recent_trade_flow",
    "recent_trade_volume",
    "recent_volatility",
    "recent_mid_move",
    "side_is_buy",
}


def run_audits(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str], markout_horizons: list[int]) -> pd.DataFrame:
    rows = []

    def add(check: str, passed: bool, detail: str) -> None:
        rows.append({"check": check, "passed": bool(passed), "detail": detail})

    add("no_overlapping_obs_ids_train_validation", set(train["obs_id"]).isdisjoint(set(validation["obs_id"])), "")
    add("no_overlapping_obs_ids_train_test", set(train["obs_id"]).isdisjoint(set(test["obs_id"])), "")
    add("no_overlapping_obs_ids_validation_test", set(validation["obs_id"]).isdisjoint(set(test["obs_id"])), "")
    add("chronological_train_before_validation", train["step"].max() < validation["step"].min(), f"{train['step'].max()} < {validation['step'].min()}")
    add("chronological_validation_before_test", validation["step"].max() < test["step"].min(), f"{validation['step'].max()} < {test['step'].min()}")
    add("no_negative_time_to_fill", (test["time_to_fill"].dropna() >= 0).all(), "")
    for horizon in markout_horizons:
        filled = test[(test["filled"] == 1) & test[f"markout_{horizon}"].notna()]
        add(f"markout_{horizon}_after_fill", ((filled["fill_step"] + horizon) > filled["fill_step"]).all(), "")
    add("no_duplicated_order_observations", not pd.concat([train, validation, test])["obs_id"].duplicated().any(), "")
    allowed = set(FEATURE_COLUMNS_NO_FUTURE)
    allowed.update(
        col
        for col in feature_columns
        if col.startswith("flow_persistence_")
        or col.startswith("total_depletion_")
        or col.startswith("trade_depletion_")
        or col.startswith("cancel_depletion_")
        or col.startswith("replenishment_")
        or col.startswith("flow_depletion_interaction_")
    )
    add("feature_allowlist_no_future_labels", set(feature_columns).issubset(allowed), ",".join(sorted(set(feature_columns) - allowed)))
    all_orders = pd.concat([train, validation, test])
    add("no_zero_or_negative_spread", (all_orders["spread"] > 0).all(), "")
    add("no_negative_queue_ahead", (all_orders["queue_ahead"] >= 0).all(), "")
    target_like = {"filled", "time_to_fill", "fill_step", "fill_price", "adverse_fill"}
    target_like.update(col for col in all_orders.columns if col.startswith("markout_"))
    add("target_columns_not_in_features", set(feature_columns).isdisjoint(target_like), ",".join(sorted(set(feature_columns) & target_like)))

    return pd.DataFrame(rows)
