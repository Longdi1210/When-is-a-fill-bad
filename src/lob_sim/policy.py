from __future__ import annotations

import pandas as pd


def summarize_policy(name: str, selected: pd.DataFrame, markout_col: str) -> dict:
    filled = selected[selected["filled"] == 1]
    spread_capture = selected["spread"].mean() / 2.0 if len(selected) else 0.0
    return {
        "policy": name,
        "selected_n": int(len(selected)),
        "fill_rate": float(selected["filled"].mean()) if len(selected) else 0.0,
        "avg_markout_per_fill": float(filled[markout_col].mean()) if len(filled) else 0.0,
        "avg_spread_capture_ticks": float(spread_capture),
        "expected_value_per_opportunity": float(selected["realized_posting_value"].mean()) if len(selected) else 0.0,
        "expected_value_per_fill": float(filled["realized_posting_value_if_filled"].mean()) if len(filled) else 0.0,
        "fraction_adverse_fills": float(filled["adverse_fill"].mean()) if len(filled) else 0.0,
    }


def policy_comparison(df: pd.DataFrame, markout_col: str, selection_fraction: float = 0.10) -> pd.DataFrame:
    n_select = max(1, int(len(df) * selection_fraction))
    policies = [
        summarize_policy("post_every_eligible", df, markout_col),
        summarize_policy("top_predicted_fill_probability", df.nlargest(n_select, "pred_fill_prob"), markout_col),
        summarize_policy("top_predicted_posting_value", df.nlargest(n_select, "pred_posting_value"), markout_col),
        summarize_policy("reject_predicted_toxic", df[df["pred_markout"] >= 0.0], markout_col),
    ]
    return pd.DataFrame(policies)
