from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .labels import expected_posting_value


def add_deciles(df: pd.DataFrame, column: str, output: str) -> pd.DataFrame:
    result = df.copy()
    try:
        result[output] = pd.qcut(result[column], 10, labels=False, duplicates="drop") + 1
    except ValueError:
        result[output] = 1
    return result


def binomial_ci(rate: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    return 1.96 * math.sqrt(max(rate * (1.0 - rate), 0.0) / n)


def mean_ci(values: pd.Series) -> float:
    clean = values.dropna()
    if len(clean) <= 1:
        return float("nan")
    return 1.96 * clean.std(ddof=1) / math.sqrt(len(clean))


def calibration_table(df: pd.DataFrame, pred_col: str = "pred_fill_prob") -> pd.DataFrame:
    binned = add_deciles(df, pred_col, "fill_prob_decile")
    rows = []
    for decile, group in binned.groupby("fill_prob_decile", observed=True):
        realized = float(group["filled"].mean())
        rows.append(
            {
                "fill_prob_decile": int(decile),
                "n": int(len(group)),
                "mean_pred_fill_prob": float(group[pred_col].mean()),
                "realized_fill_rate": realized,
                "realized_fill_rate_ci95": binomial_ci(realized, len(group)),
            }
        )
    return pd.DataFrame(rows)


def fill_score_bin_table(df: pd.DataFrame, pred_col: str, markout_col: str, bins: int = 10) -> pd.DataFrame:
    source = add_deciles(df, pred_col, "fill_score_bin")
    rows = []
    for bin_id, group in source.groupby("fill_score_bin", observed=True):
        filled = group[(group["filled"] == 1) & group[markout_col].notna()]
        realized = float(group["filled"].mean())
        rows.append(
            {
                "fill_score_bin": int(bin_id),
                "n": int(len(group)),
                "filled_n": int(len(filled)),
                "mean_pred_fill_prob": float(group[pred_col].mean()),
                "realized_fill_rate": realized,
                "realized_fill_rate_ci95": binomial_ci(realized, len(group)),
                "mean_signed_markout_cond_fill": float(filled[markout_col].mean()) if len(filled) else np.nan,
                "mean_signed_markout_ci95": mean_ci(filled[markout_col]) if len(filled) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def fill_toxicity_table(df: pd.DataFrame, markout_col: str, fees_ticks: float) -> pd.DataFrame:
    binned = add_deciles(df, "pred_fill_prob", "fill_prob_decile")
    rows = []
    for decile, group in binned.groupby("fill_prob_decile", observed=True):
        filled = group[group["filled"] == 1]
        adverse_rate = float(filled["adverse_fill"].mean()) if len(filled) else np.nan
        mean_markout = float(filled[markout_col].mean()) if len(filled) else np.nan
        mean_spread = float(group["spread"].mean())
        fill_rate = float(group["filled"].mean())
        value = expected_posting_value(fill_rate, mean_spread, 0.0 if np.isnan(mean_markout) else mean_markout, fees_ticks)
        trade_share = float(filled["trade_depletion_share"].mean()) if len(filled) else np.nan
        rows.append(
            {
                "fill_prob_decile": int(decile),
                "n": int(len(group)),
                "filled_n": int(len(filled)),
                "mean_pred_fill_prob": float(group["pred_fill_prob"].mean()),
                "realized_fill_rate": fill_rate,
                "realized_fill_rate_ci95": binomial_ci(fill_rate, len(group)),
                "adverse_fill_rate_cond_fill": adverse_rate,
                "adverse_fill_rate_ci95": binomial_ci(adverse_rate, len(filled)) if len(filled) else np.nan,
                "mean_signed_markout_cond_fill": mean_markout,
                "mean_signed_markout_ci95": mean_ci(filled[markout_col]) if len(filled) else np.nan,
                "mean_trade_depletion_share": trade_share,
                "expected_posting_value": value,
            }
        )
    return pd.DataFrame(rows)


def markout_decile_table(df: pd.DataFrame, markout_col: str) -> pd.DataFrame:
    filled = df[(df["filled"] == 1) & df[markout_col].notna()].copy()
    filled = add_deciles(filled, "pred_markout", "pred_markout_decile")
    rows = []
    for decile, group in filled.groupby("pred_markout_decile", observed=True):
        rows.append(
            {
                "pred_markout_decile": int(decile),
                "n": int(len(group)),
                "mean_pred_markout": float(group["pred_markout"].mean()),
                "realized_mean_markout": float(group[markout_col].mean()),
                "realized_markout_ci95": mean_ci(group[markout_col]),
            }
        )
    return pd.DataFrame(rows)


def mechanism_table(df: pd.DataFrame, markout_col: str) -> pd.DataFrame:
    filled = df[(df["filled"] == 1) & df["trade_depletion_share"].notna()].copy()
    filled["depletion_regime"] = np.where(filled["trade_depletion_share"] >= 0.5, "trade_driven", "cancel_driven")
    rows = []
    for regime, group in filled.groupby("depletion_regime"):
        rows.append(
            {
                "regime": regime,
                "n": int(len(group)),
                "mean_trade_depletion_share": float(group["trade_depletion_share"].mean()),
                "mean_time_to_fill": float(group["time_to_fill"].mean()),
                "adverse_fill_rate": float(group["adverse_fill"].mean()),
                "mean_signed_markout": float(group[markout_col].mean()),
                "mean_pred_fill_prob": float(group["pred_fill_prob"].mean()),
                "mean_pred_value": float(group["pred_posting_value"].mean()),
            }
        )
    return pd.DataFrame(rows)


def regime_boundary_table(df: pd.DataFrame, markout_col: str) -> pd.DataFrame:
    result = df.copy()
    result["flow_regime"] = np.where(result["signed_recent_trade_flow"].abs() >= result["signed_recent_trade_flow"].abs().median(), "persistent_flow", "weak_flow")
    result["vol_regime"] = np.where(result["recent_volatility"] >= result["recent_volatility"].median(), "high_vol", "low_vol")
    rows = []
    for flow_regime, group in result.groupby("flow_regime"):
        filled = group[group["filled"] == 1]
        rows.append(
            {
                "boundary": "signed_flow",
                "regime": flow_regime,
                "n": int(len(group)),
                "fill_rate": float(group["filled"].mean()),
                "adverse_fill_rate": float(filled["adverse_fill"].mean()) if len(filled) else np.nan,
                "mean_signed_markout": float(filled[markout_col].mean()) if len(filled) else np.nan,
                "mean_pred_value": float(group["pred_posting_value"].mean()),
            }
        )
    for vol_regime, group in result.groupby("vol_regime"):
        filled = group[group["filled"] == 1]
        rows.append(
            {
                "boundary": "volatility",
                "regime": vol_regime,
                "n": int(len(group)),
                "fill_rate": float(group["filled"].mean()),
                "adverse_fill_rate": float(filled["adverse_fill"].mean()) if len(filled) else np.nan,
                "mean_signed_markout": float(filled[markout_col].mean()) if len(filled) else np.nan,
                "mean_pred_value": float(group["pred_posting_value"].mean()),
            }
        )
    return pd.DataFrame(rows)


def response_surface(
    df: pd.DataFrame,
    flow_col: str,
    depletion_col: str,
    value_col: str,
    filled_only: bool = False,
    bins: int = 5,
) -> pd.DataFrame:
    source = df[df["filled"] == 1].copy() if filled_only else df.copy()
    source = source.dropna(subset=[flow_col, depletion_col, value_col])
    if source.empty:
        return pd.DataFrame()
    source["flow_bin"] = pd.qcut(source[flow_col], bins, labels=False, duplicates="drop") + 1
    source["depletion_bin"] = pd.qcut(source[depletion_col], bins, labels=False, duplicates="drop") + 1
    rows = []
    for (flow_bin, depletion_bin), group in source.groupby(["flow_bin", "depletion_bin"], observed=True):
        rows.append(
            {
                "flow_bin": int(flow_bin),
                "depletion_bin": int(depletion_bin),
                "n": int(len(group)),
                "mean_flow": float(group[flow_col].mean()),
                "mean_depletion": float(group[depletion_col].mean()),
                "mean_value": float(group[value_col].mean()),
            }
        )
    return pd.DataFrame(rows)


def interaction_contrast(df: pd.DataFrame, interaction_col: str, markout_col: str) -> dict:
    filled = df[(df["filled"] == 1) & df[markout_col].notna()].copy()
    if len(filled) < 20:
        return {"high_low_markout_contrast": np.nan, "high_group_markout": np.nan, "low_group_markout": np.nan, "n": len(filled)}
    low_threshold = filled[interaction_col].quantile(0.2)
    high_threshold = filled[interaction_col].quantile(0.8)
    low = filled[filled[interaction_col] <= low_threshold]
    high = filled[filled[interaction_col] >= high_threshold]
    return {
        "high_low_markout_contrast": float(high[markout_col].mean() - low[markout_col].mean()),
        "high_group_markout": float(high[markout_col].mean()),
        "low_group_markout": float(low[markout_col].mean()),
        "n": int(len(filled)),
    }
