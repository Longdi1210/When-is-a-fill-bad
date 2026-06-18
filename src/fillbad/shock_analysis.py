from __future__ import annotations

import numpy as np
import pandas as pd

from fillbad.multilevel_flow import DEPTH_LEVELS, potential_penetration_class, quote_survives, side_adjusted_markout_bps
from fillbad.real_validation import apply_bins, rank_correlation, train_quantile_bins


SHOCK_WINDOWS_SECONDS = [1, 2, 5, 10, 30]
RESPONSE_TIMES_SECONDS = [1, 2, 5, 10, 30, 60, 120, 300]
STRICT_TOTAL_TIMES_SECONDS = [10, 30, 60, 120, 300]
OUTCOME_HORIZONS_AFTER_ABSORPTION_SECONDS = [5, 25, 55, 115, 295]


def rolling_seconds(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    ordered = frame.sort_values("timestamp").set_index("timestamp")
    return ordered[column].rolling(f"{window}s", closed="both").sum().to_numpy()


def chronological_masks(frame: pd.DataFrame, split: pd.DataFrame) -> dict[str, pd.Series]:
    dates = pd.to_datetime(frame["timestamp"], utc=True).dt.date.astype(str)
    return {
        row.split: (dates >= str(row.start_date_utc)) & (dates <= str(row.end_date_utc))
        for row in split.itertuples(index=False)
    }


def fit_shock_thresholds(frame: pd.DataFrame, train_mask: pd.Series, score_col: str, quantile: float = 0.95) -> float:
    values = frame.loc[train_mask, score_col].replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.quantile(quantile))


def enforce_refractory(frame: pd.DataFrame, seconds: int = 30) -> pd.DataFrame:
    kept = []
    last_time_by_side = {}
    for row in frame.sort_values("timestamp").itertuples():
        last = last_time_by_side.get(row.side)
        if last is None or (row.timestamp - last).total_seconds() >= seconds:
            kept.append(row.Index)
            last_time_by_side[row.side] = row.timestamp
    return frame.loc[kept].copy()


def classify_absorption(episodes: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.DataFrame, dict]:
    out = episodes.copy()
    components = ["net_absorption_30s", "depth_recovery_5_30s", "quote_survives_30s"]
    params = {}
    score = pd.Series(0.0, index=out.index)
    for column in components[:2]:
        train = out.loc[train_mask, column].replace([np.inf, -np.inf], np.nan)
        mean = float(train.mean())
        std = float(train.std(ddof=0)) or 1.0
        params[column] = {"mean": mean, "std": std}
        score += ((out[column] - mean) / std).fillna(0.0)
    score += out["quote_survives_30s"].fillna(0.0)
    out["absorption_score"] = score
    train_score = score.loc[train_mask]
    low, high = train_score.quantile([1 / 3, 2 / 3])
    params["score_thresholds"] = {"weak_max": float(low), "strong_min": float(high)}
    out["absorption_state"] = pd.cut(
        score,
        bins=[-np.inf, low, high, np.inf],
        labels=["weak_absorption", "partial_absorption", "strong_absorption"],
        include_lowest=True,
    ).astype(str)
    return out, params


def classify_strict_absorption(episodes: pd.DataFrame, train_mask: pd.Series, absorption_window: int = 5) -> tuple[pd.DataFrame, dict]:
    out = episodes.copy()
    components = [
        f"flow_absorption_{absorption_window}s",
        f"depth_recovery_5_absorption_{absorption_window}s",
        f"spread_recovery_absorption_{absorption_window}s",
    ]
    score = pd.Series(0.0, index=out.index)
    params = {"absorption_window": absorption_window, "components": components}
    for column in components:
        train = out.loc[train_mask, column].replace([np.inf, -np.inf], np.nan).dropna()
        mean = float(train.mean())
        std = float(train.std(ddof=0)) or 1.0
        params[column] = {"mean": mean, "std": std}
        score += ((out[column] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["strict_absorption_score"] = score
    train_score = score.loc[train_mask]
    low, high = train_score.quantile([1 / 3, 2 / 3])
    params["score_thresholds"] = {"weak_max": float(low), "strong_min": float(high)}
    out["strict_absorption_state"] = pd.cut(
        score,
        bins=[-np.inf, low, high, np.inf],
        labels=["weak_absorption", "partial_absorption", "strong_absorption"],
        include_lowest=True,
    ).astype(str)
    return out, params


def clustered_difference_ci(
    frame: pd.DataFrame,
    value_col: str,
    group_col: str = "strict_absorption_state",
    high_label: str = "strong_absorption",
    low_label: str = "weak_absorption",
    block_col: str = "time_block",
    seed: int = 17,
    n_bootstrap: int = 300,
) -> dict:
    clean = frame[[value_col, group_col, block_col]].replace([np.inf, -np.inf], np.nan).dropna()
    high = clean.loc[clean[group_col] == high_label, value_col]
    low = clean.loc[clean[group_col] == low_label, value_col]
    if len(high) == 0 or len(low) == 0:
        return {"difference": np.nan, "ci_low": np.nan, "ci_high": np.nan, "high_mean": np.nan, "low_mean": np.nan}
    observed = float(high.mean() - low.mean())
    block_stats = clean.pivot_table(index=block_col, columns=group_col, values=value_col, aggfunc=["sum", "count"], fill_value=0.0)
    blocks = block_stats.index.to_numpy()
    high_sum = block_stats.get(("sum", high_label), pd.Series(0.0, index=block_stats.index)).to_numpy(dtype=float)
    low_sum = block_stats.get(("sum", low_label), pd.Series(0.0, index=block_stats.index)).to_numpy(dtype=float)
    high_count = block_stats.get(("count", high_label), pd.Series(0.0, index=block_stats.index)).to_numpy(dtype=float)
    low_count = block_stats.get(("count", low_label), pd.Series(0.0, index=block_stats.index)).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(blocks), size=len(blocks))
        hs = high_sum[sampled].sum()
        hc = high_count[sampled].sum()
        ls = low_sum[sampled].sum()
        lc = low_count[sampled].sum()
        if hc > 0 and lc > 0:
            draws.append(float(hs / hc - ls / lc))
    if not draws:
        return {"difference": observed, "ci_low": observed, "ci_high": observed, "high_mean": float(high.mean()), "low_mean": float(low.mean())}
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    return {
        "difference": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "high_mean": float(high.mean()),
        "low_mean": float(low.mean()),
    }


def empirical_p_value(real: float, null_values: np.ndarray, two_sided: bool = True) -> float:
    clean = np.asarray(null_values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0 or not np.isfinite(real):
        return np.nan
    if two_sided:
        exceed = np.sum(np.abs(clean) >= abs(real))
    else:
        exceed = np.sum(clean >= real)
    return float((1 + exceed) / (1 + len(clean)))


def robust_location_summary(values: pd.Series) -> dict:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) == 0:
        return {"mean": np.nan, "median": np.nan, "trimmed_mean": np.nan, "winsorized_mean": np.nan, "top_1pct_share": np.nan}
    lower, upper = clean.quantile([0.01, 0.99])
    trimmed = clean[(clean >= lower) & (clean <= upper)]
    winsorized = clean.clip(lower, upper)
    total_abs = clean.abs().sum()
    top_n = max(1, int(np.ceil(0.01 * len(clean))))
    top_share = float(clean.abs().nlargest(top_n).sum() / total_abs) if total_abs else 0.0
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "trimmed_mean": float(trimmed.mean()),
        "winsorized_mean": float(winsorized.mean()),
        "top_1pct_share": top_share,
    }


def effective_nonoverlap_count(timestamps: pd.Series, horizon_seconds: int) -> int:
    count = 0
    last = None
    for timestamp in pd.to_datetime(timestamps, utc=True).sort_values():
        if last is None or (timestamp - last).total_seconds() >= horizon_seconds:
            count += 1
            last = timestamp
    return count


def top_minus_bottom(frame: pd.DataFrame, signal: str, target: str, train_mask: pd.Series, eval_mask: pd.Series) -> dict:
    edges = train_quantile_bins(frame[signal], train_mask)
    bins = apply_bins(frame[signal], edges)
    valid = eval_mask & frame[signal].notna() & frame[target].notna()
    low = frame.loc[valid & (bins <= 2), target]
    high = frame.loc[valid & (bins >= 9), target]
    return {
        "top_minus_bottom": float(high.mean() - low.mean()),
        "rank_correlation": rank_correlation(frame.loc[valid, signal], frame.loc[valid, target]),
        "low_count": int(low.count()),
        "high_count": int(high.count()),
        "count": int(valid.sum()),
    }


def linear_projection(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> dict:
    train_clean = train[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    test_clean = test[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train_clean) < 20 or len(test_clean) < 20:
        return {"r2": np.nan, "mae": np.nan, "rank_correlation": np.nan, "count": len(test_clean)}
    mean = train_clean[features].mean()
    std = train_clean[features].std(ddof=0).replace(0, 1.0)
    x_train = ((train_clean[features] - mean) / std).fillna(0.0).to_numpy()
    x_test = ((test_clean[features] - mean) / std).fillna(0.0).to_numpy()
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_test = np.column_stack([np.ones(len(x_test)), x_test])
    coef = np.linalg.lstsq(x_train, train_clean[target].to_numpy(), rcond=None)[0]
    pred = x_test @ coef
    y = test_clean[target].to_numpy()
    denom = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - pred) ** 2) / denom if denom > 0 else np.nan
    return {
        "r2": float(r2),
        "mae": float(np.mean(np.abs(y - pred))),
        "rank_correlation": rank_correlation(pd.Series(pred), pd.Series(y)),
        "count": int(len(test_clean)),
        **{f"coef_{name}": float(value) for name, value in zip(["intercept"] + features, coef)},
    }


def stratified_absorption_null(episodes: pd.DataFrame, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = episodes.copy()
    strata_columns = [c for c in ["date", "side", "shock_intensity_bin", "pre_depth_bin", "spread_regime", "volatility_regime", "time_of_day_block"] if c in out.columns]
    if not strata_columns:
        strata_columns = ["date", "side", "penetration_class"]
    strata = [out[column] for column in strata_columns]
    group_key = pd.Series(list(zip(*strata)), index=out.index)
    outcome_cols = [
        c
        for c in out.columns
        if c.startswith("future_markout_after_absorption_")
        or c.startswith("future_quote_survives_")
        or c.startswith("total_quote_survives_")
        or c.startswith("episode_markout_from_preshock_")
        or c.startswith("adverse_future_markout_")
        or c.startswith("markout_")
    ]
    for _, idx in out.groupby(group_key).groups.items():
        idx = np.array(list(idx))
        if len(idx) > 1:
            shuffled = rng.permutation(idx)
            out.loc[idx, outcome_cols] = out.loc[shuffled, outcome_cols].to_numpy()
    return out
