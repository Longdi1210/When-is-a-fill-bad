from __future__ import annotations

import numpy as np
import pandas as pd

from fillbad.multilevel_flow import DEPTH_LEVELS, potential_penetration_class, quote_survives, side_adjusted_markout_bps
from fillbad.real_validation import apply_bins, rank_correlation, train_quantile_bins


SHOCK_WINDOWS_SECONDS = [1, 2, 5, 10, 30]
RESPONSE_TIMES_SECONDS = [1, 2, 5, 10, 30, 60, 120, 300]


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
    strata = [out["date"], out["side"], out["penetration_class"]]
    group_key = pd.Series(list(zip(*strata)), index=out.index)
    for _, idx in out.groupby(group_key).groups.items():
        idx = np.array(list(idx))
        if len(idx) > 1:
            shuffled = rng.permutation(idx)
            for col in [c for c in out.columns if c.startswith("markout_")]:
                out.loc[idx, col] = out.loc[shuffled, col].to_numpy()
    return out
