from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
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
]


def chronological_split(df: pd.DataFrame, train_fraction: float, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["step", "side"]).reset_index(drop=True)
    steps = sorted(ordered["step"].unique())
    train_step_end = steps[int(len(steps) * train_fraction) - 1]
    validation_step_end = steps[int(len(steps) * (train_fraction + validation_fraction)) - 1]
    train = ordered[ordered["step"] <= train_step_end].copy()
    validation = ordered[(ordered["step"] > train_step_end) & (ordered["step"] <= validation_step_end)].copy()
    test = ordered[ordered["step"] > validation_step_end].copy()
    return train, validation, test


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )


def preprocessor_for(feature_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                feature_columns,
            )
        ],
        remainder="drop",
    )


def fit_fill_model_with_features(train: pd.DataFrame, feature_columns: list[str]) -> Pipeline:
    estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=17)
    return Pipeline([("features", preprocessor_for(feature_columns)), ("model", estimator)]).fit(train[feature_columns], train["filled"])


def fit_markout_model_with_features(train_filled: pd.DataFrame, target: str, feature_columns: list[str]) -> Pipeline:
    clean = train_filled.dropna(subset=[target])
    estimator = Ridge(alpha=1.0)
    return Pipeline([("features", preprocessor_for(feature_columns)), ("model", estimator)]).fit(clean[feature_columns], clean[target])


def predict_fill_with_features(model: Pipeline, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return model.predict_proba(df[feature_columns])[:, 1]


def predict_markout_with_features(model: Pipeline, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return model.predict(df[feature_columns])


def fit_fill_model(train: pd.DataFrame, model_name: str = "logistic") -> Pipeline:
    if model_name == "gradient_boosting":
        estimator = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, random_state=17)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=17)
    return Pipeline([("features", _preprocessor()), ("model", estimator)]).fit(train[FEATURE_COLUMNS], train["filled"])


def fit_markout_model(train_filled: pd.DataFrame, target: str, model_name: str = "ridge") -> Pipeline:
    clean = train_filled.dropna(subset=[target])
    if model_name == "gradient_boosting":
        estimator = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, random_state=19)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)]).fit(clean[FEATURE_COLUMNS], clean[target])
    estimator = Ridge(alpha=1.0)
    return Pipeline([("features", _preprocessor()), ("model", estimator)]).fit(clean[FEATURE_COLUMNS], clean[target])


def predict_fill(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    if hasattr(model[-1], "predict_proba"):
        return model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    return model.predict(df[FEATURE_COLUMNS])


def predict_markout(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return model.predict(df[FEATURE_COLUMNS])


def fill_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "roc_auc": roc_auc_score(y_true, y_pred) if y_true.nunique() > 1 else np.nan,
        "brier_score": brier_score_loss(y_true, y_pred),
        "log_loss": log_loss(y_true, np.clip(y_pred, 1e-6, 1 - 1e-6), labels=[0, 1]),
        "fill_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }


def markout_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    clean = pd.DataFrame({"y": y_true, "pred": y_pred}).dropna()
    if clean.empty:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "mean_signed_error": np.nan,
            "rank_correlation": np.nan,
            "n": 0,
        }
    corr = clean["y"].corr(clean["pred"], method="spearman") if len(clean) > 2 else np.nan
    return {
        "mae": mean_absolute_error(clean["y"], clean["pred"]),
        "rmse": mean_squared_error(clean["y"], clean["pred"], squared=False),
        "mean_signed_error": float((clean["pred"] - clean["y"]).mean()),
        "rank_correlation": corr,
        "n": int(len(clean)),
    }
