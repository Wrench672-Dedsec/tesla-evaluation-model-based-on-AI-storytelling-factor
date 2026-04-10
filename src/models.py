"""
models.py
=========
Valuation models for Tesla stock.

Pipeline
--------
1. LASSO regression – variable selection (LassoCV) + baseline predictions.
2. Ridge regression  – regularised linear baseline.
3. Random Forest     – ensemble of decision trees (bagging variant).
4. Bagging           – BaggingRegressor wrapping ExtraTreesRegressor.
5. Gradient Boosting – XGBoost (falls back to sklearn GBM if unavailable).
6. Neural Network    – MLPRegressor (3-layer feed-forward).

Each model is fitted inside a Pipeline that includes:
  - Median imputation (handles NaN in financial ratios)
  - Standard scaling
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    selected_features: list[str]
    cv_rmse: float
    cv_r2: float
    test_rmse: float
    test_r2: float
    feature_importance: pd.Series = field(default_factory=pd.Series)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

N_SPLITS = 5


def _base_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def _cv_scores(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    kf = KFold(n_splits=N_SPLITS, shuffle=False)
    neg_mse = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_squared_error")
    r2 = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")
    rmse = np.sqrt(-neg_mse.mean())
    return float(rmse), float(r2.mean())


def _test_scores(
    pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[float, float]:
    from sklearn.metrics import mean_squared_error, r2_score

    y_pred = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    return rmse, r2


# ---------------------------------------------------------------------------
# 1. LASSO – variable selection + baseline
# ---------------------------------------------------------------------------


def fit_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_alphas: int = 100,
) -> ModelResult:
    """Fit LassoCV, extract selected features, retrain on the selected subset,
    and report CV and test metrics.

    Two-phase approach:
      Phase 1 – run LassoCV on all features to find the optimal α and
                 the non-zero coefficient features.
      Phase 2 – refit a pipeline using only the selected features so that
                 predict() always receives the correct number of columns.
    """
    kf = KFold(n_splits=N_SPLITS, shuffle=False)

    # ---- Phase 1: selection ------------------------------------------------
    lasso_cv = LassoCV(n_alphas=n_alphas, cv=kf, max_iter=10_000, random_state=42)
    selection_pipe = _base_pipeline(lasso_cv)

    logger.info("Fitting LASSO (LassoCV) for feature selection …")
    selection_pipe.fit(X_train.values, y_train.values)

    coef = selection_pipe.named_steps["model"].coef_
    selected = [c for c, v in zip(X_train.columns, coef) if abs(v) > 1e-8]

    # Fall back to top-5 by absolute magnitude if nothing is selected
    if not selected:
        fallback_top_idx = np.argsort(np.abs(coef))[-5:]
        selected = [X_train.columns[i] for i in fallback_top_idx]

    logger.info("LASSO selected %d / %d features.", len(selected), X_train.shape[1])

    # ---- Phase 2: refit on selected features only --------------------------
    best_alpha = selection_pipe.named_steps["model"].alpha_
    from sklearn.linear_model import Lasso

    lasso_final = Lasso(alpha=best_alpha, max_iter=10_000)
    pipe = _base_pipeline(lasso_final)
    pipe.fit(X_train[selected].values, y_train.values)

    cv_rmse, cv_r2 = _cv_scores(pipe, X_train[selected].values, y_train.values)
    test_rmse, test_r2 = _test_scores(pipe, X_test[selected].values, y_test.values)

    # Feature importance: absolute values of the refitted coefficients.
    # Also include all zero-coefficient features ranked last so that
    # fit_all_models can always build an ordered list of all columns.
    coef_final = pipe.named_steps["model"].coef_
    importance_selected = pd.Series(np.abs(coef_final), index=selected)

    # Recover all-feature importance from the selection-phase coefficients
    all_importance = pd.Series(np.abs(coef), index=X_train.columns).sort_values(ascending=False)
    # Overwrite selected entries with refitted values (may differ slightly)
    all_importance.update(importance_selected)
    importance = all_importance.sort_values(ascending=False)

    return ModelResult(
        name="LASSO",
        pipeline=pipe,
        selected_features=selected,
        cv_rmse=cv_rmse,
        cv_r2=cv_r2,
        test_rmse=test_rmse,
        test_r2=test_r2,
        feature_importance=importance,
    )


# ---------------------------------------------------------------------------
# 2. Ridge – regularised linear baseline
# ---------------------------------------------------------------------------


def fit_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_features: list[str] | None = None,
) -> ModelResult:
    cols = selected_features if selected_features else list(X_train.columns)
    pipe = _base_pipeline(Ridge(alpha=1.0))

    logger.info("Fitting Ridge regression …")
    pipe.fit(X_train[cols].values, y_train.values)

    cv_rmse, cv_r2 = _cv_scores(pipe, X_train[cols].values, y_train.values)
    test_rmse, test_r2 = _test_scores(pipe, X_test[cols].values, y_test.values)

    coef = pipe.named_steps["model"].coef_
    importance = pd.Series(np.abs(coef), index=cols).sort_values(ascending=False)

    return ModelResult(
        name="Ridge",
        pipeline=pipe,
        selected_features=cols,
        cv_rmse=cv_rmse,
        cv_r2=cv_r2,
        test_rmse=test_rmse,
        test_r2=test_r2,
        feature_importance=importance,
    )


# ---------------------------------------------------------------------------
# 3. Random Forest
# ---------------------------------------------------------------------------


def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_features: list[str] | None = None,
    n_estimators: int = 300,
) -> ModelResult:
    cols = selected_features if selected_features else list(X_train.columns)
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    pipe = _base_pipeline(rf)

    logger.info("Fitting Random Forest (%d trees) …", n_estimators)
    pipe.fit(X_train[cols].values, y_train.values)

    cv_rmse, cv_r2 = _cv_scores(pipe, X_train[cols].values, y_train.values)
    test_rmse, test_r2 = _test_scores(pipe, X_test[cols].values, y_test.values)

    importance = pd.Series(
        pipe.named_steps["model"].feature_importances_, index=cols
    ).sort_values(ascending=False)

    return ModelResult(
        name="Random Forest",
        pipeline=pipe,
        selected_features=cols,
        cv_rmse=cv_rmse,
        cv_r2=cv_r2,
        test_rmse=test_rmse,
        test_r2=test_r2,
        feature_importance=importance,
    )


# ---------------------------------------------------------------------------
# 4. Bagging (BaggingRegressor with ExtraTreesRegressor as base)
# ---------------------------------------------------------------------------


def fit_bagging(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_features: list[str] | None = None,
    n_estimators: int = 50,
) -> ModelResult:
    cols = selected_features if selected_features else list(X_train.columns)
    base = ExtraTreesRegressor(
        n_estimators=10,
        max_features="sqrt",
        min_samples_leaf=10,
        random_state=0,
    )
    bag = BaggingRegressor(
        estimator=base,
        n_estimators=n_estimators,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )
    pipe = _base_pipeline(bag)

    logger.info("Fitting Bagging (%d estimators) …", n_estimators)
    pipe.fit(X_train[cols].values, y_train.values)

    cv_rmse, cv_r2 = _cv_scores(pipe, X_train[cols].values, y_train.values)
    test_rmse, test_r2 = _test_scores(pipe, X_test[cols].values, y_test.values)

    # Approximate importance using BaggingRegressor.estimators_features_
    # which stores the column indices seen by each base estimator.
    bag_model = pipe.named_steps["model"]
    importances = np.zeros(len(cols))
    for est, feat_idx in zip(bag_model.estimators_, bag_model.estimators_features_):
        imp = est.feature_importances_  # shape: (len(feat_idx),)
        for local_i, global_i in enumerate(feat_idx):
            importances[global_i] += imp[local_i]
    importances /= importances.sum() + 1e-9
    importance = pd.Series(importances, index=cols).sort_values(ascending=False)

    return ModelResult(
        name="Bagging",
        pipeline=pipe,
        selected_features=cols,
        cv_rmse=cv_rmse,
        cv_r2=cv_r2,
        test_rmse=test_rmse,
        test_r2=test_r2,
        feature_importance=importance,
    )


# ---------------------------------------------------------------------------
# 5. Boosting (XGBoost preferred, sklearn GBM fallback)
# ---------------------------------------------------------------------------


def fit_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_features: list[str] | None = None,
    n_estimators: int = 300,
) -> ModelResult:
    cols = selected_features if selected_features else list(X_train.columns)

    try:
        from xgboost import XGBRegressor  # type: ignore[import]

        booster = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
            verbosity=0,
        )
        model_name = "XGBoost"
    except ImportError:
        logger.warning("XGBoost not available – using sklearn GradientBoostingRegressor.")
        booster = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        )
        model_name = "GradientBoosting"

    pipe = _base_pipeline(booster)

    logger.info("Fitting %s (%d estimators) …", model_name, n_estimators)
    pipe.fit(X_train[cols].values, y_train.values)

    cv_rmse, cv_r2 = _cv_scores(pipe, X_train[cols].values, y_train.values)
    test_rmse, test_r2 = _test_scores(pipe, X_test[cols].values, y_test.values)

    importance = pd.Series(
        pipe.named_steps["model"].feature_importances_, index=cols
    ).sort_values(ascending=False)

    return ModelResult(
        name=model_name,
        pipeline=pipe,
        selected_features=cols,
        cv_rmse=cv_rmse,
        cv_r2=cv_r2,
        test_rmse=test_rmse,
        test_r2=test_r2,
        feature_importance=importance,
    )


# ---------------------------------------------------------------------------
# 6. Neural Network (MLP)
# ---------------------------------------------------------------------------


def fit_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_features: list[str] | None = None,
) -> ModelResult:
    cols = selected_features if selected_features else list(X_train.columns)
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
    pipe = _base_pipeline(mlp)

    logger.info("Fitting Neural Network (MLP 128-64-32) …")
    pipe.fit(X_train[cols].values, y_train.values)

    cv_rmse, cv_r2 = _cv_scores(pipe, X_train[cols].values, y_train.values)
    test_rmse, test_r2 = _test_scores(pipe, X_test[cols].values, y_test.values)

    return ModelResult(
        name="Neural Network (MLP)",
        pipeline=pipe,
        selected_features=cols,
        cv_rmse=cv_rmse,
        cv_r2=cv_r2,
        test_rmse=test_rmse,
        test_r2=test_r2,
    )


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def fit_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_features: list[str] | None = None,
    min_ensemble_features: int = 20,
) -> list[ModelResult]:
    """Fit all six models and return a list of ModelResult objects.

    Feature routing
    ---------------
    - LASSO & Ridge use the features with non-zero LASSO coefficients
      (strict linear-model selection).
    - Ensemble models (RF, Bagging, Boosting, MLP) receive the broader
      set of top-*min_ensemble_features* features ranked by LASSO
      coefficient magnitude, which lets them exploit complex interactions
      while still benefiting from LASSO's ranking guidance.
    """
    lasso = fit_lasso(X_train, y_train, X_test, y_test)

    # Strict LASSO selection for linear models
    lasso_strict = lasso.selected_features or list(X_train.columns)

    # Broader selection for ensemble models: top-N by coefficient magnitude
    if selected_features:
        ensemble_feats = selected_features
    else:
        n_take = max(min_ensemble_features, len(lasso_strict))
        n_take = min(n_take, len(X_train.columns))
        ensemble_feats = list(
            lasso.feature_importance.reindex(X_train.columns).sort_values(ascending=False).head(n_take).index
        )
        if not ensemble_feats:
            ensemble_feats = list(X_train.columns[:n_take])

    logger.info(
        "Ensemble feature set: %d features (strict LASSO: %d)",
        len(ensemble_feats),
        len(lasso_strict),
    )

    results = [
        lasso,
        fit_ridge(X_train, y_train, X_test, y_test, lasso_strict),
        fit_random_forest(X_train, y_train, X_test, y_test, ensemble_feats),
        fit_bagging(X_train, y_train, X_test, y_test, ensemble_feats),
        fit_boosting(X_train, y_train, X_test, y_test, ensemble_feats),
        fit_neural_network(X_train, y_train, X_test, y_test, ensemble_feats),
    ]
    return results
