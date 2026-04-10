"""
tests/test_pipeline.py
======================
Unit and integration tests for the Tesla valuation model pipeline.

Tests cover:
- Data collection (synthetic path)
- Feature engineering (all factor groups)
- NLP factor construction
- Model fitting (LASSO, Random Forest, Boosting, Neural Network)
- Evaluation utilities
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_collection import fetch_financial_data, fetch_price_data
from src.feature_engineering import (
    add_company_factors,
    add_fundamental_factors,
    add_technical_factors,
    build_feature_matrix,
)
from src.models import (
    fit_bagging,
    fit_boosting,
    fit_lasso,
    fit_neural_network,
    fit_random_forest,
    fit_ridge,
)
from src.nlp_factors import build_nlp_factors, score_text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def price_df():
    """Small synthetic price frame (3 years of daily data)."""
    return fetch_price_data(start="2021-01-01", end="2023-12-31", cache=False)


@pytest.fixture(scope="session")
def fundamental_df():
    return fetch_financial_data(cache=False)


@pytest.fixture(scope="session")
def nlp_df(price_df):
    return build_nlp_factors(price_df, cache=False)


@pytest.fixture(scope="session")
def feature_df(price_df, fundamental_df, nlp_df):
    return build_feature_matrix(price_df, fundamental_df, nlp_df, target_horizon=21)


@pytest.fixture(scope="session")
def split_data(feature_df):
    df = feature_df.dropna(thresh=int(feature_df.shape[1] * 0.5))
    X = df.drop(columns=["Target"])
    y = df["Target"]
    split = int(len(X) * 0.8)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# ---------------------------------------------------------------------------
# Data collection tests
# ---------------------------------------------------------------------------


class TestDataCollection:
    def test_price_df_shape(self, price_df):
        assert not price_df.empty
        assert {"Open", "High", "Low", "Close", "Volume"} <= set(price_df.columns)
        # At least 500 trading days in a 3-year window
        assert len(price_df) >= 500

    def test_price_df_positive(self, price_df):
        assert (price_df["Close"] > 0).all()
        assert (price_df["Volume"] > 0).all()

    def test_price_df_index(self, price_df):
        assert isinstance(price_df.index, pd.DatetimeIndex)

    def test_fundamental_df_shape(self, fundamental_df):
        assert not fundamental_df.empty
        assert len(fundamental_df) >= 4  # at least 4 quarters

    def test_fundamental_columns(self, fundamental_df):
        assert "Total Revenue" in fundamental_df.columns


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------


class TestTechnicalFactors:
    def test_columns_added(self, price_df):
        out = add_technical_factors(price_df)
        expected = ["MA_5", "MA_20", "RSI_14", "MACD", "BB_pct", "HV_21", "ATR_14"]
        for col in expected:
            assert col in out.columns, f"Missing column: {col}"

    def test_rsi_bounds(self, price_df):
        out = add_technical_factors(price_df)
        rsi = out["RSI_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_no_inf(self, price_df):
        out = add_technical_factors(price_df)
        numeric = out.select_dtypes(include=np.number)
        assert not np.isinf(numeric.values).any()


class TestFundamentalFactors:
    def test_fundamental_columns_added(self, price_df, fundamental_df):
        tech = add_technical_factors(price_df)
        out = add_fundamental_factors(tech, fundamental_df)
        expected = ["PS_ratio", "PB_ratio", "EV_EBITDA", "Gross_Margin", "Debt_to_Equity"]
        for col in expected:
            assert col in out.columns, f"Missing column: {col}"

    def test_market_cap_positive(self, price_df, fundamental_df):
        tech = add_technical_factors(price_df)
        out = add_fundamental_factors(tech, fundamental_df)
        assert (out["Market_Cap"].dropna() > 0).all()


class TestCompanyFactors:
    def test_beta_computed(self, price_df, fundamental_df):
        tech = add_technical_factors(price_df)
        fund = add_fundamental_factors(tech, fundamental_df)
        out = add_company_factors(fund)
        assert "Beta_252" in out.columns
        assert "Beta_63" in out.columns

    def test_log_market_cap(self, price_df, fundamental_df):
        tech = add_technical_factors(price_df)
        fund = add_fundamental_factors(tech, fundamental_df)
        out = add_company_factors(fund)
        assert "Log_Market_Cap" in out.columns


class TestFeatureMatrix:
    def test_target_column(self, feature_df):
        assert "Target" in feature_df.columns

    def test_no_ohlcv_leak(self, feature_df):
        """Raw OHLCV columns must be removed to prevent target leakage."""
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col not in feature_df.columns

    def test_sufficient_rows(self, feature_df):
        assert len(feature_df) >= 300

    def test_nlp_columns_present(self, feature_df):
        nlp_cols = [c for c in feature_df.columns if c.startswith("NLP_")]
        assert len(nlp_cols) >= 3


# ---------------------------------------------------------------------------
# NLP factor tests
# ---------------------------------------------------------------------------


class TestNLPFactors:
    def test_columns_present(self, nlp_df):
        expected = [
            "NLP_news_sentiment",
            "NLP_earnings_sentiment",
            "NLP_social_sentiment",
            "NLP_combined_sentiment",
            "NLP_sentiment_momentum",
            "NLP_narrative_shift",
        ]
        for col in expected:
            assert col in nlp_df.columns, f"Missing NLP column: {col}"

    def test_score_bounds(self, nlp_df):
        _TOLERANCE = 0.01  # small numeric slack for floating-point rounding
        for col in ["NLP_news_sentiment", "NLP_earnings_sentiment", "NLP_social_sentiment"]:
            values = nlp_df[col].dropna()
            assert (values >= -1 - _TOLERANCE).all() and (values <= 1 + _TOLERANCE).all(), (
                f"{col} out of [-1, 1] range"
            )

    def test_score_text(self):
        score = score_text("Tesla delivers record profits and strong growth!")
        assert -1 <= score <= 1

    def test_score_text_negative(self):
        score = score_text("Tesla misses earnings badly, terrible quarter.")
        assert score < 0.1  # should lean negative

    def test_earnings_sentiment_length(self, nlp_df, price_df):
        assert len(nlp_df) == len(price_df)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestLASSOModel:
    def test_lasso_runs(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        result = fit_lasso(X_tr, y_tr, X_te, y_te)
        assert result.name == "LASSO"
        assert isinstance(result.selected_features, list)
        assert len(result.selected_features) > 0

    def test_lasso_metrics_finite(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        result = fit_lasso(X_tr, y_tr, X_te, y_te)
        assert np.isfinite(result.cv_rmse)
        assert np.isfinite(result.test_rmse)
        assert np.isfinite(result.cv_r2)
        assert np.isfinite(result.test_r2)

    def test_lasso_feature_importance(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        result = fit_lasso(X_tr, y_tr, X_te, y_te)
        assert not result.feature_importance.empty


class TestRidgeModel:
    def test_ridge_runs(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        result = fit_ridge(X_tr, y_tr, X_te, y_te)
        assert result.name == "Ridge"
        assert np.isfinite(result.test_r2)


class TestRandomForestModel:
    def test_rf_runs(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        result = fit_random_forest(X_tr, y_tr, X_te, y_te, n_estimators=50)
        assert result.name == "Random Forest"
        assert np.isfinite(result.test_r2)

    def test_rf_feature_importance(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        result = fit_random_forest(X_tr, y_tr, X_te, y_te, n_estimators=50)
        assert len(result.feature_importance) > 0
        assert abs(result.feature_importance.sum() - 1.0) < 1e-5


class TestBaggingModel:
    def test_bagging_runs(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        result = fit_bagging(X_tr, y_tr, X_te, y_te, n_estimators=20)
        assert result.name == "Bagging"
        assert np.isfinite(result.test_r2)


class TestBoostingModel:
    def test_boosting_runs(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        result = fit_boosting(X_tr, y_tr, X_te, y_te, n_estimators=50)
        assert result.name in ("XGBoost", "GradientBoosting")
        assert np.isfinite(result.test_r2)


class TestNeuralNetworkModel:
    def test_nn_runs(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        result = fit_neural_network(X_tr, y_tr, X_te, y_te)
        assert "Neural Network" in result.name
        assert np.isfinite(result.test_r2)


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_summary_table(self, split_data):
        from src.evaluation import summary_table

        X_tr, X_te, y_tr, y_te = split_data
        results = [fit_lasso(X_tr, y_tr, X_te, y_te)]
        tbl = summary_table(results)
        assert "Test R²" in tbl.columns
        assert "CV RMSE" in tbl.columns
        assert len(tbl) == 1
