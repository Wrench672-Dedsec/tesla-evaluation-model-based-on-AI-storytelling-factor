"""
main.py
=======
End-to-end runner for the Tesla AI Storytelling Valuation Model.

Steps
-----
1. Fetch / load Tesla price and financial data
2. Build NLP narrative sentiment factors
3. Engineer fundamental, technical, and company-level factors
4. Construct feature matrix and split train / test
5. Screen variables with LASSO and build baseline model
6. Fit Random Forest, Bagging, Boosting, and Neural Network models
7. Evaluate and compare all models
8. Save figures and summary CSV to results/

Usage
-----
    python main.py [--start START] [--end END] [--horizon HORIZON]
                   [--no-cache] [--log-level {DEBUG,INFO,WARNING}]
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# Make the src package importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_collection import fetch_financial_data, fetch_price_data
from src.evaluation import (
    plot_factor_correlation,
    plot_feature_importance,
    plot_lasso_path,
    plot_model_comparison,
    plot_nlp_sentiment,
    plot_predicted_vs_actual,
    print_summary,
)
from src.feature_engineering import build_feature_matrix
from src.models import fit_all_models
from src.nlp_factors import build_nlp_factors

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tesla AI Storytelling Valuation Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", default="2018-01-01", help="Data start date (YYYY-MM-DD)")
    p.add_argument("--end", default="2024-12-31", help="Data end date (YYYY-MM-DD)")
    p.add_argument("--horizon", type=int, default=21, help="Forward return horizon (trading days)")
    p.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of data used for test")
    p.add_argument("--ticker", default="TSLA", help="Stock ticker symbol")
    p.add_argument("--no-cache", action="store_true", help="Ignore and overwrite data cache")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    logger = logging.getLogger("main")
    cache = not args.no_cache

    # ------------------------------------------------------------------
    # 1. Data collection
    # ------------------------------------------------------------------
    logger.info("=== Step 1: Data Collection ===")
    price_df = fetch_price_data(
        ticker=args.ticker, start=args.start, end=args.end, cache=cache
    )
    fundamental_df = fetch_financial_data(ticker=args.ticker, cache=cache)
    logger.info(
        "Price data: %d rows (%s → %s)",
        len(price_df),
        price_df.index.min().date(),
        price_df.index.max().date(),
    )
    logger.info("Fundamental data: %d quarters", len(fundamental_df))

    # ------------------------------------------------------------------
    # 2. NLP narrative factors
    # ------------------------------------------------------------------
    logger.info("=== Step 2: NLP Narrative Factors ===")
    nlp_df = build_nlp_factors(price_df, ticker=args.ticker, cache=cache)
    plot_nlp_sentiment(nlp_df)

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    logger.info("=== Step 3: Feature Engineering ===")
    feature_df = build_feature_matrix(
        price_df=price_df,
        fundamental_df=fundamental_df,
        nlp_df=nlp_df,
        target_horizon=args.horizon,
    )
    logger.info("Feature matrix shape: %s", feature_df.shape)

    # Correlation heat-map (before removing columns)
    plot_factor_correlation(feature_df)

    # ------------------------------------------------------------------
    # 4. Train / test split (temporal – no shuffle)
    # ------------------------------------------------------------------
    logger.info("=== Step 4: Train / Test Split ===")
    # Drop rows with too many NaN (e.g. first 252 days with rolling windows)
    feature_df.dropna(thresh=int(feature_df.shape[1] * 0.5), inplace=True)

    target_col = "Target"
    X = feature_df.drop(columns=[target_col])
    y = feature_df[target_col]

    split_idx = int(len(X) * (1 - args.test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(
        "Train: %d rows (%s → %s)",
        len(X_train),
        X_train.index.min().date(),
        X_train.index.max().date(),
    )
    logger.info(
        "Test : %d rows (%s → %s)",
        len(X_test),
        X_test.index.min().date(),
        X_test.index.max().date(),
    )

    # ------------------------------------------------------------------
    # 5–6. LASSO screening + all models
    # ------------------------------------------------------------------
    logger.info("=== Step 5-6: Model Training ===")
    results = fit_all_models(X_train, y_train, X_test, y_test)

    # LASSO path visualisation (use full column set)
    plot_lasso_path(X_train, y_train)

    # ------------------------------------------------------------------
    # 7. Evaluation
    # ------------------------------------------------------------------
    logger.info("=== Step 7: Model Evaluation ===")
    print_summary(results)
    plot_model_comparison(results)
    plot_feature_importance(results)
    plot_predicted_vs_actual(results, X_test, y_test)

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    from src.evaluation import summary_table

    tbl = summary_table(results)
    best = tbl["Test R²"].idxmax()
    print(f"\n✅  Optimal model: {best}  (Test R² = {tbl.loc[best, 'Test R²']:.4f})")
    print(f"   All figures saved to: results/figures/")
    print(f"   Summary CSV saved to: results/model_summary.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    args = parse_args()
    _setup_logging(args.log_level)
    run(args)
