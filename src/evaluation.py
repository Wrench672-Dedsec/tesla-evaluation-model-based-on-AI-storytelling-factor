"""
evaluation.py
=============
Model evaluation, comparison visualisation, and reporting utilities.

Outputs
-------
- Console comparison table
- results/figures/model_comparison.png      – bar chart of CV & test metrics
- results/figures/feature_importance.png    – top-20 features per model
- results/figures/lasso_path.png            – LASSO regularisation path
- results/figures/predicted_vs_actual.png   – scatter plot for each model
- results/model_summary.csv                 – machine-readable summary
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # non-interactive backend for headless environments

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FIG_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

_PALETTE = sns.color_palette("tab10")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def summary_table(results: list) -> pd.DataFrame:
    """Build a DataFrame comparing all model results."""
    rows = []
    for r in results:
        rows.append(
            {
                "Model": r.name,
                "CV RMSE": round(r.cv_rmse, 6),
                "CV R²": round(r.cv_r2, 4),
                "Test RMSE": round(r.test_rmse, 6),
                "Test R²": round(r.test_r2, 4),
                "# Features": len(r.selected_features),
            }
        )
    df = pd.DataFrame(rows).set_index("Model")
    return df


def print_summary(results: list) -> None:
    df = summary_table(results)
    print("\n" + "=" * 72)
    print("  Tesla Valuation Model – Performance Comparison")
    print("=" * 72)
    print(df.to_string())
    print("=" * 72)

    best_r2 = df["Test R²"].idxmax()
    best_rmse = df["Test RMSE"].idxmin()
    print(f"\n  Best Test R²   → {best_r2} ({df.loc[best_r2, 'Test R²']:.4f})")
    print(f"  Best Test RMSE → {best_rmse} ({df.loc[best_rmse, 'Test RMSE']:.6f})")
    print()

    csv_path = RESULTS_DIR / "model_summary.csv"
    df.to_csv(csv_path)
    logger.info("Summary saved to %s", csv_path)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_model_comparison(results: list) -> None:
    """Bar chart of CV R², Test R², and Test RMSE for all models."""
    df = summary_table(results).reset_index()
    models = df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Tesla Valuation Model – Performance Comparison", fontsize=14, fontweight="bold")

    # R² panel
    ax = axes[0]
    ax.bar(x - width / 2, df["CV R²"], width, label="CV R²", color=_PALETTE[0], alpha=0.85)
    ax.bar(x + width / 2, df["Test R²"], width, label="Test R²", color=_PALETTE[1], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title("R² Score (higher = better)")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # RMSE panel
    ax = axes[1]
    ax.bar(x - width / 2, df["CV RMSE"], width, label="CV RMSE", color=_PALETTE[2], alpha=0.85)
    ax.bar(x + width / 2, df["Test RMSE"], width, label="Test RMSE", color=_PALETTE[3], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("RMSE (log-return units)")
    ax.set_title("RMSE (lower = better)")
    ax.legend()

    plt.tight_layout()
    _save(fig, "model_comparison.png")


def plot_feature_importance(results: list, top_n: int = 20) -> None:
    """Horizontal bar charts for top-N feature importances per model."""
    models_with_imp = [r for r in results if not r.feature_importance.empty]
    ncols = 2
    nrows = (len(models_with_imp) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 5))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    fig.suptitle("Feature Importance (top 20 per model)", fontsize=14, fontweight="bold")

    for i, r in enumerate(models_with_imp):
        ax = axes_flat[i]
        imp = r.feature_importance.head(top_n)
        sns.barplot(x=imp.values, y=imp.index, ax=ax, palette="viridis", orient="h")
        ax.set_title(r.name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Importance")
        ax.tick_params(labelsize=7)

    for j in range(len(models_with_imp), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    _save(fig, "feature_importance.png")


def plot_lasso_path(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Plot the LASSO regularisation path (coefficient vs log α)."""
    from sklearn.linear_model import lasso_path
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    Xp = scl.fit_transform(imp.fit_transform(X_train.values))
    yp = y_train.values

    try:
        alphas, coefs, _ = lasso_path(Xp, yp, n_alphas=80, max_iter=5000)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not compute LASSO path: %s", exc)
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for coef_path in coefs:
        ax.plot(-np.log10(alphas + 1e-12), coef_path, linewidth=0.8, alpha=0.7)

    ax.set_xlabel("-log10(α)  →  decreasing regularisation")
    ax.set_ylabel("Coefficient value")
    ax.set_title("LASSO Regularisation Path – Tesla Valuation Features")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    _save(fig, "lasso_path.png")


def plot_predicted_vs_actual(
    results: list,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Scatter plot of predicted vs actual forward returns for each model."""
    ncols = 3
    nrows = (len(results) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 5))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    fig.suptitle("Predicted vs Actual Forward Log-Returns", fontsize=14, fontweight="bold")

    for i, r in enumerate(results):
        ax = axes_flat[i]
        cols = r.selected_features
        y_pred = r.pipeline.predict(X_test[cols].values)
        ax.scatter(y_test.values, y_pred, alpha=0.3, s=10, color=_PALETTE[i % len(_PALETTE)])
        lim = max(abs(y_test).max(), abs(y_pred).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1.2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{r.name}\nTest R²={r.test_r2:.4f}", fontsize=9)

    for j in range(len(results), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    _save(fig, "predicted_vs_actual.png")


def plot_nlp_sentiment(nlp_df: pd.DataFrame) -> None:
    """Time-series plot of NLP sentiment factors."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Tesla – NLP / AI Narrative Sentiment Factors", fontsize=13, fontweight="bold")

    channels = [
        ("NLP_news_sentiment", "News Sentiment", _PALETTE[0]),
        ("NLP_earnings_sentiment", "Earnings-Call Sentiment", _PALETTE[1]),
        ("NLP_social_sentiment", "Social Sentiment", _PALETTE[2]),
    ]
    for ax, (col, label, color) in zip(axes, channels):
        if col not in nlp_df.columns:
            continue
        ax.plot(nlp_df.index, nlp_df[col], color=color, linewidth=0.8, alpha=0.9, label=label)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.fill_between(
            nlp_df.index,
            nlp_df[col],
            0,
            where=nlp_df[col] >= 0,
            alpha=0.2,
            color="green",
        )
        ax.fill_between(
            nlp_df.index,
            nlp_df[col],
            0,
            where=nlp_df[col] < 0,
            alpha=0.2,
            color="red",
        )
        ax.set_ylabel("Score [-1, 1]")
        ax.legend(loc="upper left")

    if "NLP_combined_sentiment" in nlp_df.columns:
        axes[-1].plot(
            nlp_df.index,
            nlp_df["NLP_combined_sentiment"],
            color="purple",
            linewidth=1.2,
            label="Combined (weighted)",
        )
        axes[-1].legend(loc="upper left")

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    _save(fig, "nlp_sentiment.png")


def plot_factor_correlation(feature_df: pd.DataFrame) -> None:
    """Heat map of pairwise factor correlations (sample of 30 features)."""
    num_cols = feature_df.select_dtypes(include=np.number).columns
    sample = list(num_cols[: min(30, len(num_cols))])
    corr = feature_df[sample].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        annot=False,
        linewidths=0.3,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Factor Correlation Heat Map", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    _save(fig, "factor_correlation.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, filename: str) -> None:
    path = FIG_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", path)
