# Tesla Valuation Model Based on AI Storytelling Factor

A machine-learning valuation framework for **Tesla (TSLA)** stock that combines
four factor groups and compares six models to find the optimal predictor of
21-day forward log-returns.

---

## Factor Groups

| Group | Examples |
|---|---|
| **Fundamental** | P/E, P/S, EV/EBITDA, Gross Margin, ROE, Debt-to-Equity, Revenue growth YoY |
| **Technical** | MA (5/20/60), RSI, MACD, Bollinger Bands, ATR, Historical Volatility, Momentum |
| **Company-level** | Rolling Beta (63/252 d), Market Cap, Correlation with market, Short-interest proxy |
| **NLP / AI Narrative** | News sentiment (TextBlob), Earnings-call tone, Social sentiment, Sentiment momentum |

---

## Models

1. **LASSO** (LassoCV) – variable screening + linear baseline  
2. **Ridge** – regularised linear model on LASSO-selected features  
3. **Random Forest** – 300-tree ensemble on top-20 LASSO features  
4. **Bagging** – BaggingRegressor wrapping ExtraTreesRegressor  
5. **XGBoost / GBM** – gradient boosting (XGBoost if installed, sklearn GBM fallback)  
6. **Neural Network** – 3-layer MLP (128 → 64 → 32, ReLU, Adam, early stopping)  

---

## Project Layout

```
.
├── main.py                        # End-to-end runner (CLI)
├── requirements.txt
├── src/
│   ├── data_collection.py         # yfinance download + synthetic fallback
│   ├── feature_engineering.py     # Technical / fundamental / company-level factors
│   ├── nlp_factors.py             # NLP sentiment construction
│   ├── models.py                  # All six model implementations
│   └── evaluation.py              # Metrics, plots, summary CSV
├── notebooks/
│   └── tesla_valuation_model.ipynb
├── tests/
│   └── test_pipeline.py           # 31 pytest unit/integration tests
├── data/
│   ├── raw/                       # Cached price & fundamental parquet files
│   └── processed/                 # Cached NLP factor parquet files
└── results/
    ├── model_summary.csv          # CV & test metrics for all models
    └── figures/                   # 6 publication-quality PNG charts
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (uses synthetic data when offline)
python main.py

# 3. Or specify your own date range / horizon
python main.py --start 2019-01-01 --end 2024-12-31 --horizon 21

# 4. Run unit tests
pytest tests/test_pipeline.py -v
```

---

## Generated Outputs

| File | Description |
|---|---|
| `results/model_summary.csv` | CV RMSE, CV R², Test RMSE, Test R² for all models |
| `results/figures/nlp_sentiment.png` | News / earnings / social sentiment time-series |
| `results/figures/factor_correlation.png` | Pairwise feature correlation heat-map |
| `results/figures/lasso_path.png` | LASSO regularisation path |
| `results/figures/model_comparison.png` | Bar chart comparing all six models |
| `results/figures/feature_importance.png` | Top-20 feature importances per model |
| `results/figures/predicted_vs_actual.png` | Scatter plots of predicted vs actual returns |

---

## Data Sources

- **Price & financials**: Yahoo Finance via `yfinance` (auto-fallback to synthetic data when offline)  
- **NLP sentiment**: TextBlob on live yfinance news headlines; hand-crafted earnings-call  
  tone trajectory; synthetic social media proxy correlated with news  

> The synthetic data generator simulates TSLA-calibrated GBM price dynamics  
> (μ ≈ 35 % annualised, σ ≈ 60 %) and realistic quarterly fundamental trajectories  
> so the entire pipeline runs without internet access.
