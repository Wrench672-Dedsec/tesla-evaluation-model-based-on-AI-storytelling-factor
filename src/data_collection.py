"""
data_collection.py
==================
Fetch Tesla price history and financial data.

Primary source: yfinance (TSLA)
Fallback:       Synthetic data generator that replicates realistic
                Tesla price and financial dynamics when the network
                is unavailable.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_price_data(
    ticker: str = "TSLA",
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    cache: bool = True,
) -> pd.DataFrame:
    """Return daily OHLCV price data for *ticker*.

    Tries yfinance first; falls back to synthetic data if the network is
    unavailable or returns an empty frame.
    """
    cache_file = RAW_DIR / f"{ticker}_price.parquet"
    if cache and cache_file.exists():
        logger.info("Loading price data from cache: %s", cache_file)
        return pd.read_parquet(cache_file)

    df = _fetch_yfinance(ticker, start, end)
    if df is None or df.empty:
        logger.warning("yfinance unavailable – using synthetic price data.")
        df = _synthetic_price_data(start, end)

    if cache:
        df.to_parquet(cache_file)
    return df


def fetch_financial_data(
    ticker: str = "TSLA",
    cache: bool = True,
) -> pd.DataFrame:
    """Return quarterly fundamental data for *ticker*.

    Falls back to synthetic fundamental data when offline.
    """
    cache_file = RAW_DIR / f"{ticker}_fundamentals.parquet"
    if cache and cache_file.exists():
        logger.info("Loading fundamental data from cache: %s", cache_file)
        return pd.read_parquet(cache_file)

    df = _fetch_yfinance_fundamentals(ticker)
    if df is None or df.empty:
        logger.warning("yfinance fundamentals unavailable – using synthetic data.")
        df = _synthetic_fundamental_data()

    if cache:
        df.to_parquet(cache_file)
    return df


# ---------------------------------------------------------------------------
# Internal helpers – yfinance
# ---------------------------------------------------------------------------


def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf

        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if raw.empty:
            return None

        # Flatten multi-level columns that yfinance ≥ 0.2 may produce
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.index = pd.to_datetime(raw.index)
        raw.index.name = "Date"
        return raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    except Exception as exc:  # noqa: BLE001
        logger.debug("yfinance download failed: %s", exc)
        return None


def _fetch_yfinance_fundamentals(ticker: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)

        income = stock.quarterly_financials
        balance = stock.quarterly_balance_sheet
        cashflow = stock.quarterly_cashflow

        if income is None or income.empty:
            return None

        # Transpose so dates are rows
        income = income.T
        balance = balance.T if (balance is not None and not balance.empty) else pd.DataFrame()
        cashflow = cashflow.T if (cashflow is not None and not cashflow.empty) else pd.DataFrame()

        df = income.join(balance, how="outer", rsuffix="_bal")
        df = df.join(cashflow, how="outer", rsuffix="_cf")
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        df.sort_index(inplace=True)
        return df
    except Exception as exc:  # noqa: BLE001
        logger.debug("yfinance fundamentals failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Internal helpers – synthetic data generation
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _synthetic_price_data(start: str, end: str) -> pd.DataFrame:
    """Simulate realistic Tesla daily OHLCV data with trend, cycles, and
    volatility clustering.
    """
    dates = pd.bdate_range(start=start, end=end, freq="B")
    n = len(dates)

    # Geometric Brownian Motion parameters (calibrated to TSLA history)
    mu = 0.35 / 252       # annualised drift ≈ 35 %
    sigma = 0.60 / np.sqrt(252)  # annualised vol ≈ 60 %
    S0 = 20.0             # starting price (split-adjusted)

    # GARCH-like volatility clustering
    vol = np.ones(n) * sigma
    for t in range(1, n):
        shock = _RNG.standard_normal()
        vol[t] = np.sqrt(0.9 * vol[t - 1] ** 2 + 0.1 * (shock * sigma) ** 2)

    returns = mu + vol * _RNG.standard_normal(n)
    prices = S0 * np.exp(np.cumsum(returns))

    # Add plausible intraday spread
    high = prices * (1 + np.abs(_RNG.normal(0, 0.02, n)))
    low = prices * (1 - np.abs(_RNG.normal(0, 0.02, n)))
    open_ = low + _RNG.uniform(0, 1, n) * (high - low)
    volume = (_RNG.lognormal(mean=14.5, sigma=0.6, size=n)).astype(int)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": prices,
            "Volume": volume,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def _synthetic_fundamental_data() -> pd.DataFrame:
    """Simulate quarterly Tesla fundamentals (2018–2024)."""
    dates = pd.date_range(start="2018-03-31", end="2024-12-31", freq="QE")
    n = len(dates)

    # Revenue grows from ~3 B to ~25 B quarterly (realistic TSLA arc)
    rev_growth = np.linspace(3e9, 25e9, n) * (1 + _RNG.normal(0, 0.05, n))
    gross_margin = np.clip(np.linspace(0.12, 0.22, n) + _RNG.normal(0, 0.02, n), 0.05, 0.35)
    ebitda_margin = gross_margin - 0.08 + _RNG.normal(0, 0.01, n)
    net_income = rev_growth * np.clip(ebitda_margin - 0.04, -0.10, 0.20)

    # Balance sheet items
    total_assets = rev_growth * np.linspace(3.0, 5.0, n) * (1 + _RNG.normal(0, 0.03, n))
    total_debt = total_assets * np.linspace(0.45, 0.25, n)
    equity = total_assets - total_debt
    cash = equity * np.linspace(0.30, 0.45, n) * (1 + _RNG.normal(0, 0.04, n))

    capex = -rev_growth * np.linspace(0.25, 0.10, n) * (1 + _RNG.normal(0, 0.05, n))
    fcf = net_income - capex * 0.8

    return pd.DataFrame(
        {
            "Total Revenue": rev_growth,
            "Gross Profit": rev_growth * gross_margin,
            "EBITDA": rev_growth * ebitda_margin,
            "Net Income": net_income,
            "Total Assets": total_assets,
            "Total Debt": total_debt,
            "Stockholders Equity": equity,
            "Cash And Cash Equivalents": cash,
            "Capital Expenditure": capex,
            "Free Cash Flow": fcf,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )
