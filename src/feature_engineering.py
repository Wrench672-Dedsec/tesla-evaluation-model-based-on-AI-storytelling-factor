"""
feature_engineering.py
=======================
Build fundamental, technical, and company-level factor columns from raw
price and financial data.

Factor groups
-------------
Fundamental  : valuation ratios, profitability, solvency
Technical    : moving averages, momentum, volatility, oscillators
Company-level: market cap, beta, capital structure
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Technical factors
# ---------------------------------------------------------------------------


def add_technical_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append technical indicators to a daily OHLCV DataFrame.

    Parameters
    ----------
    df : DataFrame with columns Open, High, Low, Close, Volume and a
         DatetimeIndex.

    Returns
    -------
    DataFrame with additional technical factor columns.
    """
    out = df.copy()
    close = out["Close"]
    volume = out["Volume"]
    high = out["High"]
    low = out["Low"]

    # Moving averages
    for w in (5, 10, 20, 60, 120):
        out[f"MA_{w}"] = close.rolling(w).mean()
        out[f"MA_ratio_{w}"] = close / out[f"MA_{w}"] - 1

    # Exponential moving averages (MACD components)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    # RSI
    out["RSI_14"] = _rsi(close, 14)
    out["RSI_28"] = _rsi(close, 28)

    # Bollinger Bands
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["BB_upper"] = ma20 + 2 * std20
    out["BB_lower"] = ma20 - 2 * std20
    out["BB_pct"] = (close - out["BB_lower"]) / (out["BB_upper"] - out["BB_lower"] + 1e-9)
    out["BB_width"] = (out["BB_upper"] - out["BB_lower"]) / (ma20 + 1e-9)

    # Momentum / rate of change
    for lag in (5, 10, 20, 60):
        out[f"Momentum_{lag}"] = close.pct_change(lag)

    # Historical volatility (annualised)
    log_ret = np.log(close / close.shift(1))
    for w in (10, 21, 63):
        out[f"HV_{w}"] = log_ret.rolling(w).std() * np.sqrt(252)

    # Volume features
    out["Volume_MA20"] = volume.rolling(20).mean()
    out["Volume_ratio"] = volume / (out["Volume_MA20"] + 1e-9)
    out["Volume_change"] = volume.pct_change(1)

    # Average True Range (ATR)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR_14"] = tr.rolling(14).mean()
    out["ATR_ratio"] = out["ATR_14"] / (close + 1e-9)

    # 52-week high/low ratio
    out["Price_to_52w_high"] = close / close.rolling(252).max()
    out["Price_to_52w_low"] = close / (close.rolling(252).min() + 1e-9)

    # Log returns (target-adjacent feature for models that need it)
    out["Log_return_1d"] = log_ret
    out["Log_return_5d"] = np.log(close / close.shift(5))

    return out


# ---------------------------------------------------------------------------
# Fundamental factors  (quarterly → daily via forward-fill)
# ---------------------------------------------------------------------------


def add_fundamental_factors(
    price_df: pd.DataFrame,
    fundamental_df: pd.DataFrame,
    shares_outstanding: float = 3.2e9,
) -> pd.DataFrame:
    """Merge quarterly fundamentals into the daily price frame.

    Parameters
    ----------
    price_df : Daily OHLCV (with technical factors already added).
    fundamental_df : Quarterly financial statements.
    shares_outstanding : Approximate shares outstanding (default TSLA ~3.2 B).
    """
    # Quarterly metrics → daily (forward-fill)
    q = fundamental_df.copy()
    q.index = pd.to_datetime(q.index)
    q = q.resample("B").last().ffill()

    out = price_df.copy()
    close = out["Close"]
    market_cap = close * shares_outstanding

    # Valuation ratios
    out["Market_Cap"] = market_cap
    _safe_merge(out, q, "Total Revenue", "Revenue_TTM", window_sum=4)
    _safe_merge(out, q, "Gross Profit", "GrossProfit_TTM", window_sum=4)
    _safe_merge(out, q, "EBITDA", "EBITDA_TTM", window_sum=4)
    _safe_merge(out, q, "Net Income", "NetIncome_TTM", window_sum=4)
    _safe_merge(out, q, "Total Assets", "Total_Assets")
    _safe_merge(out, q, "Total Debt", "Total_Debt")
    _safe_merge(out, q, "Stockholders Equity", "Equity")
    _safe_merge(out, q, "Cash And Cash Equivalents", "Cash")
    _safe_merge(out, q, "Free Cash Flow", "FCF_TTM", window_sum=4)

    # Derived ratios
    eps_ttm = out["NetIncome_TTM"] / shares_outstanding
    out["PE_ratio"] = close / eps_ttm.where(eps_ttm > 0)
    out["PS_ratio"] = market_cap / out["Revenue_TTM"].where(out["Revenue_TTM"] > 0)
    out["PB_ratio"] = market_cap / out["Equity"].where(out["Equity"] > 0)

    ev = market_cap + out["Total_Debt"] - out["Cash"]
    out["EV"] = ev
    out["EV_EBITDA"] = ev / out["EBITDA_TTM"].where(out["EBITDA_TTM"] > 0)
    out["EV_Revenue"] = ev / out["Revenue_TTM"].where(out["Revenue_TTM"] > 0)
    out["EV_FCF"] = ev / out["FCF_TTM"].where(out["FCF_TTM"] > 0)

    # Profitability
    out["Gross_Margin"] = out["GrossProfit_TTM"] / out["Revenue_TTM"].where(out["Revenue_TTM"] > 0)
    out["EBITDA_Margin"] = out["EBITDA_TTM"] / out["Revenue_TTM"].where(out["Revenue_TTM"] > 0)
    out["Net_Margin"] = out["NetIncome_TTM"] / out["Revenue_TTM"].where(out["Revenue_TTM"] > 0)
    out["ROE"] = out["NetIncome_TTM"] / out["Equity"].where(out["Equity"] > 0)
    out["ROA"] = out["NetIncome_TTM"] / out["Total_Assets"].where(out["Total_Assets"] > 0)

    # Revenue growth YoY (requires 4-quarter lag)
    rev = q.get("Total Revenue", pd.Series(dtype=float))
    if not rev.empty:
        rev_aligned = rev.reindex(price_df.index, method="ffill")
        rev_lag = rev_aligned.shift(252)
        out["Revenue_growth_YoY"] = (rev_aligned - rev_lag) / (rev_lag.abs() + 1e-9)

    # Solvency
    out["Debt_to_Equity"] = out["Total_Debt"] / out["Equity"].where(out["Equity"] > 0)
    out["Debt_to_Assets"] = out["Total_Debt"] / out["Total_Assets"].where(out["Total_Assets"] > 0)
    out["Cash_to_Debt"] = out["Cash"] / (out["Total_Debt"] + 1e-9)

    return out


# ---------------------------------------------------------------------------
# Company-level factors
# ---------------------------------------------------------------------------


def add_company_factors(
    df: pd.DataFrame,
    market_index_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add company-level factors (beta, short interest proxy, etc.).

    Parameters
    ----------
    df : Daily frame with Close and (optionally) fundamental columns already
         added.
    market_index_df : DataFrame with a 'Close' column representing a broad
                      market index. If None, synthetic S&P 500 returns are
                      computed from the Tesla data (correlation proxy only).
    """
    out = df.copy()
    log_ret = np.log(out["Close"] / out["Close"].shift(1))

    # Beta (rolling 252-day regression against market)
    if market_index_df is not None and not market_index_df.empty:
        mkt = np.log(market_index_df["Close"] / market_index_df["Close"].shift(1))
        mkt = mkt.reindex(out.index, method="ffill")
    else:
        # Synthetic market: use a smooth version of TSLA returns as proxy
        rng = np.random.default_rng(0)
        noise = pd.Series(
            rng.normal(0, 0.008, len(out)), index=out.index, name="mkt_ret"
        )
        mkt = log_ret * 0.4 + noise  # lower beta than raw TSLA

    out["Beta_252"] = _rolling_beta(log_ret, mkt, 252)
    out["Beta_63"] = _rolling_beta(log_ret, mkt, 63)

    # Correlation with market
    out["Corr_market_63"] = log_ret.rolling(63).corr(mkt)

    # Short-interest proxy: high vol / momentum divergence (no live data)
    out["Short_interest_proxy"] = (
        out.get("HV_21", log_ret.rolling(21).std() * np.sqrt(252))
        / (out["Close"] / out["Close"].rolling(63).max() + 1e-9)
    )

    # Earnings surprise proxy (distance of close from 20-day pre-earnings avg)
    out["Price_jump_proxy"] = (out["Close"] / out["Close"].rolling(5).mean() - 1).abs()

    # Log market cap (already in fundamental factors; re-derive if missing)
    if "Market_Cap" not in out.columns:
        out["Market_Cap"] = out["Close"] * 3.2e9
    out["Log_Market_Cap"] = np.log(out["Market_Cap"] + 1e-9)

    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / (var + 1e-9)


def _safe_merge(
    out: pd.DataFrame,
    q: pd.DataFrame,
    col: str,
    alias: str,
    window_sum: Optional[int] = None,
) -> None:
    if col not in q.columns:
        out[alias] = np.nan
        return
    series = q[col].reindex(out.index, method="ffill")
    if window_sum:
        # Rolling sum to approximate TTM
        series = q[col].rolling(window_sum).sum().reindex(out.index, method="ffill")
    out[alias] = series


def build_feature_matrix(
    price_df: pd.DataFrame,
    fundamental_df: pd.DataFrame,
    nlp_df: Optional[pd.DataFrame] = None,
    target_horizon: int = 21,
) -> pd.DataFrame:
    """Combine all factor groups into a single modelling DataFrame.

    Parameters
    ----------
    price_df : Raw daily OHLCV.
    fundamental_df : Quarterly fundamentals.
    nlp_df : DataFrame with NLP sentiment columns aligned to trading dates.
    target_horizon : Forward return horizon in trading days (default 21 ≈ 1 month).

    Returns
    -------
    DataFrame with all features and a 'Target' column (forward log-return).
    """
    df = add_technical_factors(price_df)
    df = add_fundamental_factors(df, fundamental_df)
    df = add_company_factors(df)

    # Merge NLP factors
    if nlp_df is not None and not nlp_df.empty:
        nlp_aligned = nlp_df.reindex(df.index, method="ffill")
        df = pd.concat([df, nlp_aligned], axis=1)

    # Target: forward log-return over horizon
    df["Target"] = np.log(df["Close"].shift(-target_horizon) / df["Close"])

    # Drop raw OHLCV columns that would leak the target
    df = df.drop(columns=["Open", "High", "Low", "Close", "Volume"], errors="ignore")

    # Remove rows with no target (last `horizon` rows)
    df = df.dropna(subset=["Target"])

    logger.info("Feature matrix: %d rows × %d columns", *df.shape)
    return df
