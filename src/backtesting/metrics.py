"""Performance metric calculations for backtest results."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_total_return(equity_curve: pd.Series) -> float:
    """Return the cumulative return over the full backtest."""
    if equity_curve.empty:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)


def calculate_sharpe_ratio(daily_returns: pd.Series, annualization_factor: int = 252) -> float:
    """Return the annualized Sharpe ratio using daily returns."""
    cleaned_returns = daily_returns.dropna()
    if cleaned_returns.empty:
        return 0.0

    volatility = cleaned_returns.std(ddof=0)
    if volatility == 0:
        return 0.0

    return float(np.sqrt(annualization_factor) * cleaned_returns.mean() / volatility)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Return the worst drawdown observed in the equity curve."""
    if equity_curve.empty:
        return 0.0

    rolling_peak = equity_curve.cummax()
    drawdown = equity_curve / rolling_peak - 1
    return float(drawdown.min())


def summarize_performance(equity_curve: pd.Series) -> dict[str, float]:
    """Return a compact metrics dictionary for the equity curve."""
    daily_returns = equity_curve.pct_change().fillna(0.0)
    return {
        "total_return": calculate_total_return(equity_curve),
        "daily_return_mean": float(daily_returns.mean()),
        "daily_return_std": float(daily_returns.std(ddof=0)),
        "sharpe_ratio": calculate_sharpe_ratio(daily_returns),
        "max_drawdown": calculate_max_drawdown(equity_curve),
    }
