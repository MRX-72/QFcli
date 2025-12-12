"""
Metrics Module
Core financial metrics calculations.
"""

import numpy as np
import pandas as pd
from typing import Union


def cumulative_return(prices: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate cumulative return over the entire period.
    
    Formula: R_cum = (P_end - P_start) / P_start
    
    Args:
        prices: Series or array of closing prices
    
    Returns:
        Cumulative return as a decimal (e.g., 0.25 for 25% gain)
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    p_start = prices[0]
    p_end = prices[-1]
    
    cum_return = (p_end - p_start) / p_start
    return cum_return


def volatility(returns: pd.Series, annualize: bool = True) -> float:
    """
    Calculate volatility (standard deviation of returns).
    
    Formula: σ = sqrt(1/(N-1) * Σ(r_t - r_mean)^2)
    Annualization: σ_ann = σ * sqrt(252)
    
    Args:
        returns: Series of daily returns
        annualize: If True, annualize the volatility (default: True)
    
    Returns:
        Volatility as a decimal
    """
    # Remove NaN values
    clean_returns = returns.dropna()
    
    # Calculate standard deviation
    vol = clean_returns.std()
    
    # Annualize if requested (252 trading days per year)
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualize: bool = True) -> float:
    """
    Calculate Sharpe ratio (risk-adjusted return).
    
    Formula: Sharpe = (r_mean - r_f) / σ
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate as decimal (default: 0.0)
        annualize: If True, annualize the ratio (default: True)
    
    Returns:
        Sharpe ratio
    """
    # Remove NaN values
    clean_returns = returns.dropna()
    
    # Calculate mean return
    mean_return = clean_returns.mean()
    
    # Calculate volatility (non-annualized for daily calculation)
    vol = clean_returns.std()
    
    if vol == 0:
        return 0.0
    
    # Convert annual risk-free rate to daily
    daily_rf = risk_free_rate / 252 if annualize else risk_free_rate
    
    # Calculate Sharpe ratio
    sharpe = (mean_return - daily_rf) / vol
    
    # Annualize if requested
    if annualize:
        sharpe = sharpe * np.sqrt(252)
    
    return sharpe


def max_drawdown(prices: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate maximum drawdown (worst peak-to-trough decline).
    
    Formula: MDD = min((P_t - max(P_0..P_t)) / max(P_0..P_t))
    
    Args:
        prices: Series or array of closing prices
    
    Returns:
        Maximum drawdown as a negative decimal (e.g., -0.15 for 15% drawdown)
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(prices)
    
    # Calculate drawdown at each point
    drawdown = (prices - running_max) / running_max
    
    # Return the maximum (most negative) drawdown
    mdd = np.min(drawdown)
    
    return mdd


def risk_score(vol: float) -> str:
    """
    Convert volatility into categorical risk rating.
    
    Thresholds (annualized volatility):
    - High Risk: > 30%
    - Moderate: 15% - 30%
    - Stable: < 15%
    
    Args:
        vol: Annualized volatility as decimal
    
    Returns:
        Risk category as string
    """
    if vol > 0.30:
        return "HIGH RISK"
    elif vol > 0.15:
        return "MODERATE"
    else:
        return "STABLE"


def average_return(returns: pd.Series, annualize: bool = True) -> float:
    """
    Calculate average return.
    
    Args:
        returns: Series of daily returns
        annualize: If True, annualize the return (default: True)
    
    Returns:
        Average return as a decimal
    """
    clean_returns = returns.dropna()
    avg_return = clean_returns.mean()
    
    if annualize:
        avg_return = avg_return * 252
    
    return avg_return
