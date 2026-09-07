"""
Metrics Module
Core financial metrics calculations.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict


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


def annual_return(equity_curve: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate annualized return (CAGR) from an equity curve.

    Formula:
        CAGR = (end / start) ** (252 / n_days) - 1

    Args:
        equity_curve: Series of cumulative wealth values (start ~ 1.0)

    Returns:
        Annualized return as a decimal
    """
    if isinstance(equity_curve, np.ndarray):
        equity_curve = pd.Series(equity_curve)

    clean = equity_curve.dropna()
    if len(clean) < 2:
        return 0.0

    start, end = clean.iloc[0], clean.iloc[-1]
    if start <= 0:
        return 0.0

    days = len(clean)
    return (end / start) ** (252 / days) - 1


def daily_return_stats(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate statistics about daily returns.
    
    Args:
        returns: Series of daily returns
    
    Returns:
        Dictionary with 'best_day', 'worst_day', 'positive_days', 'total_days', 'win_ratio'
    """
    clean_returns = returns.dropna()
    
    if clean_returns.empty:
        return {
            'best_day': np.nan,
            'worst_day': np.nan,
            'win_ratio': np.nan
        }
        
    best_day = clean_returns.max()
    worst_day = clean_returns.min()
    
    positive_days = (clean_returns > 0).sum()
    total_days = len(clean_returns)
    win_ratio = positive_days / total_days if total_days > 0 else 0
    
    return {
        'best_day': best_day,
        'worst_day': worst_day,
        'win_ratio': win_ratio
    }


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualize: bool = True) -> float:
    """
    Calculate the Sortino ratio (return per unit of downside risk).

    Unlike the Sharpe ratio, only below-target (downside) volatility is
    penalized, so it distinguishes healthy returns from merely volatile ones.

    Formula:
        downside_dev = sqrt(mean(min(r_t - r_f_daily, 0)^2))
        Sortino = (mean(r) - r_f_daily) / downside_dev * sqrt(252)

    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate as decimal
        annualize: Scale to an annualized ratio (default: True)

    Returns:
        Sortino ratio, or 0.0 if there is no downside deviation
    """
    clean = returns.dropna()
    if clean.empty:
        return 0.0

    daily_rf = risk_free_rate / 252 if annualize else risk_free_rate
    excess = clean - daily_rf

    downside = excess[excess < 0]
    if downside.empty:
        return 0.0

    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    if downside_dev == 0:
        return 0.0

    ratio = float(np.mean(excess) / downside_dev)
    return ratio * np.sqrt(252) if annualize else ratio


def value_at_risk(returns: pd.Series, confidence: float = 0.95, annualize: bool = True) -> float:
    """
    Calculate historical Value at Risk (VaR).

    The worst expected loss over the period at a given confidence level,
    estimated directly from the empirical return distribution rather than a
    normal assumption.

    Formula:
        VaR = -percentile(returns, (1 - confidence))
    annualized: VaR * sqrt(252)

    Args:
        returns: Series of daily returns
        confidence: Confidence level between 0 and 1 (default: 0.95)
        annualize: Annualize the result (default: True)

    Returns:
        VaR as a positive decimal (e.g., 0.03 means a 3% expected max loss)
    """
    clean = returns.dropna()
    if clean.empty:
        return 0.0

    quantile = 1.0 - confidence
    daily_var = -float(np.percentile(clean.values, quantile * 100))
    daily_var = max(daily_var, 0.0)

    return daily_var * np.sqrt(252) if annualize else daily_var


def beta(returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate beta (systematic risk) of an asset relative to a market.

    Formula:
        beta = Cov(r_asset, r_market) / Var(r_market)

    Args:
        returns: Series of daily returns
        market_returns: Series of daily market returns

    Returns:
        Beta coefficient, or NaN if insufficient data
    """
    clean = pd.concat([returns, market_returns], axis=1, join="inner").dropna()
    if len(clean) < 2:
        return np.nan

    var_market = float(np.var(clean.iloc[:, 1], ddof=1))
    if var_market == 0:
        return np.nan

    cov = float(np.cov(clean.iloc[:, 0], clean.iloc[:, 1])[0, 1])
    return cov / var_market


def yearly_returns(prices: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    """
    Calculate calendar-year returns from a daily price series.

    Args:
        prices: Series of closing prices indexed by date

    Returns:
        Dictionary mapping "YYYY" -> total return for that calendar year
    """
    if isinstance(prices, np.ndarray):
        return {}

    clean = prices.dropna()
    if clean.empty:
        return {}

    by_year = clean.groupby(clean.index.year)
    yearly = {}
    for year, group in by_year:
        if len(group) < 2:
            continue
        p_start = group.iloc[0]
        p_end = group.iloc[-1]
        if p_start == 0:
            continue
        yearly[str(year)] = (p_end - p_start) / p_start
    return yearly


def rolling_volatility(returns: pd.Series, window: int = 60, annualize: bool = True) -> pd.Series:
    """
    Rolling window realized volatility.

    Args:
        returns: Series of daily returns
        window: Rolling window length (default: 60)
        annualize: Scale each window by sqrt(252) (default: True)

    Returns:
        Series of rolling volatility (same index, NaN for warmup)
    """
    rv = returns.dropna().rolling(window=window).std(ddof=1)
    return rv * np.sqrt(252) if annualize else rv


def rolling_sharpe(returns: pd.Series, window: int = 60, risk_free_rate: float = 0.0) -> pd.Series:
    """
    Rolling window Sharpe ratio (annualized).

    Args:
        returns: Series of daily returns
        window: Rolling window length (default: 60)
        risk_free_rate: Annual risk-free rate as decimal

    Returns:
        Series of rolling Sharpe ratios
    """
    daily_rf = risk_free_rate / 252
    roll = returns.dropna().rolling(window=window)
    mean = roll.mean() - daily_rf
    std = roll.std(ddof=1)
    return (mean / std) * np.sqrt(252)


def monte_carlo_forecast(
    returns: pd.Series,
    horizon: int = 252,
    n_sims: int = 5000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Bootstrap Monte Carlo forecast of total return over a horizon.

    Daily returns are resampled with replacement (block of history drawn
    pathwise), compounded forward for 'horizon' days. Report the P5/P50/P95
    of the terminal wealth multiplier.

    Args:
        returns: Historical daily returns
        horizon: Number of days to simulate (default: 252)
        n_sims: Number of simulated paths (default: 5000)
        seed: Deterministic RNG seed

    Returns:
        Dict with p5, p50, p95 (terminal wealth multipliers), n_sims
    """
    clean = returns.dropna().to_numpy()
    if len(clean) < 30:
        return {'p5': np.nan, 'p50': np.nan, 'p95': np.nan, 'n_sims': n_sims}

    rng = np.random.default_rng(seed)
    draws = rng.choice(clean, size=(n_sims, horizon), replace=True)
    total = np.prod(1 + draws, axis=1)
    p5, p50, p95 = np.percentile(total, [5, 50, 95])
    return {
        'p5': float(p5),
        'p50': float(p50),
        'p95': float(p95),
        'n_sims': n_sims,
    }


def bootstrap_conf_interval(
    sample: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Percentile bootstrap confidence interval for an arbitrary sample statistic.

    Args:
        sample: Array of observations
        n_boot: Number of resamples
        seed: Deterministic RNG seed
        confidence: Confidence level (default: 0.95)

    Returns:
        Dict with ci_low, ci_high, n_boot
    """
    sample = np.asarray(sample, dtype=float)
    sample = sample[~np.isnan(sample)]
    n = len(sample)
    if n < 2:
        return {'ci_low': np.nan, 'ci_high': np.nan, 'n_boot': n_boot}

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(sample, size=n, replace=True).mean()

    alpha = (1 - confidence) / 2
    low, high = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return {'ci_low': float(low), 'ci_high': float(high), 'n_boot': n_boot}


def worst_rolling_return(returns: pd.Series, window: int = 30) -> float:
    """
    Worst contiguous N-day cumulative return over the history.

    Useful as an empirically-grounded stress scenario.

    Args:
        returns: Series of daily returns
        window: Window length (default: 30)

    Returns:
        Most negative rolling N-day return (decimal)
    """
    clean = returns.dropna()
    if len(clean) < window:
        if clean.empty:
            return 0.0
        return float((1 + clean).prod() - 1)
    roll = ((1 + clean).rolling(window).apply(np.prod, raw=True))
    return float(roll.min())
