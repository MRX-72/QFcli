"""
Analysis Module
Single source of truth for per-stock quantitative analysis.
Used by both single-stock mode and stock comparison.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .data_fetcher import (
    fetch_stock_data,
    calculate_daily_returns,
    get_current_price,
    get_date_range
)
from .metrics import (
    cumulative_return,
    volatility,
    sharpe_ratio,
    max_drawdown,
    risk_score,
    average_return,
    daily_return_stats,
    sortino_ratio,
    value_at_risk,
    beta,
    yearly_returns
)
from .indicators import (
    calculate_all_smas,
    moving_average_signals,
    rate_of_change,
    calculate_rsi,
    calculate_macd,
    bollinger_bands,
    average_true_range,
    golden_death_cross
)


def analyze_single_stock(
    ticker: str,
    period: str = '1y',
    risk_free_rate: float = 0.0,
    benchmark_returns: Optional[pd.Series] = None
) -> Dict:
    """
    Analyze a single stock and return a flat dictionary of every metric.

    The returned dict carries the flat fields used by the comparison logic
    (ticker, company_name, current_price, cumulative_return, ...) and the
    nested fields used by the display layer (sma_values, signals, rsi, macd,
    bollinger_bands, daily_stats, yearly_returns, trend). Each dict is
    self-contained, so callers never reslice the raw DataFrame again.

    Args:
        ticker: Stock symbol
        period: Historical data period
        risk_free_rate: Annual risk-free rate as decimal
        benchmark_returns: Optional Series of market returns to compute beta

    Returns:
        Flat dict of all computed metrics

    Raises:
        ValueError: If data cannot be fetched or analyzed
    """
    df, company_name = fetch_stock_data(ticker, period)

    prices = df['Close']
    current_price = get_current_price(df)
    start_date, end_date = get_date_range(df)

    returns = calculate_daily_returns(prices)

    cum_return = cumulative_return(prices)
    vol = volatility(returns, annualize=True)
    sharpe = sharpe_ratio(returns, risk_free_rate=risk_free_rate, annualize=True)
    downside = sortino_ratio(returns, risk_free_rate=risk_free_rate, annualize=True)
    var_95 = value_at_risk(returns, confidence=0.95, annualize=True)
    mdd = max_drawdown(prices)
    avg_ret = average_return(returns, annualize=True)

    sma_values = calculate_all_smas(prices, windows=[20, 50, 200])
    signals = moving_average_signals(current_price, sma_values)
    roc = rate_of_change(prices, period=12)

    rsi = calculate_rsi(prices)
    macd_data = calculate_macd(prices)
    bb_data = bollinger_bands(prices)
    daily_stats = daily_return_stats(returns)

    atr = average_true_range(df)
    cross = golden_death_cross(prices)
    yearly = yearly_returns(prices)

    risk_rating = risk_score(vol)

    bullish_count = sum(1 for signal in signals.values() if signal == "BULLISH")
    total_signals = sum(1 for signal in signals.values() if signal in ("BULLISH", "BEARISH"))

    beta_value = None
    if benchmark_returns is not None and len(benchmark_returns) >= 2:
        beta_est = beta(returns, benchmark_returns)
        beta_value = None if np.isnan(beta_est) else float(beta_est)

    return {
        'ticker': ticker.upper(),
        'company_name': company_name,
        'current_price': current_price,
        'start_date': start_date,
        'end_date': end_date,
        'cumulative_return': cum_return,
        'avg_return': avg_ret,
        'volatility': vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': downside,
        'value_at_risk': var_95,
        'max_drawdown': mdd,
        'sma_values': sma_values,
        'signals': signals,
        'bullish_count': bullish_count,
        'total_signals': total_signals,
        'roc': roc,
        'atr': atr,
        'golden_cross': cross,
        'yearly_returns': yearly,
        'beta': beta_value,
        'risk_score': risk_rating,
        'rsi': rsi,
        'macd': macd_data,
        'bollinger_bands': bb_data,
        'daily_stats': daily_stats,
        'trend': {
            'prices': [round(float(p), 2) for p in prices.tail(60)],
            'high': float(prices.max()),
            'low': float(prices.min())
        }
    }