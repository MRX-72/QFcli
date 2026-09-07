"""
Analysis Module
Single source of truth for per-stock quantitative analysis.
Used by both single-stock mode and stock comparison.
"""

from typing import Dict

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
    daily_return_stats
)
from .indicators import (
    calculate_all_smas,
    moving_average_signals,
    rate_of_change,
    calculate_rsi,
    calculate_macd,
    bollinger_bands
)


def analyze_single_stock(ticker: str, period: str = '1y', risk_free_rate: float = 0.0) -> Dict:
    """
    Analyze a single stock and return a flat dictionary of every metric.

    The returned dict carries the flat fields used by the comparison logic
    (ticker, company_name, current_price, cumulative_return, ...) and the
    nested fields used by the display layer (sma_values, signals, rsi, macd,
    bollinger_bands, daily_stats). Each dict is self-contained, so callers
    never reslice the raw DataFrame again.

    Args:
        ticker: Stock symbol
        period: Historical data period
        risk_free_rate: Annual risk-free rate as decimal

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
    mdd = max_drawdown(prices)
    avg_ret = average_return(returns, annualize=True)

    sma_values = calculate_all_smas(prices, windows=[20, 50, 200])
    signals = moving_average_signals(current_price, sma_values)
    roc = rate_of_change(prices, period=12)

    rsi = calculate_rsi(prices)
    macd_data = calculate_macd(prices)
    bb_data = bollinger_bands(prices)
    daily_stats = daily_return_stats(returns)

    risk_rating = risk_score(vol)

    bullish_count = sum(1 for signal in signals.values() if signal == "BULLISH")
    total_signals = sum(1 for signal in signals.values() if signal in ("BULLISH", "BEARISH"))

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
        'max_drawdown': mdd,
        'sma_values': sma_values,
        'signals': signals,
        'bullish_count': bullish_count,
        'total_signals': total_signals,
        'roc': roc,
        'risk_score': risk_rating,
        'rsi': rsi,
        'macd': macd_data,
        'bollinger_bands': bb_data,
        'daily_stats': daily_stats
    }