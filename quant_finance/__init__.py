"""
Quant Finance Module
A quantitative finance toolkit for stock analysis.
"""

__version__ = "1.1.0"
__author__ = "MRX-72"

from .data_fetcher import fetch_stock_data, calculate_daily_returns
from .metrics import (
    cumulative_return,
    volatility,
    sharpe_ratio,
    max_drawdown,
    risk_score
)
from .indicators import (
    simple_moving_average,
    moving_average_signals,
    rate_of_change
)
from .analysis import analyze_single_stock
from .output import format_results, format_comparison_results
from .comparison import compare_stocks

__all__ = [
    'fetch_stock_data',
    'calculate_daily_returns',
    'cumulative_return',
    'volatility',
    'sharpe_ratio',
    'max_drawdown',
    'risk_score',
    'simple_moving_average',
    'moving_average_signals',
    'rate_of_change',
    'analyze_single_stock',
    'format_results',
    'format_comparison_results',
    'compare_stocks'
]