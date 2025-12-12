"""
Quant Finance Mini-Module
A quantitative finance toolkit for stock analysis.
"""

__version__ = "1.0.0"
__author__ = "QFcli"

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
from .output import format_results, display_summary, format_comparison_results
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
    'format_results',
    'display_summary',
    'format_comparison_results',
    'compare_stocks'
]
