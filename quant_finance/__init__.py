"""
Quant Finance Module
A quantitative finance toolkit for stock analysis.
"""

__version__ = "1.3.0"
__author__ = "MRX-72"

from .data_fetcher import fetch_stock_data, calculate_daily_returns, fetch_benchmark_returns
from .metrics import (
    cumulative_return,
    volatility,
    sharpe_ratio,
    max_drawdown,
    risk_score,
    sortino_ratio,
    value_at_risk,
    beta,
    yearly_returns,
    annual_return,
    rolling_volatility,
    rolling_sharpe,
    monte_carlo_forecast,
    bootstrap_conf_interval,
    worst_rolling_return
)
from .indicators import (
    simple_moving_average,
    moving_average_signals,
    rate_of_change,
    average_true_range,
    golden_death_cross
)
from .analysis import analyze_single_stock
from .output import format_results, format_comparison_results
from .comparison import compare_stocks
from .portfolio import (
    min_variance,
    tangency_portfolio,
    efficient_portfolio,
    efficient_frontier,
    correlation_matrix,
    stress_test,
    build_portfolio_report
)
from .statistics import (
    sharpe_significance,
    sharpe_bootstrap_ci,
    ljung_box,
    jarque_bera,
    adf_test
)
from .backtest import run_backtest, BUILTIN_STRATEGIES

__all__ = [
    'fetch_stock_data',
    'calculate_daily_returns',
    'fetch_benchmark_returns',
    'cumulative_return',
    'volatility',
    'sharpe_ratio',
    'max_drawdown',
    'risk_score',
    'sortino_ratio',
    'value_at_risk',
    'beta',
    'yearly_returns',
    'annual_return',
    'rolling_volatility',
    'rolling_sharpe',
    'monte_carlo_forecast',
    'bootstrap_conf_interval',
    'worst_rolling_return',
    'simple_moving_average',
    'moving_average_signals',
    'rate_of_change',
    'average_true_range',
    'golden_death_cross',
    'analyze_single_stock',
    'format_results',
    'format_comparison_results',
    'compare_stocks',
    'min_variance',
    'tangency_portfolio',
    'efficient_portfolio',
    'efficient_frontier',
    'correlation_matrix',
    'stress_test',
    'build_portfolio_report',
    'sharpe_significance',
    'sharpe_bootstrap_ci',
    'ljung_box',
    'jarque_bera',
    'adf_test',
    'run_backtest',
    'BUILTIN_STRATEGIES'
]