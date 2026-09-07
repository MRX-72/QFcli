"""
Backtest Module
A small event-free vectorized backtester for daily equity strategies.

Strategies produce *signal* series (target exposure in -1..1). The engine
shifts signals by one day to avoid lookahead, applies a transaction-cost and
slippage model on day-to-day weight changes, and reports the standard
performance and risk statistics against a buy-and-hold baseline.

Everything is pure pandas/numpy and deterministic, so the whole module is
testable offline with synthetic price series.
"""

from typing import Callable, Dict

import numpy as np
import pandas as pd

from .metrics import (
    annual_return,
    volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    daily_return_stats
)


def sma_cross_strategy(
    prices: pd.Series,
    fast: int = 20,
    slow: int = 50,
    **kwargs
) -> pd.Series:
    """
    Long/flat trend strategy: hold when SMA(fast) > SMA(slow).

    Returns signal series (1 or 0). Trades happen the day after the crossover,
    so this is a clean, transparent baseline momentum rule.
    """
    fast_sma = prices.rolling(window=fast).mean()
    slow_sma = prices.rolling(window=slow).mean()
    return (fast_sma > slow_sma).astype(float)


def rsi_reversion_strategy(
    prices: pd.Series,
    window: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    **kwargs
) -> pd.Series:
    """
    Mean-reversion strategy: long when RSI drops below the oversold line,
    exit to flat when RSI crosses above the overbought line.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)

    position = pd.Series(np.nan, index=prices.index, dtype=float)
    position[rsi < oversold] = 1.0
    position[rsi > overbought] = 0.0
    position = position.ffill().fillna(0.0)
    return position


def momentum_strategy(
    prices: pd.Series,
    lookback: int = 20,
    hold: int = 10,
    **kwargs
) -> pd.Series:
    """
    The classic Jegadeesh-Titman rule simplified: score each day by trailing
    ROC, long if in the top percentile, plus a holding period to reduce noise.
    """
    roc = prices.pct_change(lookback)
    score = roc.rolling(hold).mean().fillna(0.0)
    threshold = score.quantile(0.70)
    return (score >= threshold).astype(float)


BUILTIN_STRATEGIES: Dict[str, Callable] = {
    'sma_cross': sma_cross_strategy,
    'rsi_reversion': rsi_reversion_strategy,
    'momentum': momentum_strategy,
}


def load_custom_strategy(path: str, func_name: str = 'custom_strategy') -> Callable:
    """
    Load a user-supplied strategy from a Python file.

    The file must define a callable named ``func_name`` with the signature
    ``fn(prices: pd.Series, **kwargs) -> pd.Series`` that returns a signal
    series in -1..1 (positive = long, negative = short, 0 = flat).

    Loading executes arbitrary Python, so only load files you trust.

    Args:
        path: Path to a .py file
        func_name: Name of the strategy callable to import (default: custom_strategy)

    Returns:
        The strategy callable

    Raises:
        ValueError: if the file can't be loaded or the function is missing/not callable
    """
    import importlib.util
    import pathlib
    import sys

    p = pathlib.Path(path).expanduser().resolve()
    if not p.is_file():
        raise ValueError(f"Strategy file not found: {path}")
    if not p.suffix == '.py':
        raise ValueError(f"Strategy file must be a .py file, got: {p.suffix or '(none)'}")

    spec = importlib.util.spec_from_file_location(f"qfcli_custom_{p.stem}", p)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not create module spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ValueError(f"Error executing strategy file {path}: {e}") from e
    finally:
        sys.modules.pop(f"qfcli_custom_{p.stem}", None)

    fn = getattr(module, func_name, None)
    if fn is None:
        raise ValueError(
            f"Strategy file must define a function named '{func_name}' "
            f"receiving (prices, **kwargs) and returning a signal Series."
        )
    if not callable(fn):
        raise ValueError(f"'{func_name}' in {path} is not callable.")
    return fn


def parse_strategy_params(pairs) -> Dict:
    """
    Parse ``key=value`` CLI tokens into a kwargs dict for a strategy.

    Values are coerced to int/float/bool when possible, else kept as strings.
    """
    params: Dict = {}
    for token in pairs:
        if '=' not in token:
            raise ValueError(f"Invalid strategy parameter '{token}' (expected key=value)")
        key, _, raw = token.partition('=')
        key = key.strip()
        raw = raw.strip()
        params[key] = _coerce_param(raw)
    return params


def _coerce_param(raw: str):
    low = raw.lower()
    if low == 'true':
        return True
    if low == 'false':
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def run_backtest(
    prices: pd.Series,
    strategy: Callable,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    risk_free_rate: float = 0.0,
    **strategy_kwargs
) -> Dict:
    """
    Run a vectorized daily backtest of a strategy against a price series.

    Position is the strategy signal shifted one day forward (no lookahead).
    Each time exposure changes, a transaction cost is charged against the
    traded notional. Returns a flat dict of performance stats plus the
    buy-and-hold baseline for context.

    Args:
        prices: Series of daily closing prices
        strategy: Callable(prices, **kwargs) -> pd.Series of signals in -1..1
        cost_bps: Round-trip commission in basis points (e.g. 5 == 0.05%)
        slippage_bps: Slippage per trade in basis points
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino
        **strategy_kwargs: Passed through to the strategy

    Returns:
        Flat dict with strategy and baseline performance metrics
    """
    clean = prices.dropna()
    if len(clean) < 3:
        raise ValueError("Not enough price data to backtest.")

    returns = clean.pct_change()

    target = strategy(clean, **strategy_kwargs)
    target = target.reindex(clean.index).fillna(0.0).clip(-1.0, 1.0)

    # One-day shift: trade at the open of the following bar
    position = target.shift(1).fillna(0.0)

    gross = position * returns

    turnover = position.diff().abs().fillna(0.0)
    cost_per_day = turnover * ((cost_bps + slippage_bps) / 10000.0)
    net = (gross - cost_per_day).dropna()

    def _stats(series: pd.Series, label: str, per_year: float = 252) -> Dict:
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        sr = sharpe_ratio(series, risk_free_rate=risk_free_rate, annualize=True)
        so = sortino_ratio(series, risk_free_rate=risk_free_rate, annualize=True)
        vol = volatility(series, annualize=True)
        eq = (1 + series).cumprod()
        mdd = max_drawdown(eq)
        stats = daily_return_stats(series)
        ann = annual_return(eq)
        return {
            f'{label}_total_return': float((1 + series).cumprod().iloc[-1] - 1),
            f'{label}_annual_return': ann,
            f'{label}_volatility': vol,
            f'{label}_sharpe': sr,
            f'{label}_sortino': so,
            f'{label}_max_drawdown': mdd,
            f'{label}_best_day': stats['best_day'],
            f'{label}_worst_day': stats['worst_day'],
            f'{label}_win_ratio': stats['win_ratio'],
        }

    result = _stats(net, 'strategy')
    result.update(_stats(returns, 'baseline'))

    # study-level stats
    n_trades = int((position.diff().abs() > 1e-9).sum())
    total_cost = float(cost_per_day.sum())
    if np.isnan(total_cost):
        total_cost = 0.0
    avg_daily_turnover = float(turnover.mean()) if len(turnover) else 0.0

    result.update({
        'n_days': int(len(net)),
        'n_trades': n_trades,
        'total_cost': total_cost,
        'avg_daily_turnover': avg_daily_turnover,
        'days_in_market': float((position > 1e-9).mean()),
    })
    return result


def prepare_backtest_data(df: pd.DataFrame, period: str = '1y') -> pd.Series:
    """
    Extract the close-price series used by the backtester.

    Args:
        df: DataFrame with Close column
        period: Unused (kept for parity with fetch); included for clarity

    Returns:
        Series of closing prices
    """
    return df['Close']