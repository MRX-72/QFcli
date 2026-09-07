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

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import (
    annual_return,
    volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    daily_return_stats,
    active_performance
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


def size_position(
    signal: pd.Series,
    returns: pd.Series,
    sizer: str = 'fixed',
    sizer_kwargs: Optional[Dict] = None
) -> pd.Series:
    """
    Apply a position-sizing scheme to a target signal series.

    The signal (target exposure in -1..1) is the *what*; the sizer is the
    *how much*. This never expands the signal beyond the strategy's stated
    direction - it only scales the magnitude.

    Schemes:
        fixed      - signal * weight (default weight 1.0, i.e. unchanged)
        target_vol - scale exposure so that rolling realized volatility
                     (windows trades) times exposure approximates a target
                     annual volatility, capped at max_leverage. Volatility
                     targets position size inversely, so it naturally cuts
                     exposure in choppy markets (a classical risk-parity
                     podding approach). With ``slow_window`` set (default 0),
                     exposure is instead scaled by the SLOW realized vol
                     (Moreira-Muir style vol management) so positions do not
                     chase last week's vol; the fast target is retained as an
                     upper bound, so a sudden vol spike still cuts exposure
                     immediately (crash guard). ``slow_window`` must be >=
                     ``window``.
        kelly      - fractional Kelly on trailing win-rate / average win /
                     average loss over the trailing window:
                     f = win_rate - (1 - win_rate) / (avg_win / avg_loss),
                     capped to [0, 1] and multiplied by ``fraction`` (default
                     0.25, i.e. quarter-Kelly). This is a heuristic sizing
                     rule built on the daily outcome distribution; it is NOT
                     a guarantee of optimality, and it only reduces risk.

    Only data available up to each day is used (no lookahead).

    Args:
        signal: Target exposure series in -1..1
        returns: Daily returns series aligned to signal
        sizer: 'fixed', 'target_vol' or 'kelly'
        sizer_kwargs: See the scheme descriptions above

    Returns:
        Sized position series (same index as signal)
    """
    kw = sizer_kwargs or {}
    s = signal.fillna(0.0).clip(-1.0, 1.0)

    if sizer == 'fixed':
        return s * float(kw.get('weight', 1.0))

    if sizer == 'target_vol':
        target = float(kw.get('target_vol', 0.15))
        window = int(kw.get('window', 20))
        slow_window = int(kw.get('slow_window', 0))
        max_lev = float(kw.get('max_leverage', 1.0))
        if slow_window and slow_window < window:
            raise ValueError(
                f"slow_window ({slow_window}) must be >= window ({window}) "
                "for target_vol sizing.")
        vol = returns.rolling(window=window).std(ddof=1) * np.sqrt(252)
        fast_exposure = (target / vol.replace(0, np.nan)).clip(upper=max_lev)
        exposure = fast_exposure
        if slow_window:
            slow_vol = returns.rolling(window=slow_window).std(ddof=1) * np.sqrt(252)
            slow_exposure = target / slow_vol.replace(0, np.nan)
            # vol-management: smooth MM scaling, but never more aggressive than
            # the fast vol-target allows (crash guard); until the slow window
            # fills, fall back to the fast scaling rather than a flat 1.0
            exposure = slow_exposure.clip(upper=fast_exposure)
            exposure = exposure.where(exposure.notna(), fast_exposure).clip(upper=max_lev)
        exposure = exposure.reindex(s.index).fillna(1.0).clip(upper=max_lev)
        return s * exposure

    if sizer == 'kelly':
        fraction = float(kw.get('fraction', 0.25))
        window = int(kw.get('window', 252))
        # masked rolling means need a floor on observed days; otherwise every
        # window that contains any masked-out day would be NaN forever
        min_periods = max(10, min(window, window // 4))
        pos_mask = returns > 0
        traded = returns.abs() > 0
        win_rate = pos_mask.astype(float).rolling(window, min_periods=min_periods).mean()
        avg_win = returns.where(pos_mask).rolling(window, min_periods=min_periods).mean()
        avg_loss = (-returns).where(traded & ~pos_mask).rolling(window, min_periods=min_periods).mean()
        ratio = avg_win / avg_loss.replace(0, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            f = win_rate - (1 - win_rate) / ratio
        f = f.replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=1.0)
        f = f.fillna(0.0)
        return s * (f * fraction)

    raise ValueError(f"Unknown sizer '{sizer}' (choose fixed, target_vol or kelly).")


def run_backtest(
    prices: pd.Series,
    strategy: Callable,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    risk_free_rate: float = 0.0,
    sizer: str = 'fixed',
    sizer_kwargs: Optional[Dict] = None,
    **strategy_kwargs
) -> Dict:
    """
    Run a vectorized daily backtest of a strategy against a price series.

    Position is the strategy signal shifted one day forward (no lookahead).
    Each time exposure changes, a transaction cost is charged against the
    traded notional. Returns a flat dict of performance stats plus the
    buy-and-hold baseline for context, and a benchmark-adjusted evaluation
    (alpha, beta, hit rate, information ratio vs buy-and-hold) that makes it
    easy to tell whether the strategy actually beats just holding the asset.

    Args:
        prices: Series of daily closing prices
        strategy: Callable(prices, **kwargs) -> pd.Series of signals in -1..1
        cost_bps: Round-trip commission in basis points (e.g. 5 == 0.05%)
        slippage_bps: Slippage per trade in basis points
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino
        sizer: Position-sizing scheme ('fixed', 'target_vol' or 'kelly')
        sizer_kwargs: Passed to size_position()
        **strategy_kwargs: Passed through to the strategy

    Returns:
        Flat dict with strategy and baseline performance metrics plus the
        ``strategy_alpha``/``strategy_hit_rate`` style benchmark-adjusted keys.
        Also carries ``_strategy_daily_returns`` and ``_baseline_daily_returns``
        (underscore keys, used internally by walk_forward and stripped from
        JSON output).
    """
    clean = prices.dropna()
    if len(clean) < 3:
        raise ValueError("Not enough price data to backtest.")

    returns = clean.pct_change()

    target = strategy(clean, **strategy_kwargs)
    target = target.reindex(clean.index).fillna(0.0).clip(-1.0, 1.0)
    target = size_position(target, returns, sizer=sizer, sizer_kwargs=sizer_kwargs)

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

    # benchmark-adjusted evaluation vs buy-and-hold
    active = active_performance(net, returns, risk_free_rate=risk_free_rate)
    result.update({f'strategy_{k}': v for k, v in active.items()})

    # daily return streams for walk-forward stitching (underscore -> hidden in JSON)
    result['_strategy_daily_returns'] = net
    result['_baseline_daily_returns'] = returns
    return result


def parse_param_grid(text: str) -> List[Dict]:
    """
    Parse a ``key=v1,v2;key2=v3,v4`` grid spec into a cartesian param grid.

    Example: ``--grid "fast=10,20;slow=40,80"`` produces
    ``[{fast:10, slow:40}, {fast:10, slow:80}, {fast:20, slow:40}, {fast:20, slow:80}]``.
    Values are coerced like ``--strategy-param`` (int/float/bool/str).

    Args:
        text: Grid specification string

    Returns:
        List of param dicts (one per grid point)

    Raises:
        ValueError: on malformed specs
    """
    from itertools import product

    if not text or not text.strip():
        raise ValueError("--grid requires a non-empty spec (e.g. 'fast=10,20;slow=40,80').")
    axes: Dict[str, list] = {}
    for clause in text.split(';'):
        clause = clause.strip()
        if not clause:
            continue
        if '=' not in clause:
            raise ValueError(f"Invalid grid clause '{clause}' (expected key=v1,v2,v3).")
        key, _, raw = clause.partition('=')
        key = key.strip()
        values = [_coerce_param(v.strip()) for v in raw.split(',') if v.strip()]
        if not values:
            raise ValueError(f"Grid clause '{clause}' has no values.")
        axes[key] = values

    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in product(*(axes[k] for k in keys))]


def _select_best_params(
    train: pd.Series,
    strategy: Callable,
    param_grid: List[Dict],
    selection_metric: str,
    sizer: str,
    sizer_kwargs: Optional[Dict],
    cost_bps: float,
    slippage_bps: float,
    risk_free_rate: float
) -> Tuple[Dict, float]:
    """
    Pick the grid point with the best training-window score (no lookahead).

    Returns (params, best_score). NaN/invalid training scores are treated as
    -inf so degenerate configurations are never selected.
    """
    best = None
    best_score = -np.inf
    for params in param_grid:
        res = run_backtest(
            train, strategy,
            cost_bps=cost_bps, slippage_bps=slippage_bps,
            risk_free_rate=risk_free_rate,
            sizer=sizer, sizer_kwargs=sizer_kwargs,
            **params
        )
        score = res[f'strategy_{selection_metric}']
        if score is None or (isinstance(score, float) and np.isnan(score)):
            score = -np.inf
        if score > best_score:
            best_score = score
            best = params
    return best, best_score


def _ensemble_weights(scores: List[float], method: str, topk: Optional[int]) -> np.ndarray:
    """
    Compute non-negative per-param weights for out-of-sample blending.

    Schemes:
        best  - one-hot argmax (single best config)
        equal - equal weights across the whole grid
        rank  - weights proportional to the rank of each config's training
                score (ties averaged). The best config gets the most weight but
                alternatives are not thrown away - a cheap form of robustness
                against choosing the wrong winner.
        topk  - equal weights on the top-k configs only (default k = half the
                grid, at least 2), everything else zero.

    Weights always sum to 1.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot combine an empty grid.")
    arr = pd.Series(scores, dtype=float)

    if method == 'best':
        best = int(arr.idxmax())
        w = np.zeros(n)
        w[best] = 1.0
        return w

    if method == 'equal':
        return np.full(n, 1.0 / n)

    if method == 'rank':
        ranks = arr.rank(method='average', ascending=True).to_numpy()
        return ranks / ranks.sum()

    if method == 'topk':
        ranks = arr.rank(method='average', ascending=True).to_numpy()
        k = int(min(max(int(topk) if topk else max(2, n // 2), 1), n))
        keep = ranks >= (n - k + 1)   # best k configs (ties at the cut stay in)
        w = keep.astype(float)
        return w / w.sum()

    raise ValueError(f"Unknown ensemble method '{method}' "
                     f"(choose 'best', 'equal', 'rank' or 'topk').")


def walk_forward(
    prices: pd.Series,
    strategy: Callable,
    param_grid: List[Dict],
    sizer: str = 'fixed',
    sizer_kwargs: Optional[Dict] = None,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    risk_free_rate: float = 0.0,
    train_frac: float = 0.6,
    step: int = 63,
    selection_metric: str = 'sharpe',
    min_test: int = 30,
    ensemble: str = 'best',
    topk: Optional[int] = None
) -> Dict:
    """
    Walk-forward (expanding-window) validation of a strategy's parameters.

    The history is split into in-sample (train) / out-of-sample (test)
    segments. For each fold the parameter combination with the best *training*
    score (default annualized Sharpe) is selected, then the strategy is run
    *without re-tuning* on the following test window. Out-of-sample daily
    returns are stitched across folds into one honest out-of-sample equity
    curve and evaluated against the buy-and-hold baseline (including alpha,
    hit rate and information ratio via active_performance()).

    This is the defensible way to answer *"does tuning actually survive out of
    sample?"*: parameters are chosen on data the test fold has never seen.

    By default the single best grid point (by training score) is run out of
    sample. With ``ensemble`` you can instead blend every grid point's OOS
    returns, which is significantly more robust than betting on one winner:
        - 'equal' averages all configs (average model, lowest variance)
        - 'rank' weights configs by their training-score rank (ties averaged)
        - 'topk' averages the top-k configs only
    Blends never add alpha that the individual configs don't have - they only
    reduce the variance of *parameter selection*. The returns that only the
    best config would have produced are still reported
    (``best_oos_total_return``) so the two can be compared honestly.

    Note: the exact day between two adjacent folds is not traded, so the
    stitched equity curve compounds across a small non-traded gap.

    Args:
        prices: Series of daily closing prices
        strategy: Strategy callable
        param_grid: List of param dicts to evaluate per fold (see parse_param_grid)
        sizer / sizer_kwargs / cost_bps / slippage_bps / risk_free_rate:
            Shared with run_backtest()
        train_frac: Fraction of history used as initial training data
        step: Days added per fold to the in-sample window
        selection_metric: Strategy metric maximized on the training window
            (e.g. 'sharpe', 'sortino', 'annual_return')
        min_test: Minimum test-window length in days
        ensemble: How to combine grid points out of sample: 'best' (default),
            'equal', 'rank' or 'topk'
        topk: Number of configs averaged by 'topk' (default: half the grid, min 2)

    Returns:
        Dict with aggregate out-of-sample stats, benchmark-adjusted evaluation,
        fold-level details and the chosen selection_metric / ensemble method
    """
    clean = prices.dropna()
    n = len(clean)
    if n < 100:
        raise ValueError("Walk-forward needs at least 100 price observations.")
    if not param_grid:
        raise ValueError("param_grid must contain at least one parameter combination.")

    oos_start = int(n * train_frac)
    if oos_start < 60:
        raise ValueError("train_frac leaves too little training data; raise it or pass more history.")

    folds: List[Dict] = []
    oos_parts: List[pd.Series] = []
    best_parts: List[pd.Series] = []
    baseline_parts: List[pd.Series] = []

    start = oos_start
    while start + min_test <= n:
        test_end = min(start + step, n)
        if test_end - start < min_test:
            break

        train = clean.iloc[:start]
        test = clean.iloc[start:test_end]

        best, best_score = _select_best_params(
            train, strategy, param_grid, selection_metric,
            sizer, sizer_kwargs, cost_bps, slippage_bps, risk_free_rate
        )

        # run every grid point out of sample so the fold can be blended
        per_param: List[Tuple[Dict, float, pd.Series, pd.Series]] = []
        for params in param_grid:
            oos_res = run_backtest(
                test, strategy,
                cost_bps=cost_bps, slippage_bps=slippage_bps,
                risk_free_rate=risk_free_rate,
                sizer=sizer, sizer_kwargs=sizer_kwargs,
                **params
            )
            # training score (recomputed cheaply on the train window) is used
            # for rank/topk weighting
            train_res = run_backtest(
                train, strategy,
                cost_bps=cost_bps, slippage_bps=slippage_bps,
                risk_free_rate=risk_free_rate,
                sizer=sizer, sizer_kwargs=sizer_kwargs,
                **params
            )
            tscore = train_res[f'strategy_{selection_metric}']
            if tscore is None or (isinstance(tscore, float) and np.isnan(tscore)):
                tscore = -np.inf
            per_param.append((params, float(tscore),
                              oos_res['_strategy_daily_returns'],
                              oos_res['_baseline_daily_returns']))

        scores = [s for _, s, _, _ in per_param]
        weights = _ensemble_weights(scores, ensemble, topk)

        blended = pd.Series(0.0, index=per_param[0][2].index, dtype=float)
        best_daily = per_param[0][2]
        for (params, _, oos_d, _), w in zip(per_param, weights):
            blended = blended + w * oos_d.fillna(0.0)
            if params == best:
                best_daily = oos_d
        blended = blended.replace([np.inf, -np.inf], np.nan)

        baseline_daily = per_param[0][3]

        oos_total = float((1 + blended.dropna()).cumprod().iloc[-1] - 1) \
            if len(blended.dropna()) else 0.0
        best_total = float((1 + best_daily.dropna()).cumprod().iloc[-1] - 1) \
            if len(best_daily.dropna()) else 0.0
        baseline_total = float((1 + baseline_daily.dropna()).cumprod().iloc[-1] - 1) \
            if len(baseline_daily.dropna()) else 0.0

        folds.append({
            'test_start': int(start),
            'test_end': test_end,
            'best_params': best,
            'n_params': len(param_grid),
            'ensemble': ensemble,
            'selected_train_score': float(best_score),
            'oos_n_days': int(len(blended.dropna())),
            'oos_strategy_total_return': oos_total,
            'oos_best_total_return': best_total,
            'oos_baseline_total_return': baseline_total,
            'beat_baseline': bool(oos_total > baseline_total),
        })
        oos_parts.append(blended)
        best_parts.append(best_daily)
        baseline_parts.append(baseline_daily)
        start += step

    if not folds:
        raise ValueError("No complete walk-forward folds could be built; reduce train_frac or step.")

    oos_returns = pd.concat(oos_parts)
    best_returns = pd.concat(best_parts)
    oos_baseline = pd.concat(baseline_parts)

    def _agg(series: pd.Series) -> Dict:
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        eq = (1 + series).cumprod()
        return {
            'total_return': float(eq.iloc[-1] - 1),
            'annual_return': annual_return(eq),
            'sharpe': sharpe_ratio(series, risk_free_rate=risk_free_rate, annualize=True),
        }

    agg = _agg(oos_returns)
    best_agg = _agg(best_returns)
    base = _agg(oos_baseline)
    active = active_performance(oos_returns, oos_baseline, risk_free_rate=risk_free_rate)

    return {
        'oos_total_return': agg['total_return'],
        'oos_annual_return': agg['annual_return'],
        'oos_sharpe': agg['sharpe'],
        'best_oos_total_return': best_agg['total_return'],
        'best_oos_annual_return': best_agg['annual_return'],
        'baseline_total_return': base['total_return'],
        'baseline_annual_return': base['annual_return'],
        'alpha': active['alpha'],
        'beta_to_baseline': active['beta_to_baseline'],
        'active_return': active['active_return'],
        'information_ratio': active['information_ratio'],
        'hit_rate': active['hit_rate'],
        'oos_days': int(len(oos_returns.replace([np.inf, -np.inf], np.nan).dropna())),
        'n_folds': len(folds),
        'fold_beat_rate': float(np.mean([f['beat_baseline'] for f in folds])),
        'selection_metric': selection_metric,
        'ensemble': ensemble,
        'folds': folds,
    }


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