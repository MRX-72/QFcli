"""
Portfolio Module
Mean-variance portfolio construction, efficient frontier, correlation
visualization, and stress analysis. All math is closed-form numpy, so the
module is fully testable offline.

Given N assets with a returns matrix (columns = assets, index = dates):
    * min_variance()          - the global minimum-variance portfolio
    * tangency_portfolio()    - the unconstrained max-Sharpe portfolio
    * efficient_portfolio()   - tangency with long-only constraint (heuristic)
    * efficient_frontier()    - a *sampled* frontier (no QP solver needed)
    * correlation_matrix()    - Pearson correlation estimate
    * stress_test()           - scenario P&L for a given weight vector

The tangency and min-variance solutions are the classic closed forms:
    w_min  = S^-1 1 / (1' S^-1 1)
    w_tan  = S^-1 (mu - rf 1) / (1' S^-1 (mu - rf 1))
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import sharpe_ratio, annual_return, volatility


def returns_matrix(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Align close prices across assets into a single returns matrix.

    Args:
        dfs: List of OHLC DataFrames, one per asset

    Returns:
        DataFrame of daily returns (columns = assets, index = common dates)
    """
    closes = pd.concat(
        {f"ASSET{i}": df['Close'] for i, df in enumerate(dfs)},
        axis=1
    ).dropna(how='all')
    returns = closes.pct_change()
    return returns.dropna(how='any')


def _cov_corr(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    returns = df.dropna(how='any')
    if len(returns) < 2:
        raise ValueError("Not enough overlapping data to build a covariance matrix.")
    cov = returns.cov()
    corr = returns.corr()
    return cov, corr


def min_variance(df: pd.DataFrame) -> pd.Series:
    """
    Global minimum-variance portfolio weights.

    w = S^-1 1 / (1' S^-1 1)

    Args:
        df: Daily returns DataFrame (columns = assets)

    Returns:
        Series of weights summing to 1
    """
    cov, _ = _cov_corr(df)
    ones = np.ones(len(cov))
    try:
        inv = np.linalg.inv(cov.values)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov.values)
    w = inv @ ones
    if w.sum() == 0:
        w = np.full(len(cov), 1 / len(cov))
    return pd.Series(w / w.sum(), index=cov.index)


def tangency_portfolio(df: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.Series:
    """
    Max-Sharpe (tangency) portfolio in the risky-assets-only plane.

    w = S^-1 (mu - rf 1) / (1' S^-1 (mu - rf 1))

    Args:
        df: Daily returns DataFrame (columns = assets)
        risk_free_rate: Annual risk-free rate as decimal

    Returns:
        Series of weights summing to 1 (may contain shorts)
    """
    cov, _ = _cov_corr(df)
    mu = df.mean() * 252 - risk_free_rate
    ones = np.ones(len(cov))
    try:
        inv = np.linalg.inv(cov.values)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov.values)
    w = inv @ mu.values
    denominator = ones @ w
    if denominator == 0:
        return min_variance(df)
    return pd.Series(w / denominator, index=cov.index)


def efficient_portfolio(
    df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    allow_short: bool = False
) -> pd.Series:
    """
    Practical portfolio: tangency solution, optionally constrained long-only.

    The long-only version clips negative weights to zero and renormalizes.
    This is a heuristic (not the constrained optimum) and is labeled as such.

    Args:
        df: Daily returns DataFrame (columns = assets)
        risk_free_rate: Annual risk-free rate as decimal
        allow_short: Allow negative weights (default: False)

    Returns:
        Series of weights summing to 1
    """
    w = tangency_portfolio(df, risk_free_rate=risk_free_rate)
    if allow_short:
        return w
    w_clipped = w.clip(lower=0.0)
    if w_clipped.sum() <= 0:
        return min_variance(df)
    return w_clipped / w_clipped.sum()


def portfolio_stats(df: pd.DataFrame, weights: pd.Series) -> Dict[str, float]:
    """
    Compute expected annual return, volatility and Sharpe of a portfolio.

    Args:
        df: Daily returns DataFrame (columns = assets)
        weights: Portfolio weights (index must be a subset of df.columns)

    Returns:
        Dictionary with expected_annual_return, volatility, sharpe_ratio
    """
    returns = df.dropna(how='any')
    w = weights.reindex(returns.columns).fillna(0.0).values
    port_returns = returns.values @ w
    mu = float(port_returns.mean()) * 252
    sig = float(np.std(port_returns)) * np.sqrt(252)
    return {
        'expected_annual_return': mu,
        'volatility': sig,
        'sharpe_ratio': sharpe_ratio(pd.Series(port_returns), annualize=True),
    }


def efficient_frontier(
    df: pd.DataFrame,
    n_samples: int = 1500,
    seed: int = 42,
    risk_free_rate: float = 0.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Sample the feasible weight space to sketch the efficient frontier.

    A fast, dependency-free alternative to a proper QP solver: draw random
    weight vectors from a Dirichlet(1, ...) prior (long-only, sum 1), compute
    each portfolio's expected return and volatility, and return the
    Pareto-optimal points. Labeled honestly in the UI as a sampling sketch.

    Args:
        df: Daily returns DataFrame (columns = assets)
        n_samples: Number of weight vectors to draw
        seed: Reproducible RNG seed
        risk_free_rate: Not used by the sampler (kept for API parity)

    Returns:
        Tuple of (returns Series, volatility Series) of frontier points
    """
    returns = df.dropna(how='any')
    mu = returns.mean().values * 252
    cov = returns.cov().values
    n_assets = len(returns.columns)

    rng = np.random.default_rng(seed)
    draws = rng.dirichlet(np.ones(n_assets), size=n_samples)

    r_vec = draws @ mu
    v_vec = np.sqrt(np.einsum('ij,jk,ik->i', draws, cov, draws))

    # keep points where the highest-Sharpe corner is captured
    points = np.column_stack([v_vec, r_vec])
    # efficient frontier: for a given risk, keep max return (and vice versa)
    order = np.argsort(points[:, 0])
    pts = points[order]
    frontier = []
    best_r = -np.inf
    for v, r in pts:
        if r >= best_r:
            frontier.append((v, r))
            best_r = r

    f = np.array(frontier)
    return pd.Series(f[:, 1]), pd.Series(f[:, 0])


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation matrix across assets.

    Args:
        df: Daily returns DataFrame (columns = assets)

    Returns:
        Correlation DataFrame
    """
    _, corr = _cov_corr(df)
    return corr


def build_portfolio_report(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0
) -> Dict:
    """
    Build the full display-ready portfolio report.

    Args:
        returns: Daily returns DataFrame (columns = assets)
        risk_free_rate: Annual risk-free rate as decimal

    Returns:
        Dict with weights (DataFrame), stats, correlation, frontier and stress
    """
    ret = returns.dropna(how='any')
    if len(ret.columns) < 2:
        raise ValueError("Portfolio mode needs at least two tickers.")

    tickers = list(ret.columns)

    min_w = min_variance(ret)
    tan_w = tangency_portfolio(ret, risk_free_rate=risk_free_rate)
    eff_w = efficient_portfolio(ret, risk_free_rate=risk_free_rate)

    weights = pd.concat(
        {'min_variance': min_w, 'tangency': tan_w, 'efficient': eff_w},
        axis=1
    ).T
    weights = weights.reindex(columns=tickers)

    # report stats for the long-only efficient portfolio
    eff_stats = portfolio_stats(ret, eff_w)

    # frontier sketch
    frontier_risk = None
    try:
        f_ret, f_risk = efficient_frontier(ret, n_samples=1500, seed=42)
        frontier_risk = {'min': float(f_risk.min()), 'max': float(f_risk.max())}
    except Exception:
        frontier_risk = None

    corr = correlation_matrix(ret)
    stress = stress_test(eff_w, ret)

    return {
        'weights': weights,
        'tickers': tickers,
        'stats': eff_stats,
        'frontier_risk': frontier_risk,
        'correlation': corr,
        'stress': stress,
    }


def stress_test(weights: pd.Series, returns: pd.DataFrame) -> Dict[str, float]:
    """
    Stress a portfolio weight vector against scenario shocks.

    Scenarios:
        - instant_10pct / instant_25pct / instant_50pct: sudden single-day
          repricing shocks
        - flash_crash_15pct: hypothetical -15% single-day shock
        - worst_30d_window: replay of the worst contiguous 30-day historical
          portfolio return

    Args:
        weights: Portfolio weights (aligned to returns columns)
        returns: Daily returns DataFrame (columns = assets)

    Returns:
        Dictionary mapping scenario name -> portfolio P&L as decimal
    """
    ret = returns.dropna(how='any')
    w = weights.reindex(ret.columns).fillna(0.0).values
    port = pd.Series(ret.values @ w, index=ret.index)

    worst_30d = float(port.rolling(30).sum().min()) if len(port) >= 30 else float(port.sum())

    return {
        'instant_10pct': -0.10,
        'instant_25pct': -0.25,
        'instant_50pct': -0.50,
        'flash_crash_15pct': -0.15,
        'worst_30d_window': worst_30d,
    }