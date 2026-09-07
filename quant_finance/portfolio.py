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
    * covariance_estimator()  - sample or Ledoit-Wolf style shrinkage covariance
    * implied_returns()       - reverse-optimized equilibrium returns
    * black_litterman()       - Black-Litterman "light" posterior construction
    * correlation_matrix()    - Pearson correlation estimate
    * stress_test()           - scenario P&L for a given weight vector

The tangency and min-variance solutions are the classic closed forms:
    w_min  = S^-1 1 / (1' S^-1 1)
    w_tan  = S^-1 (mu - rf 1) / (1' S^-1 (mu - rf 1))

Covariance estimation defaults to Ledoit-Wolf style shrinkage toward a
constant-correlation target, which produces a well-conditioned S^-1. Pass
``method='sample'`` to use the raw Pearson covariance instead.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .metrics import sharpe_ratio


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


def _cov_corr(df: pd.DataFrame, method: str = 'lw') -> Tuple[pd.DataFrame, pd.DataFrame]:
    returns = df.dropna(how='any')
    if len(returns) < 2:
        raise ValueError("Not enough overlapping data to build a covariance matrix.")
    cov = covariance_estimator(returns, method=method)
    corr = returns.corr()
    return cov, corr


def covariance_estimator(
    df: pd.DataFrame,
    method: str = 'lw',
    n_factors: Optional[int] = None,
    min_expl_var: float = 0.8
) -> pd.DataFrame:
    """
    Estimate a covariance matrix from daily returns.

    Args:
        df: Daily returns DataFrame (columns = assets)
        method: 'sample' for the raw Pearson covariance, 'lw' for
            Ledoit-Wolf style shrinkage toward a constant-correlation target
            (default), or 'factor' for a low-rank PCA factor-model covariance
            (common + idiosyncratic components). Shrinkage and the factor model
            both guard against singular/ill-conditioned estimates produced by
            short overlapping histories.
        n_factors: Number of factors for method='factor' (see pca_factor_model)
        min_expl_var: Explained-variance cutoff for method='factor'

    Returns:
        Covariance DataFrame
    """
    returns = df.dropna(how='any')
    if len(returns) < 2:
        raise ValueError("Not enough overlapping data to build a covariance matrix.")
    if method == 'sample':
        return returns.cov()
    if method == 'lw':
        return shrinkage_covariance(returns)
    if method == 'factor':
        from .factor_model import factor_model_cov
        return factor_model_cov(returns, n_factors=n_factors, min_expl_var=min_expl_var)
    raise ValueError(f"Unknown covariance estimator '{method}' "
                     f"(expected 'sample', 'lw' or 'factor').")


def shrinkage_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Ledoit-Wolf style shrinkage covariance toward a constant-correlation target.

    The sample covariance is shrunk toward F, where F keeps each asset's own
    variance but replaces cross-variances with the average pair-wise
    correlation:

        F_ii = S_ii      F_ij = rho * sqrt(S_ii * S_jj)   (i != j)

    The shrinkage intensity delta is estimated from sample moments
    (delta -> 0 as T grows; delta -> 1 when the sample covariance is noisy),
    then the matrix is rescaled by T/(T-1) to stay unbiased. The whole matrix
    stays symmetric and positive semi-definite and the inversion in
    tangency/min-variance becomes numerically safe.

    Note: this is a numpy-only simplification of the full Ledoit-Wolf (2004)
    estimator - the target is treated as fixed rather than jointly estimated -
    which is exactly the constant-correlation shrinkage used in practice.

    Args:
        returns: Daily returns DataFrame (columns = assets)

    Returns:
        Shrunk covariance DataFrame
    """
    x = returns.to_numpy(dtype=float)
    T, p = x.shape
    if T < 2 or p < 1:
        raise ValueError("Not enough data for shrinkage covariance.")
    if p == 1:
        return returns.cov()

    xc = x - x.mean(axis=0)
    S_mle = (xc.T @ xc) / T

    var = np.diag(S_mle).copy()
    sqrt_var = np.sqrt(np.maximum(var, 0.0))
    corr = S_mle / np.outer(sqrt_var, sqrt_var)
    np.fill_diagonal(corr, 1.0)
    rho = np.clip((corr.sum() - p) / (p * (p - 1.0)), -1.0, 1.0)

    # constant-correlation target F
    F = np.outer(sqrt_var, sqrt_var) * rho
    np.fill_diagonal(F, var)

    # pi_ij is the sampling variance of the covariance estimator S_ij, i.e.
    # Var(w_ij)/T. With demeaned data: (1/T)sum_t w_tij^2 - s_ij^2 estimates
    # Var(w_ij), so divide by (T-1) to get Var(S_ij). The diagonal is excluded:
    # the constant-correlation target keeps the diagonal fixed (F_ii = S_ii),
    # so only cross-variances should drive delta.
    w2 = np.einsum('ti,tj,ti,tj->ij', xc, xc, xc, xc) / T
    pi = (w2 - S_mle ** 2) / (T - 1.0)
    np.fill_diagonal(pi, 0.0)

    gamma = float(np.sum((F - S_mle) ** 2))
    delta = float(np.clip(pi.sum() / gamma if gamma > 0 else 0.0, 0.0, 1.0))

    shrunk = delta * F + (1 - delta) * S_mle
    out = pd.DataFrame(shrunk * T / (T - 1.0), index=returns.columns, columns=returns.columns)
    return out


def _safe_inv(cov: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(cov)


def min_variance(df: pd.DataFrame, method: str = 'lw') -> pd.Series:
    """
    Global minimum-variance portfolio weights.

    w = S^-1 1 / (1' S^-1 1)

    Args:
        df: Daily returns DataFrame (columns = assets)
        method: Covariance estimator ('sample' or 'lw', default: lw)

    Returns:
        Series of weights summing to 1
    """
    cov, _ = _cov_corr(df, method=method)
    ones = np.ones(len(cov))
    w = _safe_inv(cov.values) @ ones
    if w.sum() == 0:
        w = np.full(len(cov), 1 / len(cov))
    return pd.Series(w / w.sum(), index=cov.index)


def tangency_portfolio(
    df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    method: str = 'lw',
    mu: Optional[pd.Series] = None,
    cov: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Max-Sharpe (tangency) portfolio in the risky-assets-only plane.

    w = S^-1 (mu - rf 1) / (1' S^-1 (mu - rf 1))

    Args:
        df: Daily returns DataFrame (columns = assets)
        risk_free_rate: Annual risk-free rate as decimal
        method: Covariance estimator ('sample' or 'lw', default: lw)
        mu: Optional annualized *excess* expected-return vector. When None it
            is estimated as ``df.mean() * 252 - risk_free_rate``. Used by
            black_litterman() to substitute the posterior mean.
        cov: Optional covariance DataFrame override (used by black_litterman)

    Returns:
        Series of weights summing to 1 (may contain shorts)
    """
    if cov is None:
        cov = covariance_estimator(df, method=method)
    if mu is None:
        mu = df.mean() * 252 - risk_free_rate
    ones = np.ones(len(cov))
    inv = _safe_inv(cov.values)
    w = inv @ mu.values
    denominator = ones @ w
    if denominator == 0:
        return min_variance(df, method=method)
    return pd.Series(w / denominator, index=cov.index)


def efficient_portfolio(
    df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    allow_short: bool = False,
    method: str = 'lw',
    mu: Optional[pd.Series] = None,
    cov: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Practical portfolio: tangency solution, optionally constrained long-only.

    The long-only version clips negative weights to zero and renormalizes.
    This is a heuristic (not the constrained optimum) and is labeled as such.

    Args:
        df: Daily returns DataFrame (columns = assets)
        risk_free_rate: Annual risk-free rate as decimal
        allow_short: Allow negative weights (default: False)
        method: Covariance estimator ('sample' or 'lw', default: lw)
        mu: Optional annualized excess expected-return vector (see tangency_portfolio)
        cov: Optional covariance DataFrame override

    Returns:
        Series of weights summing to 1
    """
    w = tangency_portfolio(df, risk_free_rate=risk_free_rate, method=method, mu=mu, cov=cov)
    if allow_short:
        return w
    w_clipped = w.clip(lower=0.0)
    if w_clipped.sum() <= 0:
        return min_variance(df, method=method)
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


def implied_returns(
    cov: pd.DataFrame,
    weights: pd.Series,
    risk_aversion: float = 2.5
) -> pd.Series:
    """
    Reverse-optimize the equilibrium *excess* returns implied by a portfolio.

    Black-Litterman's key idea: if the market holds the reference portfolio
    ``weights`` under mean-variance preferences, then its expected excess
    returns must satisfy  pi = lam * Sigma * w.  Working backwards from an
    agreed reference portfolio (e.g. market caps, or equal weight as a
    stand-in) yields a far more stable expected-return prior than raw
    historical means.

    Args:
        cov: Annualized covariance DataFrame
        weights: Reference portfolio weights (aligned to cov.columns)
        risk_aversion: Global risk-aversion coefficient lambda (default: 2.5)

    Returns:
        Series of annualized excess expected returns
    """
    w = weights.reindex(cov.index).fillna(0.0).values
    return pd.Series(risk_aversion * (cov.values @ w), index=cov.index)


def black_litterman(
    returns: pd.DataFrame,
    prior_weights: Optional[pd.Series] = None,
    views: Optional[Dict[str, float]] = None,
    view_confidence: Optional[Union[float, List[float]]] = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    cov_method: str = 'lw',
    risk_free_rate: float = 0.0,
    prior_returns: Optional[pd.Series] = None,
    prior_label: str = 'implied'
) -> Dict:
    """
    Black-Litterman "light" portfolio construction, numpy-only.

    Steps:
        1. Estimate the covariance with shrinkage (default 'lw').
        2. Build the prior mean: reverse-optimize implied equilibrium returns
           from a reference portfolio (default: equal weight). If
           ``prior_returns`` is given (e.g. a Fama-French factor-model prior),
           it replaces the reverse-optimized prior entirely.
        3. If absolute *views* (asset -> target annual excess return) are
           given, blend them into the posterior mean with the standard
           closed-form update:
               mu_post = pi + tau*S P' (P tau*S P' + Omega)^-1 (q - P pi)
        4. Return the tangency weights on (mu_post, S).

    Without views the posterior equals the prior. With the default implied
    prior, the tangency portfolio reproduces the reference weights
    (*views are the only source of tilts*). When a factor-model ``prior_returns``
    is substituted, that clean reproduction property no longer holds - the prior
    itself now tilts the portfolio.

    ``view_confidence`` is the per-view error *variance* in annual return units;
    smaller values trust the view more (default is 0.0025, i.e. a 5% error
    standard deviation).

    Args:
        returns: Daily returns DataFrame (columns = assets)
        prior_weights: Reference portfolio for step 2 (default: equal weight)
        views: Mapping asset -> target annual excess return (decimal)
        view_confidence: Per-view error variance (float = same for all, or list)
        risk_aversion: Risk-aversion coefficient for step 2 (default: 2.5)
        tau: Confidence scale on the prior covariance (default: 0.05)
        cov_method: Covariance estimator for step 1 ('sample', 'lw' or 'factor')
        risk_free_rate: Annual risk-free rate (deflates the tangency weights)
        prior_returns: Optional annualized excess expected-return prior Series
            replacing the reverse-optimized implied returns (e.g. from
            factor_model.ff_expected_returns)
        prior_label: Label for the prior source ('implied', 'ff', ...)

    Returns:
        Dict with weights (Series, sum to 1), implied_returns, posterior_returns,
        prior_returns (the prior actually used), prior_label, prior_weights and
        cov (annualized)
    """
    ret = returns.dropna(how='any')
    if len(ret.columns) < 2:
        raise ValueError("Black-Litterman needs at least two assets.")

    cov = covariance_estimator(ret, method=cov_method) * 252.0
    n = len(cov)

    if prior_weights is None:
        prior_weights = pd.Series(np.full(n, 1.0 / n), index=cov.index)
    else:
        prior_weights = prior_weights.reindex(cov.index).fillna(0.0)

    pi_implied = implied_returns(cov, prior_weights, risk_aversion=risk_aversion)

    if prior_returns is not None:
        pi = prior_returns.reindex(cov.index).fillna(0.0)
    else:
        pi = pi_implied

    if views:
        assets = sorted(views.keys())
        for a in assets:
            if a not in cov.index:
                raise ValueError(f"View asset '{a}' is not in the portfolio.")
        P = np.zeros((len(assets), n))
        Q = np.zeros(len(assets))
        for row, a in enumerate(assets):
            P[row, cov.index.get_loc(a)] = 1.0
            Q[row] = views[a]
        if view_confidence is None:
            omega_diag = np.full(len(assets), 0.0025)
        elif isinstance(view_confidence, (int, float)):
            omega_diag = np.full(len(assets), float(view_confidence))
        else:
            omega_diag = np.asarray(view_confidence, dtype=float)
            if len(omega_diag) != len(assets):
                raise ValueError("view_confidence must match the number of views.")
        Omega = np.diag(omega_diag)
        tau_s = tau * cov.values
        A = P @ tau_s @ P.T + Omega
        post = pi.values + tau_s @ P.T @ _safe_inv(A) @ (Q - P @ pi.values)
        posterior_returns = pd.Series(post, index=cov.index)
    else:
        posterior_returns = pi

    weights = tangency_portfolio(
        ret,
        risk_free_rate=risk_free_rate,
        method='sample',
        mu=posterior_returns,
        cov=cov
    )

    return {
        'weights': weights,
        'implied_returns': pi_implied,
        'prior_returns': pi,
        'prior_label': prior_label,
        'posterior_returns': posterior_returns,
        'prior_weights': prior_weights,
        'cov': cov,
    }


def black_litterman_weights(
    returns: pd.DataFrame,
    prior_weights: Optional[pd.Series] = None,
    views: Optional[Dict[str, float]] = None,
    **kwargs
) -> pd.Series:
    """Convenience wrapper returning only the BL weights Series."""
    return black_litterman(returns, prior_weights=prior_weights, views=views, **kwargs)['weights']


def build_portfolio_report(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    bl: Optional[Dict] = None,
    cov_method: str = 'lw',
    ff: Optional[Dict] = None
) -> Dict:
    """
    Build the full display-ready portfolio report.

    Args:
        returns: Daily returns DataFrame (columns = assets)
        risk_free_rate: Annual risk-free rate as decimal
        bl: Optional result dict from black_litterman(); when given a
            'black_litterman' weights row and the implied/posterior expected
            returns are attached to the report
        cov_method: Covariance estimator for the mean-variance builders
            ('sample', 'lw' or 'factor')
        ff: Optional dict from factor_model.ff_betas()/factor_premia()/... used
            to render the Fama-French exposure panel and (when combined with
            black_litterman) a factor-model expected-return prior

    Returns:
        Dict with weights (DataFrame), stats, correlation, frontier, stress,
        optional black_litterman detail and optional factor model / ff panels
    """
    ret = returns.dropna(how='any')
    if len(ret.columns) < 2:
        raise ValueError("Portfolio mode needs at least two tickers.")

    tickers = list(ret.columns)

    min_w = min_variance(ret, method=cov_method)
    tan_w = tangency_portfolio(ret, risk_free_rate=risk_free_rate, method=cov_method)
    eff_w = efficient_portfolio(ret, risk_free_rate=risk_free_rate, method=cov_method)

    blocks: Dict[str, pd.Series] = {
        'min_variance': min_w,
        'tangency': tan_w,
        'efficient': eff_w,
    }
    if bl is not None:
        blocks['black_litterman'] = bl['weights']

    weights = pd.concat(blocks, axis=1).T
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

    report: Dict = {
        'weights': weights,
        'tickers': tickers,
        'stats': eff_stats,
        'frontier_risk': frontier_risk,
        'correlation': corr,
        'stress': stress,
        'cov_method': cov_method,
    }
    if bl is not None:
        report['bl'] = {
            'implied_returns': {
                t: float(v) for t, v in bl['implied_returns'].items()
            },
            'prior_returns': {
                t: float(v) for t, v in bl['prior_returns'].items()
            },
            'prior_label': bl['prior_label'],
            'posterior_returns': {
                t: float(v) for t, v in bl['posterior_returns'].items()
            },
            'prior_weights': {
                t: float(v) for t, v in bl['prior_weights'].items()
            },
        }
    if cov_method == 'factor':
        from .factor_model import pca_factor_model
        try:
            fm = pca_factor_model(ret)
            report['factor_model'] = {
                'n_factors': fm['n_factors'],
                'explained_variance': fm['explained_variance'],
                'idiosyncratic_std': {
                    t: float(v) for t, v in fm['idiosyncratic_std'].items()
                },
            }
        except ValueError:
            report['factor_model'] = None
    if ff is not None:
        report['ff'] = {
            'betas': ff['betas'].to_dict(orient='index'),
            'alpha_annual': {
                t: float(v) for t, v in ff['alpha_annual'].items()
            },
            'premia': {
                f: float(v) for f, v in ff['premia'].items()
            },
        }
    return report


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