"""
Factor Model Module
Statistical (PCA) risk models and an optional Fama-French overlay.

Two complementary layers, both pure numpy/pandas:

1. PCA risk model (offline).  A "statistical factor model" decomposes the
   return covariance of a panel into a low-rank *common* component driven by
   K latent factors plus an *idiosyncratic* diagonal:
       Sigma = L * Sigma_f * L' + Psi
   The loading matrix L and factor returns are estimated by an eigen
   decomposition of the cross-asset correlation matrix (principal components).
   The resulting factor-model covariance is PSD by construction and is a
   classical alternative to shrinkage covariance in mean-variance inputs.

   Honest caveat: PCA factors are atheoretical. They describe common *variance*;
   they carry no economic label and say nothing about expected returns. Use them
   to de-noise covariance, not to explain why assets move together.

2. Fama-French overlay (optional, network).  fetch_ff_factors() downloads the
   daily Fama-French market/size/value research factors (Ken French Data
   Library), ff_betas() regresses each asset on those factors, and
   ff_expected_returns() converts factor exposures times historical premia into
   an expected-return prior - a factor-model counterpart to the Black-Litterman
   reverse-optimized prior.

The Fama-French download is the only network dependency in this module and is
*optional*: everything else runs offline on synthetic data.
"""

import io
import os
import pathlib
import time
import urllib.request
import zipfile
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

FF_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)

FF_CACHE_DIR = os.environ.get(
    "QFCLI_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".qfcli", "cache"),
)
FF_CACHE_TTL_SECONDS = int(os.environ.get("QFCLI_CACHE_TTL", str(6 * 60 * 60)))


def pca_factor_model(
    returns: pd.DataFrame,
    n_factors: Optional[int] = None,
    min_expl_var: float = 0.8
) -> Dict:
    """
    Fit a K-factor statistical risk model via PCA of the correlation matrix.

    Returns (all in daily return units):
        n_factors             - number of factors retained
        eigenvalues           - sorted eigenvalues of the correlation matrix
        explained_variance    - fraction of total variance explained (cumulative)
        loadings              - factor loadings L (returns per unit factor)
        factor_returns        - the K latent factor return series (unit variance)
        factor_covariance     - diagonal factor covariance (eigenvalues)
        idiosyncratic_std     - daily idiosyncratic standard deviation per asset
        common_covariance     - L Sigma_f L' (rank-K common component)
        covariance            - full model covariance = common + idiosyncratic
        correlation           - model correlation matrix (PSD)

    Args:
        returns: Daily returns DataFrame (columns = assets)
        n_factors: Number of factors to retain. When None, the smallest k is
            chosen so that the retained eigenvalues explain at least
            ``min_expl_var`` (default 0.8) of total variance, capped at
            p-1 factors.
        min_expl_var: Explained-variance cutoff used when n_factors is None.

    Raises:
        ValueError: if there are fewer than two assets or too few observations.
    """
    ret = returns.dropna(how='any')
    if len(ret.columns) < 2:
        raise ValueError("A factor model needs at least two assets.")
    if len(ret) < 3:
        raise ValueError("Not enough overlapping observations for a factor model.")

    n_assets = len(ret.columns)
    std = ret.std(ddof=1).replace(0, np.nan)
    z = (ret - ret.mean()) / std
    z = z.fillna(0.0)

    corr = ret.corr().values
    eigvals, eigvecs = np.linalg.eigh(corr)  # ascending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order].copy()
    eigvecs = eigvecs[:, order].copy()
    eigvals = np.maximum(eigvals, 0.0)

    total = float(eigvals.sum()) if eigvals.sum() > 0 else 1.0
    if n_factors is None:
        cum = np.cumsum(eigvals) / total
        k = int(np.searchsorted(cum, min_expl_var) + 1)
        k = int(min(max(k, 1), n_assets - 1))
    else:
        k = int(min(max(int(n_factors), 1), n_assets))

    # standardized loadings B with B B' ≈ corr; factor scores have unit variance
    sqrt_l = np.sqrt(eigvals[:k])
    B = eigvecs[:, :k] * sqrt_l  # p x k, unit-variance factor loadings (std. space)
    scores = z.values @ eigvecs[:, :k] / sqrt_l[np.newaxis, :]  # T x k

    psi = np.maximum(1.0 - np.sum(B * B, axis=1), 0.0)  # idiosyncratic var, std. space

    D = np.asarray(std.fillna(0.0).to_numpy(dtype=float))
    L = B * D[:, None]                    # loadings in return units
    cov_factor = np.diag(eigvals[:k])     # factor covariance
    common = L @ cov_factor @ L.T
    idio_var_daily = psi * D * D
    model = common + np.diag(idio_var_daily)
    model = (model + model.T) / 2.0

    cov = pd.DataFrame(model, index=ret.columns, columns=ret.columns)
    corr_model = _cov_to_corr(cov)

    return {
        'n_factors': k,
        'eigenvalues': list(map(float, eigvals[:k])),
        'explained_variance': float(cum[k - 1]) if n_factors is None
                              else float(np.cumsum(eigvals)[min(k, len(eigvals)) - 1] / total),
        'loadings': pd.DataFrame(L, index=ret.columns, columns=[f'F{i+1}' for i in range(k)]),
        'factor_returns': pd.DataFrame(scores, index=ret.index,
                                       columns=[f'F{i+1}' for i in range(k)]),
        'factor_covariance': cov_factor,
        'idiosyncratic_std': pd.Series(np.sqrt(idio_var_daily), index=ret.columns),
        'common_covariance': pd.DataFrame(common, index=ret.columns, columns=ret.columns),
        'covariance': cov,
        'correlation': corr_model,
    }


def factor_model_cov(
    returns: pd.DataFrame,
    n_factors: Optional[int] = None,
    min_expl_var: float = 0.8
) -> pd.DataFrame:
    """Convenience wrapper returning only the factor-model covariance matrix."""
    return pca_factor_model(returns, n_factors=n_factors,
                            min_expl_var=min_expl_var)['covariance']


def _cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    d = np.sqrt(np.diag(cov.to_numpy()))
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = cov.to_numpy() / np.outer(d, d)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def _ff_cache_path() -> pathlib.Path:
    return pathlib.Path(FF_CACHE_DIR) / "ff_factors.pkl"


def _load_cached_factors() -> Optional[pd.DataFrame]:
    path = _ff_cache_path()
    if not path.exists():
        return None
    try:
        if time.time() - path.stat().st_mtime > FF_CACHE_TTL_SECONDS:
            return None
        return pd.read_pickle(path)
    except Exception:
        return None


def _save_cached_factors(df: pd.DataFrame) -> None:
    try:
        path = _ff_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(df, path)
    except Exception:
        pass


def parse_ff_csv(text: str) -> pd.DataFrame:
    """
    Parse the Ken French daily factors text (percent format) into a DataFrame.

    The Ken French library publishes the daily research factors as continuation
    CSV where the first row is a header (e.g. ``, Mkt-RF, SMB, HML, RF``) and
    each following row starts with a YYYYMMDD date, values in *percent*:
        ``19260701,0.10,-0.05,0.03,0.005``

    Only the rows between the header and the trailing empty table are kept.
    Values are converted to decimals (percent / 100) and the first column is
    parsed into a DatetimeIndex.

    Args:
        text: Raw text content of the factor file (the unzipped CSV)

    Returns:
        DataFrame of daily factor excess returns plus 'RF' (columns
        'Mkt-RF', 'SMB', 'HML', 'RF'), datetime-indexed

    Raises:
        ValueError: if no recognisable header/data rows are found
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header_idx = None
    header_cells: List[str] = []
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.split(',')]
        if cells[0].startswith(',') or cells[0] == '':
            lowered = [c.lower() for c in cells]
            if any('mkt' in c or 'rf' in c for c in lowered):
                header_idx = i
                header_cells = cells
                break
    if header_idx is None:
        raise ValueError("Could not find the Fama-French factor header.")

    rows: List[List[str]] = []
    for line in lines[header_idx + 1:]:
        cells = [c.strip() for c in line.split(',')]
        if len(cells) < 2:
            break
        if not cells[0].isdigit():
            break
        rows.append(cells)

    if not rows:
        raise ValueError("No Fama-French data rows found after the header.")

    data = np.array([[float(c) for c in row[1:]] for row in rows])
    frame = pd.DataFrame(
        data / 100.0,
        columns=[c.strip() for c in header_cells[1:len(header_cells)][: data.shape[1]]],
        index=pd.to_datetime([r[0] for r in rows], format='%Y%m%d'),
    )
    if len(frame.columns) < 2 or 'RF' not in frame.columns.astype(str):
        raise ValueError("Unexpected Fama-French factor layout.")
    return frame


def fetch_ff_factors(use_cache: bool = True) -> pd.DataFrame:
    """
    Download and parse the daily Fama-French 3-factor research data (network).

    Downloads the Ken French CSV zip, extracts the daily factors, converts the
    percent values to decimals and caches the result on disk (6h TTL, shared
    with the stock cache directory).

    Args:
        use_cache: Use/save the on-disk cache

    Returns:
        DataFrame of daily 'Mkt-RF', 'SMB', 'HML', 'RF'

    Raises:
        ValueError: if the download or parse fails
    """
    if use_cache:
        cached = _load_cached_factors()
        if cached is not None:
            return cached
    try:
        with urllib.request.urlopen(FF_URL, timeout=30) as resp:
            raw = resp.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = zf.namelist()[0]
            text = zf.read(name).decode('utf-8', errors='replace')
        frame = parse_ff_csv(text)
    except Exception as e:
        raise ValueError(
            f"Could not fetch Fama-French factors from {FF_URL}: {e}"
        ) from e
    if use_cache:
        _save_cached_factors(frame)
    return frame


def ff_betas(returns: pd.DataFrame, factors: pd.DataFrame) -> Dict:
    """
    Regress each asset's excess returns on the Fama-French factors.

    For each asset: (r_t - RF_t) = alpha + sum_k beta_k * F_kt + e_t (OLS); a
    'RF' column, if present, is subtracted from the dependent variable and is
    never treated as a factor. Returns the per-factor betas and the annualized
    intercept (alpha).

    Args:
        returns: Daily returns DataFrame (columns = assets)
        factors: Daily factor returns DataFrame (columns e.g. 'Mkt-RF', 'SMB',
            'HML', plus optional 'RF'; values already decimal)

    Returns:
        Dict with 'betas' (DataFrame, rows = factors, cols = assets) and
        'alpha_annual' (Series per asset)
    """
    if getattr(returns.index, 'tz', None) is not None:
        returns = returns.copy()
        returns.index = returns.index.tz_localize(None)
    common = returns.index.intersection(factors.index)
    if len(common) < len(factors.columns) + 3:
        raise ValueError("Not enough overlapping observations for factor betas.")

    rf_col = next((c for c in factors.columns if str(c).upper() == 'RF'), None)
    left = [c for c in factors.columns if c != rf_col]
    X = factors.loc[common[0]: common[-1], left].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(X)), X])
    cols = list(left)

    y0 = returns.loc[common[0]: common[-1]]
    if rf_col is not None:
        rf = factors.loc[common[0]: common[-1], rf_col].to_numpy(dtype=float)
    else:
        rf = np.zeros(len(X))

    betas = {}
    alpha = {}
    for asset in returns.columns:
        y = y0[asset].to_numpy(dtype=float) - rf
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha[asset] = float(coef[0] * 252)   # annualized excess-return alpha
        betas[asset] = {f: float(b) for f, b in zip(cols, coef[1:])}

    beta_df = pd.DataFrame(betas).T   # rows = assets, cols = factors
    return {
        'betas': beta_df.T,           # rows = factors, cols = assets
        'alpha_annual': pd.Series(alpha),
    }


def factor_premia(factors: pd.DataFrame, annualize: float = 252) -> pd.Series:
    """
    Historical annualized mean excess returns of each factor.

    Args:
        factors: Daily factor returns DataFrame
        annualize: Trading days per year (default 252)

    Returns:
        Series of annualized premia per factor (excluding an 'RF' column if one
        is present, since RF is a rate, not a risk premia)
    """
    cols = [c for c in factors.columns if str(c).upper() != 'RF']
    return factors[cols].mean() * annualize


def ff_expected_returns(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    premia: Optional[pd.Series] = None,
    rf: float = 0.0
) -> pd.Series:
    """
    Factor-model expected returns: E[r_i] = rf + sum_k beta_ik * premia_k.

    Uses the asset's OLS betas against the Fama-French factors and either the
    supplied risk premia or the historical annualized factor means.

    Args:
        returns: Daily returns DataFrame (columns = assets)
        factors: Daily factor returns DataFrame
        premia: Optional annualized premia Series (index matching factor names)
        rf: Annual risk-free rate to add back (default 0.0)

    Returns:
        Series of annualized expected returns per asset
    """
    fit = ff_betas(returns, factors)
    if premia is None:
        premia = factor_premia(factors)
    premia = premia.reindex(fit['betas'].index).fillna(0.0)
    expected = fit['betas'].values.T @ premia.values + rf
    return pd.Series(expected, index=returns.columns)