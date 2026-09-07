"""
Statistics Module
Classical statistical tests used in quantitative work.

Implements dependency-free versions (plain numpy/scipy-free) of:

    * sharpe_significance  - Jobson & Korkie z-test on the Sharpe ratio
    * sharpe_bootstrap_ci  - percentile bootstrap confidence interval
    * ljung_box            - autocorrelation portmanteau test (Q statistic)
    * jarque_bera          - normality test (skewness/kurtosis)
    * adf_test             - Augmented Dickey-Fuller stationarity test

Each function returns a dict with 'statistic', 'p_value' (or critical
values for ADF), and a human 'verdict' string. All are deterministic and
testable offline against synthetic data.
"""

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd


def norm_pvalue_2sided(z: float) -> float:
    """Two-sided p-value for a standard normal z-statistic."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def _clean(returns) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    return arr[~np.isnan(arr)]


def sharpe_significance(returns, risk_free_rate: float = 0.0, annualize: bool = True) -> Dict:
    """
    Jobson & Korkie (1981) test that the Sharpe ratio equals zero.

    The standard error incorporates skewness and excess kurtosis of returns:

        SE = sqrt((1 - skew*SR + (kurt-1)/4 * SR^2) / T)

    Args:
        returns: Series/array of daily returns
        risk_free_rate: Annual risk-free rate as decimal
        annualize: Use the annualized ratio in the z-stat (default True)

    Returns:
        Dict with statistic, p_value, verdict
    """
    r = _clean(returns)
    T = len(r)
    if T < 30:
        return {'statistic': np.nan, 'p_value': np.nan, 'verdict': 'INSUFFICIENT DATA'}

    daily_rf = risk_free_rate / 252 if annualize else risk_free_rate
    mu = r.mean() - daily_rf
    sigma = r.std(ddof=1)
    if sigma == 0:
        return {'statistic': np.nan, 'p_value': 1.0, 'verdict': 'NOT SIGNIFICANT (zero vol)'}

    sr = mu / sigma

    skew = float(pd.Series(r).skew())
    kurt_excess = float(pd.Series(r).kurt())  # pandas: excess kurtosis (gamma4 - 3)

    # Lo (2002) i.i.d. variance for the Sharpe estimator:
    #   Var(SR) ~ (1 - skew*SR + (kurtosis - 1)/4 * SR^2) / T
    # where kurtosis here is the raw fourth-moment measure (gamma4).
    se = np.sqrt((1 - skew * sr + (kurt_excess + 2) / 4 * sr ** 2) / T)
    if se <= 0:
        se = 1e-12

    z = sr / se
    p = norm_pvalue_2sided(z)
    p = max(min(p, 1.0), 0.0)

    verdict = 'SIGNIFICANT' if p < 0.05 else 'NOT SIGNIFICANT'
    return {'statistic': float(z), 'p_value': float(p), 'verdict': verdict}


def sharpe_bootstrap_ci(
    returns,
    risk_free_rate: float = 0.0,
    n_boot: int = 2000,
    seed: int = 42,
    annualize: bool = True
) -> Dict:
    """
    Percentile bootstrap confidence interval for the Sharpe ratio.

    Args:
        returns: Series/array of daily returns
        risk_free_rate: Annual risk-free rate as decimal
        n_boot: Number of bootstrap resamples
        seed: Reproducible RNG seed
        annualize: Scale to annualized Sharpe (default True)

    Returns:
        Dict with observed, ci_low, ci_high
    """
    r = _clean(returns)
    T = len(r)
    if T < 30:
        return {'observed': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'n_boot': n_boot}

    daily_rf = risk_free_rate / 252 if annualize else risk_free_rate
    rng = np.random.default_rng(seed)

    def _sharpe(sample):
        sigma = sample.std(ddof=1)
        if sigma == 0:
            return 0.0
        sr = (sample.mean() - daily_rf) / sigma
        return sr * np.sqrt(252) if annualize else sr

    observed = _sharpe(r)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = _sharpe(rng.choice(r, size=T, replace=True))

    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    return {
        'observed': float(observed),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'n_boot': n_boot,
    }


def ljung_box(returns, lags: int = 10) -> Dict:
    """
    Ljung-Box portmanteau test for serial autocorrelation (white noise H0).

    Q = T(T+2) * sum_{k=1}^{lags} rho_k^2 / (T - k)

    Args:
        returns: Series/array of daily returns
        lags: Number of lags to test (default: 10)

    Returns:
        Dict with statistic (Q), p_value, verdict
    """
    r = _clean(returns)
    T = len(r)
    if T <= lags:
        return {'statistic': np.nan, 'p_value': np.nan, 'verdict': 'INSUFFICIENT DATA'}

    r = r - r.mean()
    acf = np.correlate(r, r, mode='full')[T - 1:]
    acf /= acf[0]
    rho = acf[1:lags + 1]

    q = T * (T + 2) * np.sum(rho ** 2 / (T - np.arange(1, lags + 1)))
    p = 1 - _chi2_cdf(q, lags)

    verdict = 'AUTOCORRELATED' if p < 0.05 else 'NO AUTOCORRELATION'
    return {'statistic': float(q), 'p_value': float(p), 'verdict': verdict}


def jarque_bera(returns) -> Dict:
    """
    Jarque-Bera normality test on skewness and excess kurtosis.

    JB = (T/6) * (S^2 + (K-3)^2/4)

    Args:
        returns: Series/array of daily returns

    Returns:
        Dict with statistic, p_value, verdict
    """
    r = _clean(returns)
    T = len(r)
    if T < 8:
        return {'statistic': np.nan, 'p_value': np.nan, 'verdict': 'INSUFFICIENT DATA'}

    s = float(pd.Series(r).skew())
    k = float(pd.Series(r).kurt())  # excess kurtosis

    jb = T / 6 * (s ** 2 + (k) ** 2 / 4)
    p = 1 - _chi2_cdf(jb, 2)

    verdict = 'NOT NORMAL' if p < 0.05 else 'NORMAL (cannot reject)'
    return {'statistic': float(jb), 'p_value': float(p), 'verdict': verdict}


def adf_test(series, max_lag: int = 10) -> Dict:
    """
    Augmented Dickey-Fuller stationarity test (intercept-only model).

    Runs the regression:
        dY_t = c + phi Y_{t-1} + sum_{i=1}^{p} beta_i dY_{t-i} + eps_t

    and reports the tau statistic (phi / SE(phi)) against tabulated critical
    values. Rejecting the null (unit root) means the series is stationary.

    Args:
        series: Series/array of price-like data
        max_lag: Max lag order considered for the augmentation (default 10)

    Returns:
        Dict with statistic, critical_values {1%,5%,10%}, verdict, used_lags
    """
    y = _clean(series)
    n = len(y)
    if n < 30:
        return {'statistic': np.nan, 'critical_values': {}, 'verdict': 'INSUFFICIENT DATA', 'used_lags': 0}

    d = np.diff(y)
    y_lag = y[:-1]

    best = None  # (s2, tau, lag)
    n = len(d)
    for lag in range(max_lag + 1):
        if lag > n - 2:
            break
        ys = d[lag:]
        cols = [y_lag[lag:], np.ones(len(ys))]
        for k in range(1, lag + 1):
            cols.append(d[lag - k:n - k])
        X = np.column_stack(cols)
        if len(ys) < 3:
            continue
        try:
            beta_vec, _, _, _ = np.linalg.lstsq(X, ys, rcond=None)
            resid = ys - X @ beta_vec
            df = len(ys) - X.shape[1]
            if df <= 0:
                continue
            s2 = resid @ resid / df
            # tau = phi_hat / SE(phi_hat), SE from (X'X)^-1 * s2
            xtx_inv00 = np.linalg.inv(X.T @ X)[0, 0]
            se_phi = np.sqrt(abs(s2) * xtx_inv00)
            tau = beta_vec[0] / se_phi
            if best is None or s2 < best[0]:
                best = (s2, tau, lag)
        except Exception:
            continue

    if best is None:
        return {'statistic': np.nan, 'critical_values': {}, 'verdict': 'INSUFFICIENT DATA', 'used_lags': 0}

    _, tau, used_lags = best
    critical = {'1%': -3.43, '5%': -2.86, '10%': -2.57}

    if tau < critical['5%']:
        verdict = 'STATIONARY (no unit root)'
    elif tau < critical['10%']:
        verdict = 'WEAKLY STATIONARY'
    else:
        verdict = 'NON-STATIONARY (unit root)'

    return {
        'statistic': float(tau),
        'critical_values': critical,
        'verdict': verdict,
        'used_lags': used_lags,
    }


# ----------------------------------------------------------------------------
# chi-squared survival function via numerical integration (regularized gamma)
# ----------------------------------------------------------------------------

def _gamma_p(a: float, x: float, iters: int = 200) -> float:
    """
    Lower regularized incomplete gamma P(a, x), via continued fraction.

    Relative error is well under 1e-9 for the range we use, and it removes
    any SciPy dependency.
    """
    if x < a + 1:
        # power series
        ap = a
        del_, sum_ = 1.0 / a, 1.0 / a
        for _ in range(iters):
            ap += 1
            del_ *= x / ap
            sum_ += del_
            if abs(del_) < abs(sum_) * 1e-12:
                break
        fac = np.exp(-x + a * np.log(x) - _lgamma(a))
        return sum_ * fac if sum_ < 1.0 else min(1.0, sum_ * fac)
    else:
        # continued fraction
        b = x + 1.0 - a
        c = 1.0 / 1e-30
        d = 1.0 / b
        h = d
        for i in range(1, iters):
            an = -i * (i - a)
            b += 2
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            del_ = d * c
            h *= del_
            if abs(del_ - 1.0) < 1e-12:
                break
        fac = np.exp(-x + a * np.log(x) - _lgamma(a))
        return 1.0 - fac * h if fac * h < 1.0 else max(0.0, 1.0 - fac * h)


def _lgamma(x: float) -> float:
    """Natural log-gamma via Lanczos approximation."""
    coefficients = [
        676.5203681218851, -1259.1392167224028, 771.32342877765313,
        -176.61502916214059, 12.507343278686905, -0.13857109526572012,
        9.9843695780195716e-6, 1.5056327351493116e-7,
    ]
    x -= 1
    a = 0.99999999999980993
    for i, c in enumerate(coefficients):
        a += c / (x + i + 1)
    t = x + len(coefficients) - 0.5
    return 0.5 * np.log(2 * np.pi) + (x + 0.5) * np.log(t) - t + np.log(a)


def _chi2_cdf(x: float, k: int) -> float:
    """CDF of the chi-squared distribution with k degrees of freedom."""
    if x <= 0:
        return 0.0
    return _gamma_p(k / 2.0, x / 2.0)