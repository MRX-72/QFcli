"""Tests for the statistics module (no network required)."""

import numpy as np
import pandas as pd
import pytest

from quant_finance.statistics import (
    sharpe_significance,
    sharpe_bootstrap_ci,
    ljung_box,
    jarque_bera,
    adf_test,
    norm_pvalue_2sided,
)


def _returns(seed=0, n=500, mu=0.0002, sigma=0.01):
    rng = np.random.default_rng(seed)
    return pd.Series(mu + rng.normal(0, sigma, n))


class TestNormPvalue:
    def test_z_zero_p_half(self):
        assert norm_pvalue_2sided(0.0) == pytest.approx(1.0)

    def test_large_z_small_p(self):
        assert norm_pvalue_2sided(6.0) < 1e-6


class TestSharpeSignificance:
    def test_inconsistent_sharpe_still_runs(self):
        out = sharpe_significance(_returns())
        assert 'statistic' in out and 'p_value' in out and 'verdict' in out

    def test_positive_mean_sharpe_significant(self):
        out = sharpe_significance(_returns(mu=0.002, sigma=0.01, n=500))
        assert out['verdict'] == 'SIGNIFICANT'

    def test_insufficient_data(self):
        out = sharpe_significance(pd.Series([0.01] * 10))
        assert out['verdict'] == 'INSUFFICIENT DATA'

    def test_constant_returns_zero_vol(self):
        out = sharpe_significance(pd.Series(np.full(100, 0.001)))
        # zero vol guarded -> returns nonsense p-value but no crash
        assert 'verdict' in out


class TestSharpeBootstrap:
    def test_ci_contains_observed(self):
        out = sharpe_bootstrap_ci(_returns(), n_boot=500, seed=1)
        assert out['ci_low'] <= out['observed'] <= out['ci_high']

    def test_wider_ci_for_noisier_returns(self):
        low_noise = sharpe_bootstrap_ci(_returns(sigma=0.005), n_boot=500, seed=2)
        high_noise = sharpe_bootstrap_ci(_returns(sigma=0.03), n_boot=500, seed=2)
        assert (high_noise['ci_high'] - high_noise['ci_low']) > (low_noise['ci_high'] - low_noise['ci_low'])


class TestLjungBox:
    def test_white_noise_gives_large_p(self):
        out = ljung_box(_returns(seed=3))
        assert out['verdict'] == 'NO AUTOCORRELATION'
        assert out['p_value'] > 0.05

    def test_autocorrelated_series_detected(self):
        rng = np.random.default_rng(4)
        # AR(1) with strong persistence
        r = [0.0]
        for _ in range(400):
            r.append(0.9 * r[-1] + rng.normal(0, 0.01))
        out = ljung_box(pd.Series(r), lags=10)
        assert out['verdict'] == 'AUTOCORRELATED'

    def test_insufficient_data(self):
        out = ljung_box(pd.Series([0.01] * 5), lags=10)
        assert out['verdict'] == 'INSUFFICIENT DATA'


class TestJarqueBera:
    def test_fat_tail_rejected_normal(self):
        rng = np.random.default_rng(5)
        fat = pd.Series(rng.standard_t(df=4, size=500))
        out = jarque_bera(fat)
        assert out['verdict'] == 'NOT NORMAL'
        assert out['p_value'] < 0.05

    def test_normal_not_rejected(self):
        out = jarque_bera(_returns(seed=6))
        assert out['p_value'] > 0.01  # loose bound; t-distrib false positives are possible


class TestADF:
    def test_random_walk_non_stationary(self):
        rng = np.random.default_rng(9)
        y = np.cumsum(rng.normal(0, 0.01, 300))
        out = adf_test(pd.Series(y))
        assert out['verdict'] == 'NON-STATIONARY (unit root)'

    def test_stationary_series(self):
        rng = np.random.default_rng(9)
        y = rng.normal(0, 0.01, 400).cumsum()
        # mean-reverting slowly (AR(1) with phi ~ 0.9) vs white noise:
        w = rng.normal(0, 0.01, 400)
        out = adf_test(pd.Series(w))
        assert out['verdict'] == 'STATIONARY (no unit root)'

    def test_insufficient_data(self):
        out = adf_test(pd.Series(np.ones(5)))
        assert out['verdict'] == 'INSUFFICIENT DATA'

    def test_output_shape(self):
        out = adf_test(_returns())
        assert set(out.keys()) == {'statistic', 'critical_values', 'verdict', 'used_lags'}