"""Tests for the portfolio module (no network required)."""

import numpy as np
import pandas as pd
import pytest

from quant_finance.portfolio import (
    returns_matrix,
    min_variance,
    tangency_portfolio,
    efficient_portfolio,
    portfolio_stats,
    efficient_frontier,
    correlation_matrix,
    covariance_estimator,
    shrinkage_covariance,
    implied_returns,
    black_litterman,
    black_litterman_weights,
    stress_test,
    build_portfolio_report,
)


def _returns(n=252, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    a = 0.0004 + rng.normal(0, 0.01, n)
    b = 0.0008 + rng.normal(0, 0.015, n)
    c = -0.0002 + rng.normal(0, 0.008, n)
    return pd.DataFrame({'A': a, 'B': b, 'C': c}, index=idx)


class TestReturnsMatrix:
    def test_aligns_common_dates(self):
        df1 = pd.DataFrame({'Close': np.linspace(100, 110, 50)},
                           index=pd.date_range("2024-01-01", periods=50, freq="B"))
        df2 = pd.DataFrame({'Close': np.linspace(50, 40, 50)},
                           index=pd.date_range("2024-01-01", periods=50, freq="B"))
        mat = returns_matrix([df1, df2])
        assert mat.shape[0] == 49
        assert mat.shape[1] == 2


class TestWeights:
    def test_min_variance_sums_to_one(self):
        w = min_variance(_returns())
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_min_variance_zero_weights_for_identical_assets(self):
        # two identical assets -> min variance splits evenly
        rng = np.random.default_rng(1)
        idx = pd.date_range("2024-01-01", periods=252, freq="B")
        r = rng.normal(0.001, 0.01, 252)
        df = pd.DataFrame({'X': r, 'Y': r}, index=idx)
        w = min_variance(df)
        assert w['X'] == pytest.approx(0.5, abs=1e-6)

    def test_tangency_sums_to_one(self):
        w = tangency_portfolio(_returns(), risk_free_rate=0.02)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_efficient_portfolio_long_only_no_negatives(self):
        w = efficient_portfolio(_returns(), risk_free_rate=0.04)
        assert (w >= -1e-9).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_efficient_allows_shorts(self):
        w = efficient_portfolio(_returns(), risk_free_rate=0.04, allow_short=True)
        assert (w.values < -1e-9).any()  # at least one short in tangency


class TestStats:
    def test_weights_drive_expected_return(self):
        df = _returns()
        w = pd.Series({'A': 1.0, 'B': 0.0, 'C': 0.0})
        stats = portfolio_stats(df, w)
        # all-in on A -> expected return == A's own annualized mean
        assert stats['expected_annual_return'] == pytest.approx(df['A'].mean() * 252, rel=1e-6)


class TestFrontier:
    def test_returns_correlated_series(self):
        f_ret, f_risk = efficient_frontier(_returns(), n_samples=300, seed=5)
        assert len(f_ret) > 1
        assert (f_risk > 0).all()

    def test_higher_return_on_upper_frontier(self):
        _, f_risk = efficient_frontier(_returns(), n_samples=300, seed=5)
        assert f_risk.is_monotonic_increasing


class TestCorrelation:
    def test_diagonal_ones(self):
        corr = correlation_matrix(_returns())
        assert corr.loc['A', 'A'] == pytest.approx(1.0)

    def test_symmetric(self):
        corr = correlation_matrix(_returns())
        assert np.allclose(corr.values, corr.values.T)


class TestShrinkageCovariance:
    def _near_collinear(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        true = np.array([[1.0, 0.99, 0.3],
                         [0.99, 1.02, 0.31],
                         [0.3, 0.31, 0.6]])
        chol = np.linalg.cholesky(true)
        X = rng.normal(size=(n, 3)) @ chol.T
        return pd.DataFrame(X, columns=['A', 'B', 'C'])

    def test_matches_sample_for_identical_assets_scale(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0.001, 0.02, 300)
        df = pd.DataFrame({'X': r, 'Y': r * 2})
        lw = shrinkage_covariance(df)
        # identical scaling: shrunk diag ~ sample diag (scaled by factor 4)
        assert lw.loc['Y', 'Y'] == pytest.approx(4 * lw.loc['X', 'X'], rel=0.1)

    def test_defines_identity_of_single_asset(self):
        df = _returns().iloc[:, [0]]
        lw = shrinkage_covariance(df)
        assert lw.loc['A', 'A'] == pytest.approx(df['A'].cov(df['A']), rel=1e-6)

    def test_improves_conditioning_when_near_collinear(self):
        df = self._near_collinear()
        sample = covariance_estimator(df, method='sample').values
        shrunk = shrinkage_covariance(df).values
        assert np.linalg.cond(shrunk) < np.linalg.cond(sample)

    def test_positive_semidefinite(self):
        df = self._near_collinear()
        eig = np.linalg.eigvalsh(shrinkage_covariance(df).values)
        assert eig.min() > -1e-10

    def test_shrinks_less_with_more_data(self):
        # deviation from the sample covariance grows with the shrinkage
        # intensity; more data should shrink less
        def _deviation(n):
            df = self._near_collinear(n=n, seed=3)
            sample = covariance_estimator(df, method='sample').values
            shrunk = shrinkage_covariance(df).values
            return float(np.linalg.norm(shrunk - sample))
        assert _deviation(120) > _deviation(1200)

    def test_beats_sample_when_target_is_close_to_truth(self):
        # near-diagonal truth: the constant-correlation target is close, so
        # shrinkage wins on average over seeds
        true = np.diag([1.0, 1.5, 0.8])
        err_sample, err_shrunk = [], []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            X = rng.normal(size=(250, 3)) @ np.linalg.cholesky(true).T
            df = pd.DataFrame(X, columns=['A', 'B', 'C'])
            err_sample.append(np.linalg.norm(covariance_estimator(df, 'sample').values - true))
            err_shrunk.append(np.linalg.norm(shrinkage_covariance(df).values - true))
        assert np.mean(err_shrunk) < np.mean(err_sample)

    def test_lw_default_used_by_min_variance(self):
        w = min_variance(_returns(seed=2))
        assert w.sum() == pytest.approx(1.0, abs=1e-9)


class TestImpliedReturns:
    def test_monotonic_in_risk_aversion(self):
        cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=['A', 'B'], columns=['A', 'B'])
        w = pd.Series({'A': 0.6, 'B': 0.4})
        low = implied_returns(cov, w, risk_aversion=1.0)
        high = implied_returns(cov, w, risk_aversion=3.0)
        assert (high > low).all()

    def test_known_value(self):
        cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=['A', 'B'], columns=['A', 'B'])
        w = pd.Series({'A': 0.5, 'B': 0.5})
        pi = implied_returns(cov, w, risk_aversion=2.0)
        expected = 2.0 * np.array([0.04 * 0.5 + 0.01 * 0.5, 0.01 * 0.5 + 0.09 * 0.5])
        assert np.allclose(pi.values, expected)


class TestBlackLitterman:
    def _df(self, n=500, seed=0):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        return pd.DataFrame(0.0004 + rng.normal(0, 0.01, size=(n, 3)),
                            columns=['A', 'B', 'C'], index=idx)

    def test_no_views_returns_reference_weights(self):
        bl = black_litterman(self._df())
        w = bl['weights']
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        # equal-weight prior reproduces ~equal weights under the model
        for asset in ('A', 'B', 'C'):
            assert w[asset] == pytest.approx(1 / 3, abs=0.05)

    def test_view_tilts_weights_monotonically(self):
        df = self._df(seed=3)
        base = black_litterman(df)['weights']['A']
        weights = []
        for v in (0.02, 0.08, 0.20):
            weights.append(black_litterman(df, views={'A': v})['weights']['A'])
        assert weights[0] < weights[1] < weights[2]
        assert weights[2] > base

    def test_view_pulls_posterior_toward_view(self):
        df = self._df(seed=9)
        bl_no = black_litterman(df)
        bl_with = black_litterman(df, views={'A': 0.06}, view_confidence=1e-4)
        assert bl_with['posterior_returns']['A'] > bl_no['implied_returns']['A']
        assert bl_with['posterior_returns']['A'] > 0.05   # close to the view

    def test_layout_keys(self):
        bl = black_litterman(self._df())
        for key in ('weights', 'implied_returns', 'posterior_returns',
                    'prior_weights', 'cov'):
            assert key in bl

    def test_unknown_view_asset_raises(self):
        with pytest.raises(ValueError):
            black_litterman(self._df(), views={'ZZZ': 0.1})

    def test_black_litterman_weight_helper(self):
        w = black_litterman_weights(self._df())
        assert w.sum() == pytest.approx(1.0, abs=1e-9)


class TestStress:
    def test_instant_shocks_are_exact(self):
        df = _returns()
        w = pd.Series({'A': 0.5, 'B': 0.5, 'C': 0.0})
        stress = stress_test(w, df)
        assert stress['instant_10pct'] == pytest.approx(-0.10)
        assert stress['instant_50pct'] == pytest.approx(-0.50)
        assert stress['worst_30d_window'] <= 0.0


class TestReport:
    def test_build_portfolio_report_structure(self):
        df = _returns(n=300)
        report = build_portfolio_report(df, risk_free_rate=0.03)
        assert set(report['weights'].index) == {'min_variance', 'tangency', 'efficient'}
        assert 'expected_annual_return' in report['stats']
        assert report['correlation'].shape == (3, 3)
        assert 'worst_30d_window' in report['stress']

    def test_build_requires_two_assets(self):
        df = _returns().iloc[:, :1]
        with pytest.raises(ValueError):
            build_portfolio_report(df)


class TestFactorCovariance:
    def test_factor_covariance_psd(self):
        df = _returns(n=500)
        cov = covariance_estimator(df, method='factor')
        eig = np.linalg.eigvalsh(cov.to_numpy())
        assert eig.min() >= -1e-9

    def test_factor_covariance_finite_aligned(self):
        df = _returns(n=400)
        cov = covariance_estimator(df, method='factor')
        assert cov.index.equals(df.columns)
        assert np.isfinite(cov.to_numpy()).all()

    def test_factor_covariance_reduced_rank_common(self):
        df = _returns(n=400)
        cov1 = covariance_estimator(df, method='factor', n_factors=1)
        cov3 = covariance_estimator(df, method='factor', n_factors=3)
        # more factors -> richer (closer-to-sample) estimate
        assert np.linalg.norm(cov3.to_numpy() - df.cov().to_numpy()) <= \
            np.linalg.norm(cov1.to_numpy() - df.cov().to_numpy()) + 1e-9

    def test_optimizers_accept_factor_method(self):
        df = _returns(n=500, seed=3)
        w_min = min_variance(df, method='factor')
        w_tan = tangency_portfolio(df, method='factor')
        assert w_min.sum() == pytest.approx(1.0, abs=1e-9)
        assert w_tan.sum() == pytest.approx(1.0, abs=1e-9)

    def test_unknown_method_raises_message_lists_factor(self):
        df = _returns()
        with pytest.raises(ValueError, match='factor'):
            covariance_estimator(df, method='bogus')


class TestPriorReturns:
    def test_custom_prior_replaces_implied_no_views(self):
        df = _returns(n=300, seed=5)
        prior = pd.Series({'A': 0.10, 'B': 0.05, 'C': 0.02})
        bl = black_litterman(df, prior_returns=prior, prior_label='ff')
        # no views -> posterior equals the supplied prior
        assert bl['prior_returns'].equals(prior)
        assert bl['posterior_returns'].equals(prior)
        assert bl['implied_returns'].index.equals(prior.index)
        assert not bl['implied_returns'].equals(prior)   # implied differs
        assert bl['prior_label'] == 'ff'
        assert bl['weights'].sum() == pytest.approx(1.0, abs=1e-9)

    def test_custom_prior_with_views_still_blends(self):
        df = _returns(n=300, seed=6)
        prior = pd.Series({'A': 0.03, 'B': 0.03, 'C': 0.03})
        bl = black_litterman(df, prior_returns=prior, views={'A': 0.06},
                             view_confidence=1e-4)
        assert bl['posterior_returns']['A'] > prior['A']
        assert bl['weights'].sum() == pytest.approx(1.0, abs=1e-9)

    def test_report_factor_panel(self):
        df = _returns(n=400)
        report = build_portfolio_report(df, cov_method='factor')
        assert report['cov_method'] == 'factor'
        assert report['factor_model']['n_factors'] >= 1
        assert 'idiosyncratic_std' in report['factor_model']

    def test_report_bl_prior_detail(self):
        df = _returns(n=300)
        import pandas as pd
        prior = pd.Series({'A': 0.06, 'B': 0.04, 'C': 0.02})
        bl = black_litterman(df, prior_returns=prior, prior_label='ff')
        report = build_portfolio_report(df, bl=bl)
        assert report['bl']['prior_label'] == 'ff'
        assert set(report['bl'].keys()) == {
            'implied_returns', 'prior_returns', 'prior_label',
            'posterior_returns', 'prior_weights',
        }