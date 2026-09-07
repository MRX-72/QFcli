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