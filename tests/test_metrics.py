"""Tests for the metrics module (no network required)."""

import numpy as np
import pandas as pd
import pytest

from quant_finance.metrics import (
    cumulative_return,
    volatility,
    sharpe_ratio,
    max_drawdown,
    risk_score,
    average_return,
    daily_return_stats,
    sortino_ratio,
    value_at_risk,
    beta,
    yearly_returns,
    active_performance
)


def _series(values):
    return pd.Series(values, dtype=float)


class TestCumulativeReturn:
    def test_positive_gain(self):
        s = _series([100.0, 125.0])
        assert cumulative_return(s) == pytest.approx(0.25)

    def test_loss(self):
        s = _series([50.0, 40.0])
        assert cumulative_return(s) == pytest.approx(-0.2)

    def test_flat(self):
        s = _series([10.0, 10.0, 10.0])
        assert cumulative_return(s) == pytest.approx(0.0)

    def test_accepts_ndarray(self):
        assert cumulative_return(np.array([10.0, 12.0])) == pytest.approx(0.2)


class TestVolatility:
    def test_constant_returns_zero(self):
        assert volatility(_series([1.0, 1.0, 1.0])) == 0.0

    def test_annualized_scales_by_sqrt252(self):
        returns = pd.Series(np.full(100, 0.02))
        expected = returns.std() * np.sqrt(252)
        assert volatility(returns) == pytest.approx(expected)

    def test_non_annualized(self):
        returns = pd.Series([0.01, -0.01, 0.02, -0.02])
        assert volatility(returns, annualize=False) == pytest.approx(returns.std())


class TestSharpeRatio:
    def test_zero_volatility_returns_zero(self):
        assert sharpe_ratio(_series([1.0, 1.0, 1.0])) == 0.0

    def test_positive_excess_return(self):
        returns = pd.Series(np.full(252, 0.0005))
        sharpe = sharpe_ratio(returns, risk_free_rate=0.0)
        assert sharpe == pytest.approx(0.0005 / returns.std() * np.sqrt(252))

    def test_risk_free_rate_shifts_result(self):
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0.001, 0.01, 252))
        low = sharpe_ratio(returns, risk_free_rate=0.0)
        high = sharpe_ratio(returns, risk_free_rate=0.10)
        assert high < low


class TestMaxDrawdown:
    def test_monotonic_up_no_drawdown(self):
        s = _series([100.0, 110.0, 120.0])
        assert max_drawdown(s) == pytest.approx(0.0)

    def test_known_drawdown(self):
        # Peak 100 -> trough 70 => -30%
        s = _series([100.0, 120.0, 70.0, 90.0])
        assert max_drawdown(s) == pytest.approx(-0.4166667, rel=1e-3)


class TestRiskScore:
    def test_stable(self):
        assert risk_score(0.05) == "STABLE"

    def test_moderate(self):
        assert risk_score(0.20) == "MODERATE"

    def test_high(self):
        assert risk_score(0.45) == "HIGH RISK"

    def test_boundaries(self):
        assert risk_score(0.15) == "STABLE"
        assert risk_score(0.30) == "MODERATE"


class TestAverageReturn:
    def test_annualized_scales_by_252(self):
        returns = pd.Series(np.full(252, 0.001))
        assert average_return(returns) == pytest.approx(0.252)


class TestDailyReturnStats:
    def test_best_worst_win_ratio(self):
        s = _series([0.01, -0.02, 0.005, 0.0, 0.03, -0.01])
        stats = daily_return_stats(s)
        assert stats['best_day'] == pytest.approx(0.03)
        assert stats['worst_day'] == pytest.approx(-0.02)
        # positives: 0.01, 0.005, 0.03 = 3 of 6
        assert stats['win_ratio'] == pytest.approx(3 / 6)

    def test_empty_returns_returns_nan(self):
        stats = daily_return_stats(pd.Series([], dtype=float))
        assert np.isnan(stats['best_day'])
        assert np.isnan(stats['worst_day'])
        assert np.isnan(stats['win_ratio'])


class TestSortinoRatio:
    def test_empty_returns_zero(self):
        assert sortino_ratio(pd.Series([], dtype=float)) == 0.0

    def test_no_downside_returns_zero(self):
        # All returns above the risk-free rate -> downside dev = 0 -> 0.0
        returns = pd.Series(np.full(252, 0.001))
        assert sortino_ratio(returns, risk_free_rate=0.0) == 0.0

    def test_downside_penalized_by_target(self):
        rng = np.random.default_rng(11)
        # skewed: frequent small positives + rarer larger negatives (net positive)
        returns = pd.Series(np.where(rng.random(500) < 0.7, 0.004, -0.006))
        assert returns.mean() > 0.0
        sortino = sortino_ratio(returns, risk_free_rate=0.0)
        assert sortino > 0.0

    def test_positive_ratio_when_rf_creates_downside(self):
        # oscillating returns, net-positive vs a modest daily risk-free rate
        returns = pd.Series(np.where(np.arange(252) % 2 == 0, 0.003, -0.001))
        assert sortino_ratio(returns, risk_free_rate=0.05) > 0.0

    def test_higher_risk_free_lowers_sortino(self):
        returns = pd.Series(np.where(np.arange(252) % 2 == 0, 0.003, -0.001))
        low = sortino_ratio(returns, risk_free_rate=0.0)
        high = sortino_ratio(returns, risk_free_rate=0.10)
        assert high < low

    def test_annualized_scales_by_sqrt252(self):
        returns = pd.Series(np.where(np.arange(252) % 2 == 0, 0.003, -0.001))
        daily = sortino_ratio(returns, risk_free_rate=0.0, annualize=False)
        ann = sortino_ratio(returns, risk_free_rate=0.0, annualize=True)
        assert ann == pytest.approx(daily * np.sqrt(252), rel=1e-6)


class TestValueAtRisk:
    def test_known_percentile(self):
        # n=101 => the 1% quantile lands exactly on index 1 (value -0.05)
        returns = pd.Series([-0.10, -0.05] + [0.01] * 99, dtype=float)
        var = value_at_risk(returns, confidence=0.99, annualize=False)
        assert var == pytest.approx(0.05, rel=1e-6)

    def test_matches_numpy_percentile(self):
        rng = np.random.default_rng(2)
        returns = pd.Series(rng.normal(0.001, 0.01, 500))
        expected = -np.percentile(returns.values, 5)
        assert value_at_risk(returns, confidence=0.95, annualize=False) == pytest.approx(max(0.0, expected), rel=1e-9)

    def test_annualized_scales(self):
        returns = pd.Series([-0.02] * 10 + [0.001] * 242, dtype=float)
        daily = value_at_risk(returns, confidence=0.95, annualize=False)
        ann = value_at_risk(returns, confidence=0.95, annualize=True)
        assert ann == pytest.approx(daily * np.sqrt(252), rel=1e-6)

    def test_empty_returns_zero(self):
        assert value_at_risk(pd.Series([], dtype=float)) == 0.0

    def test_no_losses_is_zero(self):
        returns = pd.Series(np.full(100, 0.005))
        assert value_at_risk(returns, confidence=0.95, annualize=False) == 0.0


class TestBeta:
    def test_identical_to_market_is_one(self):
        rng = np.random.default_rng(3)
        r = pd.Series(rng.normal(0.001, 0.01, 252))
        assert beta(r, r) == pytest.approx(1.0)

    def test_half_market_movement(self):
        rng = np.random.default_rng(4)
        market = pd.Series(rng.normal(0.001, 0.01, 252))
        asset = pd.Series(market.values * 0.5)
        assert beta(asset, market) == pytest.approx(0.5, rel=1e-3)

    def test_constant_market_is_nan(self):
        r = pd.Series(np.random.default_rng(5).normal(0, 0.01, 100))
        m = pd.Series(np.full(100, 0.0))
        assert np.isnan(beta(r, m))

    def test_misaligned_joined_inner(self):
        rng = np.random.default_rng(6)
        idx = pd.date_range("2025-01-01", periods=100, freq="B")
        r = pd.Series(rng.normal(0, 0.01, 100), index=idx)
        m = pd.Series(rng.normal(0, 0.01, 100), index=idx + pd.Timedelta(days=200))
        # no overlapping returns -> NaN
        assert np.isnan(beta(r, m))


class TestYearlyReturns:
    def test_known_two_year_returns(self):
        idx = pd.date_range("2023-01-01", periods=252, freq="B")
        boom = pd.Series(np.linspace(100, 110, 252), index=idx)
        idx2 = pd.date_range("2024-01-01", periods=252, freq="B")
        bear = pd.Series(np.linspace(110, 99, 252), index=idx2)
        prices = pd.concat([boom, bear])
        yearly = yearly_returns(prices)
        assert yearly['2023'] == pytest.approx((110 - 100) / 100)
        assert yearly['2024'] == pytest.approx((99 - 110) / 110)

    def test_empty_returns_empty_dict(self):
        assert yearly_returns(pd.Series([], dtype=float)) == {}

class TestActivePerformance:
    def _noisy(self, level, seed=0):
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(level, 0.005, 300))

    def test_all_keys_present(self):
        out = active_performance(self._noisy(0.001), self._noisy(0.0005))
        for key in ('alpha', 'beta_to_baseline', 'active_return',
                    'information_ratio', 'hit_rate'):
            assert key in out

    def test_hit_rate_one_when_always_better(self):
        strat = pd.Series(np.full(200, 0.01))
        base = pd.Series(np.full(200, 0.005))
        out = active_performance(strat, base)
        assert out['hit_rate'] == pytest.approx(1.0)

    def test_hit_rate_zero_when_identical_to_baseline(self):
        strat = pd.Series(np.full(200, 0.01))
        out = active_performance(strat, strat)
        assert out['hit_rate'] == pytest.approx(0.0)
        assert out['active_return'] == pytest.approx(0.0, abs=1e-12)

    def test_alpha_positive_for_beat_every_day(self):
        strat = self._noisy(0.002)
        base = self._noisy(0.0005, seed=5)
        out = active_performance(strat, base)
        assert out['alpha'] > 0
        assert out['hit_rate'] > 0.5

    def test_beta_consistent_with_regression(self):
        rng = np.random.default_rng(7)
        base = pd.Series(rng.normal(0.0005, 0.01, 400))
        strat = pd.Series(0.0001 + 1.3 * base.values + rng.normal(0, 0.002, 400))
        out = active_performance(strat, base)
        assert out['beta_to_baseline'] == pytest.approx(1.3, rel=0.05)
        assert out['alpha'] == pytest.approx(0.0001 * 252, abs=0.06)

    def test_constant_returns_zero_variance_guard(self):
        # constant baseline returns -> degenerate regression handled cleanly
        out = active_performance(pd.Series([0.01] * 100), pd.Series([0.01] * 100))
        assert np.isnan(out['beta_to_baseline'])
        assert out['alpha'] == pytest.approx(0.01 * 252)

    def test_insufficient_overlap_returns_nan(self):
        idx = pd.date_range("2025-01-01", periods=100, freq="B")
        a = pd.Series(np.zeros(100), index=idx)
        b = pd.Series(np.zeros(100), index=idx + pd.Timedelta(days=250))
        out = active_performance(a, b)
        assert np.isnan(out['alpha']) and np.isnan(out['hit_rate'])
