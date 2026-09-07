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
    daily_return_stats
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