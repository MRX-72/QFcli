"""Tests for the unified analysis module (network mocked)."""

import numpy as np
import pandas as pd
import pytest

from quant_finance import analysis


def _fake_df(n=260, start=100.0, drift=0.05):
    """Deterministic synthetic OHLCV frame with a mild uptrend."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = start * np.cumprod(1 + drift / 252 + rng.normal(0, 0.01, n))
    opens = np.maximum(closes * (1 + rng.normal(0, 0.002, n)), 1.0)
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.002, n)))
    vols = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=idx,
    )


@pytest.fixture(autouse=True)
def _mock_fetch(monkeypatch):
    def fake_fetch(ticker, period="1y"):
        return _fake_df(), "FakeCorp Inc."

    monkeypatch.setattr(analysis, "fetch_stock_data", fake_fetch)


class TestAnalyzeSingleStock:
    def test_returns_flat_dict_with_all_metrics(self):
        result = analysis.analyze_single_stock("TEST")
        expected_keys = {
            'ticker', 'company_name', 'current_price', 'start_date', 'end_date',
            'cumulative_return', 'avg_return', 'volatility', 'sharpe_ratio',
            'sortino_ratio', 'value_at_risk', 'atr', 'max_drawdown',
            'sma_values', 'signals', 'bullish_count',
            'total_signals', 'roc', 'risk_score', 'rsi', 'macd',
            'golden_cross', 'yearly_returns', 'beta',
            'bollinger_bands', 'daily_stats', 'trend'
        }
        assert expected_keys <= set(result.keys())

    def test_trend_has_prices_and_bounds(self):
        result = analysis.analyze_single_stock("TEST")
        trend = result['trend']
        assert len(trend['prices']) == 60
        assert trend['high'] >= max(trend['prices'])
        assert trend['low'] <= min(trend['prices'])

    def test_beta_none_by_default(self):
        assert analysis.analyze_single_stock("TEST")['beta'] is None

    def test_beta_computed_with_benchmark(self):
        import pandas as pd
        rng = np.random.default_rng(1)
        n = 260
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        # market series that mirrors the synthetic price's drift direction
        benchmark = pd.Series(rng.normal(0.0002, 0.01, n), index=idx)
        result = analysis.analyze_single_stock("TEST", benchmark_returns=benchmark)
        assert result['beta'] is not None

    def test_yearly_returns_is_dict(self):
        yearly = analysis.analyze_single_stock("TEST")['yearly_returns']
        assert isinstance(yearly, dict)

    def test_ticker_uppercased(self):
        assert analysis.analyze_single_stock("test")["ticker"] == "TEST"

    def test_company_name_present(self):
        assert analysis.analyze_single_stock("TEST")["company_name"] == "FakeCorp Inc."

    def test_metric_values_are_finite(self):
        result = analysis.analyze_single_stock("TEST")
        for key in (
            'current_price', 'cumulative_return', 'avg_return', 'volatility',
            'sharpe_ratio', 'max_drawdown', 'roc', 'rsi'
        ):
            assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"

    def test_signals_use_valid_labels(self):
        result = analysis.analyze_single_stock("TEST")
        valid = {"BULLISH", "BEARISH", "INSUFFICIENT DATA"}
        assert set(result['signals'].values()) <= valid

    def test_rsi_within_bounds(self):
        result = analysis.analyze_single_stock("TEST")
        assert 0 <= result["rsi"] <= 100