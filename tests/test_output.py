"""Tests for the output/export layer."""

import json

import numpy as np
import pytest

from quant_finance.output import (
    export_to_dict,
    format_price,
    format_percentage,
    sparkline_levels,
    gauge_bar
)


class TestExportToDict:
    def _metrics(self):
        return {
            'ticker': 'AAPL',
            'company_name': 'Apple Inc.',
            'current_price': 190.5,
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'cumulative_return': 0.25,
            'avg_return': 0.15,
            'volatility': 0.25,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.10,
            'sma_values': {20: 185.0, 50: 180.0, 200: 170.0},
            'signals': {20: 'BULLISH', 50: 'BULLISH', 200: 'BEARISH'},
            'roc': 0.05,
            'risk_score': 'MODERATE',
        }

    def test_exports_expected_structure(self):
        metrics = self._metrics()
        out = export_to_dict('AAPL', 'Apple Inc.', metrics)

        assert out['ticker'] == 'AAPL'
        assert out['company_name'] == 'Apple Inc.'
        assert out['data_period'] == {'start': '2024-01-01', 'end': '2024-12-31'}
        assert out['returns']['sharpe_ratio'] == 1.2
        assert out['moving_averages']['sma_20']['signal'] == 'BULLISH'
        assert out['risk_metrics']['risk_score'] == 'MODERATE'

    def test_json_serializable(self):
        out = export_to_dict('AAPL', 'Apple Inc.', self._metrics())
        json.dumps(out)  # must not raise

    def test_nan_values_become_none(self):
        metrics = self._metrics()
        metrics['sma_values'] = {20: float('nan'), 50: 180.0, 200: 170.0}
        metrics['signals'] = {20: 'INSUFFICIENT DATA', 50: 'BULLISH', 200: 'BEARISH'}
        out = export_to_dict('AAPL', 'Apple Inc.', metrics)
        assert out['moving_averages']['sma_20']['value'] is None
        assert out['moving_averages']['sma_50']['value'] == 180.0

    def test_new_fields_exported(self):
        metrics = self._metrics()
        metrics['sortino_ratio'] = 1.8
        metrics['value_at_risk'] = 0.04
        metrics['atr'] = 2.5
        metrics['beta'] = 1.15
        metrics['golden_cross'] = 'GOLDEN CROSS'
        metrics['trend'] = {
            'prices': [100.0, 102.0],
            'high': 105.0,
            'low': 95.0
        }
        out = export_to_dict('AAPL', 'Apple Inc.', metrics)
        assert out['returns']['sortino_ratio'] == 1.8
        assert out['risk_metrics']['value_at_risk_95'] == 0.04
        assert out['risk_metrics']['atr_14'] == 2.5
        assert out['risk_metrics']['beta'] == 1.15
        assert out['trend']['golden_cross'] == 'GOLDEN CROSS'
        assert out['trend']['price_high'] == 105.0

    def test_beta_none_exported_as_none(self):
        metrics = self._metrics()
        metrics['beta'] = None
        metrics['trend'] = {'prices': [], 'high': np.nan, 'low': np.nan}
        out = export_to_dict('AAPL', 'Apple Inc.', metrics)
        assert out['risk_metrics']['beta'] is None

    def test_simulation_and_stat_tests_exported(self):
        metrics = self._metrics()
        metrics['bootstrap_ci'] = {'ci_low': 0.0001, 'ci_high': 0.001, 'n_boot': 2000}
        mc = {'monte_carlo': {'p5': 0.5, 'p50': 1.1, 'p95': 1.9}}
        metrics.update(mc)
        metrics['stress_30d'] = -0.22
        metrics['sharpe_bootstrap'] = {'observed': 1.2, 'ci_low': 0.5, 'ci_high': 1.9, 'n_boot': 2000}
        metrics['ljung_box'] = {'statistic': 4.5, 'p_value': 0.5, 'verdict': 'NO AUTOCORRELATION'}
        metrics['jarque_bera'] = {'statistic': 1.2, 'p_value': 0.4, 'verdict': 'NORMAL (cannot reject)'}
        metrics['sharpe_significance'] = {'statistic': 2.1, 'p_value': 0.03, 'verdict': 'SIGNIFICANT'}

        out = export_to_dict('AAPL', 'Apple Inc.', metrics)
        assert out['simulation']['monte_carlo_1y']['p50'] == pytest.approx(1.1)
        assert out['simulation']['worst_30d_window'] == pytest.approx(-0.22)
        assert out['statistical_tests']['ljung_box']['verdict'] == 'NO AUTOCORRELATION'
        assert out['statistical_tests']['sharpe_significance']['p_value'] == pytest.approx(0.03)
        json.dumps(out)  # serializable


class TestSparkline:
    def test_short_series_passthrough_length(self):
        levels = sparkline_levels([1, 2, 3, 4], width=10)
        assert len(levels) == 4
        assert all(0 <= l <= 7 for l in levels)

    def test_monotonic_levels_increase(self):
        levels = sparkline_levels([1, 2, 3, 4, 5], width=5)
        assert levels == sorted(levels)
        assert levels[0] == 0
        assert levels[-1] == 7

    def test_flat_series_mid_level(self):
        assert sparkline_levels([4.0, 4.0, 4.0], width=3) == [3, 3, 3]

    def test_downsampling_respects_width(self):
        values = list(range(100))
        assert len(sparkline_levels(values, width=30)) == 30

    def test_empty_values(self):
        assert sparkline_levels([], width=10) == []


class TestGauge:
    def test_at_low_is_green_range(self):
        text = gauge_bar(30, 30, 70, width=20)
        # returns a rich Text with style containing 'green'
        styles = {span.style for span in text.spans}
        assert any('green' in str(s) for s in styles if s)

    def test_at_high_is_red_range(self):
        text = gauge_bar(70.0, 30, 70, width=20)
        styles = {span.style for span in text.spans}
        assert any('red' in str(s) for s in styles if s)

    def test_middle_is_yellow(self):
        text = gauge_bar(50.0, 30, 70, width=20)
        styles = {span.style for span in text.spans}
        assert any('yellow' in str(s) for s in styles if s)

    def test_bounds_clamping(self):
        text = gauge_bar(200.0, 30, 70, width=20)
        assert '█' in text.plain
        text2 = gauge_bar(-100.0, 30, 70, width=20)
        assert '░' in text2.plain

    def test_equal_range_center(self):
        text = gauge_bar(5.0, 5.0, 5.0, width=10)
        assert len(text.plain.replace('[', '').replace(']', '')) == 10


class TestFormatting:
    def test_format_price(self):
        assert format_price(190.5).plain == "$190.50"

    def test_format_percentage_negative(self):
        assert format_percentage(-0.10).plain == "-10.00%"

    def test_format_nan(self):
        assert format_percentage(np.nan).plain == "N/A"