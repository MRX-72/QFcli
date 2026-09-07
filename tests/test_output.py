"""Tests for the output/export layer."""

import json

import numpy as np
import pytest

from quant_finance.output import (
    export_to_dict,
    format_price,
    format_percentage
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


class TestFormatting:
    def test_format_price(self):
        assert format_price(190.5).plain == "$190.50"

    def test_format_percentage_negative(self):
        assert format_percentage(-0.10).plain == "-10.00%"

    def test_format_nan(self):
        assert format_percentage(np.nan).plain == "N/A"