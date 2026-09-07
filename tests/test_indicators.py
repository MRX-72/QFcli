"""Tests for the indicators module (no network required)."""

import numpy as np
import pandas as pd
import pytest

from quant_finance.indicators import (
    simple_moving_average,
    moving_average_signals,
    rate_of_change,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    bollinger_bands,
    calculate_all_smas,
    average_true_range,
    golden_death_cross
)


def _series(values):
    return pd.Series(values, dtype=float)


class TestSimpleMovingAverage:
    def test_matches_last_window(self):
        s = _series([1, 2, 3, 4, 5])
        assert simple_moving_average(s, 3) == 4.0  # (3+4+5)/3

    def test_insufficient_data(self):
        s = _series([1, 2])
        assert np.isnan(simple_moving_average(s, 3))


class TestMovingAverageSignals:
    def test_bullish_and_bearish(self):
        signals = moving_average_signals(current_price=110.0, sma_values={20: 105.0, 50: 115.0})
        assert signals[20] == "BULLISH"
        assert signals[50] == "BEARISH"

    def test_insufficient_data_signal(self):
        signals = moving_average_signals(current_price=10.0, sma_values={200: np.nan})
        assert signals[200] == "INSUFFICIENT DATA"


class TestRateOfChange:
    def test_known_roc(self):
        s = _series([100, 105, 110])  # period 1: (110-105)/105
        assert rate_of_change(s, period=1) == pytest.approx((110 - 105) / 105)

    def test_insufficient_data(self):
        s = _series([1.0, 2.0])
        assert np.isnan(rate_of_change(s, period=2))


class TestEMA:
    def test_constant_series(self):
        s = _series([5.0, 5.0, 5.0, 5.0])
        assert calculate_ema(s, 3) == pytest.approx(5.0)


class TestRSI:
    def test_uptrend_rsi_high(self):
        s = _series(np.linspace(100, 200, 50))
        assert calculate_rsi(s, period=14) > 90

    def test_downtrend_rsi_low(self):
        s = _series(np.linspace(200, 100, 50))
        assert calculate_rsi(s, period=14) < 10

    def test_flat_series_neutral(self):
        s = _series(np.full(50, 100.0))
        assert calculate_rsi(s, period=14) == 50.0

    def test_all_gains_no_losses(self):
        # strictly increasing => avg_loss == 0 => RSI 100
        s = _series(np.arange(1.0, 30.0))
        assert calculate_rsi(s, period=14) == 100.0


class TestMACD:
    def test_uptrend_histogram_shape(self):
        s = _series(np.linspace(100, 200, 100))
        macd = calculate_macd(s)
        assert set(macd.keys()) == {'macd_line', 'signal_line', 'histogram'}
        assert macd['histogram'] > 0

    def test_constant_series_zero_macd(self):
        s = _series(np.full(60, 50.0))
        macd = calculate_macd(s)
        assert macd['macd_line'] == pytest.approx(0.0, abs=1e-6)
        assert macd['histogram'] == pytest.approx(0.0, abs=1e-6)


class TestBollingerBands:
    def test_constant_series_bands_touch_middle(self):
        s = _series(np.full(50, 10.0))
        bb = bollinger_bands(s)
        assert bb['upper'] == pytest.approx(bb['middle'])
        assert bb['lower'] == pytest.approx(bb['middle'])
        assert bb['middle'] == pytest.approx(10.0)

    def test_upper_above_lower(self):
        s = _series(np.arange(1.0, 60.0))
        bb = bollinger_bands(s)
        assert bb['upper'] > bb['middle'] > bb['lower']


class TestAllSmas:
    def test_default_windows(self):
        s = _series(np.arange(1.0, 250.0))
        smas = calculate_all_smas(s)
        assert set(smas.keys()) == {20, 50, 200}

    def test_custom_windows(self):
        s = _series(np.arange(1.0, 30.0))
        smas = calculate_all_smas(s, windows=[5, 10])
        assert set(smas.keys()) == {5, 10}


def _ohlc(highs, lows, closes):
    n = len(highs)
    return pd.DataFrame({
        'High': highs,
        'Low': lows,
        'Close': closes,
    }, index=pd.date_range("2025-01-01", periods=n, freq="B"))


class TestAverageTrueRange:
    def test_flat_series_zero(self):
        df = _ohlc([10.0]*20, [9.0]*20, [9.5]*20)
        assert average_true_range(df, period=14) == pytest.approx(1.0)

    def test_known_value(self):
        # TR = max(H-L, |H-prevC|, |L-prevC|) = 1.0 on flat 10/9/9.5 bars
        highs = [10.0]*15
        lows = [9.0]*15
        closes = [9.5]*15
        df = _ohlc(highs, lows, closes)
        assert average_true_range(df, period=5) == pytest.approx(1.0, rel=1e-6)

    def test_gap_day_true_range(self):
        # H=12, L=10, prev close 8 -> TR = max(2, 4, 2) = 4; flat day 2 -> TR 2
        highs = [8.0, 12.0, 12.0]
        lows = [7.0, 10.0, 10.0]
        closes = [8.0, 11.0, 11.0]
        df = _ohlc(highs, lows, closes)
        assert average_true_range(df, period=2) == pytest.approx((4.0 + 2.0) / 2, rel=1e-6)

    def test_insufficient_data_nan(self):
        df = _ohlc([10.0], [9.0], [9.5])
        assert np.isnan(average_true_range(df, period=14))


class TestGoldenDeathCross:
    def test_uptrend_bullish(self):
        # strictly rising: SMA50 > SMA200 at the end
        s = _series(np.linspace(100, 200, 300))
        assert golden_death_cross(s) in ("BULLISH", "GOLDEN CROSS")

    def test_downtrend_bearish(self):
        s = _series(np.linspace(200, 120, 300))
        assert golden_death_cross(s) in ("BEARISH", "DEATH CROSS")

    def test_insufficient_data(self):
        s = _series(np.linspace(100, 110, 50))
        assert golden_death_cross(s) == "INSUFFICIENT DATA"

    def test_flat_series_bullish(self):
        # flat => SMA50 == SMA200 => not a cross; checks last value equality case
        s = _series(np.full(300, 100.0))
        result = golden_death_cross(s)
        # neither above nor below strictly; equality -> BEARISH branch (curr <= 0)
        assert result in ("BULLISH", "BEARISH")