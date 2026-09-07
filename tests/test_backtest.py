"""Tests for the backtest engine (no network required)."""

import numpy as np
import pandas as pd
import pytest

from quant_finance.backtest import (
    run_backtest,
    BUILTIN_STRATEGIES,
    prepare_backtest_data,
    load_custom_strategy,
    parse_strategy_params,
)


def _trend(seed=1, days=300, start=100.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=days, freq="B")
    prices = start * np.cumprod(1 + 0.0008 + rng.normal(0, 0.008, days))
    return pd.Series(prices, index=idx)


class TestBacktestEngine:
    def test_baseline_equals_market_performance(self):
        # Loosely: buy & hold total return close to raw price change
        prices = _trend()
        res = run_backtest(prices, BUILTIN_STRATEGIES['sma_cross'])
        raw = prices.iloc[-1] / prices.iloc[0] - 1
        assert res['baseline_total_return'] == pytest.approx(raw, rel=1e-6)

    def test_all_required_keys_present(self):
        prices = _trend()
        res = run_backtest(prices, BUILTIN_STRATEGIES['momentum'], cost_bps=5)
        for key in ('strategy_total_return', 'strategy_sharpe', 'strategy_max_drawdown',
                    'baseline_total_return', 'baseline_sharpe',
                    'n_days', 'n_trades', 'total_cost', 'days_in_market'):
            assert key in res, f"missing {key}"

    def test_costs_reduce_strategy_return(self):
        prices = _trend()
        free = run_backtest(prices, BUILTIN_STRATEGIES['sma_cross'], cost_bps=0, slippage_bps=0)
        costly = run_backtest(prices, BUILTIN_STRATEGIES['sma_cross'], cost_bps=50, slippage_bps=50)
        assert costly['strategy_total_return'] <= free['strategy_total_return']

    def test_no_lookahead_flat_series(self):
        # flat prices: signals at 0/1 but returns are 0 => 0-ish total return
        prices = pd.Series(np.full(300, 100.0), index=pd.date_range("2023-01-01", periods=300, freq="B"))
        res = run_backtest(prices, BUILTIN_STRATEGIES['sma_cross'])
        assert res['strategy_total_return'] == pytest.approx(0.0, abs=1e-12)

    def test_insufficient_data_raises(self):
        prices = pd.Series([100.0, 101.0], index=pd.date_range("2023-01-01", periods=2, freq="B"))
        with pytest.raises(ValueError):
            run_backtest(prices, BUILTIN_STRATEGIES['sma_cross'])

    def test_momentum_returns_bounded_signal(self):
        prices = _trend(seed=7)
        res = run_backtest(prices, BUILTIN_STRATEGIES['momentum'])
        assert -1.0 <= res['strategy_total_return'] <= 20.0  # sanity bound


class TestPrepareBacktestData:
    def test_extracts_close(self):
        df = pd.DataFrame({'Close': [1, 2, 3]})
        out = prepare_backtest_data(df)
        assert list(out) == [1, 2, 3]


class TestLoadCustomStrategy:
    def test_loads_valid_strategy(self, tmp_path):
        f = tmp_path / "strat.py"
        f.write_text(
            "import pandas as pd\n"
            "def custom_strategy(prices, threshold=0.5, **kwargs):\n"
            "    return (prices.pct_change().fillna(0) > threshold).astype(float)\n"
        )
        fn = load_custom_strategy(str(f))
        prices = _trend()
        out = fn(prices, threshold=0.0)
        assert isinstance(out, pd.Series)
        assert out.index.equals(prices.index)

    def test_missing_file_raises(self):
        with pytest.raises(ValueError):
            load_custom_strategy("/no/such/file.py")

    def test_non_py_raises(self, tmp_path):
        f = tmp_path / "strat.txt"
        f.write_text("x = 1")
        with pytest.raises(ValueError):
            load_custom_strategy(str(f))

    def test_missing_function_raises(self, tmp_path):
        f = tmp_path / "nofn.py"
        f.write_text("x = 1\n")
        with pytest.raises(ValueError):
            load_custom_strategy(str(f))

    def test_not_callable_raises(self, tmp_path):
        f = tmp_path / "notcall.py"
        f.write_text("custom_strategy = 42\n")
        with pytest.raises(ValueError):
            load_custom_strategy(str(f))

    def test_syntax_error_raises(self, tmp_path):
        f = tmp_path / "badsyntax.py"
        f.write_text("def custom_strategy(:\n")
        with pytest.raises(ValueError):
            load_custom_strategy(str(f))


class TestParseStrategyParams:
    def test_int_float_bool_string(self):
        out = parse_strategy_params(['fast=10', 'pct=0.5', 'flag=true', 'name=trend'])
        assert out == {'fast': 10, 'pct': 0.5, 'flag': True, 'name': 'trend'}

    def test_no_equals_raises(self):
        with pytest.raises(ValueError):
            parse_strategy_params(['naked'])

    def test_empty_returns_empty(self):
        assert parse_strategy_params([]) == {}