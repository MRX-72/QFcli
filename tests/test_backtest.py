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
    parse_param_grid,
    size_position,
    walk_forward,
    _ensemble_weights,
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


class TestPositionSizing:
    def _data(self, seed=5, days=300):
        prices = _trend(seed=seed, days=days)
        returns = prices.pct_change()
        signal = pd.Series(1.0, index=prices.index)
        return signal, returns

    def test_fixed_passthrough_with_weight(self):
        signal, returns = self._data()
        out = size_position(signal, returns, 'fixed', {'weight': 0.5})
        assert out.max() == pytest.approx(0.5)

    def test_fixed_default_is_identity(self):
        signal, returns = self._data()
        out = size_position(signal, returns, 'fixed')
        assert out.max() == pytest.approx(1.0)

    def test_allows_short_signals(self):
        signal = pd.Series(-1.0, index=pd.date_range("2023-01-01", periods=100, freq="B"))
        returns = pd.Series(np.zeros(100), index=signal.index)
        out = size_position(signal, returns, 'fixed')
        assert out.min() == pytest.approx(-1.0)

    def test_target_vol_reduces_exposure_when_vol_high(self):
        signal, returns = self._data(seed=7)
        tv = size_position(signal, returns, 'target_vol',
                           {'target_vol': 0.10, 'window': 20, 'max_leverage': 1.0})
        vol = returns.rolling(20).std() * np.sqrt(252)
        # when realized vol is very high, exposure is low
        assert float(tv.max()) <= 1.0
        high_vol_mask = vol > 0.30
        if high_vol_mask.any():
            assert float(tv[high_vol_mask].max()) < 1.0

    def test_target_vol_caps_at_max_leverage(self):
        signal, returns = self._data(seed=1)
        tv = size_position(signal, returns, 'target_vol',
                           {'target_vol': 0.50, 'window': 20, 'max_leverage': 0.8})
        assert float(tv.max()) <= 0.8 + 1e-12

    def test_kelly_bounded_and_nonnegative(self):
        signal, returns = self._data(seed=13)
        k = size_position(signal, returns, 'kelly', {'fraction': 0.25, 'window': 126})
        assert float(k.min()) >= 0.0
        assert float(k.max()) <= 1.0

    def test_kelly_grows_with_fraction(self):
        signal, returns = self._data(seed=13)
        half = size_position(signal, returns, 'kelly', {'fraction': 0.5, 'window': 126})
        quarter = size_position(signal, returns, 'kelly', {'fraction': 0.25, 'window': 126})
        assert float(half.max()) > float(quarter.max())

    def test_unknown_sizer_raises(self):
        signal, returns = self._data()
        with pytest.raises(ValueError):
            size_position(signal, returns, 'bogus')

    def test_run_backtest_supports_sizers(self):
        prices = _trend(seed=9)
        res = run_backtest(prices, BUILTIN_STRATEGIES['sma_cross'],
                           sizer='kelly', sizer_kwargs={'fraction': 0.1, 'window': 126})
        assert res['n_days'] > 0
        assert 'strategy_alpha' in res


class TestActiveEvalInBacktest:
    def test_active_keys_present(self):
        prices = _trend()
        res = run_backtest(prices, BUILTIN_STRATEGIES['sma_cross'], cost_bps=5)
        for key in ('strategy_alpha', 'strategy_beta_to_baseline',
                    'strategy_active_return', 'strategy_information_ratio',
                    'strategy_hit_rate'):
            assert key in res

    def test_daily_return_series_exposed(self):
        prices = _trend()
        res = run_backtest(prices, BUILTIN_STRATEGIES['sma_cross'])
        strat = res['_strategy_daily_returns']
        base = res['_baseline_daily_returns']
        # baseline starts with a NaN (pct_change), the strategy stream drops it
        assert strat.index.equals(base.dropna().index)
        assert len(strat) >= 1


class TestParseParamGrid:
    def test_cartesian_product(self):
        grid = parse_param_grid("fast=10,20;slow=40,80")
        assert grid == [
            {'fast': 10, 'slow': 40},
            {'fast': 10, 'slow': 80},
            {'fast': 20, 'slow': 40},
            {'fast': 20, 'slow': 80},
        ]

    def test_single_axis(self):
        assert parse_param_grid("lookback=10,20,30") == [
            {'lookback': 10}, {'lookback': 20}, {'lookback': 30}
        ]

    def test_coercion(self):
        grid = parse_param_grid("flag=true;n=3;pct=0.1")
        assert grid == [{'flag': True, 'n': 3, 'pct': 0.1}]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_param_grid("")

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            parse_param_grid("fast=10;slow")


class TestWalkForward:
    def test_structure_and_oos_stitching(self):
        prices = _trend(days=320, seed=11)
        grid = parse_param_grid("fast=5,10,20;slow=40,60")
        wf = walk_forward(prices, BUILTIN_STRATEGIES['sma_cross'], grid,
                          train_frac=0.6, step=40)
        assert wf['n_folds'] >= 2
        assert wf['oos_days'] >= 2
        for key in ('oos_total_return', 'oos_annual_return', 'oos_sharpe',
                    'baseline_total_return', 'alpha', 'beta_to_baseline',
                    'active_return', 'information_ratio', 'hit_rate',
                    'n_folds', 'fold_beat_rate', 'selection_metric', 'folds'):
            assert key in wf

    def test_oos_days_equals_sum_of_fold_days(self):
        prices = _trend(days=400, seed=3)
        grid = parse_param_grid("fast=10;slow=50")
        wf = walk_forward(prices, BUILTIN_STRATEGIES['sma_cross'], grid,
                          train_frac=0.6, step=63)
        fold_days = sum(f['oos_n_days'] for f in wf['folds'])
        assert wf['oos_days'] == fold_days

    def test_empty_param_grid_raises(self):
        prices = _trend()
        with pytest.raises(ValueError):
            walk_forward(prices, BUILTIN_STRATEGIES['sma_cross'], [],
                         train_frac=0.6, step=40)

    def test_insufficient_data_raises(self):
        prices = _trend(days=50)
        with pytest.raises(ValueError):
            walk_forward(prices, BUILTIN_STRATEGIES['sma_cross'],
                         [{'fast': 10, 'slow': 30}], step=40)


class TestEnsembleWeights:
    def test_best_is_one_hot_argmax(self):
        w = _ensemble_weights([0.1, 0.5, 0.9], 'best', None)
        assert (w == np.array([0., 0., 1.])).all()

    def test_equal_sums_to_one(self):
        assert _ensemble_weights([0.1, 0.5, 0.9], 'equal', None) == pytest.approx(
            [1 / 3, 1 / 3, 1 / 3])

    def test_rank_positive_and_sum_one(self):
        w = _ensemble_weights([0.1, 0.5, 0.9], 'rank', None)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()
        assert w[2] > w[1] > w[0]  # best config gets the most weight

    def test_topk_zeros_below_k(self):
        w = _ensemble_weights([0.1, 0.5, 0.9], 'topk', 2)
        assert w[2] == pytest.approx(0.5)
        assert w[1] == pytest.approx(0.5)
        assert w[0] == pytest.approx(0.0)

    def test_topk_default_k(self):
        w = _ensemble_weights([0.1, 0.2, 0.3, 0.4], 'topk', None)
        # default k = max(2, 4//2) = 2
        assert (w == np.array([0, 0, 0.5, 0.5])).all()

    def test_topk_caps_at_n(self):
        w = _ensemble_weights([0.1, 0.9], 'topk', 5)
        assert len(w) == 2 and w.sum() == pytest.approx(1.0)

    def test_ties_break_average(self):
        w = _ensemble_weights([0.5, 0.5, 0.0], 'rank', None)
        assert w[0] == pytest.approx(w[1])

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            _ensemble_weights([0.1, 0.9], 'bogus', None)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _ensemble_weights([], 'equal', None)


class TestWalkForwardEnsemble:
    def test_best_reproduces_best_reference(self):
        prices = _trend(days=400, seed=11)
        grid = parse_param_grid("fast=10,20;slow=40,80")
        wf = walk_forward(prices, BUILTIN_STRATEGIES['sma_cross'], grid,
                          train_frac=0.6, step=63, ensemble='best')
        assert wf['oos_total_return'] == pytest.approx(wf['best_oos_total_return'])
        assert wf['ensemble'] == 'best'

    def test_equal_blend_reports_keys(self):
        prices = _trend(days=400, seed=13)
        grid = parse_param_grid("fast=10,20;slow=40,80")
        wf = walk_forward(prices, BUILTIN_STRATEGIES['sma_cross'], grid,
                          train_frac=0.6, step=63, ensemble='equal')
        assert 'best_oos_total_return' in wf
        assert wf['ensemble'] == 'equal'
        assert wf['folds'][0]['n_params'] == len(grid)
        assert wf['folds'][0]['ensemble'] == 'equal'

    def test_best_config_reference_is_best_only(self):
        prices = _trend(days=400, seed=17)
        grid = parse_param_grid("fast=10;slow=40,80,120")
        best_only = walk_forward(prices, BUILTIN_STRATEGIES['sma_cross'], grid,
                                 train_frac=0.6, step=63, ensemble='best')
        equal = walk_forward(prices, BUILTIN_STRATEGIES['sma_cross'], grid,
                             train_frac=0.6, step=63, ensemble='equal')
        assert equal['best_oos_total_return'] == pytest.approx(
            best_only['oos_total_return'], abs=1e-9)

    def test_invalid_ensemble_raises(self):
        prices = _trend(days=400)
        with pytest.raises(ValueError):
            walk_forward(prices, BUILTIN_STRATEGIES['sma_cross'],
                         [{'fast': 10, 'slow': 40}], ensemble='bogus')