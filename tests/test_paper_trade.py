"""Offline tests for the paper-trading harness."""

import json

import numpy as np
import pandas as pd
import pytest

from quant_finance.backtest import BUILTIN_STRATEGIES, parse_param_grid
from quant_finance.paper_trade import (
    paper_trade,
    save_paper_state,
    load_paper_state,
    paper_state_path,
)


def _prices(seed=5, days=300):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=days, freq="B")
    return pd.Series(100 * np.cumprod(1 + 0.0005 + rng.normal(0, 0.008, days)), index=idx)


def _run(ticker='AAPL', days=300, ensemble='best', seed=5, train_frac=0.6):
    prices = _prices(seed=seed, days=days)
    grid = parse_param_grid("fast=10,20;slow=40,60")
    state = paper_trade(prices, BUILTIN_STRATEGIES['sma_cross'], grid,
                        train_frac=train_frac, ensemble=ensemble, cost_bps=5)
    state['ticker'] = ticker
    return state


class TestPaperTrade:
    def test_structure_and_days(self):
        state = _run()
        assert state['n_days'] == 300 - int(300 * 0.6)
        for key in ('paper_total_return', 'paper_annual_return', 'paper_sharpe',
                    'baseline_total_return', 'alpha', 'beta_to_baseline',
                    'active_return', 'information_ratio', 'hit_rate',
                    'current_params', 'param_selection_rate', 'daily_tail'):
            assert key in state

    def test_deterministic(self):
        a = _run(seed=8)
        b = _run(seed=8)
        assert a['paper_total_return'] == b['paper_total_return']

    def test_selection_rates_sum_to_one(self):
        state = _run()
        assert sum(state['param_selection_rate'].values()) == pytest.approx(1.0)

    def test_current_params_is_a_grid_point(self):
        state = _run()
        assert state['current_params'] in parse_param_grid("fast=10,20;slow=40,60")

    def test_ensembles_run(self):
        for ens in ('equal', 'rank', 'topk'):
            state = _run(ensemble=ens)
            assert np.isfinite(state['paper_total_return'])
            assert state['ensemble'] == ens

    def test_unknown_ensemble_raises(self):
        with pytest.raises(ValueError):
            paper_trade(_prices(), BUILTIN_STRATEGIES['sma_cross'],
                        [{'fast': 10, 'slow': 40}], ensemble='bogus')

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            paper_trade(_prices(days=50), BUILTIN_STRATEGIES['sma_cross'],
                        [{'fast': 10, 'slow': 40}], train_frac=0.6)

    def test_empty_grid_raises(self):
        with pytest.raises(ValueError):
            paper_trade(_prices(), BUILTIN_STRATEGIES['sma_cross'], [])

    def test_no_lookahead_property(self):
        # Day t's selection may only use data up to t-1. If that holds, the
        # *same calendar day* must produce the identical paper return whether or
        # not the harness's price series is truncated after that day.
        rng = np.random.default_rng(3)
        idx = pd.date_range("2022-01-01", periods=300, freq="B")
        prices = pd.Series(100 * np.cumprod(1 + 0.0005 + rng.normal(0, 0.008, 300)), index=idx)
        grid = parse_param_grid("fast=10,20;slow=40,80")
        full = paper_trade(prices, BUILTIN_STRATEGIES['sma_cross'], grid,
                           train_frac=0.4, ensemble='best')
        short = paper_trade(prices.iloc[:-60], BUILTIN_STRATEGIES['sma_cross'], grid,
                            train_frac=0.4, ensemble='best')
        full_by_day = {pd.Timestamp(row[0]): row[1] for row in full['daily_tail']}
        short_by_day = {pd.Timestamp(row[0]): row[1] for row in short['daily_tail']}
        overlap = sorted(set(full_by_day) & set(short_by_day))
        assert len(overlap) >= 10
        for day in overlap:
            assert full_by_day[day] == pytest.approx(short_by_day[day], abs=1e-12)


class TestStabilityFilter:
    def _run_stable(self, stable_days=1, seed=5, **kw):
        return _run(seed=seed, **kw) if stable_days == 1 else paper_trade(
            _prices(seed=seed), BUILTIN_STRATEGIES['sma_cross'],
            parse_param_grid("fast=10,20;slow=40,60"),
            train_frac=0.6, ensemble='best', stable_days=stable_days)

    def test_default_stable_days_1_has_all_keys(self):
        state = _run()
        for key in ('stable_days', 'n_switches', 'mean_days_in_force', 'days_in_force'):
            assert key in state
        assert state['stable_days'] == 1

    def test_big_stable_days_prevents_switching(self):
        state = self._run_stable(stable_days=999)
        assert state['n_switches'] == 0
        assert len(state['days_in_force']) == 1
        assert sum(state['days_in_force'].values()) == state['n_days']

    def test_big_stable_days_vs_single_config(self):
        """With stable_days >= n_days, only the initial argmax ever trades."""
        state = self._run_stable(stable_days=999)
        active_key = list(state['days_in_force'].keys())[0]
        params = json.loads(active_key)
        single = paper_trade(_prices(), BUILTIN_STRATEGIES['sma_cross'],
                             [params], train_frac=0.6, ensemble='best')
        assert state['paper_total_return'] == pytest.approx(single['paper_total_return'], abs=1e-9)

    def test_stable_days_reduces_or_equals_switches(self):
        loose = self._run_stable(stable_days=1, seed=8)
        tight = self._run_stable(stable_days=5, seed=8)
        assert tight['n_switches'] <= loose['n_switches']

    def test_in_force_days_sum_to_n_days(self):
        state = self._run_stable(stable_days=3)
        assert sum(state['days_in_force'].values()) == state['n_days']

    def test_selection_rate_still_sums_to_one(self):
        state = self._run_stable(stable_days=3)
        assert sum(state['param_selection_rate'].values()) == pytest.approx(1.0)

    def test_invalid_stable_days_raises(self):
        for bad in (0, -1, 'x'):
            with pytest.raises(ValueError):
                paper_trade(_prices(), BUILTIN_STRATEGIES['sma_cross'],
                            [{'fast': 10, 'slow': 40}], stable_days=bad)

    def test_blending_ensemble_ignores_stable_for_returns(self):
        """Ensembles blend daily returns; stable_days is a reporting-only filter."""
        prices = _prices(seed=12)
        grid = parse_param_grid("fast=10,20;slow=40,60")
        rank = paper_trade(prices, BUILTIN_STRATEGIES['sma_cross'], grid,
                           train_frac=0.6, ensemble='rank', stable_days=5)
        rank_raw = paper_trade(prices, BUILTIN_STRATEGIES['sma_cross'], grid,
                               train_frac=0.6, ensemble='rank')
        assert rank['paper_total_return'] == pytest.approx(rank_raw['paper_total_return'], abs=1e-9)


class TestPaperStateFile:
    def test_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv('QFCLI_PAPER_DIR', str(tmp_path))
        state = _run()
        path = paper_state_path('AAPL', 'sma_cross')
        save_paper_state(path, state)
        back = load_paper_state(path)
        assert back['n_days'] == state['n_days']
        assert back['current_params'] == state['current_params']
        assert back['stable_days'] == 1
        assert json.dumps(back)  # JSON-serializable

    def test_missing_state_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv('QFCLI_PAPER_DIR', str(tmp_path))
        assert load_paper_state(paper_state_path('NOPE', 'sma_cross')) is None

    def test_prevents_slash_collisions(self, tmp_path, monkeypatch):
        monkeypatch.setenv('QFCLI_PAPER_DIR', str(tmp_path))
        p1 = paper_state_path('AAPL', 'custom:my_strat.py')
        assert '/' not in p1.name
        assert p1.suffix == '.json'