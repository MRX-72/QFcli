"""
Paper Trading Module
A monitoring harness that replays, day by day, what walk-forward-selected
parameters would have traded - as if the strategy had been re-run every
morning.

The contract is deliberately modest and honest:

- Every morning (bar) the parameter combination with the best *training* score
  on the data available up to the prior close is selected (or blended, when an
  ensemble is used) and the resulting position takes that day's return. No
  future information is ever touched.
- The equity curve is therefore exactly what the walk-forward logic *would*
  have traded, day by day, with hindsight-free signals and the same
  cost/slippage model as the backtester.
- State is persisted to disk, so re-running the command repeatedly extends the
  paper history as new bars arrive. This is a *monitoring* harness - it does
  not place orders, and nothing here is investment advice.
"""

import json
import os
import pathlib
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .backtest import run_backtest, _ensemble_weights
from .metrics import annual_return, sharpe_ratio, active_performance

PAPER_DIR = os.environ.get(
    "QFCLI_PAPER_DIR",
    os.path.join(os.path.expanduser("~"), ".qfcli", "paper"),
)

DEFAULT_TRAIN_FRAC = 0.6
OOS_TAIL = 504  # daily rows kept in the persisted state

_ENSEMBLE_CHOICES = ('best', 'equal', 'rank', 'topk')


def _params_key(params: Dict) -> str:
    import json as _json
    return _json.dumps(dict(sorted(params.items())), sort_keys=True)


def _iso(dt) -> str:
    return pd.Timestamp(dt).isoformat()


def paper_trade(
    prices: pd.Series,
    strategy,
    param_grid: List[Dict],
    sizer: str = 'fixed',
    sizer_kwargs: Optional[Dict] = None,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    risk_free_rate: float = 0.0,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    ensemble: str = 'best',
    topk: Optional[int] = None,
    selection_metric: str = 'sharpe'
) -> Dict:
    """
    Replay the walk-forward-selected strategy day by day as a paper account.

    For each out-of-sample day ``t``:
        1. Score every grid point on the expanding train window ending at
           ``t-1`` (no lookahead of day ``t``).
        2. Combine the grid via ``ensemble`` into per-config weights.
        3. Each config trades day ``t`` with its own position (signal computed
           on data through ``t-1``, shifted one bar), then returns are blended
           with those weights and recorded as the paper day return.
        4. The buy-and-hold return of the asset for the same day is recorded.

    The persisted state also flags how often each config was the day's argmax
    (``param_selection_rate``) so the harness is honest about which parameters
    kept winning.

    Args:
        prices: Daily close price series
        strategy: Strategy callable (same conventions as run_backtest)
        param_grid: List of param dicts (cartesian via parse_param_grid)
        sizer / sizer_kwargs / cost_bps / slippage_bps / risk_free_rate:
            Shared with run_backtest()
        train_frac: Initial fraction of history treated as in-sample
        ensemble: 'best', 'equal', 'rank' or 'topk' (see walk_forward)
        topk: k used by 'topk'
        selection_metric: Metric maximized on the training window

    Returns:
        Dict of paper-account statistics plus the daily history tail
    """
    if ensemble not in _ENSEMBLE_CHOICES:
        raise ValueError(f"Unknown ensemble '{ensemble}' "
                         f"(choose {', '.join(_ENSEMBLE_CHOICES)}).")
    clean = prices.dropna()
    n = len(clean)
    oos_start = int(n * train_frac)
    if oos_start < 60 or n - oos_start < 1:
        raise ValueError(
            "Not enough data for paper trading; pass a longer history or "
            "lower --train-frac."
        )
    if not param_grid:
        raise ValueError("param_grid must contain at least one parameter combination.")

    returns = clean.pct_change()

    paper_days: List[float] = []
    baseline_days: List[float] = []
    days_index: List[object] = []
    selection: Dict[str, int] = {}
    current_params = None

    for t in range(oos_start, n):
        train = clean.iloc[:t]      # data available before day t's bar
        scores: Dict = {}

        for params in param_grid:
            res = run_backtest(
                train, strategy,
                cost_bps=cost_bps, slippage_bps=slippage_bps,
                risk_free_rate=risk_free_rate,
                sizer=sizer, sizer_kwargs=sizer_kwargs,
                **params
            )
            s = res[f'strategy_{selection_metric}']
            if s is None or (isinstance(s, float) and np.isnan(s)):
                s = -np.inf
            scores[_params_key(params)] = (params, float(s))

        ordered = [scores[k][1] for k in scores]
        weights = _ensemble_weights(ordered, ensemble, topk)

        best_key = max(scores, key=lambda k: scores[k][1])
        selection[best_key] = selection.get(best_key, 0) + 1

        # day t net return = position decided on data through t-1 (shift(1))
        day_net = 0.0
        for (params, _), w in zip(scores.values(), weights):
            full = run_backtest(
                clean.iloc[:t + 1], strategy,
                cost_bps=cost_bps, slippage_bps=slippage_bps,
                risk_free_rate=risk_free_rate,
                sizer=sizer, sizer_kwargs=sizer_kwargs,
                **params
            )
            series = full['_strategy_daily_returns']
            day_net += w * float(series.iloc[-1]) if len(series) else 0.0
        day_net = day_net if np.isfinite(day_net) else 0.0

        paper_days.append(day_net)
        baseline_days.append(float(returns.iloc[t]))
        days_index.append(clean.index[t])

    # current recommended config: re-select on the *entire* history so the
    # next, not-yet-seen bar can use today's pick
    best_score = -np.inf
    for params in param_grid:
        res = run_backtest(
            clean, strategy,
            cost_bps=cost_bps, slippage_bps=slippage_bps,
            risk_free_rate=risk_free_rate,
            sizer=sizer, sizer_kwargs=sizer_kwargs,
            **params
        )
        s = res[f'strategy_{selection_metric}']
        if s is None or (isinstance(s, float) and np.isnan(s)):
            s = -np.inf
        if s > best_score:
            best_score = s
            current_params = params

    paper = pd.Series(paper_days, index=pd.DatetimeIndex(days_index))
    baseline = pd.Series(baseline_days, index=paper.index)

    paper_eq = (1 + paper).cumprod()
    base_eq = (1 + baseline).cumprod()
    active = active_performance(paper, baseline, risk_free_rate=risk_free_rate)

    total_sel = sum(selection.values()) or 1
    selection_rate = {k: v / total_sel for k, v in sorted(selection.items())}

    tail = min(len(paper), OOS_TAIL)
    daily_tail = [
        [paper.index[i].isoformat(), paper.iloc[i], baseline.iloc[i]]
        for i in range(len(paper))[-tail:]
    ]

    return {
        'strategy': 'n/a',
        'ensemble': ensemble,
        'topk': topk,
        'grid': param_grid,
        'n_params': len(param_grid),
        'train_frac': train_frac,
        'sizer': sizer,
        'selection_metric': selection_metric,
        'start': _iso(paper.index[0]),
        'end': _iso(paper.index[-1]),
        'n_days': int(len(paper)),
        'paper_total_return': float(paper_eq.iloc[-1] - 1),
        'paper_annual_return': annual_return(paper_eq),
        'paper_sharpe': sharpe_ratio(paper, risk_free_rate=risk_free_rate, annualize=True),
        'baseline_total_return': float(base_eq.iloc[-1] - 1),
        'baseline_annual_return': annual_return(base_eq),
        'alpha': active['alpha'],
        'beta_to_baseline': active['beta_to_baseline'],
        'active_return': active['active_return'],
        'information_ratio': active['information_ratio'],
        'hit_rate': active['hit_rate'],
        'current_params': current_params,
        'param_selection_rate': selection_rate,
        'daily_tail': daily_tail,
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }


def paper_state_path(ticker: str, strategy_label: str) -> pathlib.Path:
    """Cache path for a ticker/strategy paper account."""
    name = f"{ticker.upper()}_{strategy_label.replace('/', '_')}".replace(' ', '_')
    return pathlib.Path(PAPER_DIR) / f"{name}.json"


def save_paper_state(path: pathlib.Path, state: Dict) -> None:
    """Persist a paper-account state dict (best effort, atomic-ish write)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.json.tmp')
        with open(tmp, 'w') as fh:
            json.dump(state, fh, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        raise OSError(f"Could not write paper state to {path}: {e}") from e


def load_paper_state(path: pathlib.Path) -> Optional[Dict]:
    """Load a previously saved paper account state, or None."""
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None