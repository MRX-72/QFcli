# QFcli

<div align="center">

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/MRX-72/QFcli/actions/workflows/ci.yml/badge.svg)](https://github.com/MRX-72/QFcli/actions/workflows/ci.yml)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)]()

</div>

**QFcli** is a command-line tool for quantitative stock analysis. It pulls
OHLCV data, computes return/risk metrics and technical indicators, runs
single/multi-asset analyses plus simple backtests and portfolio optimization,
and renders the results as readable terminal tables — or clean JSON for
scripting.

Data comes live from Yahoo Finance via `yfinance` and is cached on disk to
avoid repeat downloads (disable with `--no-cache`).

## Install

Requires Python 3.9+.

```bash
git clone https://github.com/MRX-72/QFcli.git
cd QFcli
python3 -m venv venv && source venv/bin/activate
pip install -e .            # provides the `qfcli` command
```

For development with tests:

```bash
pip install -e ".[dev]"
pytest
```

## Usage

```
qfcli [TICKER]                Analyze a single stock
qfcli --compare A B           Compare two stocks side-by-side
qfcli --backtest TICKER -s sma_cross   Backtest a strategy
qfcli --portfolio A B [C...]  Portfolio optimization + stress testing
```

### Single stock

```bash
qfcli NVDA                    # default period: 1y
qfcli AAPL --period 5y        # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
qfcli SPY --rf 0.045          # risk-free rate for Sharpe ratio
qfcli NVDA --benchmark SPY    # beta vs S&P 500 ETF
qfcli AAPL --json             # machine-readable JSON
qfcli AAPL --no-cache         # bypass the on-disk cache, hit the network
```

### Comparison

```bash
qfcli --compare GOOGL META
qfcli -c BTC-USD ETH-USD --period 6mo
```

### Backtesting

```bash
qfcli --backtest NVDA -s sma_cross --cost 5 --slippage 5
qfcli --backtest AAPL -s momentum --lookback 20 --hold 15
qfcli --backtest MSFT -s rsi_reversion --json
qfcli --backtest AAPL --strategy-file my_strat.py --strategy-param threshold=0.02
qfcli --backtest NVDA -s sma_cross --sizer target_vol --sizer-target 0.15
```

Strategies (`-s/--strategy`):

- `sma_cross` — long when the fast SMA is above the slow SMA (tunable
  `--fast`/`--slow`).
- `momentum` — enter on recent return strength, hold for a fixed window
  (`--lookback`/`--hold`).
- `rsi_reversion` — fade short-term overbought/oversold extremes.

Signals are shifted one day forward (no lookahead). A transaction cost and
slippage model (`--cost`/`--slippage`, basis points) is charged against traded
turnover. Every run reports the same metrics for the buy-and-hold baseline,
plus a **vs Buy & Hold** panel that keeps performance claims honest: annualized
**alpha**, **beta** to the baseline, **active return**, **information ratio**
and the daily **hit rate** (share of days the strategy actually beats just
holding the asset). A consistent negative spread of these numbers says the
strategy is not adding value over buy-and-hold, and it should be read that way.

#### Position sizing

`--sizer` scales the strategy signal (which only says *what* to hold) into an
actual position size:

- `fixed` (default) — signal unchanged, or scaled by `weight`.
- `target_vol` — scale exposure so rolling realized volatility times exposure
  approximates `--sizer-target` (annual), capped at `--max-leverage`. Cuts
  exposure automatically in choppy markets.
- `kelly` — fractional Kelly (`--kelly-fraction`, default a conservative 0.25)
  computed from the trailing win-rate and average win/loss. Only scales risk
  down; it never increases it.

Sizing uses only past data (with a neutral 1.0 fallback during warm-up).

#### Walk-forward validation

A backtest on the full history can overfit: good-looking parameters tuned on
the whole series prove nothing out of sample. `--walk-forward` splits the
history into an in-sample segment and successive out-of-sample windows,
re-selects the best grid point on each growing training window, then runs it
untouched on the following test window:

```bash
qfcli --backtest AAPL -s sma_cross --walk-forward \
      --grid "fast=10,20,40;slow=60,80,100"
```

`--grid` is a cartesian product of `key=value1,value2` clauses. Out-of-sample
daily returns are stitched into a single equity curve and evaluated against
buy-and-hold (alpha, hit rate, information ratio, fold-by-fold detail). Note:
parameters not listed in the grid keep the strategy's defaults.

#### Custom strategies

Pass any `.py` file with `--strategy-file`. The file must define a callable
named `custom_strategy` that takes the price series plus optional kwargs and
returns a signal `Series` in `-1..1` (positive = long, negative = short,
`0` = flat):

```python
# my_strat.py
import pandas as pd

def custom_strategy(prices, threshold=0.02, **kwargs):
    """Long for 10 days after a strong 5-day rally, otherwise flat."""
    momentum = prices.pct_change(5).fillna(0.0)
    signal = (momentum > threshold).astype(float)
    return signal
```

Tune it from the command line without editing the file:

```bash
qfcli --backtest AAPL --strategy-file my_strat.py \
      --strategy-param threshold=0.01 --strategy-param lookback=10 --cost 5
```

Values are coerced (`10` → int, `0.5` → float, `true`/`false` → bool, else
string). Note: `--strategy-file` executes the file as Python, so only load
strategies you trust.

### Portfolio mode

```bash
qfcli --portfolio AAPL MSFT NVDA KO --period 2y
qfcli --portfolio AAPL MSFT --period 1y --json
qfcli --portfolio AAPL MSFT NVDA --bl --view NVDA=0.18
```

Given the historical daily returns, QFcli computes three weightings plus (on
request) a Black-Litterman allocation:

- **Min variance** — the closed-form global minimum-variance portfolio
  `S⁻¹1/(1'S⁻¹1)`.
- **Tangency** — the maximum-Sharpe portfolio from sample statistics.
- **Efficient** — a long-only tangency-style allocation via an iterative
  optimizer (label: heuristic).
- **Black-Litterman** (`--bl`) — reverse-optimized equilibrium returns from an
  equal-weight reference portfolio, optionally blended with absolute `--view`
  tilts (`TICKER=RATE`, e.g. `--view NVDA=0.18` means an 18% target excess
  return for NVDA). Per-view error variance defaults to 0.0025
  (`--view-confidence` to override); without views the posterior equals the
  prior and reproduces the reference weights, so views are the only source of
  tilt.

Covariance estimation defaults to **Ledoit-Wolf style shrinkage** toward a
constant-correlation target, which keeps the closed-form inverse well
conditioned (the classic fix for short, correlated histories). Diagnostics are
reported honestly: great-looking tangency weights with extreme short positions
read as estimation noise, not opportunity.

It also prints portfolio expected return/volatility/Sharpe, a correlation
heatmap, an estimated efficient-frontier risk range (sampled using a seeded
Dirichlet draw — an approximate sketch, not an exhaustive frontier), and
scenario stress tests (instant −10/−25/−50%, a −15% flash crash, and the worst
30-day window).

Note: asset selection is up to you — QFcli optimizes *weights* given the assets
you pass in, and both min-variance and tangency portfolios can go short.

### Statistics & simulation

Every single-stock analysis now includes a **Statistics & Simulation** table:

- **Monte Carlo 1-year forecast** (P5 / P50 / P95) via pathwise bootstrap of
  historical returns (seeded, reproducible).
- **Sharpe ratio 95% CI** from a percentile bootstrap.
- **Sharpe significance** — the Jobson–Korkie z-test for the null that the
  Sharpe is zero.
- **Ljung-Box** test for serial autocorrelation in returns.
- **Jarque-Bera** normality test.
- **ADF** unit-root (stationarity) test.

### Terminal visuals

The text output includes a few lightweight, dependency-free visualizations:

- A **price sparkline** of the last 60 sessions, with per-bar up/down coloring,
  plus rolling-volatility and rolling-Sharpe sparklines in the trend panel.
- A **trend regime panel** showing whether the 50-day average sits above or below
  the 200-day average (golden/death cross detection).
- An **RSI position gauge** inside the risk table.
- A **yearly returns table** with green/red color-coding per calendar year.

```
Price trend (last 60 sessions)
▁▂▃▂▄▅▆▇█▇▆▅▆▇█▇▅▃▂▃▄▅▆▇█▇▆▅▄▃▂▃▄▅▆▇▆▅▄▃▂▃▄▅▆▇█▇▆▅

Market regime: BULLISH (SMA-50 > SMA-200)
Period high: $332.12   Period low: $218.77
```

### Options

| Flag | Description |
|------|-------------|
| `-c, --compare T1 T2` | Compare two tickers |
| `--period`            | Data period (default `1y`) |
| `--rf RATE`           | Annual risk-free rate (default `0.0`) |
| `--benchmark SYMBOL`  | Benchmark/index for beta (e.g. `SPY`, `^GSPC`); single-stock mode only |
| `--backtest TICKER`   | Run a strategy backtest on a ticker |
| `-s, --strategy NAME` | Strategy to backtest (default `sma_cross`) |
| `--cost BPS`          | Round-trip transaction cost in basis points (default `5`) |
| `--slippage BPS`      | Slippage per trade in basis points (default `5`) |
| `--fast N` / `--slow N` | SMA windows for `sma_cross` (defaults `20`/`50`) |
| `--lookback N` / `--hold N` | Windows for `momentum` (defaults `20`/`20`) |
| `--strategy-file PATH`      | Load a custom strategy from a `.py` file (`custom_strategy(prices, **kwargs)`) |
| `--strategy-param K=V`      | Pass a parameter to the strategy (repeatable) |
| `--sizer NAME`              | Position sizing: `fixed`, `target_vol`, `kelly` (default `fixed`) |
| `--sizer-target RATE`       | Annual vol target for `target_vol` (default `0.15`) |
| `--sizer-window N`          | Rolling window (days) for sizing (default `20`) |
| `--max-leverage N`          | Max gross exposure for `target_vol` (default `1.0`) |
| `--kelly-fraction N`        | Fraction of full Kelly for the `kelly` sizer (default `0.25`) |
| `--walk-forward`            | Out-of-sample walk-forward validation of the strategy's parameters |
| `--grid SPEC`               | Param grid for `--walk-forward`, cartesian (`"fast=10,20;slow=40,80"`) |
| `--train-frac N`            | Initial in-sample fraction for `--walk-forward` (default `0.6`) |
| `--wf-step N`               | Days added per walk-forward fold (default `63`) |
| `--portfolio T1 T2 ...` | Portfolio mode: optimize a basket (min 2) |
| `--bl`              | Add a Black-Litterman weight row in portfolio mode (prior + optional views) |
| `--view K=RATE`     | Absolute expected-return view for Black-Litterman, e.g. `NVDA=0.18` (repeatable) |
| `--view-confidence N` | Per-view error variance for Black-Litterman (default `0.0025`) |
| `--no-cache`          | Bypass the on-disk data cache |
| `--json`              | Emit machine-readable JSON |

Caching: data is keyed by ticker+period and cached for 6 hours under
`~/.qfcli/cache` (override with `QFCLI_CACHE_DIR` and `QFCLI_CACHE_TTL`).

## Example output

### Text

```
Cumulative Return      +23.15%
Annualized Volatility  +27.36%
Sharpe Ratio              1.65
Max Drawdown            -12.71%
Risk Score              MODERATE
```

### JSON

```json
{
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "current_price": 319.97,
  "returns": {
    "cumulative": 0.0297,
    "sharpe_ratio": 1.7429,
    "sortino_ratio": 2.1
  },
  "risk_metrics": {
    "max_drawdown": -0.1271,
    "value_at_risk_95": 0.0241,
    "beta": 1.12,
    "risk_score": "MODERATE"
  },
  "trend": {
    "golden_cross": "BULLISH",
    "price_high": 332.12,
    "price_low": 218.77
  }
}
```

## Metrics

- **Returns**: cumulative return, average daily return, annualized return (CAGR), annualized volatility, Sharpe ratio, **Sortino ratio**, max drawdown, win ratio, **VaR (95%, historical)**, **beta vs a chosen benchmark**, rolling volatility/Sharpe
- **Technicals**: SMA (20/50/200), RSI-14, MACD (12,26,9), Bollinger Bands (20,2), rate of change (12-day), **ATR-14**, **golden/death cross detection**
- **Signals**: price-vs-moving-average bullish/bearish labels; risk bucket derived from annualized volatility (<15% stable, 15–30% moderate, >30% high)
- **Calendar**: per-year return breakdown
- **Simulation & statistics**: Monte Carlo 1y forecast, Sharpe bootstrap CI, Jobson–Korkie Sharpe significance, Ljung-Box autocorrelation, Jarque-Bera normality, ADF stationarity, worst 30-day return
- **Benchmark-adjusted**: backtest alpha, beta-to-baseline, active return, information ratio, daily hit rate vs buy-and-hold
- **Portfolio**: min-variance / tangency / efficient / Black-Litterman weights, Ledoit-Wolf shrinkage covariance, correlation matrix, frontier sketch, scenario stress tests
- **Backtesting extras**: position sizing (target-vol, fractional Kelly), walk-forward out-of-sample validation with a parameter grid

See `quant_finance/metrics.py`, `quant_finance/statistics.py`,
`quant_finance/backtest.py`, and `quant_finance/portfolio.py` for the exact
formulas and their documented assumptions.

## Development

- `quant_finance/analysis.py` is the single source of truth for per-stock
  analysis; both single-stock and comparison modes share it.
- The `--json` path round-trips `export_to_dict`, which converts non-finite
  floats to `null` so machine output is always valid JSON.
- `pytest` covers the math layer (metrics, indicators, shrinkage covariance,
  Black-Litterman, position sizing, walk-forward), the comparison scoring,
  the backtester, portfolio closed forms, the statistics module, and the JSON
  export using synthetic, deterministic data — the test suite runs without
  network access.
- CI runs the suite on macOS and Linux.

## Project structure

```
quant_finance/
  analysis.py       shared per-stock analysis logic
  cli.py            argument parsing + CLI flow
  comparison.py     head-to-head scoring and recommendations
  data_fetcher.py   yfinance wrapper + on-disk cache
  indicators.py     SMA, EMA, RSI, MACD, Bollinger Bands
  metrics.py        return/risk math + rolling + Monte Carlo
  statistics.py     J-K Sharpe test, bootstrap, Ljung-Box, JB, ADF
  backtest.py       vectorized backtester + position sizing + walk-forward
  portfolio.py      min-variance/tangency/efficient/BL + shrinkage cov + stress
  output.py         rich tables + JSON export
tests/              offline pytest suite
main.py             entry shim (python main.py == qfcli)
```

## Disclaimer

For research and education. Nothing here is investment advice.

## License

MIT — see [LICENSE](LICENSE).