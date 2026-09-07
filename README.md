# QFcli

<div align="center">

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/MRX-72/QFcli/actions/workflows/ci.yml/badge.svg)](https://github.com/MRX-72/QFcli/actions/workflows/ci.yml)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)]()

</div>

**QFcli** is a command-line tool for quantitative stock analysis. It pulls
OHLCV data, computes return/risk metrics and technical indicators, and renders
the results as readable terminal tables — or clean JSON for scripting.

Data comes live from Yahoo Finance via `yfinance`.

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
```

### Single stock

```bash
qfcli NVDA                    # default period: 1y
qfcli AAPL --period 5y        # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
qfcli SPY --rf 0.045          # risk-free rate for Sharpe ratio
qfcli NVDA --benchmark SPY    # beta vs S&P 500 ETF
qfcli AAPL --json             # machine-readable JSON
```

### Comparison

```bash
qfcli --compare GOOGL META
qfcli -c BTC-USD ETH-USD --period 6mo
```

### Terminal visuals

The text output includes a few lightweight, dependency-free visualizations:

- A **price sparkline** of the last 60 sessions, with per-bar up/down coloring.
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
| `--json`              | Emit machine-readable JSON |

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

- **Returns**: cumulative return, average daily return, annualized volatility, Sharpe ratio, **Sortino ratio**, max drawdown, win ratio, **VaR (95%, historical)**, **beta vs a chosen benchmark**
- **Technicals**: SMA (20/50/200), RSI-14, MACD (12,26,9), Bollinger Bands (20,2), rate of change (12-day), **ATR-14**, **golden/death cross detection**
- **Signals**: price-vs-moving-average bullish/bearish labels; risk bucket derived from annualized volatility (<15% stable, 15–30% moderate, >30% high)
- **Calendar**: per-year return breakdown

See `quant_finance/metrics.py` and `quant_finance/indicators.py` for the exact
formulas.

## Development

- `quant_finance/analysis.py` is the single source of truth for per-stock
  analysis; both single-stock and comparison modes share it.
- The `--json` path round-trips `export_to_dict`, which converts non-finite
  floats to `null` so machine output is always valid JSON.
- `pytest` covers the math layer (metrics, indicators), the comparison scoring,
  and the JSON export using synthetic, deterministic data — the test suite runs
  without network access.
- CI runs the suite on macOS and Linux.

## Project structure

```
quant_finance/
  analysis.py       shared per-stock analysis logic
  cli.py            argument parsing + CLI flow
  comparison.py     head-to-head scoring and recommendations
  data_fetcher.py   yfinance wrapper
  indicators.py     SMA, EMA, RSI, MACD, Bollinger Bands
  metrics.py        return/risk math
  output.py         rich tables + JSON export
tests/              offline pytest suite
main.py             entry shim (python main.py == qfcli)
```

## Disclaimer

For research and education. Nothing here is investment advice.

## License

MIT — see [LICENSE](LICENSE).