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
qfcli AAPL --json             # machine-readable JSON
```

### Comparison

```bash
qfcli --compare GOOGL META
qfcli -c BTC-USD ETH-USD --period 6mo
```

### Options

| Flag | Description |
|------|-------------|
| `-c, --compare T1 T2` | Compare two tickers |
| `--period`            | Data period (default `1y`) |
| `--rf RATE`           | Annual risk-free rate (default `0.0`) |
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
    "sharpe_ratio": 1.7429
  },
  "risk_metrics": {
    "max_drawdown": -0.1271,
    "risk_score": "MODERATE"
  }
}
```

## Metrics

- **Returns**: cumulative return, average daily return, annualized volatility, Sharpe ratio, max drawdown, win ratio
- **Technicals**: SMA (20/50/200), RSI-14, MACD (12,26,9), Bollinger Bands (20,2), rate of change (12-day)
- **Signals**: price-vs-moving-average bullish/bearish labels; risk bucket derived from annualized volatility (<15% stable, 15–30% moderate, >30% high)

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