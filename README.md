# Quant Finance CLI

A beautiful command-line tool for quantitative stock analysis with real-time data, technical indicators, and side-by-side comparisons.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Single Stock Analysis** - Comprehensive metrics including returns, volatility, Sharpe ratio, and max drawdown
- **Stock Comparison** - Side-by-side analysis of two stocks with winner determination
- **Technical Indicators** - SMA-20, SMA-50, SMA-200 with bullish/bearish signals
- **Beautiful CLI** - Color-coded output with tables and progress indicators
- **Risk Assessment** - Automated risk scoring and investor recommendations

## Installation

```bash
# Clone the repository
git clone https://github.com/MRX-72/QFcli.git
cd QFcli

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Analyze a Single Stock

```bash
python main.py AAPL
```

### Compare Two Stocks

```bash
python main.py --compare AAPL MSFT
```

### Custom Time Period

```bash
python main.py TSLA --period 2y
```

## Usage

```
python main.py [TICKER] [OPTIONS]
python main.py --compare TICKER1 TICKER2 [OPTIONS]

Options:
  --period PERIOD       Historical data period (default: 1y)
                        Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
  --rf RATE            Risk-free rate for Sharpe ratio (default: 0.0)
  -h, --help           Show help message
```

## Metrics Explained

| Metric | Description | Formula |
|--------|-------------|---------|
| **Cumulative Return** | Total profit/loss over period | (P_end - P_start) / P_start |
| **Volatility** | Annualized risk measure | σ × √252 |
| **Sharpe Ratio** | Risk-adjusted return | (r̄ - r_f) / σ |
| **Max Drawdown** | Worst peak-to-trough decline | min((P_t - max(P)) / max(P)) |
| **SMA** | Simple Moving Average | (1/n) × Σ(P_i) |

## Examples

### Growth Stock Analysis
```bash
python main.py NVDA --period 1y
```

### Tech Giants Comparison
```bash
python main.py --compare GOOGL META
```

### With Risk-Free Rate
```bash
python main.py SPY --rf 0.045
```

## Output Preview

**Single Stock Analysis:**
- Returns Analysis (cumulative return, volatility, Sharpe ratio)
- Moving Averages (SMA-20, 50, 200 with signals)
- Risk Metrics (max drawdown, ROC, risk score)

**Stock Comparison:**
- Side-by-side metric comparison
- Winner determination for each metric
- Overall score and recommendations
- Investor type suggestions

## Requirements

- Python 3.8+
- yfinance
- pandas
- numpy
- rich

## Project Structure

```
QFcli/
├── main.py                   # CLI entry point
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── quant_finance/
    ├── __init__.py          # Package initialization
    ├── data_fetcher.py      # Data retrieval
    ├── metrics.py           # Financial calculations
    ├── indicators.py        # Technical indicators
    ├── comparison.py        # Stock comparison logic
    └── output.py            # Rich formatting
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is for educational and informational purposes only. It is not financial advice. Always do your own research before making investment decisions.

## Acknowledgments

- Data provided by [Yahoo Finance](https://finance.yahoo.com/) via yfinance
- CLI styling powered by [Rich](https://github.com/Textualize/rich)

---

**Made with ❤️ for quantitative finance enthusiasts**
