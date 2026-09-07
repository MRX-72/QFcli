"""
Quant Finance CLI - CLI entry point.

A quantitative finance tool for analyzing stock tickers with various metrics.
"""

import sys
import json
import argparse

from .analysis import analyze_single_stock
from .comparison import compare_stocks
from .output import (
    format_results,
    format_comparison_results,
    export_to_dict
)
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

console = Console()

VALID_PERIODS = ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')


def print_header() -> None:
    """Print the CLI header panel."""
    header = Panel(
        "[bold cyan]QUANT FINANCE CLI[/bold cyan]\n[dim]Stock Analysis Tool[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    )
    console.print()
    console.print(header)
    console.print()


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description='Quant Finance CLI - Analyze stock tickers with quantitative metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qfcli AAPL                    # Analyze Apple stock (1 year)
  qfcli MSFT --period 2y        # Analyze Microsoft (2 years)
  qfcli TSLA --rf 0.045         # Analyze Tesla with 4.5% risk-free rate
  qfcli --compare AAPL MSFT     # Compare Apple vs Microsoft
  qfcli -c TSLA KO --period 6mo # Compare Tesla vs Coca-Cola (6 months)
  qfcli AAPL --json             # Machine-readable JSON output
        """
    )

    parser.add_argument(
        '--compare', '-c',
        nargs=2,
        metavar=('TICKER1', 'TICKER2'),
        help='Compare two stocks side-by-side'
    )

    parser.add_argument(
        'ticker',
        type=str,
        nargs='?',
        help='Stock ticker symbol (e.g., AAPL, MSFT, TSLA)'
    )

    parser.add_argument(
        '--period',
        type=str,
        default='1y',
        help=f'Historical data period (default: 1y). Options: {", ".join(VALID_PERIODS)}'
    )

    parser.add_argument(
        '--rf',
        '--risk-free-rate',
        type=float,
        default=0.0,
        dest='risk_free_rate',
        help='Annual risk-free rate for Sharpe ratio (default: 0.0). Example: 0.045 for 4.5%%'
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help='Emit results as JSON instead of rich tables'
    )

    return parser


def analyze_with_progress(ticker: str, period: str, risk_free_rate: float):
    """Run a single-stock analysis behind a spinner."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"[cyan]Analyzing {ticker.upper()}...", total=None)
        return analyze_single_stock(ticker, period, risk_free_rate)


def emit_json(data) -> None:
    """Write a dict to stdout as clean, sortable JSON."""
    sys.stdout.write(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.period not in VALID_PERIODS:
        console.print(
            f"\n[bold red]Error:[/bold red] invalid period '{args.period}'. "
            f"Valid options: {', '.join(VALID_PERIODS)}\n"
        )
        return 2

    if not args.json:
        print_header()

    if args.compare:
        ticker1, ticker2 = args.compare
        comparison = compare_stocks(ticker1, ticker2, args.period, args.risk_free_rate)

        if comparison:
            if args.json:
                emit_json(comparison)
            else:
                format_comparison_results(comparison)
            return 0
        console.print(
            "\n[bold red]Error:[/bold red] comparison failed. "
            "Check the ticker symbols and try again.\n"
        )
        return 1

    if args.ticker:
        try:
            if args.json:
                result = analyze_single_stock(args.ticker, args.period, args.risk_free_rate)
                emit_json(export_to_dict(result['ticker'], result['company_name'], result))
            else:
                result = analyze_with_progress(args.ticker, args.period, args.risk_free_rate)
                format_results(
                    result['ticker'],
                    result['company_name'],
                    result
                )
        except ValueError as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}\n")
            return 1
        return 0

    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())