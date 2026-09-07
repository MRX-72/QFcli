"""
Quant Finance CLI - CLI entry point.

A quantitative finance tool for analyzing stock tickers with various metrics.
"""

import sys
import json
import argparse

from .analysis import analyze_single_stock
from .comparison import compare_stocks
from .data_fetcher import fetch_stock_data, fetch_benchmark_returns
from .backtest import (
    BUILTIN_STRATEGIES,
    run_backtest,
    prepare_backtest_data,
    load_custom_strategy,
    parse_strategy_params,
    parse_param_grid,
    walk_forward,
)
from .portfolio import build_portfolio_report, returns_matrix, black_litterman
from .output import (
    format_results,
    format_comparison_results,
    format_backtest_results,
    format_walk_forward_results,
    format_portfolio_results,
    export_to_dict
)
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

console = Console()

VALID_PERIODS = ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
VALID_STRATEGIES = tuple(BUILTIN_STRATEGIES.keys())


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
  qfcli NVDA --benchmark SPY    # Beta vs S&P 500 ETF
  qfcli --compare AAPL MSFT     # Compare Apple vs Microsoft
  qfcli -c TSLA KO --period 6mo # Compare Tesla vs Coca-Cola (6 months)
  qfcli AAPL --json             # Machine-readable JSON output
  qfcli --backtest AAPL -s sma_cross --cost 5     # Backtest a strategy
  qfcli --backtest AAPL --strategy-file strat.py  # Backtest a custom strategy
  qfcli --backtest AAPL -s sma_cross --sizer target_vol  # Vol-target position sizing
  qfcli --backtest AAPL --walk-forward --grid "fast=10,20;slow=40,80"  # OOS validation
  qfcli --portfolio AAPL MSFT NVDA KO --period 2y # Portfolio mode
  qfcli --portfolio AAPL MSFT --bl --view NVDA=0.18 # Black-Litterman with a view
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
        '--benchmark',
        type=str,
        default=None,
        metavar='SYMBOL',
        help='Benchmark/index for beta (e.g., SPY, ^GSPC). Single-stock mode only.'
    )

    parser.add_argument(
        '--backtest',
        type=str,
        default=None,
        metavar='TICKER',
        help='Backtest a strategy on a ticker'
    )

    parser.add_argument(
        '--strategy', '-s',
        type=str,
        default='sma_cross',
        choices=VALID_STRATEGIES,
        help=f'Strategy to backtest (default: sma_cross). Options: {", ".join(VALID_STRATEGIES)}'
    )

    parser.add_argument(
        '--cost',
        type=float,
        default=5.0,
        help='Round-trip transaction cost in basis points (default: 5)'
    )

    parser.add_argument(
        '--slippage',
        type=float,
        default=5.0,
        help='Slippage per trade in basis points (default: 5)'
    )

    parser.add_argument(
        '--sizer',
        type=str,
        default='fixed',
        choices=('fixed', 'target_vol', 'kelly'),
        help="Position-sizing scheme (default: fixed). "
             "target_vol scales exposure to a volatility target; "
             "kelly applies fractional Kelly to the signal"
    )

    parser.add_argument(
        '--sizer-target',
        type=float,
        default=0.15,
        help='Annual volatility target for --sizer target_vol (default: 0.15)'
    )

    parser.add_argument(
        '--sizer-window',
        type=int,
        default=20,
        help='Rolling window (days) for target_vol sizing or kelly stats (default: 20)'
    )

    parser.add_argument(
        '--max-leverage',
        type=float,
        default=1.0,
        help='Maximum gross exposure for target_vol sizing (default: 1.0)'
    )

    parser.add_argument(
        '--kelly-fraction',
        type=float,
        default=0.25,
        help='Fraction of full Kelly applied by the kelly sizer (default: 0.25)'
    )

    parser.add_argument(
        '--walk-forward',
        action='store_true',
        help='Validate strategy parameters out of sample with expanding-window walk-forward'
    )

    parser.add_argument(
        '--grid',
        type=str,
        default=None,
        metavar='"KEY=v1,v2;K2=v3,v4"',
        help='Parameter grid for --walk-forward, parsed as a cartesian product'
    )

    parser.add_argument(
        '--train-frac',
        type=float,
        default=0.6,
        help='Initial in-sample fraction for --walk-forward (default: 0.6)'
    )

    parser.add_argument(
        '--wf-step',
        type=int,
        default=63,
        help='Days added to the in-sample window per --walk-forward fold (default: 63)'
    )

    parser.add_argument(
        '--bl',
        action='store_true',
        help='Use Black-Litterman construction in portfolio mode (prior + optional views)'
    )

    parser.add_argument(
        '--view',
        type=str,
        action='append',
        default=None,
        metavar='TICKER=RATE',
        help='Absolute expected-return view for Black-Litterman, e.g. NVDA=0.18 (repeatable)'
    )

    parser.add_argument(
        '--view-confidence',
        type=float,
        default=None,
        help='Per-view error variance for Black-Litterman views (default: 0.0025)'
    )

    parser.add_argument(
        '--strategy-file',
        type=str,
        default=None,
        metavar='PATH',
        help='Load a custom strategy from a .py file with a custom_strategy(prices, **kwargs) function'
    )

    parser.add_argument(
        '--strategy-param',
        type=str,
        action='append',
        default=None,
        metavar='KEY=VALUE',
        help='Extra key=value parameter passed to the strategy (repeatable)'
    )

    parser.add_argument(
        '--fast',
        type=int,
        default=20,
        help='Fast SMA window for sma_cross (default: 20)'
    )

    parser.add_argument(
        '--slow',
        type=int,
        default=50,
        help='Slow SMA window for sma_cross (default: 50)'
    )

    parser.add_argument(
        '--lookback',
        type=int,
        default=20,
        help='ROC lookback for momentum strategy (default: 20)'
    )

    parser.add_argument(
        '--hold',
        type=int,
        default=20,
        help='Holding period for momentum strategy (default: 20)'
    )

    parser.add_argument(
        '--portfolio',
        type=str,
        nargs='+',
        default=None,
        metavar='TICKER',
        help='Portfolio mode: analyze a basket of tickers (minimum 2)'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Bypass the on-disk data cache and hit the network'
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help='Emit results as JSON instead of rich tables'
    )

    return parser


def analyze_with_progress(ticker: str, period: str, risk_free_rate: float, benchmark_returns=None,
                          use_cache: bool = True):
    """Run a single-stock analysis behind a spinner."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"[cyan]Analyzing {ticker.upper()}...", total=None)
        return analyze_single_stock(ticker, period, risk_free_rate, benchmark_returns,
                                    use_cache=use_cache)


def emit_json(data) -> None:
    """Write a dict to stdout as clean, sortable JSON.

    Internal keys prefixed with underscore (e.g. daily return Series used for
    walk-forward stitching) are stripped so the output stays JSON-serializable.
    """
    clean = {}
    for k, v in data.items():
        if k.startswith('_'):
            continue
        if isinstance(v, dict):
            v = {kk: vv for kk, vv in v.items() if not kk.startswith('_')}
        clean[k] = v
    sys.stdout.write(json.dumps(clean, indent=2, sort_keys=True) + "\n")


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

    if args.backtest:
        ticker = args.backtest
        try:
            df, name = fetch_stock_data(ticker, args.period, use_cache=not args.no_cache)
            prices = prepare_backtest_data(df)

            extra_params = parse_strategy_params(args.strategy_param or [])

            if args.strategy_file:
                strategy = load_custom_strategy(args.strategy_file)
                strategy_label = f"custom:{args.strategy_file}"
                label = f"Custom Strategy ({args.strategy_file.rsplit('/', 1)[-1]})"
            else:
                strategy_label = args.strategy
                label = args.strategy
                strategy = BUILTIN_STRATEGIES[args.strategy]
                if args.strategy == 'sma_cross':
                    extra_params.setdefault('fast', args.fast)
                    extra_params.setdefault('slow', args.slow)
                elif args.strategy == 'momentum':
                    extra_params.setdefault('lookback', args.lookback)
                    extra_params.setdefault('hold', args.hold)

            if args.sizer == 'target_vol':
                sizer_kwargs = {'target_vol': args.sizer_target, 'window': args.sizer_window,
                                'max_leverage': args.max_leverage}
            elif args.sizer == 'kelly':
                sizer_kwargs = {'fraction': args.kelly_fraction, 'window': args.sizer_window}
            else:
                sizer_kwargs = {}

            if args.walk_forward:
                grid = parse_param_grid(args.grid) if args.grid else [dict(extra_params)]
                wf = walk_forward(
                    prices, strategy, grid,
                    sizer=args.sizer, sizer_kwargs=sizer_kwargs,
                    cost_bps=args.cost, slippage_bps=args.slippage,
                    risk_free_rate=args.risk_free_rate,
                    train_frac=args.train_frac, step=args.wf_step,
                )
                if args.json:
                    wf['strategy'] = {
                        'label': strategy_label,
                        'params': extra_params,
                        'grid': grid,
                    }
                    emit_json(wf)
                else:
                    format_walk_forward_results(wf, ticker, label)
                return 0

            result = run_backtest(
                prices,
                strategy,
                cost_bps=args.cost,
                slippage_bps=args.slippage,
                risk_free_rate=args.risk_free_rate,
                sizer=args.sizer,
                sizer_kwargs=sizer_kwargs,
                **extra_params
            )
            if args.json:
                result['strategy'] = {
                    'label': strategy_label,
                    'params': extra_params,
                    'sizer': args.sizer if args.sizer != 'fixed' else None,
                }
                emit_json(result)
            else:
                format_backtest_results(result, ticker, label)
        except ValueError as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}\n")
            return 1
        return 0

    if args.portfolio:
        if len(args.portfolio) < 2:
            console.print("\n[bold red]Error:[/bold red] --portfolio needs at least two tickers.\n")
            return 1
        if args.benchmark:
            console.print("\n[bold red]Error:[/bold red] --benchmark is not used in portfolio mode.\n")
            return 1
        try:
            dfs = []
            for t in args.portfolio:
                df, _ = fetch_stock_data(t, args.period, use_cache=not args.no_cache)
                dfs.append(df.assign(Close=df['Close']))
            ret_m = returns_matrix(dfs)
            ret_m.columns = [t.upper() for t in args.portfolio]

            bl = None
            if args.bl or args.view:
                views = {}
                for token in args.view or []:
                    if '=' not in token:
                        raise ValueError(f"Invalid view '{token}' (expected TICKER=RATE, e.g. NVDA=0.18)")
                    tk, _, rate = token.partition('=')
                    views[tk.strip().upper()] = float(rate.strip())
                bl = black_litterman(
                    ret_m,
                    views=views if views else None,
                    view_confidence=args.view_confidence,
                    risk_free_rate=args.risk_free_rate,
                )

            report = build_portfolio_report(
                ret_m,
                risk_free_rate=args.risk_free_rate,
                bl=bl,
            )
            if args.json:
                payload = {
                    'tickers': report['tickers'],
                    'weights': report['weights'].to_dict(orient='index'),
                    'stats': report['stats'],
                    'stress': report['stress'],
                }
                if bl is not None:
                    payload['bl'] = report['bl']
                emit_json(payload)
            else:
                format_portfolio_results(report)
        except ValueError as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}\n")
            return 1
        return 0

    if args.compare:
        ticker1, ticker2 = args.compare
        comparison = compare_stocks(ticker1, ticker2, args.period, args.risk_free_rate,
                                    use_cache=not args.no_cache)

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
        if args.benchmark and args.ticker.upper() == args.benchmark.upper():
            console.print("\n[bold red]Error:[/bold red] benchmark must differ from the analyzed ticker.\n")
            return 1

        try:
            benchmark_returns = None
            if args.benchmark:
                benchmark_returns = fetch_benchmark_returns(args.benchmark, args.period,
                                                            use_cache=not args.no_cache)

            if args.json:
                result = analyze_single_stock(
                    args.ticker, args.period, args.risk_free_rate, benchmark_returns,
                    use_cache=not args.no_cache
                )
                emit_json(export_to_dict(result['ticker'], result['company_name'], result))
            else:
                result = analyze_with_progress(
                    args.ticker, args.period, args.risk_free_rate, benchmark_returns,
                    use_cache=not args.no_cache
                )
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