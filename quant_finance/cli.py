"""
Quant Finance CLI - CLI entry point.

A quantitative finance tool for analyzing stock tickers with various metrics.
"""

import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .analysis import analyze_single_stock
from .backtest import (
    BUILTIN_STRATEGIES,
    load_custom_strategy,
    parse_param_grid,
    parse_strategy_params,
    prepare_backtest_data,
    run_backtest,
    walk_forward,
)
from .comparison import compare_stocks
from .data_fetcher import fetch_benchmark_returns, fetch_stock_data
from .factor_model import factor_premia, fetch_ff_factors, ff_betas, ff_expected_returns
from .output import (
    export_to_dict,
    format_backtest_results,
    format_comparison_results,
    format_paper_trade_results,
    format_portfolio_results,
    format_results,
    format_walk_forward_results,
)
from .paper_trade import load_paper_state, paper_state_path, paper_trade, save_paper_state
from .portfolio import black_litterman, build_portfolio_report, returns_matrix

console = Console()
error_console = Console(stderr=True)

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
  qfcli --backtest AAPL --walk-forward --ensemble rank --grid "fast=10;slow=40,80"  # robust blend
  qfcli --backtest AAPL -s target_vol --slow-vol-window 126 --sizer-target 0.15  # vol-managed sizing
  qfcli --paper-trade AAPL -s sma_cross --grid "fast=10,20;slow=40,60" --stable-days 5  # stable params
  qfcli --portfolio AAPL MSFT NVDA KO --period 2y # Portfolio mode
  qfcli --portfolio AAPL MSFT --bl --view NVDA=0.18 # Black-Litterman with a view
  qfcli --portfolio AAPL MSFT NVDA --cov-method factor --ff  # PCA risk model + Fama-French overlay
        """
    )

    parser.add_argument(
        'ticker',
        type=str,
        nargs='?',
        help='Stock ticker symbol (e.g., AAPL, MSFT, TSLA)'
    )

    mode = parser.add_argument_group('modes')
    mode.add_argument(
        '--compare', '-c',
        nargs=2,
        metavar=('TICKER1', 'TICKER2'),
        help='Compare two stocks side-by-side'
    )
    mode.add_argument(
        '--backtest',
        type=str,
        default=None,
        metavar='TICKER',
        help='Backtest a strategy on a ticker'
    )
    mode.add_argument(
        '--portfolio',
        type=str,
        nargs='+',
        default=None,
        metavar='TICKER',
        help='Portfolio mode: analyze a basket of tickers (minimum 2)'
    )
    mode.add_argument(
        '--paper-trade',
        type=str,
        default=None,
        metavar='TICKER',
        help='Paper-trade a strategy day by day with walk-forward-selected '
             'params and persist the account state to disk'
    )

    data = parser.add_argument_group('data')
    data.add_argument(
        '--period',
        type=str,
        default='1y',
        help=f'Historical data period (default: 1y). Options: {", ".join(VALID_PERIODS)}'
    )
    data.add_argument(
        '--rf',
        '--risk-free-rate',
        type=float,
        default=0.0,
        dest='risk_free_rate',
        help='Annual risk-free rate for Sharpe ratio (default: 0.0). Example: 0.045 for 4.5%%'
    )
    data.add_argument(
        '--benchmark',
        type=str,
        default=None,
        metavar='SYMBOL',
        help='Benchmark/index for beta (e.g., SPY, ^GSPC). Single-stock mode only.'
    )
    data.add_argument(
        '--no-cache',
        action='store_true',
        help='Bypass the on-disk data cache and hit the network'
    )

    strategy = parser.add_argument_group('strategy')
    strategy.add_argument(
        '--strategy', '-s',
        type=str,
        default='sma_cross',
        choices=VALID_STRATEGIES,
        help=f'Strategy to backtest (default: sma_cross). Options: {", ".join(VALID_STRATEGIES)}'
    )
    strategy.add_argument(
        '--strategy-file',
        type=str,
        default=None,
        metavar='PATH',
        help='Load a custom strategy from a .py file with a custom_strategy(prices, **kwargs) function'
    )
    strategy.add_argument(
        '--strategy-param',
        type=str,
        action='append',
        default=None,
        metavar='KEY=VALUE',
        help='Extra key=value parameter passed to the strategy (repeatable)'
    )
    strategy.add_argument(
        '--cost',
        type=float,
        default=5.0,
        help='Round-trip transaction cost in basis points (default: 5)'
    )
    strategy.add_argument(
        '--slippage',
        type=float,
        default=5.0,
        help='Slippage per trade in basis points (default: 5)'
    )
    strategy.add_argument(
        '--fast',
        type=int,
        default=20,
        help='Fast SMA window for sma_cross (default: 20)'
    )
    strategy.add_argument(
        '--slow',
        type=int,
        default=50,
        help='Slow SMA window for sma_cross (default: 50)'
    )
    strategy.add_argument(
        '--lookback',
        type=int,
        default=20,
        help='ROC lookback for momentum strategy (default: 20)'
    )
    strategy.add_argument(
        '--hold',
        type=int,
        default=20,
        help='Holding period for momentum strategy (default: 20)'
    )

    sizing = parser.add_argument_group('position sizing')
    sizing.add_argument(
        '--sizer',
        type=str,
        default='fixed',
        choices=('fixed', 'target_vol', 'kelly'),
        help="Position-sizing scheme (default: fixed). "
             "target_vol scales exposure to a volatility target; "
             "kelly applies fractional Kelly to the signal"
    )
    sizing.add_argument(
        '--sizer-target',
        type=float,
        default=0.15,
        help='Annual volatility target for --sizer target_vol (default: 0.15)'
    )
    sizing.add_argument(
        '--sizer-window',
        type=int,
        default=20,
        help='Rolling window (days) for target_vol sizing or kelly stats (default: 20)'
    )
    sizing.add_argument(
        '--slow-vol-window',
        type=int,
        default=0,
        help='Slow realized-vol window for vol-managed target_vol sizing '
             '(default: 0 = disabled). Position is scaled by slow vol '
             '(Moreira-Muir) and the fast target caps exposure on vol spikes.'
    )
    sizing.add_argument(
        '--max-leverage',
        type=float,
        default=1.0,
        help='Maximum gross exposure for target_vol sizing (default: 1.0)'
    )
    sizing.add_argument(
        '--kelly-fraction',
        type=float,
        default=0.25,
        help='Fraction of full Kelly applied by the kelly sizer (default: 0.25)'
    )

    validation = parser.add_argument_group('walk-forward validation')
    validation.add_argument(
        '--walk-forward',
        action='store_true',
        help='Validate strategy parameters out of sample with expanding-window walk-forward'
    )
    validation.add_argument(
        '--grid',
        type=str,
        default=None,
        metavar='"KEY=v1,v2;K2=v3,v4"',
        help='Parameter grid for --walk-forward, parsed as a cartesian product'
    )
    validation.add_argument(
        '--train-frac',
        type=float,
        default=0.6,
        help='Initial in-sample fraction for --walk-forward (default: 0.6)'
    )
    validation.add_argument(
        '--wf-step',
        type=int,
        default=63,
        help='Days added to the in-sample window per --walk-forward fold (default: 63)'
    )
    validation.add_argument(
        '--ensemble',
        type=str,
        default='best',
        choices=('best', 'equal', 'rank', 'topk'),
        help="How to combine grid points out of sample: best (default), "
             "equal, rank (rank-weighted) or topk. More robust than betting "
             "on a single winning config"
    )
    validation.add_argument(
        '--topk',
        type=int,
        default=None,
        help='Number of top configs averaged by --ensemble topk (default: half the grid, min 2)'
    )
    validation.add_argument(
        '--stable-days',
        type=int,
        default=1,
        help='Paper trading: consecutive winning days required before the '
             'active config switches (hysteresis; default 1 = switch every day)'
    )

    portfolio = parser.add_argument_group('portfolio')
    portfolio.add_argument(
        '--cov-method',
        type=str,
        default='lw',
        choices=('sample', 'lw', 'factor'),
        help="Covariance estimator in portfolio mode: sample, lw "
             "(Ledoit-Wolf shrinkage, default) or factor (PCA risk model)"
    )
    portfolio.add_argument(
        '--bl',
        action='store_true',
        help='Use Black-Litterman construction in portfolio mode (prior + optional views)'
    )
    portfolio.add_argument(
        '--view',
        type=str,
        action='append',
        default=None,
        metavar='TICKER=RATE',
        help='Absolute expected-return view for Black-Litterman, e.g. NVDA=0.18 (repeatable)'
    )
    portfolio.add_argument(
        '--view-confidence',
        type=float,
        default=None,
        help='Per-view error variance for Black-Litterman views (default: 0.0025)'
    )
    portfolio.add_argument(
        '--ff',
        action='store_true',
        help='Fama-French overlay in portfolio mode: factor exposures, risk '
             'premia, and (with --bl) a factor-model expected-return prior '
             '(downloads the daily FF research factors; may require network)'
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help='Emit results as JSON instead of rich tables'
    )

    return parser


def _fail(message: str, code: int = 1) -> int:
    """Report a user-facing error on stderr and return its exit code."""
    error_console.print(f"\n[bold red]Error:[/bold red] {message}\n")
    return code


def _warn(message: str) -> None:
    """Report a non-fatal warning on stderr."""
    error_console.print(f"\n[yellow]Warning:[/yellow] {message}\n")


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


def _resolve_strategy(args) -> Tuple[object, str, str, Dict]:
    """Resolve the strategy to run.

    Returns:
        Tuple of (strategy callable, state label, display label, params).
        CLI windows (--fast/--slow, --lookback/--hold) are applied as defaults
        so an explicit --strategy-param always wins.
    """
    params = parse_strategy_params(args.strategy_param or [])

    if args.strategy_file:
        strategy = load_custom_strategy(args.strategy_file)
        name = os.path.basename(args.strategy_file)
        return strategy, f"custom:{args.strategy_file}", f"Custom Strategy ({name})", params

    builtin_defaults = {
        'sma_cross': {'fast': args.fast, 'slow': args.slow},
        'momentum': {'lookback': args.lookback, 'hold': args.hold},
    }
    for key, value in builtin_defaults.get(args.strategy, {}).items():
        params.setdefault(key, value)
    return BUILTIN_STRATEGIES[args.strategy], args.strategy, args.strategy, params


def _sizer_kwargs(args) -> Dict:
    """Translate the sizing flags into size_position() kwargs."""
    if args.sizer == 'target_vol':
        kwargs = {
            'target_vol': args.sizer_target,
            'window': args.sizer_window,
            'max_leverage': args.max_leverage,
        }
        if args.slow_vol_window:
            kwargs['slow_window'] = args.slow_vol_window
        return kwargs
    if args.sizer == 'kelly':
        return {'fraction': args.kelly_fraction, 'window': args.sizer_window}
    return {}


def _parse_views(tokens) -> Dict[str, float]:
    """Parse repeatable --view TICKER=RATE tokens."""
    views: Dict[str, float] = {}
    for token in tokens or []:
        if '=' not in token:
            raise ValueError(f"Invalid view '{token}' (expected TICKER=RATE, e.g. NVDA=0.18)")
        ticker, _, rate = token.partition('=')
        views[ticker.strip().upper()] = float(rate.strip())
    return views


def _ff_overlay(returns, factors) -> Dict:
    """Fit Fama-French betas and annualized premia for the display layer."""
    fit = ff_betas(returns, factors)
    return {
        'betas': fit['betas'],
        'alpha_annual': fit['alpha_annual'],
        'premia': factor_premia(factors),
    }


def _run_backtest_or_paper(args) -> int:
    """Run backtest, walk-forward or paper-trade mode."""
    ticker = args.backtest or args.paper_trade
    try:
        df, _ = fetch_stock_data(ticker, args.period, use_cache=not args.no_cache)
        prices = prepare_backtest_data(df)
        strategy, strategy_label, label, extra_params = _resolve_strategy(args)
        sizer_kwargs = _sizer_kwargs(args)
        grid = parse_param_grid(args.grid) if args.grid else [dict(extra_params)]

        if args.walk_forward:
            wf = walk_forward(
                prices, strategy, grid,
                sizer=args.sizer, sizer_kwargs=sizer_kwargs,
                cost_bps=args.cost, slippage_bps=args.slippage,
                risk_free_rate=args.risk_free_rate,
                train_frac=args.train_frac, step=args.wf_step,
                ensemble=args.ensemble, topk=args.topk,
            )
            if args.json:
                wf['strategy'] = {
                    'label': strategy_label,
                    'params': extra_params,
                    'grid': grid,
                    'ensemble': args.ensemble,
                }
                emit_json(wf)
            else:
                format_walk_forward_results(wf, ticker, label)
            return 0

        if args.paper_trade:
            state = paper_trade(
                prices, strategy, grid,
                sizer=args.sizer, sizer_kwargs=sizer_kwargs,
                cost_bps=args.cost, slippage_bps=args.slippage,
                risk_free_rate=args.risk_free_rate,
                train_frac=args.train_frac, ensemble=args.ensemble,
                topk=args.topk, stable_days=args.stable_days,
            )
            state['strategy'] = strategy_label
            state['ticker'] = ticker.upper()
            state_path = paper_state_path(ticker, strategy_label)
            previous = load_paper_state(state_path)
            save_paper_state(state_path, state)
            if args.json:
                emit_json(state)
            else:
                # the daily tail only exists for state continuity; drop it from display
                display = {k: v for k, v in state.items() if k != 'daily_tail'}
                format_paper_trade_results(display, previous, ticker, label)
            return 0

        result = run_backtest(
            prices, strategy,
            cost_bps=args.cost, slippage_bps=args.slippage,
            risk_free_rate=args.risk_free_rate,
            sizer=args.sizer, sizer_kwargs=sizer_kwargs,
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
        return _fail(e)
    return 0


def _run_portfolio(args) -> int:
    """Run portfolio optimization mode."""
    if len(args.portfolio) < 2:
        return _fail("--portfolio needs at least two tickers.")

    try:
        dfs = [fetch_stock_data(t, args.period, use_cache=not args.no_cache)[0]
               for t in args.portfolio]
        returns = returns_matrix(dfs)
        returns.columns = [t.upper() for t in args.portfolio]

        ff = None
        factors = None
        if args.ff:
            try:
                factors = fetch_ff_factors(use_cache=not args.no_cache)
                ff = _ff_overlay(returns, factors)
            except ValueError as e:
                _warn(f"{e}\nProceeding without the Fama-French overlay.")

        bl = None
        if args.bl or args.view:
            prior_returns = None
            prior_label = 'implied'
            if ff is not None:
                prior_returns = ff_expected_returns(
                    returns, factors, ff['premia'], rf=args.risk_free_rate
                )
                prior_label = 'ff'
            views = _parse_views(args.view)
            bl = black_litterman(
                returns,
                views=views or None,
                view_confidence=args.view_confidence,
                risk_free_rate=args.risk_free_rate,
                cov_method=args.cov_method,
                prior_returns=prior_returns,
                prior_label=prior_label,
            )

        report = build_portfolio_report(
            returns,
            risk_free_rate=args.risk_free_rate,
            bl=bl,
            cov_method=args.cov_method,
            ff=ff,
        )
        if args.json:
            emit_json(_portfolio_payload(report))
        else:
            format_portfolio_results(report)
    except ValueError as e:
        return _fail(e)
    return 0


def _portfolio_payload(report: Dict) -> Dict:
    """Select the JSON-serializable slice of a portfolio report."""
    payload = {
        'tickers': report['tickers'],
        'weights': report['weights'].to_dict(orient='index'),
        'stats': report['stats'],
        'stress': report['stress'],
        'cov_method': report['cov_method'],
    }
    for key in ('bl', 'factor_model', 'ff'):
        if report.get(key) is not None:
            payload[key] = report[key]
    return payload


def _run_compare(args) -> int:
    """Run the two-ticker comparison mode."""
    ticker1, ticker2 = args.compare
    try:
        comparison = compare_stocks(ticker1, ticker2, args.period, args.risk_free_rate,
                                    use_cache=not args.no_cache)
    except ValueError as e:
        return _fail(e)

    if args.json:
        emit_json(comparison)
    else:
        format_comparison_results(comparison)
    return 0


def _run_single(args) -> int:
    """Run single-stock analysis mode."""
    if args.benchmark and args.ticker.upper() == args.benchmark.upper():
        return _fail("benchmark must differ from the analyzed ticker.")

    use_cache = not args.no_cache
    try:
        benchmark_returns = None
        if args.benchmark:
            benchmark_returns = fetch_benchmark_returns(args.benchmark, args.period,
                                                        use_cache=use_cache)

        if args.json:
            result = analyze_single_stock(args.ticker, args.period, args.risk_free_rate,
                                          benchmark_returns, use_cache=use_cache)
            emit_json(export_to_dict(result['ticker'], result['company_name'], result))
        else:
            result = analyze_with_progress(args.ticker, args.period, args.risk_free_rate,
                                           benchmark_returns, use_cache=use_cache)
            format_results(result['ticker'], result['company_name'], result)
    except ValueError as e:
        return _fail(e)
    return 0


def _validate_args(args) -> Optional[int]:
    """Return an exit code for invalid flag combinations, else None."""
    if args.period not in VALID_PERIODS:
        return _fail(f"invalid period '{args.period}'. "
                     f"Valid options: {', '.join(VALID_PERIODS)}", code=2)
    if args.paper_trade and args.walk_forward:
        return _fail("--walk-forward cannot be combined with --paper-trade.")
    if args.benchmark and not args.ticker:
        return _fail("--benchmark is only valid in single-stock mode.")
    return None


def main() -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    invalid = _validate_args(args)
    if invalid is not None:
        return invalid

    if not args.json:
        print_header()

    if args.backtest or args.paper_trade:
        return _run_backtest_or_paper(args)

    if args.portfolio:
        return _run_portfolio(args)

    if args.compare:
        return _run_compare(args)

    if args.ticker:
        return _run_single(args)

    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
