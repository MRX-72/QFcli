"""
Quant Finance CLI - Main Entry Point

A quantitative finance tool for analyzing stock tickers with various metrics.
"""

import sys
import argparse
from typing import Optional

from quant_finance.data_fetcher import (
    fetch_stock_data,
    calculate_daily_returns,
    get_current_price,
    get_date_range
)
from quant_finance.metrics import (
    cumulative_return,
    volatility,
    sharpe_ratio,
    max_drawdown,
    risk_score,
    average_return,
    daily_return_stats
)
from quant_finance.indicators import (
    calculate_all_smas,
    moving_average_signals,
    rate_of_change,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    bollinger_bands
)
from quant_finance.output import format_results, format_comparison_results
from quant_finance.comparison import compare_stocks
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

console = Console()


def analyze_stock(ticker: str, period: str = '1y', risk_free_rate: float = 0.0) -> Optional[dict]:
    """
    Perform complete quantitative analysis on a stock ticker.
    
    Args:
        ticker: Stock symbol to analyze
        period: Historical data period (default: '1y')
        risk_free_rate: Annual risk-free rate for Sharpe ratio (default: 0.0)
    
    Returns:
        Dictionary with all metrics, or None if analysis fails
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # Step 1: Fetch data
            task = progress.add_task(f"[cyan]Fetching data for {ticker.upper()}...", total=None)
            df, company_name = fetch_stock_data(ticker, period)
            progress.update(task, completed=True)
            
            # Step 2: Extract price data
            prices = df['Close']
            current_price = get_current_price(df)
            start_date, end_date = get_date_range(df)
            
            # Step 3: Calculate returns
            task = progress.add_task("[cyan]Calculating returns...", total=None)
            returns = calculate_daily_returns(prices)
            progress.update(task, completed=True)
            
            # Step 4: Calculate core metrics
            task = progress.add_task("[cyan]Computing financial metrics...", total=None)
            cum_return = cumulative_return(prices)
            vol = volatility(returns, annualize=True)
            sharpe = sharpe_ratio(returns, risk_free_rate=risk_free_rate, annualize=True)
            mdd = max_drawdown(prices)
            avg_ret = average_return(returns, annualize=True)
            progress.update(task, completed=True)
            
            progress.update(task, completed=True)
            
            # Step 5: Calculate technical indicators
            task = progress.add_task("[cyan]Analyzing technical indicators...", total=None)
            sma_values = calculate_all_smas(prices, windows=[20, 50, 200])
            signals = moving_average_signals(current_price, sma_values)
            roc = rate_of_change(prices, period=12)
            
            # New indicators
            rsi = calculate_rsi(prices)
            macd_data = calculate_macd(prices)
            bb_data = bollinger_bands(prices)
            daily_stats = daily_return_stats(returns)
            
            progress.update(task, completed=True)
            
            # Step 6: Calculate risk score
            risk_rating = risk_score(vol)
        
        # Step 7: Compile results
        metrics = {
            'current_price': current_price,
            'start_date': start_date,
            'end_date': end_date,
            'cumulative_return': cum_return,
            'avg_return': avg_ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': mdd,
            'sma_values': sma_values,
            'signals': signals,
            'roc': roc,
            'risk_score': risk_rating,
            'rsi': rsi,
            'macd': macd_data,
            'bollinger_bands': bb_data,
            'daily_stats': daily_stats
        }
        
        return {
            'ticker': ticker,
            'company_name': company_name,
            'metrics': metrics
        }
    
    except ValueError as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        return None
    except Exception as e:
        console.print(f"\n[bold red]❌ Unexpected error:[/bold red] {e}")
        return None


def main():
    """
    Main CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description='Quant Finance CLI - Analyze stock tickers with quantitative metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py AAPL                    # Analyze Apple stock (1 year)
  python main.py MSFT --period 2y        # Analyze Microsoft (2 years)
  python main.py TSLA --rf 0.045         # Analyze Tesla with 4.5%% risk-free rate
  python main.py --compare AAPL MSFT     # Compare Apple vs Microsoft
  python main.py -c TSLA KO --period 6mo # Compare Tesla vs Coca-Cola (6 months)
        """
    )
    
    # Comparison mode
    parser.add_argument(
        '--compare', '-c',
        nargs=2,
        metavar=('TICKER1', 'TICKER2'),
        help='Compare two stocks side-by-side'
    )
    
    # Single stock mode
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
        help='Historical data period (default: 1y). Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max'
    )
    
    parser.add_argument(
        '--rf',
        '--risk-free-rate',
        type=float,
        default=0.0,
        dest='risk_free_rate',
        help='Annual risk-free rate for Sharpe ratio (default: 0.0). Example: 0.045 for 4.5%%'
    )
    
    args = parser.parse_args()
    
    # Print header with Rich
    header = Panel(
        "[bold cyan]QUANT FINANCE CLI[/bold cyan]\n[dim]Stock Analysis Tool[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    )
    console.print()
    console.print(header)
    console.print()
    
    # Comparison mode
    if args.compare:
        ticker1, ticker2 = args.compare
        
        comparison = compare_stocks(ticker1, ticker2, args.period, args.risk_free_rate)
        
        if comparison:
            format_comparison_results(comparison)
            return 0
        else:
            console.print("\n[bold red]❌ Comparison failed.[/bold red] Please check the ticker symbols and try again.\n")
            return 1
    
    # Single stock mode
    elif args.ticker:
        result = analyze_stock(args.ticker, args.period, args.risk_free_rate)
        
        if result:
            # Format and display results
            format_results(
                result['ticker'],
                result['company_name'],
                result['metrics']
            )
            
            return 0
        else:
            console.print("\n[bold red]❌ Analysis failed.[/bold red] Please check the ticker symbol and try again.\n")
            return 1
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
