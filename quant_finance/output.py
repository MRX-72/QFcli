"""
Output Module
Format and display analysis results using Rich library for beautiful CLI output.
"""

from typing import Dict, Any
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.style import Style

console = Console()


def get_color_for_percentage(value: float) -> str:
    """
    Get color based on percentage value.
    
    Args:
        value: Percentage value
    
    Returns:
        Color name for Rich
    """
    if np.isnan(value):
        return "dim"
    return "green" if value >= 0 else "red"


def get_color_for_signal(signal: str) -> str:
    """
    Get color for moving average signal.
    
    Args:
        signal: Signal string (BULLISH, BEARISH, etc.)
    
    Returns:
        Color name for Rich
    """
    if signal == "BULLISH":
        return "green"
    elif signal == "BEARISH":
        return "red"
    else:
        return "yellow"


def format_percentage(value: float, decimals: int = 2, colored: bool = True) -> Text:
    """
    Format a decimal value as a percentage with color.
    
    Args:
        value: Decimal value (e.g., 0.25 for 25%)
        decimals: Number of decimal places (default: 2)
        colored: Whether to apply color (default: True)
    
    Returns:
        Rich Text object with formatted percentage
    """
    if np.isnan(value):
        return Text("N/A", style="dim")
    
    percentage = value * 100
    sign = "+" if percentage >= 0 else ""
    text = f"{sign}{percentage:.{decimals}f}%"
    
    if colored:
        color = get_color_for_percentage(value)
        return Text(text, style=color)
    else:
        return Text(text)


def format_price(value: float, decimals: int = 2) -> Text:
    """
    Format a price value.
    
    Args:
        value: Price value
        decimals: Number of decimal places (default: 2)
    
    Returns:
        Rich Text object with formatted price
    """
    if np.isnan(value):
        return Text("N/A", style="dim")
    
    return Text(f"${value:.{decimals}f}", style="cyan")


def format_ratio(value: float, decimals: int = 2, colored: bool = True) -> Text:
    """
    Format a ratio value.
    
    Args:
        value: Ratio value
        decimals: Number of decimal places (default: 2)
        colored: Whether to apply color
    
    Returns:
        Rich Text object with formatted ratio
    """
    if np.isnan(value):
        return Text("N/A", style="dim")
    
    text = f"{value:.{decimals}f}"
    
    if colored and value > 0:
        return Text(text, style="green")
    elif colored and value < 0:
        return Text(text, style="red")
    else:
        return Text(text)


def create_returns_table(metrics: Dict[str, Any]) -> Table:
    """
    Create a Rich table for returns analysis.
    
    Args:
        metrics: Dictionary containing metrics
    
    Returns:
        Rich Table object
    """
    table = Table(title="Returns Analysis", box=box.ROUNDED, show_header=False, title_style="bold cyan")
    
    table.add_column("Metric", style="bold", width=30)
    table.add_column("Value", justify="right")
    
    table.add_row("Cumulative Return", format_percentage(metrics['cumulative_return']))
    table.add_row("Average Daily Return", format_percentage(metrics.get('avg_return', 0) / 252))
    table.add_row("Annualized Volatility", format_percentage(metrics['volatility']))
    table.add_row("Sharpe Ratio", format_ratio(metrics['sharpe_ratio']))
    
    return table


    return table


def create_daily_stats_table(metrics: Dict[str, Any]) -> Table:
    """
    Create a Rich table for daily return statistics.
    """
    table = Table(title="Daily Return Stats", box=box.ROUNDED, show_header=False, title_style="bold blue")
    
    table.add_column("Metric", style="bold", width=30)
    table.add_column("Value", justify="right")
    
    stats = metrics['daily_stats']
    
    table.add_row("Best Day", format_percentage(stats['best_day']))
    table.add_row("Worst Day", format_percentage(stats['worst_day']))
    table.add_row("Win Ratio", format_percentage(stats['win_ratio']))
    
    return table


def create_ma_table(metrics: Dict[str, Any]) -> Table:
    """
    Create a Rich table for moving averages.
    
    Args:
        metrics: Dictionary containing metrics
    
    Returns:
        Rich Table object
    """
    table = Table(title="Moving Averages", box=box.ROUNDED, title_style="bold magenta")
    
    table.add_column("Period", style="bold", justify="center")
    table.add_column("SMA Value", justify="right")
    table.add_column("Signal", justify="center")
    
    sma_values = metrics['sma_values']
    signals = metrics['signals']
    
    for window in [20, 50, 200]:
        sma = sma_values.get(window, np.nan)
        signal = signals.get(window, "N/A")
        
        if not np.isnan(sma):
            signal_text = Text(signal, style=get_color_for_signal(signal))
            table.add_row(f"SMA-{window}", format_price(sma), signal_text)
        else:
            table.add_row(f"SMA-{window}", Text("N/A", style="dim"), Text("INSUFFICIENT DATA", style="yellow"))
    
    return table


    return table


def create_technical_table(metrics: Dict[str, Any]) -> Table:
    """
    Create a Rich table for technical indicators (RSI, MACD, BB).
    """
    table = Table(title="Technical Indicators", box=box.ROUNDED, show_header=False, title_style="bold yellow")
    
    table.add_column("Metric", style="bold", width=30)
    table.add_column("Value", justify="right")
    
    # RSI
    rsi = metrics['rsi']
    # RSI Logic: 30-70 is standard, >70 overbought (red), <30 oversold (green or red depending on strategy, usually warning)
    # Let's use simple coloring: Green for neutral, Red for extreme
    rsi_color = "green" if 30 <= rsi <= 70 else "red"
    table.add_row("RSI (14)", Text(f"{rsi:.2f}", style=rsi_color))
    
    # MACD
    macd = metrics['macd']
    macd_text = f"{macd['macd_line']:.2f} / {macd['signal_line']:.2f}"
    hist_color = "green" if macd['histogram'] > 0 else "red"
    table.add_row("MACD (12,26,9)", Text(macd_text, style=hist_color))
    
    # Bollinger Bands
    bb = metrics['bollinger_bands']
    bb_text = f"U: {bb['upper']:.2f} / L: {bb['lower']:.2f}"
    table.add_row("Bollinger Bands (20,2)", bb_text)
    
    return table


def create_risk_table(metrics: Dict[str, Any]) -> Table:
    """
    Create a Rich table for risk metrics.
    
    Args:
        metrics: Dictionary containing metrics
    
    Returns:
        Rich Table object
    """
    table = Table(title="Risk Metrics", box=box.ROUNDED, show_header=False, title_style="bold yellow")
    
    table.add_column("Metric", style="bold", width=30)
    table.add_column("Value", justify="right")
    
    table.add_row("Max Drawdown", format_percentage(metrics['max_drawdown']))
    table.add_row("ROC (12-day)", format_percentage(metrics['roc']))
    
    # Risk score with color
    risk_score = metrics['risk_score']
    if risk_score == "HIGH RISK":
        risk_text = Text(risk_score, style="bold red")
    elif risk_score == "MODERATE":
        risk_text = Text(risk_score, style="bold yellow")
    else:
        risk_text = Text(risk_score, style="bold green")
    
    table.add_row("Risk Score", risk_text)
    
    return table


def format_results(ticker: str, company_name: str, metrics: Dict[str, Any]) -> None:
    """
    Display formatted results using Rich.
    
    Args:
        ticker: Stock ticker symbol
        company_name: Company name
        metrics: Dictionary containing all calculated metrics
    """
    # Header panel
    header_text = Text()
    header_text.append(f"{ticker.upper()}", style="bold cyan")
    header_text.append(" - ", style="white")
    header_text.append(company_name, style="bold white")
    
    header_panel = Panel(
        header_text,
        title="QUANT ANALYSIS",
        border_style="cyan",
        box=box.DOUBLE
    )
    
    console.print()
    console.print(header_panel)
    console.print()
    
    # Current info panel
    info_text = Text()
    info_text.append("Current Price: ", style="bold")
    info_text.append(f"${metrics['current_price']:.2f}", style="bold cyan")
    info_text.append("\n")
    info_text.append("Data Period: ", style="bold")
    info_text.append(f"{metrics['start_date']} to {metrics['end_date']}", style="dim")
    
    info_panel = Panel(info_text, border_style="blue", box=box.ROUNDED)
    console.print(info_panel)
    console.print()
    
    # Tables
    console.print(create_returns_table(metrics))
    console.print()
    console.print(create_daily_stats_table(metrics))
    console.print()
    console.print(create_ma_table(metrics))
    console.print()
    console.print(create_technical_table(metrics))
    console.print()



def create_comparison_table(comparison: Dict[str, Any]) -> Table:
    """
    Create a Rich table for stock comparison.
    
    Args:
        comparison: Dictionary with comparison results
    
    Returns:
        Rich Table object
    """
    stock1 = comparison['stock1']
    stock2 = comparison['stock2']
    winners = comparison['winners']
    
    table = Table(
        title=f"{stock1['ticker']} vs {stock2['ticker']} - Comparison",
        box=box.DOUBLE_EDGE,
        title_style="bold magenta",
        show_lines=True
    )
    
    table.add_column("Metric", style="bold cyan", width=25)
    table.add_column(stock1['ticker'], justify="right", style="white", width=20)
    table.add_column(stock2['ticker'], justify="right", style="white", width=20)
    table.add_column("Winner", justify="center", style="bold yellow", width=15)
    
    # Company names
    table.add_row(
        "Company",
        Text(stock1['company_name'][:18], style="dim"),
        Text(stock2['company_name'][:18], style="dim"),
        ""
    )
    
    # Current prices
    table.add_row(
        "Current Price",
        format_price(stock1['current_price']),
        format_price(stock2['current_price']),
        ""
    )
    
    # Section separator
    table.add_section()
    
    # Returns Analysis
    table.add_row(
        Text("RETURNS", style="bold cyan"),
        "",
        "",
        ""
    )
    
    # Cumulative Return
    winner = winners['cumulative_return']
    table.add_row(
        "Cumulative Return",
        format_percentage(stock1['cumulative_return']),
        format_percentage(stock2['cumulative_return']),
        Text(winner if winner != 'TIE' else "TIE", style="green" if winner == stock1['ticker'] else "cyan")
    )
    
    # Volatility
    winner = winners['volatility']
    table.add_row(
        "Volatility",
        format_percentage(stock1['volatility']),
        format_percentage(stock2['volatility']),
        Text(winner if winner != 'TIE' else "TIE", style="green" if winner == stock1['ticker'] else "cyan")
    )
    
    # Sharpe Ratio
    winner = winners['sharpe_ratio']
    table.add_row(
        "Sharpe Ratio",
        format_ratio(stock1['sharpe_ratio']),
        format_ratio(stock2['sharpe_ratio']),
        Text(winner if winner != 'TIE' else "TIE", style="green" if winner == stock1['ticker'] else "cyan")
    )
    
    table.add_section()
    
    # Risk Metrics
    table.add_row(
        Text("RISK METRICS", style="bold yellow"),
        "",
        "",
        ""
    )
    
    # Max Drawdown
    winner = winners['max_drawdown']
    table.add_row(
        "Max Drawdown",
        format_percentage(stock1['max_drawdown']),
        format_percentage(stock2['max_drawdown']),
        Text(winner if winner != 'TIE' else "TIE", style="green" if winner == stock1['ticker'] else "cyan")
    )
    
    # ROC
    winner = winners['roc']
    table.add_row(
        "ROC (12-day)",
        format_percentage(stock1['roc']),
        format_percentage(stock2['roc']),
        Text(winner if winner != 'TIE' else "TIE", style="green" if winner == stock1['ticker'] else "cyan")
    )
    
    # Risk Score
    winner = winners['risk_score']
    risk1_text = Text(stock1['risk_score'], style="red" if "HIGH" in stock1['risk_score'] else "yellow" if "MODERATE" in stock1['risk_score'] else "green")
    risk2_text = Text(stock2['risk_score'], style="red" if "HIGH" in stock2['risk_score'] else "yellow" if "MODERATE" in stock2['risk_score'] else "green")
    
    table.add_row(
        "Risk Score",
        risk1_text,
        risk2_text,
        Text(winner if winner != 'TIE' else "TIE", style="green" if winner == stock1['ticker'] else "cyan")
    )
    
    table.add_section()
    
    # Moving Averages
    table.add_row(
        Text("SIGNALS", style="bold magenta"),
        "",
        "",
        ""
    )
    
    winner = winners['bullish_signals']
    bullish1 = Text(f"{stock1['bullish_count']}/{stock1['total_signals']} Bullish", style="green" if stock1['bullish_count'] > 0 else "dim")
    bullish2 = Text(f"{stock2['bullish_count']}/{stock2['total_signals']} Bullish", style="green" if stock2['bullish_count'] > 0 else "dim")
    
    table.add_row(
        "Bullish Signals",
        bullish1,
        bullish2,
        Text(winner if winner != 'TIE' else "TIE", style="green" if winner == stock1['ticker'] else "cyan")
    )
    
    return table


def create_recommendation_panel(comparison: Dict[str, Any]) -> Panel:
    """
    Create a Rich panel for recommendations.
    
    Args:
        comparison: Dictionary with comparison results
    
    Returns:
        Rich Panel object
    """
    stock1 = comparison['stock1']
    stock2 = comparison['stock2']
    rec = comparison['recommendation']
    
    content = Text()
    
    # Overall score
    content.append(f"Overall Score: ", style="bold white")
    content.append(f"{stock1['ticker']} ({rec['score1']})", style="bold cyan")
    content.append(" vs ", style="white")
    content.append(f"{stock2['ticker']} ({rec['score2']})", style="bold cyan")
    content.append("\n\n")
    
    # Stock 1 strengths
    if rec['strengths1']:
        content.append(f"{stock1['ticker']} Strengths:\n", style="bold green")
        for strength in rec['strengths1']:
            content.append(f"  + {strength}\n", style="green")
        content.append("\n")
    
    # Stock 2 strengths
    if rec['strengths2']:
        content.append(f"{stock2['ticker']} Strengths:\n", style="bold cyan")
        for strength in rec['strengths2']:
            content.append(f"  + {strength}\n", style="cyan")
        content.append("\n")
    
    # Overall recommendation
    content.append("Overall: ", style="bold yellow")
    content.append(f"{rec['recommendation']}\n", style="white")
    
    # Investor suggestions
    if rec['investor_suggestions']:
        content.append("\nInvestor Type Recommendations:\n", style="bold magenta")
        for suggestion in rec['investor_suggestions']:
            content.append(f"  - {suggestion}\n", style="white")
    
    return Panel(
        content,
        title="RECOMMENDATION",
        border_style="yellow",
        box=box.DOUBLE
    )


def format_comparison_results(comparison: Dict[str, Any]) -> None:
    """
    Display formatted comparison results using Rich.
    
    Args:
        comparison: Dictionary with comparison results from compare_stocks()
    """
    console.print()
    console.print(create_comparison_table(comparison))
    console.print()
    console.print(create_recommendation_panel(comparison))
    console.print()


def display_summary(results: Any) -> None:
    """
    Print formatted summary to console.
    This function is kept for backward compatibility but does nothing
    since Rich formatting is now handled in format_results.
    
    Args:
        results: Formatted results (unused with Rich)
    """
    pass  # Rich formatting is done in format_results/format_comparison_results


def export_to_dict(ticker: str, company_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Export results as a structured dictionary (useful for JSON export).
    
    Args:
        ticker: Stock ticker symbol
        company_name: Company name
        metrics: Dictionary containing all calculated metrics
    
    Returns:
        Structured dictionary with all results
    """
    return {
        'ticker': ticker.upper(),
        'company_name': company_name,
        'current_price': metrics['current_price'],
        'data_period': {
            'start': metrics['start_date'],
            'end': metrics['end_date']
        },
        'returns': {
            'cumulative': metrics['cumulative_return'],
            'average_daily': metrics.get('avg_return', 0) / 252,
            'annualized_volatility': metrics['volatility'],
            'sharpe_ratio': metrics['sharpe_ratio']
        },
        'moving_averages': {
            f'sma_{window}': {
                'value': metrics['sma_values'].get(window, None),
                'signal': metrics['signals'].get(window, None)
            }
            for window in [20, 50, 200]
        },
        'risk_metrics': {
            'max_drawdown': metrics['max_drawdown'],
            'roc_12d': metrics['roc'],
            'risk_score': metrics['risk_score']
        }
    }
