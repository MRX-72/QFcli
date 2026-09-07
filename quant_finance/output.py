"""
Output Module
Format and display analysis results using Rich library for beautiful CLI output.
"""

from typing import Dict, Any
import json
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

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


def sparkline_levels(values: list, width: int = 40) -> list:
    """
    Map a list of values to sparkline heights (0..7) at a given width.

    Values are downsampled to 'width' buckets, each bucket rendered at its
    last value, then normalized to the 8 vertical block levels.

    Args:
        values: List of numeric values (ordered oldest -> newest)
        width: Target number of columns (default: 40)

    Returns:
        List of integer levels in range 0..7
    """
    if not values or width <= 0:
        return []

    n = len(values)
    if n <= width:
        sampled = values
    else:
        idx = np.linspace(0, n - 1, width).astype(int)
        sampled = [values[i] for i in idx]

    lo = float(min(sampled))
    hi = float(max(sampled))

    if hi == lo:
        return [3] * len(sampled)

    return [
        int(round((v - lo) / (hi - lo) * 7)) for v in sampled
    ]


_BLOCKS = "▁▂▃▄▅▆▇█"


def render_sparkline(values: list, width: int = 40) -> Text:
    """
    Render a unicode block sparkline with per-bar up/down coloring.

    Args:
        values: List of numeric values (ordered oldest -> newest)
        width: Target number of columns (default: 40)

    Returns:
        Rich Text object
    """
    levels = sparkline_levels(values, width)
    text = Text()
    prev = None
    for i, level in enumerate(levels):
        char = _BLOCKS[min(max(level, 0), 7)]
        color = "dim"
        if i > 0 and prev is not None:
            color = "green" if values[i] >= values[i - 1] else "red"
        elif i > 0:
            color = "dim"
        text.append(char, style=color)
        prev = values[i]
    return text


def gauge_bar(value: float, low: float, high: float, width: int = 20) -> Text:
    """
    Render a horizontal gauge showing where 'value' sits between low and high.

    The filled region is colored by zone: green at/below low (favorable,
    e.g. oversold), red at/above high (e.g. overbought), yellow in between.

    Args:
        value: Current value
        low: Range minimum
        high: Range maximum
        width: Total number of columns (default: 20)

    Returns:
        Rich Text object
    """
    if high == low:
        frac = 0.5
    else:
        frac = (value - low) / (high - low)
    frac = min(max(frac, 0.0), 1.0)

    filled = int(round(frac * width))
    text = Text()
    if value <= low:
        color = "green"
    elif value >= high:
        color = "red"
    else:
        color = "yellow"
    text.append("[" + "█" * filled + "░" * (width - filled) + "]", style=color)
    return text


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
    table.add_row("Sortino Ratio", format_ratio(metrics.get('sortino_ratio', np.nan)))
    
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
    var_val = metrics.get('value_at_risk', np.nan)
    if not np.isnan(var_val):
        table.add_row("VaR (95%, annualized)", Text(f"{var_val*100:.2f}%", style="bold yellow"))
    table.add_row("ROC (12-day)", format_percentage(metrics['roc']))
    table.add_row("ATR (14)", format_price(metrics.get('atr', np.nan)))

    beta_val = metrics.get('beta')
    if beta_val is None or (isinstance(beta_val, float) and np.isnan(beta_val)):
        table.add_row("Beta (vs benchmark)", Text("N/A", style="dim"))
    else:
        beta_color = "green" if 0.5 <= beta_val <= 1.5 else "red"
        table.add_row("Beta (vs benchmark)", Text(f"{beta_val:.2f}", style=beta_color))

    # RSI position gauge
    rsi_val = metrics.get('rsi', np.nan)
    if not np.isnan(rsi_val):
        table.add_row(
            "RSI position",
            gauge_bar(rsi_val, 30, 70, width=12)
        )
    
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


def create_trend_panel(metrics: Dict[str, Any]) -> Panel:
    """
    Create a Rich panel with a price sparkline and the trend regime.
    
    The last 60 closes are rendered as a sparkline, with 52-week (period)
    high/low annotations and the golden/death cross status.

    Args:
        metrics: Dictionary containing metrics

    Returns:
        Rich Panel object
    """
    trend = metrics.get('trend', {}).get('prices', [])
    
    content = Text()
    content.append("Price trend (last 60 sessions)\n", style="bold")
    content.append(render_sparkline(trend, width=60))
    content.append("\n\n", style="none")

    cross = metrics.get('golden_cross', 'INSUFFICIENT DATA')
    if cross == "GOLDEN CROSS":
        cross_text = Text(f"{cross}", style="bold green")
    elif cross == "DEATH CROSS":
        cross_text = Text(f"{cross}", style="bold red")
    elif cross == "BULLISH":
        cross_text = Text("BULLISH (SMA-50 > SMA-200)", style="green")
    elif cross == "BEARISH":
        cross_text = Text("BEARISH (SMA-50 < SMA-200)", style="red")
    else:
        cross_text = Text("INSUFFICIENT DATA", style="yellow")
    content.append("Market regime: ", style="bold")
    content.append(cross_text)
    content.append("\n")

    high = metrics.get('trend', {}).get('high', np.nan)
    low = metrics.get('trend', {}).get('low', np.nan)
    content.append("Period high: ", style="bold")
    content.append(f"${high:.2f}", style="cyan")
    content.append("   Period low: ", style="bold")
    content.append(f"${low:.2f}", style="cyan")
    content.append("\n")

    rolling_vol = metrics.get('trend', {}).get('rolling_vol', [])
    rolling_sharpe = metrics.get('trend', {}).get('rolling_sharpe', [])
    if rolling_vol:
        content.append("Rolling 60d vol:\n", style="dim")
        content.append(render_sparkline(rolling_vol, width=60))
        content.append("\n")
    if rolling_sharpe:
        content.append("Rolling 60d Sharpe:\n", style="dim")
        content.append(render_sparkline(rolling_sharpe, width=60))
        content.append("\n")

    return Panel(
        content,
        title="TREND",
        border_style="magenta",
        box=box.ROUNDED
    )


def create_yearly_returns_table(metrics: Dict[str, Any]) -> Table:
    """
    Create a Rich table of calendar-year returns, color-coded.

    Args:
        metrics: Dictionary containing metrics

    Returns:
        Rich Table object
    """
    table = Table(title="Yearly Returns", box=box.ROUNDED, show_header=False, title_style="bold cyan")
    table.add_column("Year", style="bold", width=12)
    table.add_column("Return", justify="right")

    yearly = metrics.get('yearly_returns', {})
    for year in sorted(yearly.keys(), reverse=True):
        table.add_row(Text(year, style="bold"), format_percentage(yearly[year]))

    if not yearly:
        table.add_row(Text("No full calendar years in period", style="dim"))
        
    return table


def create_statistics_table(metrics: Dict[str, Any]) -> Table:
    """
    Create a Rich table of statistical tests and simulations.

    Args:
        metrics: Dictionary containing metrics

    Returns:
        Rich Table object
    """
    table = Table(title="Statistics & Simulation", box=box.ROUNDED, show_header=False, title_style="bold cyan")
    table.add_column("Metric", style="bold", width=34)
    table.add_column("Value", justify="right")

    mc = metrics.get('monte_carlo', {})
    if mc.get('p50') is not None and not (isinstance(mc.get('p50'), float) and np.isnan(mc['p50'])):
        table.add_row(
            "Monte Carlo 1y (P5 / P50 / P95)",
            Text(
                f"{mc['p5']*100:+.1f}% / {mc['p50']*100:+.1f}% / {mc['p95']*100:+.1f}%",
                style="yellow"
            )
        )

    sb = metrics.get('sharpe_bootstrap', {})
    if sb.get('observed') is not None and not np.isnan(sb['observed']):
        table.add_row(
            "Sharpe (bootstrap 95% CI)",
            Text(f"{sb['observed']:.2f}  [{sb['ci_low']:.2f}, {sb['ci_high']:.2f}]", style="cyan")
        )

    for label, key, verdict_key in (
        ('Sharpe significance (J-K)', 'sharpe_significance', 'verdict'),
        ('Ljung-Box (10 lags)', 'ljung_box', 'verdict'),
        ('Jarque-Bera normality', 'jarque_bera', 'verdict'),
    ):
        entry = metrics.get(key, {})
        verdict = entry.get(verdict_key, 'INSUFFICIENT DATA')
        color = "red" if 'SIGNIFICANT' in verdict or 'NOT NORMAL' in verdict or 'AUTOCORRELATED' in verdict else "green"
        p = entry.get('p_value', np.nan)
        p_str = "N/A" if np.isnan(p) else f"{p:.4f}"
        table.add_row(f"{label}", Text(f"{verdict} (p={p_str})", style=color))

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
    
    # Trend sparkline panel
    if metrics.get('trend', {}).get('prices'):
        console.print(create_trend_panel(metrics))
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
    console.print(create_risk_table(metrics))
    console.print()
    console.print(create_yearly_returns_table(metrics))
    console.print()
    console.print(create_statistics_table(metrics))
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
    content.append("Overall Score: ", style="bold white")
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
    def round_value(value, decimals: int = 4):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return None
        return round(value, decimals)

    sma = {}
    for window in [20, 50, 200]:
        raw = metrics['sma_values'].get(window, None)
        sma[f'sma_{window}'] = {
            'value': round_value(raw),
            'signal': metrics['signals'].get(window, None)
        }

    beta_val = metrics.get('beta')
    beta_export = None if beta_val is None or not np.isfinite(beta_val) else round_value(beta_val)

    def _clean_stat(entry):
        if not isinstance(entry, dict):
            return None
        stat = entry.get('statistic')
        p = entry.get('p_value')
        return {
            'statistic': None if stat is None or np.isnan(stat) else round(stat, 4),
            'p_value': None if p is None or np.isnan(p) else round(p, 4),
            'verdict': entry.get('verdict'),
        }

    return {
        'ticker': ticker.upper(),
        'company_name': company_name,
        'current_price': round_value(metrics['current_price']),
        'data_period': {
            'start': metrics['start_date'],
            'end': metrics['end_date']
        },
        'returns': {
            'cumulative': round_value(metrics['cumulative_return']),
            'average_daily': round_value(metrics.get('avg_return', 0) / 252),
            'annualized_volatility': round_value(metrics['volatility']),
            'sharpe_ratio': round_value(metrics['sharpe_ratio']),
            'sortino_ratio': round_value(metrics.get('sortino_ratio', np.nan))
        },
        'moving_averages': sma,
        'risk_metrics': {
            'max_drawdown': round_value(metrics['max_drawdown']),
            'value_at_risk_95': round_value(metrics.get('value_at_risk', np.nan)),
            'atr_14': round_value(metrics.get('atr', np.nan)),
            'beta': beta_export,
            'roc_12d': round_value(metrics['roc']),
            'risk_score': metrics['risk_score']
        },
        'trend': {
            'golden_cross': metrics.get('golden_cross', 'INSUFFICIENT DATA'),
            'price_high': round_value(metrics.get('trend', {}).get('high', np.nan)),
            'price_low': round_value(metrics.get('trend', {}).get('low', np.nan))
        },
        'simulation': {
            'monte_carlo_1y': {
                'p5': floor_mc(metrics, 'p5'),
                'p50': floor_mc(metrics, 'p50'),
                'p95': floor_mc(metrics, 'p95'),
            },
            'returns_ci': _round_stat(metrics.get('bootstrap_ci', {})),
            'worst_30d_window': round_value(metrics.get('stress_30d', np.nan)),
        },
        'statistical_tests': {
            'sharpe_bootstrap_ci': _round_stat(metrics.get('sharpe_bootstrap', {})),
            'sharpe_significance': _clean_stat(metrics.get('sharpe_significance', {})),
            'ljung_box': _clean_stat(metrics.get('ljung_box', {})),
            'jarque_bera': _clean_stat(metrics.get('jarque_bera', {})),
        }
    }


def floor_mc(metrics, key):
    mc = metrics.get('monte_carlo', {})
    val = mc.get(key, np.nan)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), 4)


def _round_stat(entry):
    if not isinstance(entry, dict):
        return None
    out = {}
    for k, v in entry.items():
        if isinstance(v, float) and np.isnan(v):
            out[k] = None
        elif isinstance(v, float):
            out[k] = round(v, 4)
        else:
            out[k] = v
    return out


def format_backtest_results(result: Dict[str, Any], ticker: str, strategy_name: str) -> None:
    """
    Display formatted backtest results using Rich.

    Args:
        result: Flat dict from run_backtest() with strategy_*/baseline_* keys
        ticker: Stock ticker symbol
        strategy_name: Human-readable strategy label
    """
    table = Table(
        title=f"{ticker.upper()} - {strategy_name} backtest",
        box=box.DOUBLE_EDGE,
        title_style="bold magenta",
        show_lines=True
    )
    table.add_column("Metric", style="bold", width=30)
    table.add_column("Strategy", justify="right", style="cyan")
    table.add_column("Buy & Hold", justify="right", style="dim")

    def _cell(v, fmt):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return Text("N/A")
        return fmt(v)

    _row = lambda label, skey, bkey, fmt: table.add_row(
        label, _cell(result.get(skey), fmt), _cell(result.get(bkey), fmt))

    _row("Total Return", 'strategy_total_return', 'baseline_total_return', format_percentage)
    _row("Annualized Return", 'strategy_annual_return', 'baseline_annual_return', format_percentage)
    _row("Volatility", 'strategy_volatility', 'baseline_volatility', format_percentage)
    _row("Sharpe Ratio", 'strategy_sharpe', 'baseline_sharpe', format_ratio)
    _row("Sortino Ratio", 'strategy_sortino', 'baseline_sortino', format_ratio)
    _row("Max Drawdown", 'strategy_max_drawdown', 'baseline_max_drawdown', format_percentage)
    _row("Win Ratio", 'strategy_win_ratio', 'baseline_win_ratio', format_percentage)
    _row("Best Day", 'strategy_best_day', 'baseline_best_day', format_percentage)
    _row("Worst Day", 'strategy_worst_day', 'baseline_worst_day', format_percentage)

    stats_table = Table(title="Trading Stats", box=box.ROUNDED, show_header=False, title_style="bold yellow")
    stats_table.add_column("Metric", style="bold", width=30)
    stats_table.add_column("Value", justify="right")
    stats_table.add_row("Trading Days", str(result.get('n_days', 'N/A')))
    stats_table.add_row("Number of Trades", str(result.get('n_trades', 0)))
    stats_table.add_row("Days in Market", format_percentage(result.get('days_in_market', 0.0)))
    stats_table.add_row("Avg Daily Turnover", format_percentage(result.get('avg_daily_turnover', 0.0)))
    stats_table.add_row("Total Transaction Cost", format_percentage(result.get('total_cost', 0.0)))

    console.print()
    console.print(table)
    console.print()
    console.print(stats_table)
    console.print()

    active_table = Table(title="vs Buy & Hold", box=box.ROUNDED, show_header=False,
                         title_style="bold cyan")
    active_table.add_column("Metric", style="bold", width=30)
    active_table.add_column("Value", justify="right")
    active_table.add_row(
        "Alpha (annualized)",
        Text("N/A", style="dim") if result.get('strategy_alpha') is None
        or (isinstance(result.get('strategy_alpha'), float) and np.isnan(result['strategy_alpha']))
        else format_percentage(result['strategy_alpha'])
    )
    active_table.add_row("Active Return (annualized)", format_percentage(result.get('strategy_active_return', np.nan)))
    active_table.add_row(
        "Information Ratio",
        Text("N/A", style="dim") if result.get('strategy_information_ratio') is None
        or (isinstance(result.get('strategy_information_ratio'), float) and np.isnan(result['strategy_information_ratio']))
        else format_ratio(result['strategy_information_ratio'])
    )
    hit = result.get('strategy_hit_rate')
    if hit is None or (isinstance(hit, float) and np.isnan(hit)):
        hit_text = Text("N/A", style="dim")
    else:
        hit_text = format_percentage(hit)
    active_table.add_row("Hit Rate (days beating baseline)", hit_text)
    beta = result.get('strategy_beta_to_baseline')
    active_table.add_row(
        "Beta (to baseline)",
        Text("N/A", style="dim") if beta is None or (isinstance(beta, float) and np.isnan(beta)) else Text(f"{beta:.2f}")
    )
    console.print(active_table)
    console.print()


def format_walk_forward_results(result: Dict[str, Any], ticker: str, strategy_name: str) -> None:
    """
    Display formatted walk-forward (out-of-sample) validation results.

    Args:
        result: Dict from backtest.walk_forward()
        ticker: Stock ticker symbol
        strategy_name: Human-readable strategy label
    """
    console.print()
    panel = Panel(
        "[dim]Parameters are tuned on each trailing in-sample window and then run "
        "untouched on the following out-of-sample window. Alpha / hit rate / "
        "information ratio are measured against the buy-and-hold baseline on "
        "out-of-sample days only.[/dim]",
        title=f"WALK-FORWARD - {ticker.upper()} - {strategy_name}",
        border_style="magenta",
        box=box.ROUNDED,
    )
    console.print(panel)
    console.print()

    table = Table(title="Out-of-Sample Performance", box=box.ROUNDED, show_lines=True,
                  title_style="bold cyan")
    table.add_column("Metric", style="bold", width=30)
    table.add_column("Value", justify="right")

    def _num(key, fmt, decimals=2):
        v = result.get(key)
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return Text("N/A", style="dim")
        return fmt(v, decimals) if callable(fmt) else fmt(v)

    table.add_row("OOS Total Return", _num('oos_total_return', format_percentage))
    table.add_row("OOS Annual Return", _num('oos_annual_return', format_percentage))
    table.add_row("OOS Sharpe", _num('oos_sharpe', format_ratio))
    table.add_row("Buy & Hold Total Return", _num('baseline_total_return', format_percentage))
    table.add_row("Buy & Hold Annual Return", _num('baseline_annual_return', format_percentage))
    table.add_row("Alpha (annualized)", _num('alpha', format_percentage))
    table.add_row("Active Return (annualized)", _num('active_return', format_percentage))
    table.add_row("Information Ratio", _num('information_ratio', format_ratio))
    table.add_row("Hit Rate (days beating baseline)", _num('hit_rate', format_percentage))

    table.add_section()
    table.add_row("Folds", Text(str(result.get('n_folds', 'N/A')), style="cyan"))
    table.add_row("Out-of-Sample Days", Text(str(result.get('oos_days', 'N/A')), style="cyan"))
    table.add_row(
        "Folds Beating Buy & Hold",
        Text(f"{result.get('fold_beat_rate', float('nan')) * 100:.0f}%", style="cyan")
    )
    table.add_row("Selection Metric", Text(str(result.get('selection_metric', 'sharpe')), style="yellow"))

    ensemble = result.get('ensemble', 'best')
    ensemble_text = {
        'best': "best single config",
        'equal': "equal weights across grid",
        'rank': "rank-weighted across grid",
        'topk': "top-k average across grid",
    }.get(ensemble, ensemble)
    table.add_row("OOS Combination", Text(str(ensemble_text), style="magenta"))
    if ensemble != 'best' and 'best_oos_total_return' in result:
        table.add_row(
            "Best config OOS return (ref)",
            format_percentage(result.get('best_oos_total_return', np.nan))
        )
    console.print(table)
    console.print()

    if result.get('folds'):
        folds = result['folds']
        fold_table = Table(title="Per-Fold Detail", box=box.ROUNDED, title_style="bold yellow")
        fold_table.add_column("Fold", justify="center", style="bold")
        fold_table.add_column("Test Window", justify="center", style="dim")
        fold_table.add_column("Best Params", justify="center")
        fold_table.add_column("OOS Return", justify="right")
        fold_table.add_column("Baseline", justify="right")
        fold_table.add_column("Beats Baseline", justify="center")
        for i, f in enumerate(folds, 1):
            params = ", ".join(f"{k}={v}" for k, v in sorted(f['best_params'].items())) or "n/a"
            fmt_ret = lambda v: Text("N/A", style="dim") if v is None else format_percentage(v)
            beat = "yes" if f['beat_baseline'] else "no"
            beat_style = "green" if f['beat_baseline'] else "red"
            fold_table.add_row(
                str(i),
                Text(f"{f['test_start']}-{f['test_end']}", style="dim"),
                Text(params, style="yellow"),
                fmt_ret(f['oos_strategy_total_return']),
                fmt_ret(f['oos_baseline_total_return']),
                Text(beat, style=beat_style),
            )
        console.print(fold_table)
        console.print()


def _cell_bg(value: float) -> str:
    """Map correlation [-1,1] to a Rich background color."""
    t = (value + 1) / 2
    if t < 0.25:
        return "red"
    if t < 0.45:
        return "yellow"
    if t < 0.55:
        return "white"
    if t < 0.75:
        return "green"
    return "dark_green"


def create_correlation_table(corr: pd.DataFrame) -> Table:
    """
    Build a terminal heatmap of the asset correlation matrix.

    Args:
        corr: Correlation DataFrame (columns = assets)

    Returns:
        Rich Table object with color-coded cells
    """
    table = Table(title="Correlation Matrix", box=box.SQUARE, title_style="bold cyan")
    table.add_column("Asset", style="bold")
    for col in corr.columns:
        table.add_column(col, justify="center", style="bold")

    for idx in corr.index:
        row = [Text(idx, style="bold")]
        for col in corr.columns:
            val = corr.loc[idx, col]
            cell = Text(f"{val:+.2f}", style="black on " + _cell_bg(val))
            row.append(cell)
        table.add_row(*row)

    return table


def format_portfolio_results(data: Dict[str, Any]) -> None:
    """
    Display formatted portfolio results using Rich.

    Args:
        data: Dictionary from build_portfolio_report()
    """
    console.print()

    weights = data['weights']
    weights_table = Table(title="Portfolio Weights", box=box.ROUNDED, show_lines=True,
                          title_style="bold cyan")
    weights_table.add_column("Strategy", style="bold", width=22)
    for ticker in weights.columns:
        weights_table.add_column(ticker, justify="right", style="cyan")

    for strategy in weights.index:
        if strategy in ('min_variance', 'tangency', 'efficient', 'black_litterman'):
            if strategy == 'black_litterman':
                label_text = Text("Black-Litterman", style="bold magenta")
                cells = [Text(f"{w*100:.1f}%", style="magenta") for w in weights.loc[strategy]]
                weights_table.add_row(label_text, *cells)
            else:
                weights_table.add_row(
                    Text(strategy.replace('_', ' ').title()),
                    *[Text(f"{w*100:.1f}%", style="white") for w in weights.loc[strategy]]
                )
    console.print(weights_table)
    console.print()

    bl = data.get('bl')
    if bl:
        prior_label = bl.get('prior_label', 'implied')
        title = "Black-Litterman Expected Returns"
        if prior_label == 'ff':
            title += " (prior: Fama-French factor model)"
        bl_show_implied = prior_label != 'implied'
        bl_table = Table(title=title, box=box.ROUNDED,
                         title_style="bold magenta")
        bl_table.add_column("Asset", style="bold")
        if bl_show_implied:
            bl_table.add_column("Implied (reverse-opt)", justify="right")
        bl_table.add_column("Prior", justify="right")
        bl_table.add_column("Posterior", justify="right")
        for asset in bl['prior_returns']:
            row = [Text(asset, style="bold")]
            if bl_show_implied:
                row.append(format_percentage(bl['implied_returns'][asset], colored=True))
            row.append(format_percentage(bl['prior_returns'][asset], colored=True))
            row.append(format_percentage(bl['posterior_returns'][asset], colored=True))
            bl_table.add_row(*row)
        console.print(bl_table)
        console.print()

    ff = data.get('ff')
    if ff:
        columns = list(ff['betas'].keys())
        ff_table = Table(title="Fama-French Factor Exposures (daily OLS betas)",
                         box=box.ROUNDED, title_style="bold yellow")
        ff_table.add_column("Asset", style="bold")
        for c in columns:
            ff_table.add_column(c, justify="right")
        ff_table.add_column("Alpha (ann.)", justify="right")
        for asset in data['tickers']:
            row = [Text(asset, style="bold")]
            for c in columns:
                v = ff['betas'][c].get(asset)
                row.append(Text("N/A", style="dim") if v is None else Text(f"{v:.2f}"))
            alpha = ff['alpha_annual'].get(asset, np.nan)
            row.append(format_percentage(alpha))
            ff_table.add_row(*row)
        console.print(ff_table)
        console.print()
        if ff.get('premia'):
            premia_table = Table(title="Factor Premia (annualized)", box=box.ROUNDED,
                                 show_header=False, title_style="bold yellow")
            premia_table.add_column("Factor", style="bold")
            premia_table.add_column("Premia", justify="right")
            for f, v in ff['premia'].items():
                premia_table.add_row(Text(f, style="bold"), format_percentage(v))
            console.print(premia_table)
            console.print()

    fm = data.get('factor_model')
    if fm:
        fm_table = Table(title="PCA Risk Model", box=box.ROUNDED, show_header=False,
                         title_style="bold cyan")
        fm_table.add_column("Metric", style="bold", width=28)
        fm_table.add_column("Value", justify="right")
        fm_table.add_row("Factors Retained", Text(str(fm['n_factors']), style="cyan"))
        fm_table.add_row("Explained Variance",
                         Text(f"{fm['explained_variance'] * 100:.1f}%", style="cyan"))
        for asset, idio in fm['idiosyncratic_std'].items():
            fm_table.add_row(f"Idiosyncratic std ({asset})",
                             format_percentage(idio))
        console.print(fm_table)
        console.print()

    # Portfolio stats
    stats = data['stats']
    stats_table = Table(title="Portfolio Characteristics", box=box.ROUNDED, show_header=False,
                        title_style="bold yellow")
    stats_table.add_column("Metric", style="bold", width=28)
    stats_table.add_column("Value", justify="right")
    for label, key, fmt in (('Expected Annual Return', 'expected_annual_return', format_percentage),
                            ('Annualized Volatility', 'volatility', format_percentage),
                            ('Sharpe Ratio', 'sharpe_ratio', format_ratio)):
        stats_table.add_row(label, fmt(stats[key]))

    if data.get('frontier_risk') is not None:
        fr = data['frontier_risk']
        stats_table.add_row("Frontier Risk Range", Text(f"{fr['min']:.1%} - {fr['max']:.1%}"))

    console.print(stats_table)
    console.print()

    if data.get('correlation') is not None:
        console.print(create_correlation_table(data['correlation']))
        console.print()

    stress = data.get('stress')
    if stress:
        stress_table = Table(title="Stress Scenarios", box=box.ROUNDED, show_header=False,
                             title_style="bold red")
        stress_table.add_column("Scenario", style="bold", width=28)
        stress_table.add_column("Portfolio P&L", justify="right")
        for name, value in stress.items():
            stress_table.add_row(Text(name.replace('_', ' ')), format_percentage(value))
        console.print(stress_table)
        console.print()


def _params_text(params) -> str:
    if not params:
        return "n/a"
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))


def format_paper_trade_results(
    state: Dict[str, Any],
    previous: Dict[str, Any],
    ticker: str,
    strategy_label: str
) -> None:
    """
    Display the paper-trading harness state.

    Args:
        state: Fresh state dict from paper_trade.paper_trade()
        previous: Previously persisted state (or None)
        ticker: Ticker symbol
        strategy_label: Strategy label
    """
    panel = Panel(
        "[dim]The strategy was re-selected every trading day on trailing data "
        "only, then run on the next bar with the same costs as the backtester. "
        "This is a monitoring harness - it does not place orders.[/dim]",
        title=f"PAPER TRADING - {ticker.upper()} - {strategy_label}",
        border_style="magenta",
        box=box.ROUNDED,
    )
    console.print()
    console.print(panel)
    console.print()

    table = Table(title="Paper Account", box=box.ROUNDED, show_header=False,
                  show_lines=True, title_style="bold cyan")
    table.add_column("Metric", style="bold", width=32)
    table.add_column("Value", justify="right")

    table.add_row("Horizon", Text(f"{state.get('start')} -> {state.get('end')}", style="cyan"))
    table.add_row("Days Tracked", Text(str(state.get('n_days', 0)), style="cyan"))
    table.add_row("Ensemble", Text(str(state.get('ensemble'))))

    # param drift vs the previous run
    if previous and previous.get('current_params') and state.get('current_params'):
        same = previous['current_params'] == state['current_params']
        table.add_row(
            "Current Params",
            Text(_params_text(state['current_params']), style="yellow")
            + Text(("  (unchanged)" if same else "  (changed since last run)"),
                   style="dim" if same else "bold")
        )
    else:
        table.add_row("Current Params", Text(_params_text(state.get('current_params')), style="yellow"))

    if previous:
        new_days = state.get('n_days', 0) - previous.get('n_days', 0)
        table.add_row(
            "New Days Since Last Run",
            Text(f"{new_days:+d}" if new_days else "none yet", style="cyan" if new_days else "dim")
        )

    stable = state.get('stable_days', 1)
    if stable > 1:
        table.add_row("Stability Filter", Text(f"switch after {stable} winning days", style="magenta"))
        table.add_row("Param Switches",
                      Text(str(state.get('n_switches', 0)), style="cyan"))
        table.add_row("Mean Days In Force",
                      Text(f"{state.get('mean_days_in_force', np.nan):.1f}", style="cyan"))

    table.add_section()
    table.add_row("Paper Total Return", format_percentage(state.get('paper_total_return', np.nan)))
    table.add_row("Paper Annual Return", format_percentage(state.get('paper_annual_return', np.nan)))
    table.add_row("Paper Sharpe", format_ratio(state.get('paper_sharpe', np.nan)))
    table.add_row("Buy & Hold Total Return",
                  format_percentage(state.get('baseline_total_return', np.nan)))
    table.add_row("Buy & Hold Annual Return",
                  format_percentage(state.get('baseline_annual_return', np.nan)))

    table.add_section()
    table.add_row("Alpha (annualized)", format_percentage(state.get('alpha', np.nan)))
    table.add_row("Active Return (annualized)",
                  format_percentage(state.get('active_return', np.nan)))
    table.add_row("Information Ratio", format_ratio(state.get('information_ratio', np.nan)))
    table.add_row("Hit Rate (days beating baseline)",
                  format_percentage(state.get('hit_rate', np.nan)))

    console.print(table)
    console.print()

    selection = state.get('param_selection_rate') or {}
    if selection:
        rate_table = Table(title="Config Selection Rate (days as argmax)",
                           box=box.ROUNDED, title_style="bold yellow")
        rate_table.add_column("Config", style="bold")
        rate_table.add_column("% of days selected", justify="right")
        for key, frac in selection.items():
            try:
                params = json.loads(key)
                label = _params_text(params)
            except Exception:
                label = str(key)
            rate_table.add_row(Text(label, style="yellow"), Text(f"{frac*100:.1f}%"))
        console.print(rate_table)
        console.print()

    if state.get('stable_days', 1) > 1:
        force = state.get('days_in_force') or {}
        if force:
            force_table = Table(title="Config Time in Force (traded days)",
                                box=box.ROUNDED, title_style="bold magenta")
            force_table.add_column("Config", style="bold")
            force_table.add_column("Days active", justify="right")
            for key, days in force.items():
                try:
                    params = json.loads(key)
                    label = _params_text(params)
                except Exception:
                    label = str(key)
                force_table.add_row(Text(label, style="yellow"), Text(str(days)))
            console.print(force_table)
            console.print()
