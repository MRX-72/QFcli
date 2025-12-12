"""
Comparison Module
Compare two stocks across all metrics.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from .data_fetcher import (
    fetch_stock_data,
    calculate_daily_returns,
    get_current_price,
    get_date_range
)
from .metrics import (
    cumulative_return,
    volatility,
    sharpe_ratio,
    max_drawdown,
    risk_score,
    average_return
)
from .indicators import (
    calculate_all_smas,
    moving_average_signals,
    rate_of_change
)


def analyze_single_stock(ticker: str, period: str = '1y', risk_free_rate: float = 0.0) -> Optional[Dict[str, Any]]:
    """
    Analyze a single stock and return all metrics.
    
    Args:
        ticker: Stock symbol
        period: Historical data period
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Dictionary with all analysis results, or None if analysis fails
    """
    try:
        # Fetch data
        df, company_name = fetch_stock_data(ticker, period)
        
        # Extract price data
        prices = df['Close']
        current_price = get_current_price(df)
        start_date, end_date = get_date_range(df)
        
        # Calculate returns
        returns = calculate_daily_returns(prices)
        
        # Calculate core metrics
        cum_return = cumulative_return(prices)
        vol = volatility(returns, annualize=True)
        sharpe = sharpe_ratio(returns, risk_free_rate=risk_free_rate, annualize=True)
        mdd = max_drawdown(prices)
        avg_ret = average_return(returns, annualize=True)
        
        # Calculate technical indicators
        sma_values = calculate_all_smas(prices, windows=[20, 50, 200])
        signals = moving_average_signals(current_price, sma_values)
        roc = rate_of_change(prices, period=12)
        
        # Calculate risk score
        risk_rating = risk_score(vol)
        
        # Count bullish signals
        bullish_count = sum(1 for signal in signals.values() if signal == "BULLISH")
        total_signals = sum(1 for signal in signals.values() if signal in ["BULLISH", "BEARISH"])
        
        return {
            'ticker': ticker.upper(),
            'company_name': company_name,
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
            'bullish_count': bullish_count,
            'total_signals': total_signals,
            'roc': roc,
            'risk_score': risk_rating
        }
    
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return None


def determine_winners(stock1: Dict[str, Any], stock2: Dict[str, Any]) -> Dict[str, str]:
    """
    Determine which stock wins for each metric.
    
    Args:
        stock1: First stock's metrics
        stock2: Second stock's metrics
    
    Returns:
        Dictionary mapping metric names to winner ticker (or 'TIE')
    """
    winners = {}
    
    # Higher is better
    if stock1['cumulative_return'] > stock2['cumulative_return']:
        winners['cumulative_return'] = stock1['ticker']
    elif stock1['cumulative_return'] < stock2['cumulative_return']:
        winners['cumulative_return'] = stock2['ticker']
    else:
        winners['cumulative_return'] = 'TIE'
    
    # Lower volatility is better (less risky)
    if stock1['volatility'] < stock2['volatility']:
        winners['volatility'] = stock1['ticker']
    elif stock1['volatility'] > stock2['volatility']:
        winners['volatility'] = stock2['ticker']
    else:
        winners['volatility'] = 'TIE'
    
    # Higher Sharpe ratio is better
    if stock1['sharpe_ratio'] > stock2['sharpe_ratio']:
        winners['sharpe_ratio'] = stock1['ticker']
    elif stock1['sharpe_ratio'] < stock2['sharpe_ratio']:
        winners['sharpe_ratio'] = stock2['ticker']
    else:
        winners['sharpe_ratio'] = 'TIE'
    
    # Max drawdown: closer to 0 is better (less negative)
    if stock1['max_drawdown'] > stock2['max_drawdown']:
        winners['max_drawdown'] = stock1['ticker']
    elif stock1['max_drawdown'] < stock2['max_drawdown']:
        winners['max_drawdown'] = stock2['ticker']
    else:
        winners['max_drawdown'] = 'TIE'
    
    # Higher ROC is better
    if not np.isnan(stock1['roc']) and not np.isnan(stock2['roc']):
        if stock1['roc'] > stock2['roc']:
            winners['roc'] = stock1['ticker']
        elif stock1['roc'] < stock2['roc']:
            winners['roc'] = stock2['ticker']
        else:
            winners['roc'] = 'TIE'
    else:
        winners['roc'] = 'TIE'
    
    # More bullish signals is better
    if stock1['bullish_count'] > stock2['bullish_count']:
        winners['bullish_signals'] = stock1['ticker']
    elif stock1['bullish_count'] < stock2['bullish_count']:
        winners['bullish_signals'] = stock2['ticker']
    else:
        winners['bullish_signals'] = 'TIE'
    
    # Risk score: STABLE > MODERATE > HIGH RISK
    risk_order = {'STABLE': 3, 'MODERATE': 2, 'HIGH RISK': 1}
    risk1 = risk_order.get(stock1['risk_score'], 0)
    risk2 = risk_order.get(stock2['risk_score'], 0)
    
    if risk1 > risk2:
        winners['risk_score'] = stock1['ticker']
    elif risk1 < risk2:
        winners['risk_score'] = stock2['ticker']
    else:
        winners['risk_score'] = 'TIE'
    
    return winners


def calculate_overall_score(stock: Dict[str, Any], winners: Dict[str, str]) -> int:
    """
    Calculate overall score for a stock based on how many metrics it wins.
    
    Args:
        stock: Stock metrics
        winners: Winner determinations
    
    Returns:
        Score (number of wins)
    """
    score = 0
    ticker = stock['ticker']
    
    for metric, winner in winners.items():
        if winner == ticker:
            score += 1
    
    return score


def generate_recommendation(stock1: Dict[str, Any], stock2: Dict[str, Any], 
                          winners: Dict[str, str]) -> Dict[str, Any]:
    """
    Generate investment recommendation based on comparison.
    
    Args:
        stock1: First stock's metrics
        stock2: Second stock's metrics
        winners: Winner determinations
    
    Returns:
        Dictionary with recommendation details
    """
    score1 = calculate_overall_score(stock1, winners)
    score2 = calculate_overall_score(stock2, winners)
    
    # Determine strengths for each stock
    strengths1 = []
    strengths2 = []
    
    if winners['cumulative_return'] == stock1['ticker']:
        strengths1.append(f"Higher cumulative return ({stock1['cumulative_return']*100:.2f}% vs {stock2['cumulative_return']*100:.2f}%)")
    elif winners['cumulative_return'] == stock2['ticker']:
        strengths2.append(f"Higher cumulative return ({stock2['cumulative_return']*100:.2f}% vs {stock1['cumulative_return']*100:.2f}%)")
    
    if winners['volatility'] == stock1['ticker']:
        strengths1.append(f"Lower volatility ({stock1['volatility']*100:.2f}% vs {stock2['volatility']*100:.2f}%)")
    elif winners['volatility'] == stock2['ticker']:
        strengths2.append(f"Lower volatility ({stock2['volatility']*100:.2f}% vs {stock1['volatility']*100:.2f}%)")
    
    if winners['sharpe_ratio'] == stock1['ticker']:
        strengths1.append(f"Better Sharpe ratio ({stock1['sharpe_ratio']:.2f} vs {stock2['sharpe_ratio']:.2f})")
    elif winners['sharpe_ratio'] == stock2['ticker']:
        strengths2.append(f"Better Sharpe ratio ({stock2['sharpe_ratio']:.2f} vs {stock1['sharpe_ratio']:.2f})")
    
    if winners['max_drawdown'] == stock1['ticker']:
        strengths1.append(f"Smaller max drawdown ({stock1['max_drawdown']*100:.2f}% vs {stock2['max_drawdown']*100:.2f}%)")
    elif winners['max_drawdown'] == stock2['ticker']:
        strengths2.append(f"Smaller max drawdown ({stock2['max_drawdown']*100:.2f}% vs {stock1['max_drawdown']*100:.2f}%)")
    
    if winners['bullish_signals'] == stock1['ticker']:
        strengths1.append(f"More bullish signals ({stock1['bullish_count']}/{stock1['total_signals']} vs {stock2['bullish_count']}/{stock2['total_signals']})")
    elif winners['bullish_signals'] == stock2['ticker']:
        strengths2.append(f"More bullish signals ({stock2['bullish_count']}/{stock2['total_signals']} vs {stock1['bullish_count']}/{stock1['total_signals']})")
    
    if winners['risk_score'] == stock1['ticker']:
        strengths1.append(f"Better risk score ({stock1['risk_score']} vs {stock2['risk_score']})")
    elif winners['risk_score'] == stock2['ticker']:
        strengths2.append(f"Better risk score ({stock2['risk_score']} vs {stock1['risk_score']})")
    
    # Determine overall recommendation
    if score1 > score2:
        overall_winner = stock1['ticker']
        recommendation = f"{stock1['ticker']} shows stronger overall performance"
    elif score2 > score1:
        overall_winner = stock2['ticker']
        recommendation = f"{stock2['ticker']} shows stronger overall performance"
    else:
        overall_winner = 'TIE'
        recommendation = "Both stocks show similar overall performance"
    
    # Add investor type suggestions
    investor_suggestions = []
    
    # Growth vs stability
    if stock1['cumulative_return'] > stock2['cumulative_return'] and stock1['volatility'] > stock2['volatility']:
        investor_suggestions.append(f"{stock1['ticker']} for growth-focused investors (higher returns, higher risk)")
        investor_suggestions.append(f"{stock2['ticker']} for risk-averse investors (lower volatility)")
    elif stock2['cumulative_return'] > stock1['cumulative_return'] and stock2['volatility'] > stock1['volatility']:
        investor_suggestions.append(f"{stock2['ticker']} for growth-focused investors (higher returns, higher risk)")
        investor_suggestions.append(f"{stock1['ticker']} for risk-averse investors (lower volatility)")
    
    return {
        'overall_winner': overall_winner,
        'score1': score1,
        'score2': score2,
        'strengths1': strengths1,
        'strengths2': strengths2,
        'recommendation': recommendation,
        'investor_suggestions': investor_suggestions
    }


def compare_stocks(ticker1: str, ticker2: str, period: str = '1y', 
                  risk_free_rate: float = 0.0) -> Optional[Dict[str, Any]]:
    """
    Compare two stocks across all metrics.
    
    Args:
        ticker1: First stock ticker
        ticker2: Second stock ticker
        period: Historical data period
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Dictionary with comparison results, or None if comparison fails
    """
    print(f"Analyzing {ticker1.upper()}...")
    stock1 = analyze_single_stock(ticker1, period, risk_free_rate)
    
    if not stock1:
        return None
    
    print(f"Analyzing {ticker2.upper()}...")
    stock2 = analyze_single_stock(ticker2, period, risk_free_rate)
    
    if not stock2:
        return None
    
    print("Comparing metrics...")
    winners = determine_winners(stock1, stock2)
    recommendation = generate_recommendation(stock1, stock2, winners)
    
    return {
        'stock1': stock1,
        'stock2': stock2,
        'winners': winners,
        'recommendation': recommendation
    }
