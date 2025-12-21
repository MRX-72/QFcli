"""
Indicators Module
Technical indicators and signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union


def simple_moving_average(prices: Union[pd.Series, np.ndarray], window: int) -> float:
    """
    Calculate Simple Moving Average for the most recent period.
    
    Formula: SMA_n = (1/n) * Σ(P_i) for i=1 to n
    
    Args:
        prices: Series or array of closing prices
        window: Number of periods for the moving average
    
    Returns:
        SMA value for the most recent period, or NaN if insufficient data
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    if len(prices) < window:
        return np.nan
    
    # Calculate SMA for the most recent window
    sma = np.mean(prices[-window:])
    return sma


def moving_average_signals(current_price: float, sma_values: Dict[int, float]) -> Dict[int, str]:
    """
    Generate bullish/bearish signals based on price vs moving averages.
    
    Logic:
    - Price > MA → BULLISH
    - Price < MA → BEARISH
    
    Args:
        current_price: Current stock price
        sma_values: Dictionary of {window: sma_value}
    
    Returns:
        Dictionary of {window: signal}
    """
    signals = {}
    
    for window, sma in sma_values.items():
        if np.isnan(sma):
            signals[window] = "INSUFFICIENT DATA"
        elif current_price > sma:
            signals[window] = "BULLISH"
        else:
            signals[window] = "BEARISH"
    
    return signals


def rate_of_change(prices: Union[pd.Series, np.ndarray], period: int = 12) -> float:
    """
    Calculate Rate of Change (momentum indicator).
    
    Formula: ROC = (P_t - P_{t-n}) / P_{t-n}
    
    Args:
        prices: Series or array of closing prices
        period: Number of periods to look back (default: 12)
    
    Returns:
        ROC as a decimal, or NaN if insufficient data
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    if len(prices) <= period:
        return np.nan
    
    current_price = prices[-1]
    past_price = prices[-(period + 1)]
    
    roc = (current_price - past_price) / past_price
    return roc


def calculate_all_smas(prices: Union[pd.Series, np.ndarray], 
                       windows: list = [20, 50, 200]) -> Dict[int, float]:
    """
    Calculate multiple SMAs at once.
    
    Args:
        prices: Series or array of closing prices
        windows: List of window sizes (default: [20, 50, 200])
    
    Returns:
        Dictionary of {window: sma_value}
    """
    sma_values = {}
    
    for window in windows:
        sma_values[window] = simple_moving_average(prices, window)
    
    return sma_values


def bollinger_bands(prices: Union[pd.Series, np.ndarray], 
                   window: int = 20, 
                   num_std: float = 2.0) -> Dict[str, float]:
    """
    Calculate Bollinger Bands (bonus indicator).
    
    Args:
        prices: Series or array of closing prices
        window: SMA window (default: 20)
        num_std: Number of standard deviations (default: 2.0)
    
    Returns:
        Dictionary with 'upper', 'middle', 'lower' bands
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    if len(prices) < window:
        return {'upper': np.nan, 'middle': np.nan, 'lower': np.nan}
    
    # Calculate middle band (SMA)
    middle = np.mean(prices[-window:])
    
    # Calculate standard deviation
    std = np.std(prices[-window:])
    
    # Calculate upper and lower bands
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }


def calculate_ema(prices: Union[pd.Series, np.ndarray], window: int) -> float:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        prices: Series or array of closing prices
        window: Window size
    
    Returns:
        EMA value for the most recent period
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
        
    ema = prices.ewm(span=window, adjust=False).mean()
    return ema.iloc[-1]


def calculate_rsi(prices: Union[pd.Series, np.ndarray], period: int = 14) -> float:
    """
    Calculate Relative Strength Index (RSI).
    
    Formula: RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss
    
    Args:
        prices: Series or array of closing prices
        period: Lookback period (default: 14)
    
    Returns:
        RSI value (0-100)
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
        
    delta = prices.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1]


def calculate_macd(prices: Union[pd.Series, np.ndarray], 
                  fast_period: int = 12, 
                  slow_period: int = 26, 
                  signal_period: int = 9) -> Dict[str, float]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Series or array of closing prices
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)
    
    Returns:
        Dictionary with 'macd_line', 'signal_line', 'histogram'
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
        
    # Calculate Fast and Slow EMAs
    exp1 = prices.ewm(span=fast_period, adjust=False).mean()
    exp2 = prices.ewm(span=slow_period, adjust=False).mean()
    
    # MACD Line
    macd = exp1 - exp2
    
    # Signal Line
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    
    # Histogram
    hist = macd - signal
    
    return {
        'macd_line': macd.iloc[-1],
        'signal_line': signal.iloc[-1],
        'histogram': hist.iloc[-1]
    }
