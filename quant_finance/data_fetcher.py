"""
Data Fetcher Module
Handles fetching and preparing historical stock data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def fetch_stock_data(ticker: str, period: str = '1y') -> Tuple[pd.DataFrame, str]:
    """
    Fetch historical stock data for a given ticker.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
        period: Time period for historical data (default: '1y')
                Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    
    Returns:
        Tuple of (DataFrame with OHLCV data, company name)
    
    Raises:
        ValueError: If ticker is invalid or data cannot be fetched
    """
    try:
        stock = yf.Ticker(ticker.upper())
        
        # Fetch historical data
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data found for ticker '{ticker}'. Please verify the symbol.")
        
        # Get company name
        try:
            company_name = stock.info.get('longName', ticker.upper())
        except:
            company_name = ticker.upper()
        
        return df, company_name
    
    except Exception as e:
        raise ValueError(f"Error fetching data for '{ticker}': {str(e)}")


def calculate_daily_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate daily returns from price series.
    
    Formula: r_t = (P_t - P_{t-1}) / P_{t-1}
    
    Args:
        prices: Series of closing prices
    
    Returns:
        Series of daily returns (first value will be NaN)
    """
    returns = prices.pct_change()
    return returns


def get_price_array(df: pd.DataFrame, column: str = 'Close') -> np.ndarray:
    """
    Extract price array from DataFrame.
    
    Args:
        df: DataFrame with price data
        column: Column name to extract (default: 'Close')
    
    Returns:
        NumPy array of prices
    """
    return df[column].values


def get_current_price(df: pd.DataFrame) -> float:
    """
    Get the most recent closing price.
    
    Args:
        df: DataFrame with price data
    
    Returns:
        Latest closing price
    """
    return df['Close'].iloc[-1]


def get_date_range(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Get the date range of the data.
    
    Args:
        df: DataFrame with price data
    
    Returns:
        Tuple of (start_date, end_date) as strings
    """
    start_date = df.index[0].strftime('%Y-%m-%d')
    end_date = df.index[-1].strftime('%Y-%m-%d')
    return start_date, end_date
