"""
Data Fetcher Module
Handles fetching and preparing historical stock data.
"""

import hashlib
import os
import pathlib
import time

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Optional

CACHE_DIR = os.environ.get("QFCLI_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".qfcli", "cache"))
CACHE_TTL_SECONDS = int(os.environ.get("QFCLI_CACHE_TTL", str(6 * 60 * 60)))


def _cache_key(ticker: str, period: str) -> str:
    raw = f"{ticker.upper()}|{period}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def _cache_path(key: str) -> pathlib.Path:
    return pathlib.Path(CACHE_DIR) / f"{key}.pkl"


def _load_cached(key: str) -> Optional[Tuple[pd.DataFrame, str]]:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        if time.time() - path.stat().st_mtime > CACHE_TTL_SECONDS:
            return None
        payload = pd.read_pickle(path)
        # backward compatible: bare frame cached in old versions
        if isinstance(payload, pd.DataFrame):
            return payload, None
        return payload.get('df'), payload.get('name')
    except Exception:
        return None


def _save_cached(key: str, df: pd.DataFrame, name: str) -> None:
    try:
        path = _cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({'df': df, 'name': name}, path)
    except Exception:
        pass


def fetch_stock_data(ticker: str, period: str = '1y', use_cache: bool = True) -> Tuple[pd.DataFrame, str]:
    """
    Fetch historical stock data for a given ticker.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
        period: Time period for historical data (default: '1y')
                Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        use_cache: Use the on-disk cache (Ticker|Period keyed, TTL 6h)
    
    Returns:
        Tuple of (DataFrame with OHLCV data, company name)
    
    Raises:
        ValueError: If ticker is invalid or data cannot be fetched
    """
    cache_key = _cache_key(ticker, period)
    if use_cache:
        cached = _load_cached(cache_key)
        if cached is not None and cached[0] is not None:
            df, name = cached
            if not df.empty:
                return df, name if name else ticker.upper()

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

        if use_cache:
            _save_cached(cache_key, df, company_name)

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


def fetch_benchmark_returns(ticker: str, period: str = '1y', use_cache: bool = True) -> pd.Series:
    """
    Fetch daily returns for a benchmark/index (e.g. '^GSPC', 'SPY').

    Args:
        ticker: Benchmark symbol
        period: Time period (must match the asset's period)
        use_cache: Use the on-disk cache

    Returns:
        Series of daily benchmark returns

    Raises:
        ValueError: If benchmark data cannot be fetched
    """
    try:
        df, _ = fetch_stock_data(ticker, period, use_cache=use_cache)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error fetching benchmark data for '{ticker}': {str(e)}")

    if df.empty:
        raise ValueError(f"No data found for benchmark '{ticker}'.")
    return df['Close'].pct_change().dropna()
