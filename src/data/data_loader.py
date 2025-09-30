"""
Data loading module for fetching market data from various sources.

This module provides a unified interface for downloading financial data
from multiple providers including Yahoo Finance and Alpha Vantage.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf
from loguru import logger


class DataLoader:
    """
    Load financial market data from various sources.

    Supports multiple data providers and caching for efficient retrieval.

    Attributes:
        cache_dir (str): Directory for caching downloaded data
        use_cache (bool): Whether to use cached data when available

    Examples:
        >>> loader = DataLoader(use_cache=True)
        >>> data = loader.load_data(['SPY', 'QQQ'], start_date='2020-01-01')
        >>> print(data['SPY'].head())
    """

    def __init__(self, cache_dir: str = "data/raw", use_cache: bool = True):
        """
        Initialize the DataLoader.

        Args:
            cache_dir: Directory path for caching data
            use_cache: If True, check cache before downloading
        """
        self.cache_dir = cache_dir
        self.use_cache = use_cache

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        logger.info(f"DataLoader initialized with cache_dir={cache_dir}")

    def load_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        source: str = "yahoo"
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load market data for one or more symbols.

        Args:
            symbols: Single symbol or list of symbols (e.g., 'SPY' or ['SPY', 'QQQ'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1d', '1h', etc.)
            source: Data source ('yahoo', 'alpha_vantage')

        Returns:
            DataFrame (single symbol) or dict of DataFrames (multiple symbols)

        Raises:
            ValueError: If invalid date range or symbols
            ConnectionError: If unable to fetch data from source

        Examples:
            >>> loader = DataLoader()
            >>> # Single symbol
            >>> spy_data = loader.load_data('SPY', start_date='2020-01-01')
            >>> # Multiple symbols
            >>> data = loader.load_data(['SPY', 'QQQ'], start_date='2020-01-01')
        """
        # Convert single symbol to list for uniform processing
        if isinstance(symbols, str):
            symbols = [symbols]
            return_single = True
        else:
            return_single = False

        # Set default dates
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = "2015-01-01"

        # Validate date range
        self._validate_dates(start_date, end_date)

        logger.info(f"Loading data for {len(symbols)} symbols from {start_date} to {end_date}")

        # Load data based on source
        if source == "yahoo":
            data_dict = self._load_from_yahoo(symbols, start_date, end_date, interval)
        elif source == "alpha_vantage":
            data_dict = self._load_from_alpha_vantage(symbols, start_date, end_date)
        else:
            raise ValueError(f"Unsupported data source: {source}")

        # Return single DataFrame or dict based on input
        if return_single:
            return data_dict[symbols[0]]
        return data_dict

    def _load_from_yahoo(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data from Yahoo Finance.

        Args:
            symbols: List of ticker symbols
            start_date: Start date string
            end_date: End date string
            interval: Data interval

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data_dict = {}

        for symbol in symbols:
            try:
                # Check cache first
                if self.use_cache:
                    cached_data = self._load_from_cache(symbol, start_date, end_date)
                    if cached_data is not None:
                        logger.info(f"Loaded {symbol} from cache")
                        data_dict[symbol] = cached_data
                        continue

                # Download from Yahoo Finance
                logger.info(f"Downloading {symbol} from Yahoo Finance")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval)

                if df.empty:
                    logger.warning(f"No data retrieved for {symbol}")
                    continue

                # Standardize column names
                df.columns = [col.lower() for col in df.columns]

                # Save to cache
                if self.use_cache:
                    self._save_to_cache(symbol, df, start_date, end_date)

                data_dict[symbol] = df
                logger.info(f"Successfully loaded {len(df)} rows for {symbol}")

            except Exception as e:
                logger.error(f"Error loading {symbol}: {str(e)}")
                continue

        if not data_dict:
            raise ConnectionError("Failed to load data for any symbols")

        return data_dict

    def _load_from_alpha_vantage(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data from Alpha Vantage API.

        Note: Requires ALPHA_VANTAGE_API_KEY environment variable.

        Args:
            symbols: List of ticker symbols
            start_date: Start date string
            end_date: End date string

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables")

        # Implementation would use Alpha Vantage API
        # Placeholder for now
        logger.warning("Alpha Vantage integration not yet implemented")
        raise NotImplementedError("Alpha Vantage data source coming soon")

    def _validate_dates(self, start_date: str, end_date: str) -> None:
        """
        Validate date range.

        Args:
            start_date: Start date string
            end_date: End date string

        Raises:
            ValueError: If dates are invalid or in wrong order
        """
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")

            if start >= end:
                raise ValueError("start_date must be before end_date")

            if end > datetime.now():
                logger.warning("end_date is in the future, adjusting to today")

        except ValueError as e:
            raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD': {str(e)}")

    def _load_from_cache(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available.

        Args:
            symbol: Ticker symbol
            start_date: Start date string
            end_date: End date string

        Returns:
            Cached DataFrame or None if not found
        """
        cache_file = os.path.join(
            self.cache_dir,
            f"{symbol}_{start_date}_{end_date}.parquet"
        )

        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache for {symbol}: {str(e)}")
                return None

        return None

    def _save_to_cache(
        self,
        symbol: str,
        df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> None:
        """
        Save DataFrame to cache.

        Args:
            symbol: Ticker symbol
            df: DataFrame to cache
            start_date: Start date string
            end_date: End date string
        """
        cache_file = os.path.join(
            self.cache_dir,
            f"{symbol}_{start_date}_{end_date}.parquet"
        )

        try:
            df.to_parquet(cache_file)
            logger.debug(f"Saved {symbol} to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache for {symbol}: {str(e)}")

    def get_available_symbols(self) -> List[str]:
        """
        Get list of symbols available in cache.

        Returns:
            List of cached symbol names
        """
        if not os.path.exists(self.cache_dir):
            return []

        files = os.listdir(self.cache_dir)
        symbols = set()

        for file in files:
            if file.endswith(".parquet"):
                symbol = file.split("_")[0]
                symbols.add(symbol)

        return sorted(list(symbols))