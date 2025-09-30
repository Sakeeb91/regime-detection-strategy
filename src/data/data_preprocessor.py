"""
Data preprocessing module for cleaning and transforming market data.

Handles missing data, outliers, resampling, and data validation.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


class DataPreprocessor:
    """
    Preprocess and clean financial market data.

    Provides methods for handling missing values, outliers, and data validation.

    Examples:
        >>> preprocessor = DataPreprocessor()
        >>> clean_data = preprocessor.clean_data(raw_data)
        >>> validated = preprocessor.validate_data(clean_data)
    """

    def __init__(
        self,
        fill_method: str = "forward",
        outlier_std: float = 5.0,
        min_data_points: int = 100
    ):
        """
        Initialize the DataPreprocessor.

        Args:
            fill_method: Method for filling missing values ('forward', 'linear', 'mean')
            outlier_std: Standard deviations threshold for outlier detection
            min_data_points: Minimum required data points for processing
        """
        self.fill_method = fill_method
        self.outlier_std = outlier_std
        self.min_data_points = min_data_points

        logger.info(f"DataPreprocessor initialized with fill_method={fill_method}")

    def clean_data(
        self,
        df: pd.DataFrame,
        remove_outliers: bool = True,
        handle_missing: bool = True
    ) -> pd.DataFrame:
        """
        Clean and preprocess market data.

        Args:
            df: Input DataFrame with OHLCV data
            remove_outliers: Whether to detect and handle outliers
            handle_missing: Whether to fill missing values

        Returns:
            Cleaned DataFrame

        Raises:
            ValueError: If DataFrame is too small or has invalid structure

        Examples:
            >>> preprocessor = DataPreprocessor()
            >>> clean_df = preprocessor.clean_data(raw_df)
        """
        df = df.copy()

        # Validate input
        self._validate_input(df)

        logger.info(f"Cleaning data: {len(df)} rows, {len(df.columns)} columns")

        # Remove duplicates
        original_len = len(df)
        df = df[~df.index.duplicated(keep='first')]
        if len(df) < original_len:
            logger.warning(f"Removed {original_len - len(df)} duplicate rows")

        # Handle missing values
        if handle_missing:
            df = self._handle_missing_values(df)

        # Remove outliers
        if remove_outliers:
            df = self._remove_outliers(df)

        # Ensure proper data types
        df = self._ensure_data_types(df)

        logger.info(f"Data cleaning complete: {len(df)} rows remaining")

        return df

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate data quality and integrity.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, message)

        Examples:
            >>> preprocessor = DataPreprocessor()
            >>> is_valid, msg = preprocessor.validate_data(df)
            >>> if not is_valid:
            ...     print(f"Validation failed: {msg}")
        """
        # Check minimum data points
        if len(df) < self.min_data_points:
            return False, f"Insufficient data: {len(df)} < {self.min_data_points}"

        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"

        # Check for excessive missing data
        missing_pct = df.isnull().sum() / len(df)
        if (missing_pct > 0.1).any():
            cols_with_missing = missing_pct[missing_pct > 0.1].index.tolist()
            return False, f"Excessive missing data in columns: {cols_with_missing}"

        # Check for invalid OHLC relationships
        invalid_ohlc = ~(
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        )
        if invalid_ohlc.sum() > 0:
            return False, f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships"

        # Check for non-positive prices
        price_cols = ['open', 'high', 'low', 'close']
        if (df[price_cols] <= 0).any().any():
            return False, "Found non-positive prices"

        # Check for non-negative volume
        if (df['volume'] < 0).any():
            return False, "Found negative volume values"

        return True, "Data validation passed"

    def resample_data(
        self,
        df: pd.DataFrame,
        frequency: str = "1D",
        aggregation: str = "ohlc"
    ) -> pd.DataFrame:
        """
        Resample data to different frequency.

        Args:
            df: Input DataFrame
            frequency: Target frequency ('1D', '1H', '1W', etc.)
            aggregation: Aggregation method ('ohlc', 'mean', 'last')

        Returns:
            Resampled DataFrame

        Examples:
            >>> # Resample daily data to weekly
            >>> weekly_df = preprocessor.resample_data(daily_df, frequency='1W')
        """
        if aggregation == "ohlc":
            resampled = pd.DataFrame({
                'open': df['open'].resample(frequency).first(),
                'high': df['high'].resample(frequency).max(),
                'low': df['low'].resample(frequency).min(),
                'close': df['close'].resample(frequency).last(),
                'volume': df['volume'].resample(frequency).sum()
            })
        elif aggregation == "mean":
            resampled = df.resample(frequency).mean()
        elif aggregation == "last":
            resampled = df.resample(frequency).last()
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

        logger.info(f"Resampled data from {len(df)} to {len(resampled)} rows")

        return resampled.dropna()

    def calculate_returns(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        method: str = "log"
    ) -> pd.Series:
        """
        Calculate returns from price data.

        Args:
            df: Input DataFrame with price data
            price_col: Column name containing prices
            method: Return calculation method ('log', 'simple', 'pct')

        Returns:
            Series of returns

        Examples:
            >>> returns = preprocessor.calculate_returns(df, method='log')
        """
        prices = df[price_col]

        if method == "log":
            returns = np.log(prices / prices.shift(1))
        elif method == "simple":
            returns = prices.pct_change()
        elif method == "pct":
            returns = (prices - prices.shift(1)) / prices.shift(1) * 100
        else:
            raise ValueError(f"Unsupported return method: {method}")

        return returns

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            return df

        logger.info(f"Handling {missing_count} missing values using '{self.fill_method}' method")

        if self.fill_method == "forward":
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif self.fill_method == "linear":
            df = df.interpolate(method='linear', limit_direction='both')
        elif self.fill_method == "mean":
            df = df.fillna(df.mean())
        else:
            logger.warning(f"Unknown fill method '{self.fill_method}', using forward fill")
            df = df.fillna(method='ffill').fillna(method='bfill')

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and remove outliers using z-score method.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with outliers removed
        """
        # Calculate returns for outlier detection
        if 'close' in df.columns:
            returns = df['close'].pct_change()

            # Calculate z-scores
            z_scores = np.abs(stats.zscore(returns.dropna()))

            # Find outliers
            outlier_mask = z_scores > self.outlier_std

            if outlier_mask.sum() > 0:
                logger.warning(f"Detected {outlier_mask.sum()} outliers (>{self.outlier_std} std)")

                # Replace outliers with interpolated values
                outlier_indices = returns.dropna().index[outlier_mask]
                df.loc[outlier_indices, 'close'] = np.nan
                df['close'] = df['close'].interpolate(method='linear')

        return df

    def _ensure_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure proper data types for columns.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with corrected data types
        """
        # Convert numeric columns to float
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                logger.warning(f"Could not convert index to datetime: {str(e)}")

        return df

    def _validate_input(self, df: pd.DataFrame) -> None:
        """
        Validate input DataFrame structure.

        Args:
            df: Input DataFrame

        Raises:
            ValueError: If DataFrame is invalid
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        if len(df) < self.min_data_points:
            raise ValueError(
                f"Insufficient data points: {len(df)} < {self.min_data_points}"
            )