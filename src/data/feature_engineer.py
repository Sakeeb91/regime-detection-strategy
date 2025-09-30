"""
Feature engineering module for creating technical indicators and features.

Generates features for regime detection including volatility, momentum,
trend, and volume-based indicators.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger
from scipy.stats import skew, kurtosis


class FeatureEngineer:
    """
    Engineer technical and statistical features for regime detection.

    Creates comprehensive feature sets including trend, volatility, momentum,
    and volume indicators optimized for machine learning models.

    Examples:
        >>> engineer = FeatureEngineer()
        >>> features = engineer.create_features(price_data)
        >>> regime_features = engineer.extract_regime_features(features)
    """

    def __init__(
        self, windows: Optional[List[int]] = None, include_advanced: bool = True
    ):
        """
        Initialize the FeatureEngineer.

        Args:
            windows: List of lookback windows for indicators (default: [5, 10, 20, 50])
            include_advanced: Whether to include advanced features (higher computation)
        """
        self.windows = windows or [5, 10, 20, 50]
        self.include_advanced = include_advanced

        logger.info(f"FeatureEngineer initialized with windows={self.windows}")

    def create_features(
        self, df: pd.DataFrame, feature_groups: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create comprehensive feature set from OHLCV data.

        Args:
            df: Input DataFrame with OHLCV columns
            feature_groups: List of feature groups to include
                          ['trend', 'volatility', 'momentum', 'volume', 'statistical']
                          If None, includes all groups

        Returns:
            DataFrame with original data and engineered features

        Examples:
            >>> engineer = FeatureEngineer()
            >>> features = engineer.create_features(price_data)
            >>> print(features.columns)
        """
        df = df.copy()

        if feature_groups is None:
            feature_groups = [
                "trend",
                "volatility",
                "momentum",
                "volume",
                "statistical",
            ]

        logger.info(f"Creating features for groups: {feature_groups}")

        # Calculate returns (needed for many features)
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Create features by group
        if "trend" in feature_groups:
            df = self._add_trend_features(df)

        if "volatility" in feature_groups:
            df = self._add_volatility_features(df)

        if "momentum" in feature_groups:
            df = self._add_momentum_features(df)

        if "volume" in feature_groups:
            df = self._add_volume_features(df)

        if "statistical" in feature_groups:
            df = self._add_statistical_features(df)

        # Drop NaN values at the beginning
        initial_rows = len(df)
        df = df.dropna()
        logger.info(
            f"Created features: {len(df.columns)} columns, dropped {initial_rows - len(df)} rows with NaN"
        )

        return df

    def extract_regime_features(
        self, df: pd.DataFrame, n_features: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract most important features for regime detection.

        Selects subset of features most relevant for identifying market regimes.

        Args:
            df: DataFrame with all features
            n_features: Number of top features to select (None = use default set)

        Returns:
            DataFrame with selected regime-relevant features

        Examples:
            >>> regime_features = engineer.extract_regime_features(all_features)
        """
        # Define core regime features
        regime_cols = [
            "returns",
            "log_returns",
            "volatility_20",
            "volatility_50",
            "rsi_14",
            "macd",
            "macd_signal",
            "bb_width",
            "atr_14",
            "volume_ratio_20",
            "volume_std_20",
            "rolling_skew_20",
            "rolling_kurt_20",
            "trend_strength_20",
        ]

        # Filter to available columns
        available_cols = [col for col in regime_cols if col in df.columns]

        if n_features is not None:
            available_cols = available_cols[:n_features]

        logger.info(f"Extracted {len(available_cols)} regime features")

        return df[available_cols]

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend-based features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with trend features added
        """
        for window in self.windows:
            # Simple Moving Average
            df[f"sma_{window}"] = ta.sma(df["close"], length=window)

            # Exponential Moving Average
            df[f"ema_{window}"] = ta.ema(df["close"], length=window)

            # Price relative to moving average
            df[f"price_to_sma_{window}"] = df["close"] / df[f"sma_{window}"] - 1

            # Moving average slope (trend direction)
            df[f"sma_slope_{window}"] = (
                df[f"sma_{window}"].diff(5) / df[f"sma_{window}"]
            )

        # MACD
        macd_data = ta.macd(df["close"])
        if macd_data is not None and not macd_data.empty:
            df["macd"] = macd_data["MACD_12_26_9"]
            df["macd_signal"] = macd_data["MACDs_12_26_9"]
            df["macd_hist"] = macd_data["MACDh_12_26_9"]

        # ADX (Average Directional Index) - trend strength
        adx_data = ta.adx(df["high"], df["low"], df["close"])
        if adx_data is not None and not adx_data.empty:
            df["adx"] = adx_data["ADX_14"]

        # Trend strength (custom)
        for window in [20, 50]:
            df[f"trend_strength_{window}"] = (
                df["close"]
                .rolling(window)
                .apply(
                    lambda x: (
                        np.polyfit(range(len(x)), x, 1)[0]
                        if len(x) == window
                        else np.nan
                    )
                )
            )

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with volatility features added
        """
        for window in self.windows:
            # Historical volatility (annualized)
            df[f"volatility_{window}"] = df["returns"].rolling(window).std() * np.sqrt(
                252
            )

            # Parkinson volatility (using high-low range)
            df[f"parkinson_vol_{window}"] = np.sqrt(
                1
                / (4 * window * np.log(2))
                * (np.log(df["high"] / df["low"]) ** 2).rolling(window).sum()
            ) * np.sqrt(252)

        # ATR (Average True Range)
        atr_data = ta.atr(df["high"], df["low"], df["close"], length=14)
        if atr_data is not None:
            df["atr_14"] = atr_data

        # Bollinger Bands
        bb_data = ta.bbands(df["close"], length=20)
        if bb_data is not None and not bb_data.empty:
            df["bb_upper"] = bb_data["BBU_20_2.0"]
            df["bb_middle"] = bb_data["BBM_20_2.0"]
            df["bb_lower"] = bb_data["BBL_20_2.0"]
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            df["bb_position"] = (df["close"] - df["bb_lower"]) / (
                df["bb_upper"] - df["bb_lower"]
            )

        # Volatility regime indicator (rolling std of volatility)
        if "volatility_20" in df.columns:
            df["vol_of_vol"] = df["volatility_20"].rolling(20).std()

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum-based features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with momentum features added
        """
        # RSI (Relative Strength Index)
        df["rsi_14"] = ta.rsi(df["close"], length=14)

        # Stochastic Oscillator
        stoch_data = ta.stoch(df["high"], df["low"], df["close"])
        if stoch_data is not None and not stoch_data.empty:
            df["stoch_k"] = stoch_data["STOCHk_14_3_3"]
            df["stoch_d"] = stoch_data["STOCHd_14_3_3"]

        # Rate of Change
        for window in [5, 10, 20]:
            df[f"roc_{window}"] = ta.roc(df["close"], length=window)

        # Momentum (simple)
        for window in self.windows:
            df[f"momentum_{window}"] = df["close"] - df["close"].shift(window)

        # CCI (Commodity Channel Index)
        df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)

        # Williams %R
        df["williams_r"] = ta.willr(df["high"], df["low"], df["close"], length=14)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with volume features added
        """
        for window in self.windows:
            # Volume moving average
            df[f"volume_ma_{window}"] = df["volume"].rolling(window).mean()

            # Volume ratio (current / moving average)
            df[f"volume_ratio_{window}"] = df["volume"] / df[f"volume_ma_{window}"]

            # Volume standard deviation
            df[f"volume_std_{window}"] = df["volume"].rolling(window).std()

        # OBV (On-Balance Volume)
        df["obv"] = ta.obv(df["close"], df["volume"])

        # Volume-weighted average price (VWAP)
        df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

        # Money Flow Index
        df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)

        # Price-Volume Trend
        df["pvt"] = (df["returns"] * df["volume"]).cumsum()

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add statistical features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with statistical features added
        """
        for window in [20, 50]:
            # Rolling skewness
            df[f"rolling_skew_{window}"] = (
                df["returns"]
                .rolling(window)
                .apply(lambda x: skew(x) if len(x) == window else np.nan)
            )

            # Rolling kurtosis
            df[f"rolling_kurt_{window}"] = (
                df["returns"]
                .rolling(window)
                .apply(lambda x: kurtosis(x) if len(x) == window else np.nan)
            )

            # Rolling min and max returns
            df[f"rolling_min_{window}"] = df["returns"].rolling(window).min()
            df[f"rolling_max_{window}"] = df["returns"].rolling(window).max()

            # Return autocorrelation
            df[f"return_autocorr_{window}"] = (
                df["returns"]
                .rolling(window)
                .apply(lambda x: x.autocorr() if len(x) == window else np.nan)
            )

        # Hurst exponent (simplified version for mean reversion detection)
        if self.include_advanced:
            df["hurst_50"] = (
                df["close"]
                .rolling(50)
                .apply(lambda x: self._calculate_hurst(x) if len(x) == 50 else np.nan)
            )

        return df

    def _calculate_hurst(self, price_series: pd.Series) -> float:
        """
        Calculate Hurst exponent for mean reversion detection.

        Args:
            price_series: Price series

        Returns:
            Hurst exponent (0.5 = random walk, <0.5 = mean reverting, >0.5 = trending)
        """
        try:
            lags = range(2, 20)
            tau = [
                np.std(np.subtract(price_series[lag:], price_series[:-lag]))
                for lag in lags
            ]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        except Exception:
            return np.nan

    def get_feature_names(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get feature names grouped by category.

        Args:
            df: DataFrame with features

        Returns:
            Dictionary mapping category to list of feature names

        Examples:
            >>> feature_groups = engineer.get_feature_names(features_df)
            >>> print(feature_groups['volatility'])
        """
        feature_dict = {
            "trend": [],
            "volatility": [],
            "momentum": [],
            "volume": [],
            "statistical": [],
        }

        for col in df.columns:
            if any(x in col for x in ["sma", "ema", "macd", "adx", "trend"]):
                feature_dict["trend"].append(col)
            elif any(
                x in col for x in ["volatility", "vol", "atr", "bb_", "parkinson"]
            ):
                feature_dict["volatility"].append(col)
            elif any(
                x in col for x in ["rsi", "stoch", "roc", "momentum", "cci", "williams"]
            ):
                feature_dict["momentum"].append(col)
            elif any(x in col for x in ["volume", "obv", "vwap", "mfi", "pvt"]):
                feature_dict["volume"].append(col)
            elif any(
                x in col for x in ["skew", "kurt", "rolling_", "hurst", "autocorr"]
            ):
                feature_dict["statistical"].append(col)

        return feature_dict
