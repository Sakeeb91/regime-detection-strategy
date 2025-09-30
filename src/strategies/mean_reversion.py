"""
Mean reversion strategy for range-bound market regimes.

Implements Bollinger Bands and RSI-based mean reversion with
regime-specific parameterization.
"""

from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy
from ..utils import indicators as ta


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using Bollinger Bands and RSI.

    This strategy is designed for range-bound, sideways market regimes and uses:
    - Bollinger Bands for overbought/oversold levels
    - RSI for momentum confirmation
    - Z-score for mean reversion signal strength

    Attributes:
        bb_period (int): Bollinger Bands period
        bb_std (float): Bollinger Bands standard deviation
        rsi_period (int): RSI calculation period
        rsi_lower (float): RSI oversold threshold
        rsi_upper (float): RSI overbought threshold
        zscore_threshold (float): Z-score threshold for entry
        use_zscore (bool): Whether to use z-score filtering

    Examples:
        >>> strategy = MeanReversionStrategy(bb_period=20, bb_std=2.0)
        >>> signals = strategy.generate_signals(price_data)
        >>> positions = strategy.get_positions(signals)
    """

    def __init__(
        self,
        name: str = "MeanReversion",
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_lower: float = 30.0,
        rsi_upper: float = 70.0,
        zscore_threshold: float = 1.5,
        use_zscore: bool = True,
    ):
        """
        Initialize mean reversion strategy.

        Args:
            name: Strategy name
            bb_period: Bollinger Bands lookback period
            bb_std: Number of standard deviations for bands
            rsi_period: RSI calculation period
            rsi_lower: RSI oversold threshold (buy signal)
            rsi_upper: RSI overbought threshold (sell signal)
            zscore_threshold: Z-score threshold for entry
            use_zscore: Whether to use z-score filtering
        """
        super().__init__(name)
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.zscore_threshold = zscore_threshold
        self.use_zscore = use_zscore

        logger.info(
            f"MeanReversionStrategy initialized: bb_period={bb_period}, "
            f"rsi_range=[{rsi_lower}, {rsi_upper}]"
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate mean reversion signals.

        Long signal: Price below lower BB AND RSI oversold
        Short signal: Price above upper BB AND RSI overbought
        Exit: Price crosses middle BB (mean)

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Series with signals (1: long, -1: short, 0: neutral)
        """
        df = data.copy()

        # Calculate Bollinger Bands
        bb_result = ta.bbands(df["close"], length=self.bb_period, std=self.bb_std)
        df["bb_lower"] = bb_result[f"BBL_{self.bb_period}_{self.bb_std}"]
        df["bb_middle"] = bb_result[f"BBM_{self.bb_period}_{self.bb_std}"]
        df["bb_upper"] = bb_result[f"BBU_{self.bb_period}_{self.bb_std}"]
        df["bb_width"] = bb_result[f"BBB_{self.bb_period}_{self.bb_std}"]

        # Calculate RSI
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)

        # Calculate Z-score (price distance from mean in std units)
        if self.use_zscore:
            rolling_mean = df["close"].rolling(window=self.bb_period).mean()
            rolling_std = df["close"].rolling(window=self.bb_period).std()
            df["zscore"] = (df["close"] - rolling_mean) / rolling_std

        # Initialize signals
        signals = pd.Series(0, index=df.index)

        # Oversold condition: price below lower band and RSI oversold
        oversold = (df["close"] < df["bb_lower"]) & (df["rsi"] < self.rsi_lower)

        # Overbought condition: price above upper band and RSI overbought
        overbought = (df["close"] > df["bb_upper"]) & (df["rsi"] > self.rsi_upper)

        # Apply z-score filter if enabled
        if self.use_zscore:
            oversold = oversold & (df["zscore"] < -self.zscore_threshold)
            overbought = overbought & (df["zscore"] > self.zscore_threshold)

        # Exit conditions: price crosses back to middle band
        exit_long = df["close"] > df["bb_middle"]
        exit_short = df["close"] < df["bb_middle"]

        # Generate signals
        signals[oversold] = 1  # Long
        signals[overbought] = -1  # Short

        # Apply exits by forward filling and then zeroing at exit points
        signals = signals.replace(0, np.nan)

        # Forward fill positions
        current_position = signals.fillna(method="ffill").fillna(0)

        # Zero out positions at exit points
        current_position[(current_position == 1) & exit_long] = 0
        current_position[(current_position == -1) & exit_short] = 0

        # Forward fill again to maintain flat position until next signal
        signals = current_position.replace(0, np.nan).fillna(method="ffill").fillna(0)

        logger.debug(
            f"Generated {(signals == 1).sum()} long and "
            f"{(signals == -1).sum()} short signals"
        )

        return signals

    def get_positions(self, signals: pd.Series) -> pd.Series:
        """
        Convert signals to positions.

        Args:
            signals: Trading signals

        Returns:
            Position series (1: long, -1: short, 0: flat)
        """
        return signals

    def get_bb_position(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate position of price relative to Bollinger Bands.

        Returns normalized position: 0 = lower band, 0.5 = middle, 1 = upper band

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Series with BB position (0 to 1)
        """
        df = data.copy()

        bb_result = ta.bbands(df["close"], length=self.bb_period, std=self.bb_std)
        bb_lower = bb_result[f"BBL_{self.bb_period}_{self.bb_std}"]
        bb_upper = bb_result[f"BBU_{self.bb_period}_{self.bb_std}"]

        # Normalize price position between bands
        bb_position = (df["close"] - bb_lower) / (bb_upper - bb_lower)
        bb_position = bb_position.clip(0, 1)  # Clip to [0, 1]

        return bb_position

    def get_regime_suitability(self, data: pd.DataFrame) -> float:
        """
        Calculate how suitable current market regime is for mean reversion.

        Higher score = more suitable (range-bound, low trend)

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Suitability score (0 to 1)
        """
        df = data.copy()

        # Calculate ADX (lower ADX = weaker trend = better for mean reversion)
        adx_result = ta.adx(df["high"], df["low"], df["close"], length=14)
        current_adx = adx_result[f"ADX_14"].iloc[-1] if len(df) > 14 else 50

        # Calculate BB width percentile (narrower = better for mean reversion)
        bb_result = ta.bbands(df["close"], length=self.bb_period, std=self.bb_std)
        bb_width = bb_result[f"BBB_{self.bb_period}_{self.bb_std}"]
        current_width_percentile = (bb_width.iloc[-1] < bb_width).sum() / len(bb_width)

        # Combine scores (inverse ADX + width percentile)
        adx_score = 1 - (current_adx / 100)  # Lower ADX = higher score
        width_score = 1 - current_width_percentile  # Narrower bands = higher score

        suitability = (adx_score + width_score) / 2

        return float(suitability)

    def get_parameters(self) -> Dict:
        """
        Get strategy parameters.

        Returns:
            Dictionary of strategy parameters
        """
        return {
            "name": self.name,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "rsi_period": self.rsi_period,
            "rsi_lower": self.rsi_lower,
            "rsi_upper": self.rsi_upper,
            "zscore_threshold": self.zscore_threshold,
            "use_zscore": self.use_zscore,
        }
