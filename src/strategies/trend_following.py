"""
Trend following strategy for trending market regimes.

Implements a dual moving average crossover strategy with trend confirmation
using ADX and momentum indicators.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger

from .base_strategy import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following strategy using moving average crossovers and trend strength.

    This strategy is designed for trending market regimes and uses:
    - Dual moving average crossover for trend direction
    - ADX for trend strength confirmation
    - ATR for position sizing

    Attributes:
        fast_period (int): Fast moving average period
        slow_period (int): Slow moving average period
        adx_period (int): ADX calculation period
        adx_threshold (float): Minimum ADX value to enter trades
        use_stops (bool): Whether to use ATR-based stops
        atr_multiplier (float): ATR multiplier for stop loss

    Examples:
        >>> strategy = TrendFollowingStrategy(fast_period=20, slow_period=50)
        >>> signals = strategy.generate_signals(price_data)
        >>> positions = strategy.get_positions(signals)
    """

    def __init__(
        self,
        name: str = "TrendFollowing",
        fast_period: int = 20,
        slow_period: int = 50,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        use_stops: bool = True,
        atr_multiplier: float = 2.0
    ):
        """
        Initialize trend following strategy.

        Args:
            name: Strategy name
            fast_period: Fast moving average period
            slow_period: Slow moving average period
            adx_period: ADX calculation period
            adx_threshold: Minimum ADX to confirm trend
            use_stops: Whether to use stop losses
            atr_multiplier: ATR multiplier for stops
        """
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.use_stops = use_stops
        self.atr_multiplier = atr_multiplier

        logger.info(
            f"TrendFollowingStrategy initialized: fast={fast_period}, "
            f"slow={slow_period}, adx_threshold={adx_threshold}"
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trend following signals.

        Long signal: Fast MA crosses above slow MA AND ADX > threshold
        Short signal: Fast MA crosses below slow MA AND ADX > threshold
        Exit: MA crossover in opposite direction

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Series with signals (1: long, -1: short, 0: neutral)
        """
        df = data.copy()

        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_period).mean()

        # Calculate ADX for trend strength
        adx_result = ta.adx(
            df['high'],
            df['low'],
            df['close'],
            length=self.adx_period
        )
        df['adx'] = adx_result[f'ADX_{self.adx_period}']

        # Calculate ATR for volatility-based stops
        if self.use_stops:
            df['atr'] = ta.atr(
                df['high'],
                df['low'],
                df['close'],
                length=self.adx_period
            )

        # Initialize signals
        signals = pd.Series(0, index=df.index)

        # Trend condition: fast MA above/below slow MA
        bullish_trend = df['fast_ma'] > df['slow_ma']
        bearish_trend = df['fast_ma'] < df['slow_ma']

        # Strong trend condition: ADX above threshold
        strong_trend = df['adx'] > self.adx_threshold

        # Generate signals
        # Long: bullish crossover with strong trend
        signals[bullish_trend & strong_trend] = 1

        # Short: bearish crossover with strong trend
        signals[bearish_trend & strong_trend] = -1

        # Fill forward to maintain position until reversal
        signals = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)

        logger.debug(
            f"Generated {(signals == 1).sum()} long and "
            f"{(signals == -1).sum()} short signals"
        )

        return signals

    def get_positions(self, signals: pd.Series) -> pd.Series:
        """
        Convert signals to positions with optional stop loss management.

        Args:
            signals: Trading signals

        Returns:
            Position series (1: long, -1: short, 0: flat)
        """
        # For now, directly use signals as positions
        # Future enhancement: implement trailing stops using ATR
        return signals

    def get_stop_levels(self, data: pd.DataFrame, signals: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate stop loss levels using ATR.

        Args:
            data: DataFrame with OHLCV data
            signals: Position signals

        Returns:
            Dictionary with 'stop_long' and 'stop_short' series
        """
        if not self.use_stops:
            return {'stop_long': None, 'stop_short': None}

        df = data.copy()

        # Calculate ATR
        df['atr'] = ta.atr(
            df['high'],
            df['low'],
            df['close'],
            length=self.adx_period
        )

        # Calculate stop levels
        stop_long = df['close'] - (self.atr_multiplier * df['atr'])
        stop_short = df['close'] + (self.atr_multiplier * df['atr'])

        return {
            'stop_long': stop_long,
            'stop_short': stop_short
        }

    def get_parameters(self) -> Dict:
        """
        Get strategy parameters.

        Returns:
            Dictionary of strategy parameters
        """
        return {
            'name': self.name,
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'adx_period': self.adx_period,
            'adx_threshold': self.adx_threshold,
            'use_stops': self.use_stops,
            'atr_multiplier': self.atr_multiplier
        }