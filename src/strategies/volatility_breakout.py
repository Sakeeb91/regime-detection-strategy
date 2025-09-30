"""
Volatility breakout strategy for high volatility market regimes.

Implements ATR-based breakout detection with volume confirmation
and dynamic position sizing.
"""

from typing import Dict

import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy
from ..utils import indicators as ta


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Volatility breakout strategy using ATR and channel breakouts.

    This strategy is designed for high volatility regimes and uses:
    - Donchian Channels for breakout detection
    - ATR for volatility measurement and position sizing
    - Volume confirmation for breakout validity
    - Trailing stops based on ATR

    Attributes:
        lookback_period (int): Donchian channel lookback period
        atr_period (int): ATR calculation period
        atr_multiplier (float): ATR multiplier for breakout threshold
        volume_factor (float): Volume increase factor for confirmation
        use_volume_filter (bool): Whether to require volume confirmation
        trailing_stop_atr (float): ATR multiplier for trailing stops

    Examples:
        >>> strategy = VolatilityBreakoutStrategy(lookback_period=20)
        >>> signals = strategy.generate_signals(price_data)
        >>> positions = strategy.get_positions(signals)
    """

    def __init__(
        self,
        name: str = "VolatilityBreakout",
        lookback_period: int = 20,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
        volume_factor: float = 1.5,
        use_volume_filter: bool = True,
        trailing_stop_atr: float = 2.5,
    ):
        """
        Initialize volatility breakout strategy.

        Args:
            name: Strategy name
            lookback_period: Donchian channel lookback period
            atr_period: ATR calculation period
            atr_multiplier: ATR multiplier for breakout threshold
            volume_factor: Minimum volume increase for valid breakout
            use_volume_filter: Whether to require volume confirmation
            trailing_stop_atr: ATR multiplier for trailing stops
        """
        super().__init__(name)
        self.lookback_period = lookback_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.volume_factor = volume_factor
        self.use_volume_filter = use_volume_filter
        self.trailing_stop_atr = trailing_stop_atr

        logger.info(
            f"VolatilityBreakoutStrategy initialized: lookback={lookback_period}, "
            f"atr_multiplier={atr_multiplier}, volume_factor={volume_factor}"
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate volatility breakout signals.

        Long signal: Price breaks above upper Donchian + volume confirmation
        Short signal: Price breaks below lower Donchian + volume confirmation
        Exit: Trailing stop based on ATR

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Series with signals (1: long, -1: short, 0: neutral)
        """
        df = data.copy()

        # Calculate Donchian Channels
        df["donchian_upper"] = df["high"].rolling(window=self.lookback_period).max()
        df["donchian_lower"] = df["low"].rolling(window=self.lookback_period).min()
        df["donchian_middle"] = (df["donchian_upper"] + df["donchian_lower"]) / 2

        # Calculate ATR for volatility measurement
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)

        # Calculate volume confirmation
        if self.use_volume_filter:
            df["volume_ma"] = df["volume"].rolling(window=self.lookback_period).mean()
            df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # Calculate channel width (normalized by ATR)
        df["channel_width"] = (df["donchian_upper"] - df["donchian_lower"]) / df["atr"]

        # Initialize signals
        signals = pd.Series(0, index=df.index)

        # Bullish breakout: close above upper channel
        bullish_breakout = df["close"] > df["donchian_upper"].shift(1)

        # Bearish breakout: close below lower channel
        bearish_breakout = df["close"] < df["donchian_lower"].shift(1)

        # Apply volume filter if enabled
        if self.use_volume_filter:
            volume_confirmed = df["volume_ratio"] > self.volume_factor
            bullish_breakout = bullish_breakout & volume_confirmed
            bearish_breakout = bearish_breakout & volume_confirmed

        # Apply ATR filter: only trade when volatility is elevated
        high_volatility = df["atr"] > df["atr"].rolling(window=50).mean()
        bullish_breakout = bullish_breakout & high_volatility
        bearish_breakout = bearish_breakout & high_volatility

        # Entry signals
        signals[bullish_breakout] = 1
        signals[bearish_breakout] = -1

        # Calculate trailing stops
        df["trail_stop_long"] = df["close"] - (self.trailing_stop_atr * df["atr"])
        df["trail_stop_short"] = df["close"] + (self.trailing_stop_atr * df["atr"])

        # Apply exit logic with trailing stops
        signals = self._apply_trailing_stops(signals, df)

        logger.debug(
            f"Generated {(signals == 1).sum()} long and "
            f"{(signals == -1).sum()} short breakout signals"
        )

        return signals

    def _apply_trailing_stops(self, signals: pd.Series, df: pd.DataFrame) -> pd.Series:
        """
        Apply trailing stop logic to signals.

        Args:
            signals: Initial breakout signals
            df: DataFrame with price and stop data

        Returns:
            Series with signals including stop exits
        """
        result = signals.copy()
        position = 0
        trail_stop = 0

        for i in range(1, len(result)):
            # Check for new entry
            if signals.iloc[i] != 0:
                position = signals.iloc[i]
                if position == 1:
                    trail_stop = df["trail_stop_long"].iloc[i]
                else:
                    trail_stop = df["trail_stop_short"].iloc[i]
                result.iloc[i] = position
            # Update trailing stop for existing position
            elif position != 0:
                if position == 1:
                    # Update long trailing stop (raise only)
                    trail_stop = max(trail_stop, df["trail_stop_long"].iloc[i])
                    # Check if stopped out
                    if df["close"].iloc[i] < trail_stop:
                        position = 0
                        result.iloc[i] = 0
                    else:
                        result.iloc[i] = 1
                else:  # position == -1
                    # Update short trailing stop (lower only)
                    trail_stop = min(trail_stop, df["trail_stop_short"].iloc[i])
                    # Check if stopped out
                    if df["close"].iloc[i] > trail_stop:
                        position = 0
                        result.iloc[i] = 0
                    else:
                        result.iloc[i] = -1
            else:
                result.iloc[i] = 0

        return result

    def get_positions(self, signals: pd.Series) -> pd.Series:
        """
        Convert signals to positions.

        Args:
            signals: Trading signals

        Returns:
            Position series (1: long, -1: short, 0: flat)
        """
        return signals

    def calculate_position_size(
        self, data: pd.DataFrame, risk_per_trade: float = 0.02
    ) -> pd.Series:
        """
        Calculate position size based on ATR volatility.

        Uses ATR-based position sizing to normalize risk across trades.

        Args:
            data: DataFrame with OHLCV data
            risk_per_trade: Fraction of capital to risk per trade

        Returns:
            Series with position sizes (as fraction of capital)
        """
        df = data.copy()

        # Calculate ATR
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)

        # Calculate position size: risk / (ATR * multiplier)
        # Smaller positions in high volatility, larger in low volatility
        position_size = risk_per_trade / (
            df["atr"] * self.trailing_stop_atr / df["close"]
        )

        # Cap position size at reasonable limits
        position_size = position_size.clip(0.1, 2.0)

        return position_size

    def get_regime_suitability(self, data: pd.DataFrame) -> float:
        """
        Calculate how suitable current market regime is for breakout trading.

        Higher score = more suitable (high volatility, trending)

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Suitability score (0 to 1)
        """
        df = data.copy()

        # Calculate ATR percentile (higher = more volatile = better)
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
        current_atr_percentile = (df["atr"].iloc[-1] > df["atr"]).sum() / len(df["atr"])

        # Calculate price efficiency (trending vs choppy)
        price_change = abs(
            df["close"].iloc[-1] - df["close"].iloc[-self.lookback_period]
        )
        path_length = df["close"].diff().abs().iloc[-self.lookback_period :].sum()
        efficiency = price_change / path_length if path_length > 0 else 0

        # Combine scores
        volatility_score = current_atr_percentile
        trend_score = efficiency

        suitability = (volatility_score + trend_score) / 2

        return float(suitability)

    def get_parameters(self) -> Dict:
        """
        Get strategy parameters.

        Returns:
            Dictionary of strategy parameters
        """
        return {
            "name": self.name,
            "lookback_period": self.lookback_period,
            "atr_period": self.atr_period,
            "atr_multiplier": self.atr_multiplier,
            "volume_factor": self.volume_factor,
            "use_volume_filter": self.use_volume_filter,
            "trailing_stop_atr": self.trailing_stop_atr,
        }
