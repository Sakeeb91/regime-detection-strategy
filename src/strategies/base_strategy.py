"""
Base strategy class for all trading strategies.
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    All concrete strategies must implement signal generation methods.
    """

    def __init__(self, name: str):
        """
        Initialize base strategy.

        Args:
            name: Strategy name
        """
        self.name = name

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.

        Args:
            data: DataFrame with price/feature data

        Returns:
            Series with signals (1: long, -1: short, 0: neutral)
        """

    def get_positions(self, signals: pd.Series) -> pd.Series:
        """
        Convert signals to positions.

        Args:
            signals: Trading signals

        Returns:
            Position series
        """
        return signals

    def calculate_returns(self, positions: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Calculate strategy returns.

        Args:
            positions: Position series
            returns: Market returns

        Returns:
            Strategy returns
        """
        return positions.shift(1) * returns
