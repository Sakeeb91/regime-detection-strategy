"""
Strategy selector for regime-adaptive trading.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger


class StrategySelector:
    """
    Select optimal strategy based on detected market regime.

    Maps regimes to strategies and manages strategy transitions.

    Examples:
        >>> selector = StrategySelector(regime_strategy_map={0: 'trend', 1: 'mean_reversion'})
        >>> selected = selector.select_strategy(current_regime=0)
    """

    def __init__(
        self, regime_strategy_map: Dict[int, str], transition_smoothing: int = 3
    ):
        """
        Initialize strategy selector.

        Args:
            regime_strategy_map: Maps regime labels to strategy names
            transition_smoothing: Periods to smooth strategy transitions
        """
        self.regime_strategy_map = regime_strategy_map
        self.transition_smoothing = transition_smoothing

        logger.info(f"StrategySelector initialized with map: {regime_strategy_map}")

    def select_strategy(self, regime: int) -> str:
        """
        Select strategy for given regime.

        Args:
            regime: Current regime label

        Returns:
            Strategy name
        """
        return self.regime_strategy_map.get(regime, "neutral")

    def get_strategy_series(self, regimes: np.ndarray) -> pd.Series:
        """
        Get strategy assignments for regime sequence.

        Args:
            regimes: Array of regime labels

        Returns:
            Series of strategy names
        """
        strategies = [self.select_strategy(r) for r in regimes]
        return pd.Series(strategies)
