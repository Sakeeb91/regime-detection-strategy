"""
Performance metrics for strategy evaluation.
"""

import numpy as np
import pandas as pd


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252
    if excess_returns.std() == 0:
        return 0.0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Args:
        returns: Return series

    Returns:
        Maximum drawdown (negative value)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio.

    Args:
        returns: Return series
        risk_free_rate: Risk-free rate

    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    return np.sqrt(252) * excess_returns.mean() / downside_returns.std()


def calculate_calmar_ratio(returns: pd.Series) -> float:
    """
    Calculate Calmar ratio.

    Args:
        returns: Return series

    Returns:
        Calmar ratio
    """
    annual_return = returns.mean() * 252
    max_dd = abs(calculate_max_drawdown(returns))

    if max_dd == 0:
        return 0.0

    return annual_return / max_dd