"""
Visualization utilities for regime detection and strategy performance.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_regimes(
    prices: pd.Series,
    regimes: np.ndarray,
    title: str = "Market Regimes",
    figsize: tuple = (14, 7),
    save_path: Optional[str] = None,
):
    """
    Plot price series with regime overlay.

    Args:
        prices: Price series
        regimes: Regime labels
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot prices
    ax.plot(prices.index, prices.values, color="black", linewidth=1, label="Price")

    # Overlay regimes with colors
    n_regimes = len(np.unique(regimes))
    colors = plt.cm.Set2(np.linspace(0, 1, n_regimes))

    for regime in range(n_regimes):
        mask = regimes == regime
        ax.fill_between(
            prices.index,
            prices.min(),
            prices.max(),
            where=mask,
            alpha=0.3,
            color=colors[regime],
            label=f"Regime {regime}",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_equity_curve(
    returns_dict: dict,
    title: str = "Equity Curves",
    figsize: tuple = (14, 7),
    save_path: Optional[str] = None,
):
    """
    Plot equity curves for multiple strategies.

    Args:
        returns_dict: Dictionary mapping strategy names to return series
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, returns in returns_dict.items():
        equity = (1 + returns).cumprod()
        ax.plot(equity.index, equity.values, label=name, linewidth=2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
