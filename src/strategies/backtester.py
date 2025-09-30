"""
Backtesting engine for strategy evaluation.

Provides comprehensive backtesting capabilities with transaction costs,
slippage modeling, and detailed performance analytics.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy


class Backtester:
    """
    Backtesting engine for trading strategies.

    Simulates historical trading with realistic constraints including
    transaction costs, slippage, and position limits.

    Attributes:
        initial_capital (float): Starting capital for backtest
        commission (float): Commission rate per trade (as fraction)
        slippage (float): Slippage as fraction of price
        position_size (float): Position size as fraction of capital
        leverage (float): Maximum leverage allowed

    Examples:
        >>> backtester = Backtester(initial_capital=100000)
        >>> results = backtester.run(strategy, data)
        >>> print(results['total_return'])
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        position_size: float = 1.0,
        leverage: float = 1.0,
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            commission: Commission per trade (0.001 = 0.1%)
            slippage: Slippage per trade (0.0005 = 0.05%)
            position_size: Position size as fraction of capital
            leverage: Maximum leverage (1.0 = no leverage)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        self.leverage = leverage

        logger.info(
            f"Backtester initialized: capital=${initial_capital:,.0f}, "
            f"commission={commission:.3%}, slippage={slippage:.3%}"
        )

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Run backtest for a strategy.

        Args:
            strategy: Trading strategy instance
            data: DataFrame with OHLCV data
            regime_labels: Optional regime labels for analysis

        Returns:
            Dictionary with backtest results including:
                - equity_curve: Series with portfolio value over time
                - returns: Series with strategy returns
                - positions: Series with positions held
                - trades: DataFrame with trade details
                - metrics: Dictionary with performance metrics
        """
        logger.info(f"Running backtest for {strategy.name}")

        # Generate signals
        signals = strategy.generate_signals(data)
        positions = strategy.get_positions(signals)

        # Calculate returns
        market_returns = data["close"].pct_change()

        # Initialize portfolio tracking
        portfolio = self._simulate_portfolio(
            data=data, positions=positions, market_returns=market_returns
        )

        # Calculate metrics
        metrics = self._calculate_metrics(
            portfolio["equity_curve"], portfolio["returns"], positions, market_returns
        )

        # Analyze trades
        trades = self._analyze_trades(
            positions=positions, prices=data["close"], dates=data.index
        )

        # Add regime-specific analysis if available
        if regime_labels is not None:
            regime_analysis = self._analyze_by_regime(
                returns=portfolio["returns"],
                positions=positions,
                regime_labels=regime_labels,
            )
            metrics["regime_analysis"] = regime_analysis

        results = {
            "strategy_name": strategy.name,
            "equity_curve": portfolio["equity_curve"],
            "returns": portfolio["returns"],
            "positions": positions,
            "trades": trades,
            "metrics": metrics,
            "signals": signals,
        }

        logger.info(
            f"Backtest complete: Total Return={metrics['total_return']:.2%}, "
            f"Sharpe={metrics['sharpe_ratio']:.2f}, MaxDD={metrics['max_drawdown']:.2%}"
        )

        return results

    def _simulate_portfolio(
        self, data: pd.DataFrame, positions: pd.Series, market_returns: pd.Series
    ) -> Dict:
        """
        Simulate portfolio evolution with transaction costs.

        Args:
            data: OHLCV data
            positions: Position series
            market_returns: Market returns

        Returns:
            Dictionary with equity_curve and returns
        """
        # Initialize
        equity = pd.Series(self.initial_capital, index=data.index)
        cash = self.initial_capital
        shares = 0

        portfolio_returns = []

        for i in range(1, len(data)):
            prev_position = positions.iloc[i - 1]
            curr_position = positions.iloc[i]

            price = data["close"].iloc[i]
            prev_price = data["close"].iloc[i - 1]

            # Calculate unrealized P&L from existing position
            if shares != 0:
                price_change = price - prev_price
                unrealized_pnl = shares * price_change
                cash += unrealized_pnl

            # Check for position change (trade execution)
            if curr_position != prev_position:
                # Close existing position
                if shares != 0:
                    trade_value = shares * price * (1 - self.slippage * np.sign(shares))
                    commission_cost = abs(shares * price) * self.commission
                    cash += trade_value - commission_cost
                    shares = 0

                # Open new position
                if curr_position != 0:
                    position_value = cash * self.position_size * self.leverage
                    shares = (curr_position * position_value) / (
                        price * (1 + self.slippage * curr_position)
                    )
                    commission_cost = abs(shares * price) * self.commission
                    cash -= shares * price + commission_cost

            # Update equity
            equity.iloc[i] = cash + (shares * price if shares != 0 else 0)

            # Calculate return
            portfolio_returns.append(equity.iloc[i] / equity.iloc[i - 1] - 1)

        # Pad first return with 0
        portfolio_returns = [0] + portfolio_returns

        return {
            "equity_curve": equity,
            "returns": pd.Series(portfolio_returns, index=data.index),
        }

    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        returns: pd.Series,
        positions: pd.Series,
        market_returns: pd.Series,
    ) -> Dict:
        """
        Calculate performance metrics.

        Args:
            equity_curve: Portfolio equity over time
            returns: Strategy returns
            positions: Position series
            market_returns: Market returns for comparison

        Returns:
            Dictionary of performance metrics
        """
        # Basic returns
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # Annualized metrics (assuming daily data)
        trading_days = 252
        n_periods = len(returns)
        years = n_periods / trading_days

        cagr = (
            (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
            if years > 0
            else 0
        )

        # Volatility
        annual_vol = returns.std() * np.sqrt(trading_days)

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (
            (returns.mean() * trading_days) / annual_vol if annual_vol > 0 else 0
        )

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(trading_days)
        sortino_ratio = (
            (returns.mean() * trading_days) / downside_std if downside_std > 0 else 0
        )

        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate and profit factor
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]

        win_rate = (
            len(winning_trades) / len(returns[returns != 0])
            if len(returns[returns != 0]) > 0
            else 0
        )

        profit_factor = (
            winning_trades.sum() / abs(losing_trades.sum())
            if len(losing_trades) > 0 and losing_trades.sum() != 0
            else 0
        )

        # Compare to buy & hold
        buy_hold_return = (1 + market_returns).prod() - 1

        # Position statistics
        n_trades = (positions.diff() != 0).sum()
        long_positions = (positions == 1).sum()
        short_positions = (positions == -1).sum()

        return {
            "total_return": total_return,
            "cagr": cagr,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "n_trades": n_trades,
            "long_positions": long_positions,
            "short_positions": short_positions,
            "buy_hold_return": buy_hold_return,
        }

    def _analyze_trades(
        self, positions: pd.Series, prices: pd.Series, dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Analyze individual trades.

        Args:
            positions: Position series
            prices: Price series
            dates: Date index

        Returns:
            DataFrame with trade details
        """
        trades = []

        entry_date = None
        entry_price = None
        entry_position = None

        for i in range(1, len(positions)):
            prev_pos = positions.iloc[i - 1]
            curr_pos = positions.iloc[i]

            # Position opened
            if prev_pos == 0 and curr_pos != 0:
                entry_date = dates[i]
                entry_price = prices.iloc[i]
                entry_position = curr_pos

            # Position closed
            elif prev_pos != 0 and curr_pos == 0:
                exit_date = dates[i]
                exit_price = prices.iloc[i]

                # Calculate trade metrics
                if entry_position == 1:  # Long trade
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:  # Short trade
                    pnl_pct = (entry_price - exit_price) / entry_price

                holding_period = (exit_date - entry_date).days

                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "direction": "Long" if entry_position == 1 else "Short",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return": pnl_pct,
                        "holding_period": holding_period,
                    }
                )

                entry_date = None
                entry_price = None
                entry_position = None

        return pd.DataFrame(trades)

    def _analyze_by_regime(
        self, returns: pd.Series, positions: pd.Series, regime_labels: pd.Series
    ) -> Dict:
        """
        Analyze performance by market regime.

        Args:
            returns: Strategy returns
            positions: Position series
            regime_labels: Regime classification

        Returns:
            Dictionary with regime-specific metrics
        """
        regime_analysis = {}

        for regime in regime_labels.unique():
            mask = regime_labels == regime
            regime_returns = returns[mask]
            regime_positions = positions[mask]

            if len(regime_returns) > 0:
                regime_analysis[f"regime_{regime}"] = {
                    "n_periods": len(regime_returns),
                    "total_return": (1 + regime_returns).prod() - 1,
                    "sharpe": (
                        regime_returns.mean() / regime_returns.std() * np.sqrt(252)
                        if regime_returns.std() > 0
                        else 0
                    ),
                    "n_trades": (regime_positions.diff() != 0).sum(),
                    "win_rate": (
                        len(regime_returns[regime_returns > 0])
                        / len(regime_returns[regime_returns != 0])
                        if len(regime_returns[regime_returns != 0]) > 0
                        else 0
                    ),
                }

        return regime_analysis

    def compare_strategies(
        self,
        strategies: List[BaseStrategy],
        data: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.

        Args:
            strategies: List of strategy instances
            data: OHLCV data
            regime_labels: Optional regime labels

        Returns:
            DataFrame comparing strategy performance
        """
        results = []

        for strategy in strategies:
            backtest_results = self.run(strategy, data, regime_labels)
            metrics = backtest_results["metrics"]

            results.append(
                {
                    "Strategy": strategy.name,
                    "Total Return": metrics["total_return"],
                    "CAGR": metrics["cagr"],
                    "Sharpe": metrics["sharpe_ratio"],
                    "Sortino": metrics["sortino_ratio"],
                    "Max DD": metrics["max_drawdown"],
                    "Calmar": metrics["calmar_ratio"],
                    "Win Rate": metrics["win_rate"],
                    "Trades": metrics["n_trades"],
                }
            )

        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.set_index("Strategy")

        return comparison_df
