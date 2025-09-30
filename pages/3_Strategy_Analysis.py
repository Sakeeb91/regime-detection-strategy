"""
Page 3: Strategy Analysis - Compare strategy performance across regimes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy
from src.strategies.strategy_selector import StrategySelector
from src.utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown, calculate_sortino_ratio
from src.utils.ui_components import apply_professional_styling

st.set_page_config(page_title="Strategy Analysis", page_icon="💼", layout="wide")

# Apply professional styling
apply_professional_styling()

st.title("💼 Strategy Performance Analysis")
st.markdown("Compare trading strategies and analyze regime-adaptive performance")
st.markdown("---")

# Check prerequisites
if 'data' not in st.session_state:
    st.warning("⚠️ Please load data first from the Data Explorer page")
    st.stop()

if 'regimes' not in st.session_state:
    st.warning("⚠️ Please run regime detection first from the Regime Detection page")
    st.stop()

data = st.session_state['data']
regimes = st.session_state['regimes']
ticker = st.session_state.get('ticker', 'Unknown')
regime_labels = st.session_state.get('regime_labels', {})

# Sidebar configuration
st.sidebar.header("Strategy Configuration")

# Strategy selection
strategies_to_test = st.sidebar.multiselect(
    "Select Strategies",
    ["Trend Following", "Mean Reversion", "Volatility Breakout", "Regime-Adaptive"],
    default=["Trend Following", "Mean Reversion", "Regime-Adaptive"]
)

# Backtesting parameters
st.sidebar.markdown("**Backtest Parameters**")
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000, step=1000)
transaction_cost = st.sidebar.slider("Transaction Cost (bps)", 0, 50, 5) / 10000

# Run analysis button
run_button = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Advanced settings
with st.sidebar.expander("⚙️ Strategy Parameters"):
    # Trend Following
    st.markdown("**Trend Following**")
    tf_fast = st.slider("Fast MA Period", 5, 50, 20, key="tf_fast")
    tf_slow = st.slider("Slow MA Period", 50, 200, 50, key="tf_slow")

    # Mean Reversion
    st.markdown("**Mean Reversion**")
    mr_window = st.slider("Lookback Window", 10, 50, 20, key="mr_window")
    mr_threshold = st.slider("Z-Score Threshold", 1.0, 3.0, 2.0, 0.1, key="mr_threshold")

    # Volatility Breakout
    st.markdown("**Volatility Breakout**")
    vb_window = st.slider("Volatility Window", 10, 50, 20, key="vb_window")
    vb_multiplier = st.slider("Breakout Multiplier", 1.0, 3.0, 2.0, 0.1, key="vb_multiplier")

# Main content
if run_button or 'strategy_results' in st.session_state:

    if run_button:
        with st.spinner("Running strategy backtests..."):
            try:
                # Align data with regimes
                aligned_data = data.loc[regimes.index].copy()
                returns = aligned_data['close'].pct_change()

                results = {}

                # Test individual strategies
                if "Trend Following" in strategies_to_test:
                    strategy = TrendFollowingStrategy(fast_period=tf_fast, slow_period=tf_slow)
                    signals = strategy.generate_signals(aligned_data)
                    # signals is a Series, use it directly as positions
                    positions = signals if isinstance(signals, pd.Series) else signals.get('position', signals)
                    strategy_returns = positions.shift(1) * returns
                    results['Trend Following'] = strategy_returns

                if "Mean Reversion" in strategies_to_test:
                    strategy = MeanReversionStrategy(bb_period=mr_window, zscore_threshold=mr_threshold)
                    signals = strategy.generate_signals(aligned_data)
                    positions = signals if isinstance(signals, pd.Series) else signals.get('position', signals)
                    strategy_returns = positions.shift(1) * returns
                    results['Mean Reversion'] = strategy_returns

                if "Volatility Breakout" in strategies_to_test:
                    strategy = VolatilityBreakoutStrategy(lookback_period=vb_window, atr_multiplier=vb_multiplier)
                    signals = strategy.generate_signals(aligned_data)
                    positions = signals if isinstance(signals, pd.Series) else signals.get('position', signals)
                    strategy_returns = positions.shift(1) * returns
                    results['Volatility Breakout'] = strategy_returns

                # Regime-Adaptive strategy
                if "Regime-Adaptive" in strategies_to_test:
                    # Map regimes to strategies based on characteristics
                    regime_stats = st.session_state.get('regime_stats', {})

                    # Simple heuristic: positive return -> trend, negative -> mean reversion
                    regime_strategy_map = {}
                    if isinstance(regime_stats, pd.DataFrame):
                        # Handle DataFrame structure - use safe column access
                        for regime_id in regime_stats.index:
                            mean_return = regime_stats.loc[regime_id, 'mean_return'] if 'mean_return' in regime_stats.columns else 0
                            if mean_return > 0:
                                regime_strategy_map[regime_id] = TrendFollowingStrategy(fast_period=tf_fast, slow_period=tf_slow)
                            else:
                                regime_strategy_map[regime_id] = MeanReversionStrategy(bb_period=mr_window, zscore_threshold=mr_threshold)
                    else:
                        # Handle dict structure
                        for regime_id, stats in regime_stats.items():
                            mean_return = stats.get('mean_return', 0) if isinstance(stats, dict) else stats['mean_return']
                            if mean_return > 0:
                                regime_strategy_map[regime_id] = TrendFollowingStrategy(fast_period=tf_fast, slow_period=tf_slow)
                            else:
                                regime_strategy_map[regime_id] = MeanReversionStrategy(bb_period=mr_window, zscore_threshold=mr_threshold)

                    # Generate adaptive signals
                    adaptive_returns = pd.Series(0.0, index=returns.index)
                    for regime_id, strategy in regime_strategy_map.items():
                        regime_mask = regimes == regime_id
                        regime_data = aligned_data[regime_mask]
                        if len(regime_data) > 0:
                            signals = strategy.generate_signals(regime_data)
                            positions = signals if isinstance(signals, pd.Series) else signals.get('position', signals)
                            strategy_returns = positions.shift(1) * returns[regime_mask]
                            adaptive_returns[regime_mask] = strategy_returns.fillna(0)

                    results['Regime-Adaptive'] = adaptive_returns

                # Buy & Hold benchmark
                results['Buy & Hold'] = returns

                # Apply transaction costs
                for name, ret in results.items():
                    if name != 'Buy & Hold':
                        # Approximate transaction costs
                        results[name] = ret - transaction_cost

                # Calculate metrics
                metrics = {}
                for name, ret in results.items():
                    ret_clean = ret.dropna()
                    if len(ret_clean) > 0:
                        metrics[name] = {
                            'Total Return': (1 + ret_clean).prod() - 1,
                            'Annualized Return': ret_clean.mean() * 252,
                            'Volatility': ret_clean.std() * np.sqrt(252),
                            'Sharpe Ratio': calculate_sharpe_ratio(ret_clean),
                            'Sortino Ratio': calculate_sortino_ratio(ret_clean),
                            'Max Drawdown': calculate_max_drawdown(ret_clean),
                            'Win Rate': (ret_clean > 0).sum() / len(ret_clean),
                        }

                # Store results
                st.session_state['strategy_results'] = results
                st.session_state['strategy_metrics'] = metrics

                st.success(f"✅ Analyzed {len(strategies_to_test)} strategies over {len(returns)} periods")

            except Exception as e:
                st.error(f"❌ Error in strategy analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()

    # Retrieve results
    results = st.session_state['strategy_results']
    metrics = st.session_state['strategy_metrics']

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Equity Curves", "📊 Performance Metrics", "🎯 By Regime", "📉 Risk Analysis"])

    with tab1:
        st.subheader("Cumulative Returns")

        # Calculate equity curves
        fig = go.Figure()

        for name, ret in results.items():
            equity = (1 + ret.dropna()).cumprod() * initial_capital
            fig.add_trace(go.Scatter(
                x=equity.index,
                y=equity.values,
                name=name,
                mode='lines',
                line=dict(width=2)
            ))

        fig.update_layout(
            title=f"Strategy Comparison - ${initial_capital:,} Initial Capital",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=600,
            template="plotly_dark",
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Current values
        st.markdown("### Final Portfolio Values")
        final_values = []
        for name, ret in results.items():
            final_value = (1 + ret.dropna()).prod() * initial_capital
            final_values.append({
                'Strategy': name,
                'Final Value': f"${final_value:,.2f}",
                'Total Return': f"{(final_value/initial_capital - 1)*100:.2f}%",
                'Profit/Loss': f"${final_value - initial_capital:,.2f}"
            })

        df_final = pd.DataFrame(final_values)
        st.dataframe(df_final, hide_index=True, use_container_width=True)

    with tab2:
        st.subheader("Performance Metrics")

        # Create metrics DataFrame
        metrics_data = []
        for name, metric_dict in metrics.items():
            row = {'Strategy': name}
            row.update({
                'Total Return': f"{metric_dict['Total Return']*100:.2f}%",
                'Ann. Return': f"{metric_dict['Annualized Return']*100:.2f}%",
                'Volatility': f"{metric_dict['Volatility']*100:.2f}%",
                'Sharpe Ratio': f"{metric_dict['Sharpe Ratio']:.2f}",
                'Sortino Ratio': f"{metric_dict['Sortino Ratio']:.2f}",
                'Max Drawdown': f"{metric_dict['Max Drawdown']*100:.2f}%",
                'Win Rate': f"{metric_dict['Win Rate']*100:.1f}%",
            })
            metrics_data.append(row)

        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, hide_index=True, use_container_width=True)

        # Visualize key metrics
        col1, col2 = st.columns(2)

        with col1:
            # Sharpe Ratio comparison
            fig_sharpe = go.Figure(data=[
                go.Bar(
                    x=[m['Strategy'] for m in metrics_data],
                    y=[metrics[m['Strategy']]['Sharpe Ratio'] for m in metrics_data],
                    marker_color='lightblue',
                    text=[f"{metrics[m['Strategy']]['Sharpe Ratio']:.2f}" for m in metrics_data],
                    textposition='outside'
                )
            ])
            fig_sharpe.update_layout(
                title="Sharpe Ratio Comparison",
                yaxis_title="Sharpe Ratio",
                height=400,
                template="plotly_dark",
                showlegend=False
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)

        with col2:
            # Return vs Risk scatter
            fig_scatter = go.Figure()
            for name, metric_dict in metrics.items():
                fig_scatter.add_trace(go.Scatter(
                    x=[metric_dict['Volatility']*100],
                    y=[metric_dict['Annualized Return']*100],
                    mode='markers+text',
                    name=name,
                    marker=dict(size=15),
                    text=name,
                    textposition='top center'
                ))

            fig_scatter.update_layout(
                title="Return vs Risk",
                xaxis_title="Volatility (%)",
                yaxis_title="Annualized Return (%)",
                height=400,
                template="plotly_dark",
                showlegend=False
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        st.subheader("Performance by Regime")

        # Calculate returns by regime
        regime_performance = {}
        n_regimes = len(set(regimes))

        for strategy_name, strategy_returns in results.items():
            regime_performance[strategy_name] = {}
            for regime_id in range(n_regimes):
                regime_mask = regimes == regime_id
                regime_returns = strategy_returns[regime_mask].dropna()
                if len(regime_returns) > 0:
                    regime_performance[strategy_name][regime_id] = {
                        'Return': regime_returns.mean() * 252,
                        'Volatility': regime_returns.std() * np.sqrt(252),
                        'Sharpe': calculate_sharpe_ratio(regime_returns) if len(regime_returns) > 1 else 0
                    }

        # Create heatmap for returns by regime
        strategies = list(regime_performance.keys())
        regime_returns_matrix = []

        for strategy in strategies:
            row = []
            for regime_id in range(n_regimes):
                if regime_id in regime_performance[strategy]:
                    row.append(regime_performance[strategy][regime_id]['Return'] * 100)
                else:
                    row.append(0)
            regime_returns_matrix.append(row)

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=regime_returns_matrix,
            x=[regime_labels.get(i, f"Regime {i}") for i in range(n_regimes)],
            y=strategies,
            colorscale='RdYlGn',
            text=np.round(regime_returns_matrix, 1),
            texttemplate='%{text}%',
            textfont={"size": 12},
            colorbar=dict(title="Ann. Return (%)")
        ))

        fig_heatmap.update_layout(
            title="Annualized Returns by Regime (%)",
            xaxis_title="Regime",
            yaxis_title="Strategy",
            height=400,
            template="plotly_dark"
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Best strategy per regime
        st.markdown("### Optimal Strategy per Regime")
        optimal_strategies = []
        for regime_id in range(n_regimes):
            best_strategy = None
            best_return = -np.inf
            for strategy in strategies:
                if regime_id in regime_performance[strategy]:
                    ret = regime_performance[strategy][regime_id]['Return']
                    if ret > best_return:
                        best_return = ret
                        best_strategy = strategy

            optimal_strategies.append({
                'Regime': regime_labels.get(regime_id, f"Regime {regime_id}"),
                'Best Strategy': best_strategy,
                'Annualized Return': f"{best_return*100:.2f}%"
            })

        df_optimal = pd.DataFrame(optimal_strategies)
        st.dataframe(df_optimal, hide_index=True, use_container_width=True)

    with tab4:
        st.subheader("Risk Analysis")

        # Drawdown analysis
        st.markdown("### Drawdown Comparison")

        fig_dd = make_subplots(
            rows=len(results),
            cols=1,
            subplot_titles=list(results.keys()),
            vertical_spacing=0.05
        )

        for idx, (name, ret) in enumerate(results.items(), 1):
            equity = (1 + ret.dropna()).cumprod()
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max * 100

            fig_dd.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    name=name,
                    fill='tozeroy',
                    line=dict(width=1)
                ),
                row=idx,
                col=1
            )

        fig_dd.update_layout(
            height=300 * len(results),
            showlegend=False,
            template="plotly_dark"
        )
        fig_dd.update_yaxes(title_text="Drawdown (%)")

        st.plotly_chart(fig_dd, use_container_width=True)

        # Rolling metrics
        st.markdown("### Rolling Performance")

        window = st.slider("Rolling Window (days)", 20, 120, 60)

        selected_strategy = st.selectbox("Select Strategy", list(results.keys()))

        if selected_strategy:
            ret = results[selected_strategy].dropna()

            # Calculate rolling metrics
            rolling_sharpe = ret.rolling(window).apply(lambda x: calculate_sharpe_ratio(x) if len(x) > 1 else 0)
            rolling_vol = ret.rolling(window).std() * np.sqrt(252) * 100

            fig_rolling = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=["Rolling Sharpe Ratio", "Rolling Volatility (%)"],
                vertical_spacing=0.1
            )

            fig_rolling.add_trace(
                go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name="Sharpe", line=dict(color='lightblue')),
                row=1, col=1
            )

            fig_rolling.add_trace(
                go.Scatter(x=rolling_vol.index, y=rolling_vol.values, name="Volatility", line=dict(color='orange')),
                row=2, col=1
            )

            fig_rolling.update_layout(height=600, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_rolling, use_container_width=True)

else:
    st.info("👈 Configure strategies and click **Run Analysis**")

    st.markdown("""
    ### Available Strategies

    **Trend Following**
    - Follows market momentum using moving averages
    - Best in: Strong trending regimes (bull/bear)
    - Signal: Buy when fast MA > slow MA

    **Mean Reversion**
    - Exploits price deviations from average
    - Best in: Sideways/ranging regimes
    - Signal: Buy when price < -threshold, sell when > +threshold

    **Volatility Breakout**
    - Trades breakouts beyond volatility bands
    - Best in: High volatility regimes
    - Signal: Buy/sell when price breaks volatility threshold

    **Regime-Adaptive**
    - Automatically selects best strategy per regime
    - Combines multiple strategies based on market conditions
    - Optimal for: All market conditions

    ### Key Metrics Explained
    - **Sharpe Ratio**: Risk-adjusted returns (higher is better, >1 is good)
    - **Sortino Ratio**: Like Sharpe but only penalizes downside volatility
    - **Max Drawdown**: Largest peak-to-trough decline
    - **Win Rate**: Percentage of profitable periods
    """)