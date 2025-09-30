"""
Page 4: Live Dashboard - Real-time metrics and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

st.set_page_config(page_title="Live Dashboard", page_icon="🎯", layout="wide")

st.title("🎯 Live Dashboard")
st.markdown("Real-time insights and actionable recommendations")
st.markdown("---")

# Check prerequisites
if 'data' not in st.session_state:
    st.warning("⚠️ Please complete the full pipeline first: Data Explorer → Regime Detection → Strategy Analysis")
    st.stop()

data = st.session_state['data']
ticker = st.session_state.get('ticker', 'Unknown')

# Check if regime detection was run
has_regimes = 'regimes' in st.session_state
has_strategies = 'strategy_results' in st.session_state

# Auto-refresh toggle
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"### {ticker} Market Intelligence")
with col2:
    auto_refresh = st.checkbox("Auto-refresh", value=False, help="Refresh data every 60s")

if auto_refresh:
    import time
    time.sleep(60)
    st.rerun()

# Current market status
st.markdown("## 📊 Current Market Status")

# Get latest data point
latest_price = data['close'].iloc[-1]
prev_price = data['close'].iloc[-2]
price_change = ((latest_price - prev_price) / prev_price) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current Price",
        f"${latest_price:.2f}",
        f"{price_change:+.2f}%"
    )

with col2:
    # Recent volatility
    recent_returns = data['close'].pct_change().tail(20)
    volatility = recent_returns.std() * np.sqrt(252) * 100
    st.metric("20-Day Volatility", f"{volatility:.1f}%")

with col3:
    # Volume trend
    recent_volume = data['volume'].tail(20).mean()
    prev_volume = data['volume'].iloc[-40:-20].mean()
    volume_change = ((recent_volume - prev_volume) / prev_volume) * 100
    st.metric("Volume Trend", f"{recent_volume/1e6:.1f}M", f"{volume_change:+.1f}%")

with col4:
    # Price momentum
    ma20 = data['close'].rolling(20).mean().iloc[-1]
    ma50 = data['close'].rolling(50).mean().iloc[-1]
    momentum = "Bullish" if ma20 > ma50 else "Bearish"
    st.metric("Momentum (20/50)", momentum)

st.markdown("---")

# Regime insights (if available)
if has_regimes:
    regimes = st.session_state['regimes']
    regime_labels = st.session_state.get('regime_labels', {})
    probabilities = st.session_state.get('probabilities')

    st.markdown("## 🔍 Regime Intelligence")

    col1, col2, col3 = st.columns(3)

    with col1:
        current_regime = regimes.iloc[-1]
        regime_name = regime_labels.get(current_regime, f"Regime {current_regime}")
        st.metric("Current Regime", regime_name)

    with col2:
        if probabilities is not None:
            confidence = probabilities.iloc[-1, current_regime] * 100
            st.metric("Confidence", f"{confidence:.1f}%")

    with col3:
        # Days in current regime
        regime_changes = (regimes != regimes.shift()).cumsum()
        days_in_regime = (regime_changes == regime_changes.iloc[-1]).sum()
        st.metric("Days in Regime", days_in_regime)

    # Regime probability gauge
    if probabilities is not None:
        st.markdown("### Regime Probability Distribution")

        current_probs = probabilities.iloc[-1]

        fig = go.Figure()

        for i, (regime_id, prob) in enumerate(current_probs.items()):
            fig.add_trace(go.Bar(
                x=[regime_labels.get(regime_id, f"Regime {regime_id}")],
                y=[prob * 100],
                name=regime_labels.get(regime_id, f"Regime {regime_id}"),
                marker_color='lightblue' if regime_id == current_regime else 'rgba(100,100,100,0.3)',
                text=f"{prob*100:.1f}%",
                textposition='outside'
            ))

        fig.update_layout(
            yaxis_title="Probability (%)",
            height=300,
            template="plotly_dark",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Regime transition warning
        st.markdown("### Regime Transition Alert")

        # Check if regime is likely to change
        second_highest_regime = current_probs.nlargest(2).index[-1]
        second_highest_prob = current_probs.iloc[second_highest_regime]

        if second_highest_prob > 0.3:
            st.warning(f"⚠️ Potential regime transition detected! "
                      f"{regime_labels.get(second_highest_regime, f'Regime {second_highest_regime}')} "
                      f"probability at {second_highest_prob*100:.1f}%")
        else:
            st.success("✅ Regime is stable")

    st.markdown("---")

# Strategy recommendations (if available)
if has_strategies:
    st.markdown("## 💼 Strategy Recommendations")

    metrics = st.session_state['strategy_metrics']

    # Find best performing strategies
    sorted_strategies = sorted(
        metrics.items(),
        key=lambda x: x[1]['Sharpe Ratio'],
        reverse=True
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Recommended Strategies (by Sharpe Ratio)")

        for idx, (name, metric_dict) in enumerate(sorted_strategies[:3], 1):
            with st.container():
                cols = st.columns([1, 2, 2, 2, 2])
                with cols[0]:
                    st.markdown(f"**#{idx}**")
                with cols[1]:
                    st.markdown(f"**{name}**")
                with cols[2]:
                    st.markdown(f"Sharpe: {metric_dict['Sharpe Ratio']:.2f}")
                with cols[3]:
                    st.markdown(f"Return: {metric_dict['Annualized Return']*100:.1f}%")
                with cols[4]:
                    st.markdown(f"Drawdown: {metric_dict['Max Drawdown']*100:.1f}%")

    with col2:
        # Best strategy recommendation
        best_strategy = sorted_strategies[0][0]
        st.success(f"**Current Top Strategy:**\n\n{best_strategy}")

    # Performance comparison
    st.markdown("### Strategy Performance Comparison")

    strategy_names = [s[0] for s in sorted_strategies]
    sharpe_ratios = [s[1]['Sharpe Ratio'] for s in sorted_strategies]
    returns = [s[1]['Annualized Return'] * 100 for s in sorted_strategies]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Sharpe Ratio", "Annualized Return (%)"],
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    fig.add_trace(
        go.Bar(x=strategy_names, y=sharpe_ratios, marker_color='lightblue', showlegend=False),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=strategy_names, y=returns, marker_color='lightgreen', showlegend=False),
        row=1, col=2
    )

    fig.update_layout(height=400, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

# Market signals
st.markdown("## 📡 Technical Signals")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Trend Signals")
    # Moving average signals
    ma20 = data['close'].rolling(20).mean().iloc[-1]
    ma50 = data['close'].rolling(50).mean().iloc[-1]
    ma200 = data['close'].rolling(200).mean().iloc[-1]

    signals = []
    signals.append(("MA20 vs MA50", "🟢 Bullish" if ma20 > ma50 else "🔴 Bearish"))
    signals.append(("MA50 vs MA200", "🟢 Bullish" if ma50 > ma200 else "🔴 Bearish"))
    signals.append(("Price vs MA200", "🟢 Above" if latest_price > ma200 else "🔴 Below"))

    for signal, status in signals:
        st.markdown(f"**{signal}:** {status}")

with col2:
    st.markdown("### Momentum Signals")
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    if current_rsi > 70:
        rsi_signal = "🔴 Overbought"
    elif current_rsi < 30:
        rsi_signal = "🟢 Oversold"
    else:
        rsi_signal = "🟡 Neutral"

    st.markdown(f"**RSI (14):** {current_rsi:.1f} - {rsi_signal}")

    # MACD
    ema12 = data['close'].ewm(span=12).mean()
    ema26 = data['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9).mean()

    macd_signal = "🟢 Bullish" if macd.iloc[-1] > signal_line.iloc[-1] else "🔴 Bearish"
    st.markdown(f"**MACD:** {macd_signal}")

with col3:
    st.markdown("### Volatility Signals")

    # Bollinger Bands
    bb_ma = data['close'].rolling(20).mean()
    bb_std = data['close'].rolling(20).std()
    upper_band = bb_ma + (bb_std * 2)
    lower_band = bb_ma - (bb_std * 2)

    if latest_price > upper_band.iloc[-1]:
        bb_signal = "🔴 Above Upper"
    elif latest_price < lower_band.iloc[-1]:
        bb_signal = "🟢 Below Lower"
    else:
        bb_signal = "🟡 Within Bands"

    st.markdown(f"**Bollinger Bands:** {bb_signal}")

    # ATR
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(14).mean().iloc[-1]

    st.markdown(f"**ATR (14):** ${atr:.2f}")

st.markdown("---")

# Action items
st.markdown("## ✅ Action Items")

action_items = []

# Generate action items based on signals
if has_regimes:
    regime_stats = st.session_state.get('regime_stats', {})
    current_regime_stats = regime_stats.get(current_regime, {})

    if current_regime_stats.get('mean_return', 0) > 0:
        action_items.append("✅ Current regime is historically profitable - consider maintaining/increasing exposure")
    else:
        action_items.append("⚠️ Current regime shows negative historical returns - consider reducing exposure")

if current_rsi > 70:
    action_items.append("⚠️ RSI indicates overbought conditions - watch for potential reversal")
elif current_rsi < 30:
    action_items.append("✅ RSI indicates oversold conditions - potential buying opportunity")

if volatility > 30:
    action_items.append("⚠️ High volatility detected - consider position sizing and stop losses")

if has_strategies:
    best_strategy = sorted_strategies[0][0]
    action_items.append(f"💡 Recommended strategy: {best_strategy}")

if len(action_items) == 0:
    action_items.append("✅ No immediate action required - market conditions are neutral")

for item in action_items:
    st.markdown(f"- {item}")

st.markdown("---")

# Recent activity log
st.markdown("## 📋 Recent Activity")

activity_log = []

if has_regimes:
    # Find recent regime changes
    regime_changes = regimes[regimes != regimes.shift()].tail(5)
    for date, regime in regime_changes.items():
        regime_name = regime_labels.get(regime, f"Regime {regime}")
        activity_log.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Event': 'Regime Change',
            'Details': f"Transitioned to {regime_name}"
        })

# Sort by date
activity_log = sorted(activity_log, key=lambda x: x['Date'], reverse=True)

if activity_log:
    df_activity = pd.DataFrame(activity_log)
    st.dataframe(df_activity, hide_index=True, use_container_width=True)
else:
    st.info("No recent regime changes detected")

# Footer
st.markdown("---")
st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
st.markdown("*Data source: Yahoo Finance. For informational purposes only. Not financial advice.*")