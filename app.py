"""
Streamlit App: Market Regime Detection & Adaptive Strategy Selection
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.ui_components import (
    apply_professional_styling,
    render_hero_section,
    render_feature_card,
    render_metric_card
)

# Page configuration
st.set_page_config(
    page_title="Market Regime Detection",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply professional styling
apply_professional_styling()

# Hero section
render_hero_section(
    "Market Regime Intelligence",
    "AI-powered regime detection and adaptive trading strategies that evolve with market conditions"
)

st.markdown("---")

# Feature cards
st.markdown("### Core Capabilities")

col1, col2, col3, col4 = st.columns(4)

features = [
    {
        "icon": "🔍",
        "title": "Regime Detection",
        "desc": "GMM, HMM & DTW algorithms identify market states with high precision",
        "badge": "ML-Powered"
    },
    {
        "icon": "📊",
        "title": "Smart Visualization",
        "desc": "Interactive charts with regime overlays and transition analysis",
        "badge": "Real-time"
    },
    {
        "icon": "💼",
        "title": "Strategy Engine",
        "desc": "Adaptive strategies that adjust to each market regime automatically",
        "badge": "Backtested"
    },
    {
        "icon": "📈",
        "title": "Performance Analytics",
        "desc": "Comprehensive metrics: Sharpe, Sortino, drawdown, and more",
        "badge": "Pro-Grade"
    }
]

for col, feature in zip([col1, col2, col3, col4], features):
    with col:
        st.markdown(render_feature_card(
            feature['icon'],
            feature['title'],
            feature['desc'],
            feature['badge']
        ), unsafe_allow_html=True)

st.markdown("---")

# Enhanced platform stats
st.markdown("### Platform Stats")

col1, col2, col3 = st.columns(3)

metrics_data = [
    {"label": "Detection Models", "value": "3", "sublabel": "GMM • HMM • DTW", "icon": "🤖"},
    {"label": "Trading Strategies", "value": "4+", "sublabel": "Trend • Mean Reversion • Volatility • Adaptive", "icon": "💡"},
    {"label": "Data Coverage", "value": "Real-time", "sublabel": "Yahoo Finance API", "icon": "⚡"}
]

for col, metric in zip([col1, col2, col3], metrics_data):
    with col:
        st.markdown(render_metric_card(
            metric['icon'],
            metric['value'],
            metric['label'],
            metric['sublabel']
        ), unsafe_allow_html=True)

st.markdown("---")

# Quick start guide
st.markdown("### 🚀 Quick Start Guide")

steps = [
    "**Navigate to Data Explorer** → Load your ticker (SPY, AAPL, BTC-USD, etc.)",
    "**Run Regime Detection** → Choose GMM, HMM, or DTW algorithm",
    "**Analyze Strategies** → Compare performance across different regimes",
    "**Monitor Live Dashboard** → Get real-time insights and alerts"
]

for idx, step in enumerate(steps, 1):
    st.markdown(f"{idx}. {step}")

st.markdown("---")

st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.9rem;">
    Built with <span style="color: #6366F1; font-weight: 600;">Python</span> •
    <span style="color: #6366F1; font-weight: 600;">Streamlit</span> •
    <span style="color: #6366F1; font-weight: 600;">scikit-learn</span> •
    <span style="color: #6366F1; font-weight: 600;">yfinance</span>
</div>
""", unsafe_allow_html=True)