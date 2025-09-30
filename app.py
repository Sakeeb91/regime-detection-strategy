"""
Streamlit App: Market Regime Detection & Adaptive Strategy Selection
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Page configuration
st.set_page_config(
    page_title="Market Regime Detection",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main page
st.title("📈 Market Regime Detection & Adaptive Trading")
st.markdown("---")

st.markdown("""
### Welcome to the Market Regime Detection System

This application uses **machine learning** to:
- 🔍 Detect market regimes (Bull, Bear, Sideways, High Volatility)
- 📊 Visualize regime transitions over time
- 💼 Compare trading strategy performance across different regimes
- 📈 Generate actionable insights for adaptive trading

### Get Started
👈 Use the sidebar to navigate between pages:

1. **Data Explorer** - Load and analyze market data
2. **Regime Detection** - Detect and visualize market regimes
3. **Strategy Analysis** - Compare strategy performance
4. **Live Dashboard** - Real-time metrics and insights

### How It Works
1. Select a ticker symbol (e.g., SPY, AAPL, BTC-USD)
2. Choose date range for analysis
3. Run regime detection using GMM/HMM/DTW
4. View regime-colored price charts
5. Compare strategy performance (Trend Following, Mean Reversion, etc.)
6. Get performance metrics (Sharpe Ratio, Max Drawdown, etc.)

---
**Built with:** Python • Streamlit • scikit-learn • yfinance
""")

# Quick stats in columns
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Supported Models", "3", "GMM, HMM, DTW")
with col2:
    st.metric("Trading Strategies", "3+", "Trend, Mean Reversion, Volatility")
with col3:
    st.metric("Data Sources", "Yahoo Finance", "Real-time")