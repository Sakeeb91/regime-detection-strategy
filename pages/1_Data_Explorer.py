"""
Page 1: Data Explorer - Load and visualize market data
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor
from data.feature_engineer import FeatureEngineer

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

st.title("📊 Data Explorer")
st.markdown("Load and analyze market data from Yahoo Finance")
st.markdown("---")

# Sidebar inputs
st.sidebar.header("Data Configuration")

# Ticker input with popular suggestions
ticker = st.sidebar.text_input(
    "Ticker Symbol",
    value="SPY",
    help="Enter ticker symbol (e.g., SPY, AAPL, TSLA, BTC-USD)"
).upper()

# Date range with smart defaults
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=365*2),
        max_value=datetime.now()
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now()
    )

# Load data button
load_button = st.sidebar.button("🔄 Load Data", type="primary", use_container_width=True)

# Popular tickers as quick select
st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Select:**")
quick_tickers = ["SPY", "QQQ", "AAPL", "TSLA", "BTC-USD", "GLD", "TLT"]
selected_quick = st.sidebar.radio("Popular Tickers", quick_tickers, index=None, label_visibility="collapsed")

if selected_quick:
    ticker = selected_quick
    st.rerun()

# Main content
if load_button or 'data' in st.session_state:

    if load_button:
        with st.spinner(f"Loading {ticker} data from Yahoo Finance..."):
            try:
                # Load data
                loader = DataLoader(use_cache=True)
                data = loader.load_data(
                    ticker,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )

                # Preprocess
                preprocessor = DataPreprocessor()
                clean_data = preprocessor.clean_data(data)

                # Store in session state
                st.session_state['data'] = clean_data
                st.session_state['ticker'] = ticker
                st.session_state['start_date'] = start_date
                st.session_state['end_date'] = end_date

                st.success(f"✅ Successfully loaded {len(clean_data)} days of data for {ticker}")

            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                st.stop()

    # Retrieve from session state
    data = st.session_state['data']
    ticker = st.session_state['ticker']

    # Display data info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", len(data))
    with col2:
        st.metric("Start Date", data.index[0].strftime("%Y-%m-%d"))
    with col3:
        st.metric("End Date", data.index[-1].strftime("%Y-%m-%d"))
    with col4:
        returns = data['close'].pct_change()
        st.metric("Avg Daily Return", f"{returns.mean()*100:.2f}%")

    st.markdown("---")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Price Chart", "📊 Statistics", "🔧 Features"])

    with tab1:
        # Interactive price chart with Plotly
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name=ticker
        ))

        fig.update_layout(
            title=f"{ticker} Price Chart",
            yaxis_title="Price",
            xaxis_title="Date",
            height=600,
            xaxis_rangeslider_visible=False,
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Volume chart
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=data.index,
            y=data['volume'],
            name="Volume",
            marker_color='rgba(100, 149, 237, 0.5)'
        ))

        fig_vol.update_layout(
            title="Trading Volume",
            yaxis_title="Volume",
            xaxis_title="Date",
            height=300,
            template="plotly_dark"
        )

        st.plotly_chart(fig_vol, use_container_width=True)

    with tab2:
        st.subheader("Statistical Summary")

        # Calculate statistics
        returns = data['close'].pct_change().dropna()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Price Statistics**")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Close': [
                    f"${data['close'].mean():.2f}",
                    f"${data['close'].median():.2f}",
                    f"${data['close'].std():.2f}",
                    f"${data['close'].min():.2f}",
                    f"${data['close'].max():.2f}"
                ]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

        with col2:
            st.markdown("**Returns Statistics**")
            returns_stats = pd.DataFrame({
                'Metric': ['Mean Daily Return', 'Std Dev', 'Sharpe Ratio (Ann.)', 'Skewness', 'Kurtosis'],
                'Value': [
                    f"{returns.mean()*100:.4f}%",
                    f"{returns.std()*100:.4f}%",
                    f"{(returns.mean()/returns.std())*15.87:.2f}",
                    f"{returns.skew():.2f}",
                    f"{returns.kurtosis():.2f}"
                ]
            })
            st.dataframe(returns_stats, hide_index=True, use_container_width=True)

        # Returns distribution
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name="Returns Distribution",
            marker_color='rgba(100, 149, 237, 0.7)'
        ))
        fig_dist.update_layout(
            title="Daily Returns Distribution",
            xaxis_title="Return",
            yaxis_title="Frequency",
            height=400,
            template="plotly_dark"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Raw data preview
        st.subheader("Raw Data Preview")
        st.dataframe(data.tail(10), use_container_width=True)

    with tab3:
        st.subheader("Feature Engineering")

        with st.spinner("Calculating technical indicators..."):
            try:
                engineer = FeatureEngineer()
                features = engineer.create_features(data)

                st.success(f"✅ Generated {len(features.columns)} features")

                # Feature categories
                st.markdown("**Feature Categories:**")
                categories = {
                    'Trend': [col for col in features.columns if any(x in col for x in ['sma', 'ema', 'trend'])],
                    'Volatility': [col for col in features.columns if any(x in col for x in ['volatility', 'atr', 'bbands'])],
                    'Momentum': [col for col in features.columns if any(x in col for x in ['rsi', 'macd', 'momentum'])],
                    'Volume': [col for col in features.columns if 'volume' in col],
                }

                for cat, cols in categories.items():
                    st.write(f"- **{cat}**: {len(cols)} features")

                # Display features
                st.markdown("---")
                selected_features = st.multiselect(
                    "Select features to visualize",
                    options=features.columns.tolist(),
                    default=features.columns.tolist()[:5]
                )

                if selected_features:
                    fig_feat = go.Figure()
                    for feat in selected_features:
                        fig_feat.add_trace(go.Scatter(
                            x=features.index,
                            y=features[feat],
                            name=feat,
                            mode='lines'
                        ))

                    fig_feat.update_layout(
                        title="Feature Values Over Time",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        height=500,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_feat, use_container_width=True)

                # Store features in session state
                st.session_state['features'] = features

            except Exception as e:
                st.error(f"Error generating features: {str(e)}")

else:
    # Instructions when no data loaded
    st.info("👈 Configure data settings in the sidebar and click **Load Data** to begin")

    st.markdown("""
    ### Supported Tickers
    - **Stocks**: SPY, QQQ, AAPL, TSLA, MSFT, GOOGL, AMZN
    - **ETFs**: GLD, TLT, VXX, EEM
    - **Crypto**: BTC-USD, ETH-USD
    - **Indices**: ^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (NASDAQ)

    ### Data Features
    - OHLCV (Open, High, Low, Close, Volume)
    - Automatic data cleaning and validation
    - Technical indicator generation
    - Caching for faster reloads
    """)