"""
Page 2: Regime Detection - Detect and visualize market regimes
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

from src.regime_detection.gmm_detector import GMMDetector
from src.regime_detection.hmm_detector import HMMDetector
from src.data.feature_engineer import FeatureEngineer

st.set_page_config(page_title="Regime Detection", page_icon="🔍", layout="wide")

st.title("🔍 Regime Detection")
st.markdown("Identify market regimes using machine learning")
st.markdown("---")

# Check if data is loaded
if 'data' not in st.session_state:
    st.warning("⚠️ Please load data first from the Data Explorer page")
    st.stop()

data = st.session_state['data']
ticker = st.session_state.get('ticker', 'Unknown')

# Sidebar configuration
st.sidebar.header("Detection Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Detection Model",
    ["GMM (Gaussian Mixture)", "HMM (Hidden Markov)"],
    help="GMM: Fast, good for clear regimes. HMM: Better for temporal dependencies"
)

# Number of regimes
n_regimes = st.sidebar.slider(
    "Number of Regimes",
    min_value=2,
    max_value=5,
    value=3,
    help="Typical: 2-3 (Bull/Bear), 4+ for more granular analysis"
)

# Regime labels
st.sidebar.markdown("**Regime Labels** (Optional)")
regime_labels = {}
for i in range(n_regimes):
    regime_labels[i] = st.sidebar.text_input(
        f"Regime {i}",
        value=["Bull", "Bear", "Sideways", "High Volatility", "Low Volatility"][i] if i < 5 else f"Regime {i}",
        key=f"label_{i}"
    )

# Run detection button
run_button = st.sidebar.button("🚀 Detect Regimes", type="primary", use_container_width=True)

# Advanced settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    use_all_features = st.checkbox("Use All Features", value=False, help="Use all technical features vs regime-specific")
    random_state = st.number_input("Random State", value=42, help="For reproducibility")

# Main content
if run_button or 'regimes' in st.session_state:

    if run_button:
        with st.spinner(f"Running {model_type} regime detection..."):
            try:
                # Get features
                if 'features' not in st.session_state:
                    engineer = FeatureEngineer()
                    features = engineer.create_features(data)
                    st.session_state['features'] = features
                else:
                    features = st.session_state['features']

                # Extract regime features
                engineer = FeatureEngineer()
                if use_all_features:
                    regime_features = features.dropna()
                else:
                    regime_features = engineer.extract_regime_features(features)

                # Detect regimes
                if "GMM" in model_type:
                    detector = GMMDetector(n_regimes=n_regimes, random_state=random_state)
                else:
                    detector = HMMDetector(n_regimes=n_regimes, random_state=random_state)

                detector.fit(regime_features)
                regimes = detector.predict(regime_features)
                probabilities = detector.predict_proba(regime_features)

                # Get statistics
                returns = data['close'].pct_change().loc[regime_features.index]
                stats = detector.get_regime_statistics(regime_features, returns=returns)

                # Store in session state
                st.session_state['regimes'] = regimes
                st.session_state['probabilities'] = probabilities
                st.session_state['regime_stats'] = stats
                st.session_state['detector'] = detector
                st.session_state['regime_features'] = regime_features
                st.session_state['regime_labels'] = regime_labels
                st.session_state['model_type'] = model_type

                st.success(f"✅ Successfully detected {n_regimes} regimes across {len(regimes)} periods")

            except Exception as e:
                st.error(f"❌ Error in regime detection: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()

    # Retrieve from session state
    regimes = st.session_state['regimes']
    probabilities = st.session_state['probabilities']
    stats = st.session_state['regime_stats']
    regime_labels = st.session_state.get('regime_labels', {i: f"Regime {i}" for i in range(n_regimes)})

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price + Regimes", "📊 Statistics", "🎯 Probabilities", "📉 Transitions"])

    with tab1:
        st.subheader(f"{ticker} Price with Regime Overlay")

        # Create price chart with regime background colors
        fig = go.Figure()

        # Get price data aligned with regimes
        regime_df = pd.DataFrame({
            'regime': regimes,
            'close': data['close'].loc[regimes.index]
        })

        # Define colors for regimes
        colors = ['rgba(46, 204, 113, 0.3)',   # Green - Bull
                  'rgba(231, 76, 60, 0.3)',     # Red - Bear
                  'rgba(241, 196, 15, 0.3)',    # Yellow - Sideways
                  'rgba(155, 89, 182, 0.3)',    # Purple - High Vol
                  'rgba(52, 152, 219, 0.3)']    # Blue - Low Vol

        # Plot price line
        fig.add_trace(go.Scatter(
            x=regime_df.index,
            y=regime_df['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='white', width=2)
        ))

        # Add regime backgrounds
        for regime_id in range(n_regimes):
            regime_mask = regime_df['regime'] == regime_id
            regime_periods = regime_df[regime_mask]

            if len(regime_periods) > 0:
                # Find continuous regime periods
                regime_start = None
                for i, (idx, row) in enumerate(regime_df.iterrows()):
                    if row['regime'] == regime_id:
                        if regime_start is None:
                            regime_start = idx
                    else:
                        if regime_start is not None:
                            # Add shape for this regime period
                            fig.add_vrect(
                                x0=regime_start,
                                x1=idx,
                                fillcolor=colors[regime_id % len(colors)],
                                layer="below",
                                line_width=0,
                            )
                            regime_start = None

                # Handle last period
                if regime_start is not None:
                    fig.add_vrect(
                        x0=regime_start,
                        x1=regime_df.index[-1],
                        fillcolor=colors[regime_id % len(colors)],
                        layer="below",
                        line_width=0,
                    )

        # Add legend for regimes
        for regime_id in range(n_regimes):
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=colors[regime_id % len(colors)].replace('0.3', '0.8')),
                legendgroup=f'regime_{regime_id}',
                showlegend=True,
                name=regime_labels.get(regime_id, f"Regime {regime_id}")
            ))

        fig.update_layout(
            title=f"{ticker} with Market Regimes",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            template="plotly_dark",
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Current regime info
        current_regime = regimes.iloc[-1]
        current_prob = probabilities.iloc[-1]

        st.markdown("### Current Market State")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Regime", regime_labels.get(current_regime, f"Regime {current_regime}"))
        with col2:
            st.metric("Confidence", f"{current_prob[current_regime]*100:.1f}%")
        with col3:
            days_in_regime = (regimes.iloc[-20:] == current_regime).sum()
            st.metric("Days in Regime (Last 20)", days_in_regime)

    with tab2:
        st.subheader("Regime Statistics")

        # Convert stats to DataFrame
        stats_data = []
        for regime_id, regime_stats in stats.items():
            stats_data.append({
                'Regime': regime_labels.get(regime_id, f"Regime {regime_id}"),
                'Avg Return': f"{regime_stats['mean_return']*100:.3f}%",
                'Volatility': f"{regime_stats['volatility']*100:.3f}%",
                'Sharpe Ratio': f"{regime_stats['sharpe_ratio']:.2f}",
                'Frequency': f"{regime_stats['frequency']*100:.1f}%",
                'Avg Duration': f"{regime_stats['avg_duration']:.1f} days"
            })

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, hide_index=True, use_container_width=True)

        # Visualize regime performance
        col1, col2 = st.columns(2)

        with col1:
            # Returns by regime
            fig_returns = go.Figure()
            for regime_id, regime_stats in stats.items():
                fig_returns.add_trace(go.Bar(
                    x=[regime_labels.get(regime_id, f"Regime {regime_id}")],
                    y=[regime_stats['mean_return']*100],
                    name=regime_labels.get(regime_id, f"Regime {regime_id}"),
                    marker_color=colors[regime_id % len(colors)].replace('0.3', '0.8')
                ))

            fig_returns.update_layout(
                title="Average Returns by Regime",
                yaxis_title="Return (%)",
                showlegend=False,
                height=400,
                template="plotly_dark"
            )
            st.plotly_chart(fig_returns, use_container_width=True)

        with col2:
            # Regime frequency
            fig_freq = go.Figure(data=[go.Pie(
                labels=[regime_labels.get(i, f"Regime {i}") for i in range(n_regimes)],
                values=[stats[i]['frequency'] for i in range(n_regimes)],
                marker=dict(colors=[colors[i % len(colors)].replace('0.3', '0.8') for i in range(n_regimes)])
            )])

            fig_freq.update_layout(
                title="Regime Distribution",
                height=400,
                template="plotly_dark"
            )
            st.plotly_chart(fig_freq, use_container_width=True)

    with tab3:
        st.subheader("Regime Probabilities Over Time")

        # Plot probabilities
        fig_prob = go.Figure()

        for regime_id in range(n_regimes):
            fig_prob.add_trace(go.Scatter(
                x=probabilities.index,
                y=probabilities.iloc[:, regime_id],
                mode='lines',
                name=regime_labels.get(regime_id, f"Regime {regime_id}"),
                fill='tonexty' if regime_id > 0 else 'tozeroy',
                line=dict(width=0.5)
            ))

        fig_prob.update_layout(
            title="Regime Probability Evolution",
            xaxis_title="Date",
            yaxis_title="Probability",
            height=500,
            template="plotly_dark",
            hovermode='x unified'
        )

        st.plotly_chart(fig_prob, use_container_width=True)

        # Probability heatmap
        st.markdown("### Recent Probability Heatmap")
        recent_probs = probabilities.tail(30)

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=recent_probs.T,
            x=recent_probs.index.strftime('%Y-%m-%d'),
            y=[regime_labels.get(i, f"Regime {i}") for i in range(n_regimes)],
            colorscale='Viridis'
        ))

        fig_heatmap.update_layout(
            title="Last 30 Days - Regime Probabilities",
            xaxis_title="Date",
            yaxis_title="Regime",
            height=300,
            template="plotly_dark"
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

    with tab4:
        st.subheader("Regime Transitions")

        # Calculate transition matrix
        transitions = np.zeros((n_regimes, n_regimes))
        for i in range(len(regimes) - 1):
            from_regime = regimes.iloc[i]
            to_regime = regimes.iloc[i + 1]
            transitions[from_regime, to_regime] += 1

        # Normalize
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_probs = transitions / row_sums

        # Plot transition matrix
        fig_trans = go.Figure(data=go.Heatmap(
            z=transition_probs,
            x=[regime_labels.get(i, f"Regime {i}") for i in range(n_regimes)],
            y=[regime_labels.get(i, f"Regime {i}") for i in range(n_regimes)],
            colorscale='RdYlGn',
            text=np.round(transition_probs, 2),
            texttemplate='%{text}',
            textfont={"size": 14},
            colorbar=dict(title="Probability")
        ))

        fig_trans.update_layout(
            title="Regime Transition Probability Matrix",
            xaxis_title="To Regime",
            yaxis_title="From Regime",
            height=500,
            template="plotly_dark"
        )

        st.plotly_chart(fig_trans, use_container_width=True)

        # Transition statistics
        st.markdown("### Transition Insights")
        col1, col2 = st.columns(2)

        with col1:
            # Most stable regime
            stability = np.diag(transition_probs)
            most_stable = np.argmax(stability)
            st.metric(
                "Most Stable Regime",
                regime_labels.get(most_stable, f"Regime {most_stable}"),
                f"{stability[most_stable]*100:.1f}% persistence"
            )

        with col2:
            # Most volatile regime
            most_volatile = np.argmin(stability)
            st.metric(
                "Most Volatile Regime",
                regime_labels.get(most_volatile, f"Regime {most_volatile}"),
                f"{stability[most_volatile]*100:.1f}% persistence"
            )

else:
    st.info("👈 Configure detection settings and click **Detect Regimes**")

    st.markdown("""
    ### About Regime Detection Models

    **GMM (Gaussian Mixture Model)**
    - Fast and efficient
    - Assumes data comes from mixture of Gaussian distributions
    - Best for: Clear, distinct market regimes
    - Use when: You want quick results with good accuracy

    **HMM (Hidden Markov Model)**
    - Captures temporal dependencies
    - Models regime transitions probabilistically
    - Best for: Sequential patterns and regime persistence
    - Use when: Market shows clear sequential behavior

    ### Typical Regime Configurations
    - **2 Regimes**: Bull vs Bear
    - **3 Regimes**: Bull, Bear, Sideways
    - **4 Regimes**: Add High Volatility state
    - **5+ Regimes**: Very granular analysis (may overfit)
    """)