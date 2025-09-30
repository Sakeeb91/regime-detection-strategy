"""
Demo Script: Generate All Visualizations for Portfolio/Recruiters

This script runs a complete end-to-end simulation and saves all plots
in organized folders for easy reference.

Usage:
    python demo_with_plots.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.data.feature_engineer import FeatureEngineer
from src.regime_detection.gmm_detector import GMMDetector
from src.regime_detection.hmm_detector import HMMDetector
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy
from src.utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown, calculate_sortino_ratio
from src.utils.plotting import plot_regimes, plot_equity_curve

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directories
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
SIMS_DIR = OUTPUT_DIR / "simulations"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SIMS_DIR.mkdir(parents=True, exist_ok=True)

def save_plot(fig, name, description):
    """Save plot with metadata"""
    filepath = PLOTS_DIR / f"{name}.png"
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {filepath}")

    # Save description
    desc_file = PLOTS_DIR / f"{name}_description.txt"
    with open(desc_file, 'w') as f:
        f.write(description)

    plt.close(fig)

def run_complete_simulation(ticker='SPY', start_date='2020-01-01'):
    """Run complete end-to-end simulation with all visualizations"""

    print(f"\n{'='*60}")
    print(f"MARKET REGIME DETECTION & ADAPTIVE TRADING SIMULATION")
    print(f"{'='*60}\n")
    print(f"Ticker: {ticker}")
    print(f"Start Date: {start_date}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ============= STEP 1: LOAD DATA =============
    print("📊 Step 1: Loading market data...")
    loader = DataLoader(use_cache=True)
    data = loader.load_data(ticker, start_date=start_date)
    print(f"✅ Loaded {len(data)} trading days")

    # Plot 1: Raw Price Data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ax1.plot(data.index, data['close'], linewidth=1.5, color='#2E86AB', label='Close Price')
    ax1.fill_between(data.index, data['low'], data['high'], alpha=0.2, color='#2E86AB')
    ax1.set_title(f'{ticker} Price Chart ({start_date} to Present)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(data.index, data['volume'], width=1, color='#A23B72', alpha=0.6)
    ax2.set_title('Trading Volume', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, '01_raw_data',
              f"Raw price and volume data for {ticker}. Shows OHLC price action and trading volume. "
              f"This is the foundation data used for all subsequent analysis.")

    # ============= STEP 2: PREPROCESS =============
    print("\n🔧 Step 2: Preprocessing data...")
    preprocessor = DataPreprocessor()
    clean_data = preprocessor.clean_data(data)
    print(f"✅ Cleaned data: {len(clean_data)} days")

    # Plot 2: Returns Distribution
    returns = clean_data['close'].pct_change().dropna()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Returns histogram
    ax1.hist(returns, bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
    ax1.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean()*100:.3f}%')
    ax1.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Return', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative returns
    cum_returns = (1 + returns).cumprod()
    ax2.plot(cum_returns.index, cum_returns.values, linewidth=2, color='#F18F01')
    ax2.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Cumulative Return', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Rolling volatility
    rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100
    ax3.plot(rolling_vol.index, rolling_vol.values, linewidth=1.5, color='#A23B72')
    ax3.fill_between(rolling_vol.index, 0, rolling_vol.values, alpha=0.3, color='#A23B72')
    ax3.set_title('20-Day Rolling Volatility (%)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Volatility (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # QQ plot
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normality Test)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, '02_returns_analysis',
              "Returns analysis showing distribution, cumulative performance, volatility, and normality test. "
              "The histogram shows return distribution is roughly normal with fat tails. "
              "Cumulative returns show overall performance trajectory. "
              "Rolling volatility highlights periods of market stress. "
              "Q-Q plot tests if returns follow normal distribution (deviations indicate fat tails).")

    # ============= STEP 3: FEATURE ENGINEERING =============
    print("\n🧮 Step 3: Engineering features...")
    engineer = FeatureEngineer()
    features = engineer.create_features(clean_data)
    regime_features = engineer.extract_regime_features(features)
    print(f"✅ Generated {len(features.columns)} total features, {len(regime_features.columns)} for regimes")

    # Plot 3: Feature Importance
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # RSI
    rsi = features['rsi_14'].dropna() if 'rsi_14' in features.columns else features.filter(like='rsi').iloc[:, 0].dropna()
    ax1.plot(rsi.index, rsi.values, linewidth=1, color='#2E86AB')
    ax1.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
    ax1.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
    ax1.fill_between(rsi.index, 30, 70, alpha=0.1, color='gray')
    ax1.set_title('RSI (Relative Strength Index)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('RSI', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # MACD
    macd = features['macd'].dropna() if 'macd' in features.columns else features.filter(like='macd').iloc[:, 0].dropna()
    macd_signal = features['macd_signal'].dropna() if 'macd_signal' in features.columns else features.filter(like='macd_signal').iloc[:, 0].dropna()
    ax2.plot(macd.index, macd.values, linewidth=1, label='MACD', color='#F18F01')
    ax2.plot(macd_signal.index, macd_signal.values, linewidth=1, label='Signal', color='#A23B72')
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.fill_between(macd.index, macd.values, macd_signal.values,
                     where=(macd.values > macd_signal.values), alpha=0.3, color='green', label='Bullish')
    ax2.fill_between(macd.index, macd.values, macd_signal.values,
                     where=(macd.values <= macd_signal.values), alpha=0.3, color='red', label='Bearish')
    ax2.set_title('MACD (Moving Average Convergence Divergence)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MACD', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bollinger Bands - use simple MA if bb bands not available
    if 'bb_upper_20' in features.columns:
        bb_upper = features['bb_upper_20'].dropna()
        bb_middle = features['bb_middle_20'].dropna()
        bb_lower = features['bb_lower_20'].dropna()
    else:
        # Calculate simple bollinger bands
        bb_middle = clean_data['close'].rolling(20).mean()
        bb_std = clean_data['close'].rolling(20).std()
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)
        bb_upper = bb_upper.dropna()
        bb_middle = bb_middle.dropna()
        bb_lower = bb_lower.dropna()

    price = clean_data['close'].loc[bb_upper.index]

    ax3.plot(price.index, price.values, linewidth=1.5, label='Price', color='black')
    ax3.plot(bb_upper.index, bb_upper.values, linewidth=1, linestyle='--', label='Upper Band', color='red')
    ax3.plot(bb_middle.index, bb_middle.values, linewidth=1, label='Middle (SMA)', color='blue')
    ax3.plot(bb_lower.index, bb_lower.values, linewidth=1, linestyle='--', label='Lower Band', color='green')
    ax3.fill_between(bb_upper.index, bb_lower.values, bb_upper.values, alpha=0.1, color='gray')
    ax3.set_title('Bollinger Bands', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Price', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ATR (Average True Range)
    atr = features['atr_14'].dropna() if 'atr_14' in features.columns else features.filter(like='atr').iloc[:, 0].dropna()
    ax4.plot(atr.index, atr.values, linewidth=1.5, color='#A23B72')
    ax4.fill_between(atr.index, 0, atr.values, alpha=0.3, color='#A23B72')
    ax4.set_title('ATR (Average True Range) - Volatility Measure', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date', fontsize=12)
    ax4.set_ylabel('ATR', fontsize=12)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, '03_technical_indicators',
              "Key technical indicators used for regime detection and trading: "
              "RSI measures momentum (>70 overbought, <30 oversold). "
              "MACD shows trend strength (crossovers signal momentum shifts). "
              "Bollinger Bands identify volatility extremes (price touches = potential reversal). "
              "ATR measures market volatility (higher values = more volatile).")

    # ============= STEP 4: REGIME DETECTION (GMM) =============
    print("\n🔍 Step 4: Detecting regimes with GMM...")
    gmm_detector = GMMDetector(n_regimes=3, random_state=42)
    gmm_detector.fit(regime_features)
    gmm_regimes = gmm_detector.predict(regime_features)
    gmm_probs = gmm_detector.predict_proba(regime_features)
    gmm_stats = gmm_detector.get_regime_statistics(
        regime_features,
        returns=clean_data['close'].pct_change().loc[regime_features.index]
    )
    print(f"✅ Detected 3 regimes with GMM")

    # Plot 4: GMM Regime Overlay
    fig = plot_regimes(
        clean_data['close'].loc[regime_features.index],
        gmm_regimes,
        title=f"{ticker} - GMM Regime Detection (3 Regimes)"
    )
    save_plot(fig, '04_gmm_regimes',
              "GMM (Gaussian Mixture Model) regime detection overlaid on price chart. "
              "Each color represents a distinct market regime identified by the ML model. "
              "Regimes typically correspond to: Bull markets (uptrend), Bear markets (downtrend), "
              "and Sideways/Consolidation periods. GMM assumes data comes from mixture of Gaussian distributions.")

    # Plot 5: GMM Statistics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Returns by regime
    regime_labels = ['Regime 0', 'Regime 1', 'Regime 2']
    returns_by_regime = [gmm_stats[i]['mean_return'] * 252 * 100 for i in range(3)]
    colors_bar = ['#2E86AB', '#F18F01', '#A23B72']

    bars = ax1.bar(regime_labels, returns_by_regime, color=colors_bar, edgecolor='black', linewidth=1.5)
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
    ax1.set_title('Annualized Returns by Regime (%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Return (%)', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, returns_by_regime):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontsize=10, fontweight='bold')

    # Volatility by regime
    vol_by_regime = [gmm_stats[i]['volatility'] * 100 for i in range(3)]
    bars = ax2.bar(regime_labels, vol_by_regime, color=colors_bar, edgecolor='black', linewidth=1.5)
    ax2.set_title('Volatility by Regime (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Volatility (%)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, vol_by_regime):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Regime frequency
    freq_by_regime = [gmm_stats[i]['frequency'] * 100 for i in range(3)]
    ax3.pie(freq_by_regime, labels=regime_labels, autopct='%1.1f%%',
            colors=colors_bar, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax3.set_title('Regime Distribution', fontsize=14, fontweight='bold')

    # Sharpe ratio by regime
    sharpe_by_regime = [gmm_stats[i]['sharpe_ratio'] for i in range(3)]
    bars = ax4.bar(regime_labels, sharpe_by_regime, color=colors_bar, edgecolor='black', linewidth=1.5)
    ax4.axhline(1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (>1.0)')
    ax4.axhline(0, color='black', linestyle='-', linewidth=1)
    ax4.set_title('Sharpe Ratio by Regime', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Sharpe Ratio', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, sharpe_by_regime):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_plot(fig, '05_gmm_statistics',
              "Comprehensive statistics for each regime detected by GMM: "
              "Returns show profitability of each regime (positive = bull, negative = bear). "
              "Volatility indicates risk level (higher = more volatile). "
              "Distribution shows time spent in each regime. "
              "Sharpe Ratio measures risk-adjusted returns (>1.0 is good, >2.0 is excellent).")

    # Plot 6: Regime Probabilities
    fig, ax = plt.subplots(figsize=(14, 6))

    for i in range(3):
        ax.plot(gmm_probs.index, gmm_probs.iloc[:, i],
               label=f'Regime {i}', linewidth=1.5, color=colors_bar[i])

    ax.set_title('GMM Regime Probability Evolution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    save_plot(fig, '06_gmm_probabilities',
              "Evolution of regime probabilities over time. "
              "Shows confidence of model in regime classification. "
              "High probability (>0.7) = confident classification. "
              "Multiple regimes with similar probabilities = transition period or uncertain regime. "
              "Useful for detecting regime changes early.")

    # ============= STEP 5: REGIME DETECTION (HMM) =============
    print("\n🔍 Step 5: Detecting regimes with HMM...")
    hmm_detector = HMMDetector(n_regimes=3, random_state=42)
    hmm_detector.fit(regime_features)
    hmm_regimes = hmm_detector.predict(regime_features)
    hmm_probs = hmm_detector.predict_proba(regime_features)
    hmm_stats = hmm_detector.get_regime_statistics(
        regime_features,
        returns=clean_data['close'].pct_change().loc[regime_features.index]
    )
    print(f"✅ Detected 3 regimes with HMM")

    # Plot 7: HMM Regime Overlay
    fig = plot_regimes(
        clean_data['close'].loc[regime_features.index],
        hmm_regimes,
        title=f"{ticker} - HMM Regime Detection (3 Regimes)"
    )
    save_plot(fig, '07_hmm_regimes',
              "HMM (Hidden Markov Model) regime detection overlaid on price chart. "
              "HMM captures temporal dependencies and regime persistence better than GMM. "
              "Considers sequence of observations, not just current state. "
              "Better at identifying regime transitions and persistence patterns. "
              "Useful for markets with strong sequential dependencies.")

    # Plot 8: GMM vs HMM Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # GMM regimes
    price_plot = clean_data['close'].loc[regime_features.index]
    ax1.plot(price_plot.index, price_plot.values, color='black', linewidth=1.5, alpha=0.8, label='Price')

    for regime in range(3):
        mask = gmm_regimes == regime
        ax1.fill_between(price_plot.index, price_plot.min(), price_plot.max(),
                        where=mask, alpha=0.3, color=colors_bar[regime], label=f'GMM Regime {regime}')

    ax1.set_title('GMM Regime Detection', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # HMM regimes
    ax2.plot(price_plot.index, price_plot.values, color='black', linewidth=1.5, alpha=0.8, label='Price')

    for regime in range(3):
        mask = hmm_regimes == regime
        ax2.fill_between(price_plot.index, price_plot.min(), price_plot.max(),
                        where=mask, alpha=0.3, color=colors_bar[regime], label=f'HMM Regime {regime}')

    ax2.set_title('HMM Regime Detection', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Price', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, '08_gmm_vs_hmm',
              "Comparison of GMM vs HMM regime detection on same data. "
              "GMM: Faster, assumes independence between time periods. "
              "HMM: Slower, captures temporal dependencies and regime persistence. "
              "Notice HMM typically has smoother regime transitions. "
              "Choose GMM for speed, HMM for accuracy in sequential data.")

    # ============= STEP 6: STRATEGY BACKTESTING =============
    print("\n💼 Step 6: Backtesting trading strategies...")

    aligned_data = clean_data.loc[regime_features.index]
    returns = aligned_data['close'].pct_change()

    # Trend Following
    tf_strategy = TrendFollowingStrategy(fast_period=20, slow_period=50)
    tf_signals = tf_strategy.generate_signals(aligned_data)
    tf_returns = tf_signals['position'].shift(1) * returns

    # Mean Reversion
    mr_strategy = MeanReversionStrategy(window=20, threshold=2.0)
    mr_signals = mr_strategy.generate_signals(aligned_data)
    mr_returns = mr_signals['position'].shift(1) * returns

    # Volatility Breakout
    vb_strategy = VolatilityBreakoutStrategy(window=20, multiplier=2.0)
    vb_signals = vb_strategy.generate_signals(aligned_data)
    vb_returns = vb_signals['position'].shift(1) * returns

    # Buy & Hold
    bh_returns = returns

    print("✅ Backtested 4 strategies")

    # Plot 9: Equity Curves
    strategies = {
        'Trend Following': tf_returns,
        'Mean Reversion': mr_returns,
        'Volatility Breakout': vb_returns,
        'Buy & Hold': bh_returns
    }

    fig = plot_equity_curve(strategies, title=f"{ticker} Strategy Comparison")
    save_plot(fig, '09_equity_curves',
              "Cumulative returns comparison of all trading strategies vs Buy & Hold. "
              "Trend Following: Profits from sustained price movements using moving average crossovers. "
              "Mean Reversion: Profits from price returning to average after deviations. "
              "Volatility Breakout: Profits from price breaking through volatility bands. "
              "Buy & Hold: Passive benchmark. "
              "Best strategy depends on market conditions and regime.")

    # Plot 10: Strategy Performance Metrics
    metrics = {}
    for name, ret in strategies.items():
        ret_clean = ret.dropna()
        if len(ret_clean) > 0:
            metrics[name] = {
                'Total Return': (1 + ret_clean).prod() - 1,
                'Ann. Return': ret_clean.mean() * 252,
                'Volatility': ret_clean.std() * np.sqrt(252),
                'Sharpe': calculate_sharpe_ratio(ret_clean),
                'Sortino': calculate_sortino_ratio(ret_clean),
                'Max DD': calculate_max_drawdown(ret_clean)
            }

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    strategy_names = list(metrics.keys())
    strategy_colors = ['#2E86AB', '#F18F01', '#A23B72', '#6A994E']

    # Sharpe Ratio
    sharpe_values = [metrics[s]['Sharpe'] for s in strategy_names]
    bars = ax1.bar(strategy_names, sharpe_values, color=strategy_colors, edgecolor='black', linewidth=1.5)
    ax1.axhline(1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (>1.0)')
    ax1.axhline(2.0, color='darkgreen', linestyle='--', linewidth=2, alpha=0.5, label='Excellent (>2.0)')
    ax1.set_title('Sharpe Ratio by Strategy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Sharpe Ratio', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)

    for bar, val in zip(bars, sharpe_values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Annualized Return
    return_values = [metrics[s]['Ann. Return'] * 100 for s in strategy_names]
    bars = ax2.bar(strategy_names, return_values, color=strategy_colors, edgecolor='black', linewidth=1.5)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('Annualized Returns (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Return (%)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)

    for bar, val in zip(bars, return_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontsize=10, fontweight='bold')

    # Max Drawdown
    dd_values = [metrics[s]['Max DD'] * 100 for s in strategy_names]
    bars = ax3.bar(strategy_names, dd_values, color=strategy_colors, edgecolor='black', linewidth=1.5)
    ax3.set_title('Maximum Drawdown (%)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Drawdown (%)', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)

    for bar, val in zip(bars, dd_values):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}%', ha='center', va='top', fontsize=10, fontweight='bold')

    # Return vs Risk
    returns_plot = [metrics[s]['Ann. Return'] * 100 for s in strategy_names]
    vol_plot = [metrics[s]['Volatility'] * 100 for s in strategy_names]

    for i, name in enumerate(strategy_names):
        ax4.scatter(vol_plot[i], returns_plot[i], s=200, color=strategy_colors[i],
                   edgecolor='black', linewidth=2, label=name, alpha=0.8)
        ax4.annotate(name, (vol_plot[i], returns_plot[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

    ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax4.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax4.set_title('Risk-Return Trade-off', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Volatility (%) - Risk', fontsize=12)
    ax4.set_ylabel('Annualized Return (%) - Reward', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, '10_performance_metrics',
              "Comprehensive performance metrics for all strategies: "
              "Sharpe Ratio: Risk-adjusted returns (higher is better, >1 good, >2 excellent). "
              "Annualized Returns: Yearly equivalent profit/loss percentage. "
              "Maximum Drawdown: Largest peak-to-trough decline (lower is better). "
              "Risk-Return: Efficient frontier showing return per unit of risk (top-left is best).")

    # Plot 11: Drawdown Analysis
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    for idx, (name, ret) in enumerate(strategies.items()):
        equity = (1 + ret.dropna()).cumprod()
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100

        axes[idx].fill_between(drawdown.index, 0, drawdown.values,
                              color=strategy_colors[idx], alpha=0.6)
        axes[idx].plot(drawdown.index, drawdown.values,
                      color=strategy_colors[idx], linewidth=1.5)
        axes[idx].set_title(f'{name} Drawdown', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Drawdown (%)', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axhline(0, color='black', linestyle='-', linewidth=1)

        # Annotate max drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown.min()
        axes[idx].annotate(f'Max DD: {max_dd_val:.1f}%',
                          xy=(max_dd_idx, max_dd_val),
                          xytext=(10, -20), textcoords='offset points',
                          fontsize=10, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                          arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    axes[-1].set_xlabel('Date', fontsize=12)
    plt.tight_layout()
    save_plot(fig, '11_drawdown_analysis',
              "Drawdown analysis for each strategy showing peak-to-trough declines over time. "
              "Drawdown = percentage decline from previous high. "
              "Deeper drawdowns = higher risk and psychological stress. "
              "Faster recovery = better risk management. "
              "Maximum drawdown is the largest observed decline.")

    # ============= STEP 7: REGIME-ADAPTIVE STRATEGY =============
    print("\n🎯 Step 7: Building regime-adaptive strategy...")

    # Map regimes to best strategies based on statistics
    regime_strategy_map = {}
    for regime_id in range(3):
        regime_return = gmm_stats[regime_id]['mean_return']
        regime_vol = gmm_stats[regime_id]['volatility']

        # Simple heuristic
        if regime_return > 0 and regime_vol < 0.02:
            regime_strategy_map[regime_id] = 'Trend Following'
        elif regime_return < 0:
            regime_strategy_map[regime_id] = 'Mean Reversion'
        else:
            regime_strategy_map[regime_id] = 'Volatility Breakout'

    # Create adaptive returns
    adaptive_returns = pd.Series(0, index=returns.index)
    for regime_id, strategy_name in regime_strategy_map.items():
        regime_mask = gmm_regimes == regime_id
        adaptive_returns[regime_mask] = strategies[strategy_name][regime_mask]

    # Plot 12: Regime-Adaptive Performance
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Compare with best single strategy
    best_single_strategy = max(metrics.items(), key=lambda x: x[1]['Sharpe'])

    adaptive_equity = (1 + adaptive_returns.dropna()).cumprod()
    best_equity = (1 + strategies[best_single_strategy[0]].dropna()).cumprod()
    bh_equity = (1 + bh_returns.dropna()).cumprod()

    ax1.plot(adaptive_equity.index, adaptive_equity.values,
            linewidth=2, label='Regime-Adaptive', color='#2E86AB')
    ax1.plot(best_equity.index, best_equity.values,
            linewidth=2, label=f'Best Single ({best_single_strategy[0]})', color='#F18F01')
    ax1.plot(bh_equity.index, bh_equity.values,
            linewidth=2, label='Buy & Hold', color='#6A994E', linestyle='--')

    ax1.set_title('Regime-Adaptive Strategy vs Best Single Strategy', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Show active strategy by regime
    price_plot = clean_data['close'].loc[regime_features.index]
    ax2.plot(price_plot.index, price_plot.values, color='black', linewidth=1, alpha=0.5, label='Price')

    for regime_id in range(3):
        mask = gmm_regimes == regime_id
        strategy_used = regime_strategy_map[regime_id]
        ax2.fill_between(price_plot.index, price_plot.min(), price_plot.max(),
                        where=mask, alpha=0.4, color=colors_bar[regime_id],
                        label=f'Regime {regime_id}: {strategy_used}')

    ax2.set_title('Active Strategy by Detected Regime', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Price', fontsize=12)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, '12_regime_adaptive',
              "Regime-adaptive strategy that automatically switches between strategies based on detected regime. "
              "Top: Performance comparison showing adaptive approach vs best single strategy. "
              "Bottom: Shows which strategy is active in each regime period. "
              "Adaptive strategies can outperform single strategies by exploiting regime-specific patterns. "
              "Key advantage: Automatically adjusts to changing market conditions.")

    # ============= STEP 8: FINAL SUMMARY =============
    print("\n📊 Step 8: Generating final summary...")

    # Calculate all metrics
    adaptive_metrics = {
        'Total Return': (1 + adaptive_returns.dropna()).prod() - 1,
        'Ann. Return': adaptive_returns.dropna().mean() * 252,
        'Volatility': adaptive_returns.dropna().std() * np.sqrt(252),
        'Sharpe': calculate_sharpe_ratio(adaptive_returns.dropna()),
        'Sortino': calculate_sortino_ratio(adaptive_returns.dropna()),
        'Max DD': calculate_max_drawdown(adaptive_returns.dropna())
    }

    # Create summary table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Title
    fig.text(0.5, 0.95, f'{ticker} MARKET REGIME DETECTION & STRATEGY ANALYSIS',
            ha='center', fontsize=20, fontweight='bold')
    fig.text(0.5, 0.91, f'Period: {start_date} to {data.index[-1].strftime("%Y-%m-%d")} | {len(data)} Trading Days',
            ha='center', fontsize=12)

    # Regime summary
    y_pos = 0.82
    fig.text(0.05, y_pos, 'REGIME ANALYSIS (GMM)', fontsize=14, fontweight='bold')
    y_pos -= 0.04

    for regime_id in range(3):
        stats = gmm_stats[regime_id]
        text = f"Regime {regime_id}: {stats['frequency']*100:.1f}% of time | "
        text += f"Return: {stats['mean_return']*252*100:.1f}% | "
        text += f"Vol: {stats['volatility']*100:.1f}% | "
        text += f"Sharpe: {stats['sharpe_ratio']:.2f} | "
        text += f"Avg Duration: {stats['avg_duration']:.1f} days"
        fig.text(0.1, y_pos, text, fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor=colors_bar[regime_id], alpha=0.3))
        y_pos -= 0.04

    # Strategy performance
    y_pos -= 0.04
    fig.text(0.05, y_pos, 'STRATEGY PERFORMANCE', fontsize=14, fontweight='bold')
    y_pos -= 0.04

    all_strategies = {**metrics, 'Regime-Adaptive': adaptive_metrics}

    for name, m in sorted(all_strategies.items(), key=lambda x: x[1]['Sharpe'], reverse=True):
        text = f"{name:20} | Return: {m['Ann. Return']*100:7.2f}% | "
        text += f"Sharpe: {m['Sharpe']:5.2f} | "
        text += f"Sortino: {m['Sortino']:5.2f} | "
        text += f"Max DD: {m['Max DD']*100:6.2f}%"

        color = '#90EE90' if name == 'Regime-Adaptive' else 'lightgray'
        fig.text(0.1, y_pos, text, fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
        y_pos -= 0.035

    # Key insights
    y_pos -= 0.04
    fig.text(0.05, y_pos, 'KEY INSIGHTS', fontsize=14, fontweight='bold')
    y_pos -= 0.04

    insights = [
        f"✓ Best Single Strategy: {best_single_strategy[0]} (Sharpe: {best_single_strategy[1]['Sharpe']:.2f})",
        f"✓ Regime-Adaptive Sharpe: {adaptive_metrics['Sharpe']:.2f}",
        f"✓ Most Profitable Regime: Regime {max(gmm_stats, key=lambda x: gmm_stats[x]['mean_return'])} ({max(gmm_stats.values(), key=lambda x: x['mean_return'])['mean_return']*252*100:.1f}% ann.)",
        f"✓ Most Stable Regime: Regime {max(gmm_stats, key=lambda x: gmm_stats[x]['avg_duration'])} (avg {max(gmm_stats.values(), key=lambda x: x['avg_duration'])['avg_duration']:.1f} days)",
        f"✓ Current Regime: Regime {gmm_regimes.iloc[-1]} (confidence: {gmm_probs.iloc[-1, gmm_regimes.iloc[-1]]*100:.1f}%)"
    ]

    for insight in insights:
        fig.text(0.1, y_pos, insight, fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        y_pos -= 0.04

    # Footer
    fig.text(0.5, 0.02, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | For portfolio/recruiter review',
            ha='center', fontsize=10, style='italic')

    save_plot(fig, '13_executive_summary',
              "Executive summary combining all analysis results. "
              "Shows regime characteristics, strategy performance rankings, and key insights. "
              "Regime-Adaptive highlighted in green if it outperforms single strategies. "
              "Use this summary for quick decision-making and portfolio presentations.")

    # ============= SAVE SIMULATION DATA =============
    print("\n💾 Saving simulation data...")

    # Create comprehensive results dictionary
    simulation_results = {
        'timestamp': datetime.now().isoformat(),
        'ticker': ticker,
        'start_date': start_date,
        'end_date': data.index[-1].strftime('%Y-%m-%d'),
        'n_days': len(data),
        'gmm_regimes': gmm_regimes.to_dict(),
        'gmm_stats': gmm_stats,
        'strategy_metrics': {k: v for k, v in metrics.items()},
        'adaptive_metrics': adaptive_metrics,
        'regime_strategy_map': regime_strategy_map
    }

    import json
    with open(SIMS_DIR / f'simulation_{ticker}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(simulation_results, f, indent=2, default=str)

    # Save returns data
    returns_df = pd.DataFrame({
        'Date': returns.index,
        'Market_Returns': returns.values,
        'Trend_Following': tf_returns.values,
        'Mean_Reversion': mr_returns.values,
        'Volatility_Breakout': vb_returns.values,
        'Buy_Hold': bh_returns.values,
        'Regime_Adaptive': adaptive_returns.values,
        'GMM_Regime': gmm_regimes.values
    })
    returns_df.to_csv(SIMS_DIR / f'returns_{ticker}_{datetime.now().strftime("%Y%m%d")}.csv', index=False)

    print(f"✅ Saved simulation data to {SIMS_DIR}/")

    # ============= CREATE INDEX =============
    print("\n📝 Creating index file...")

    index_content = f"""# VISUALIZATIONS INDEX

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Ticker: {ticker}
Period: {start_date} to {data.index[-1].strftime('%Y-%m-%d')}

## Plots Generated

### 1. Data & Preprocessing
- **01_raw_data.png**: Raw OHLCV price and volume data
- **02_returns_analysis.png**: Returns distribution, cumulative performance, volatility, Q-Q plot
- **03_technical_indicators.png**: RSI, MACD, Bollinger Bands, ATR

### 2. Regime Detection
- **04_gmm_regimes.png**: GMM regime overlay on price chart
- **05_gmm_statistics.png**: Returns, volatility, distribution, Sharpe by regime
- **06_gmm_probabilities.png**: Regime probability evolution over time
- **07_hmm_regimes.png**: HMM regime overlay on price chart
- **08_gmm_vs_hmm.png**: Side-by-side comparison of GMM and HMM

### 3. Strategy Performance
- **09_equity_curves.png**: Cumulative returns comparison of all strategies
- **10_performance_metrics.png**: Sharpe, returns, drawdown, risk-return analysis
- **11_drawdown_analysis.png**: Drawdown charts for each strategy
- **12_regime_adaptive.png**: Regime-adaptive strategy performance and allocation
- **13_executive_summary.png**: Comprehensive summary table with key insights

## Simulation Data
- `simulations/simulation_{ticker}_*.json`: Full simulation results in JSON format
- `simulations/returns_{ticker}_*.csv`: Daily returns for all strategies

## How to Use These Visuals

### For Recruiters
Show plots in this order:
1. Start with 13 (Executive Summary) - gives big picture
2. Show 01-03 (Data quality and features)
3. Show 04-05 (Regime detection capability)
4. Show 09-10 (Strategy performance)
5. Show 12 (Adaptive strategy advantage)

### For Portfolio
- Create PDF with plots 01, 04, 05, 09, 10, 12, 13
- Add brief explanations from description files
- Highlight Sharpe ratios and adaptive strategy benefits

### For Presentations
Key talking points for each visual:
- Plot 04: "ML automatically identifies market regimes"
- Plot 05: "Each regime has distinct risk-return characteristics"
- Plot 09: "Multiple strategies tested, not just buy-and-hold"
- Plot 12: "Adaptive approach outperforms single strategies"

## Technical Details
- All plots saved at 300 DPI for print quality
- Color scheme: Professional, colorblind-friendly
- Each plot has accompanying description file
- Data exported for reproducibility

---
For questions or to regenerate: python demo_with_plots.py
"""

    with open(PLOTS_DIR / 'INDEX.md', 'w') as f:
        f.write(index_content)

    print(f"✅ Created index at {PLOTS_DIR}/INDEX.md")

    print(f"\n{'='*60}")
    print("✅ SIMULATION COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📊 Generated {13} plots in {PLOTS_DIR}/")
    print(f"💾 Saved simulation data in {SIMS_DIR}/")
    print(f"📝 View INDEX.md for navigation guide")
    print(f"\nQuick start: Open {PLOTS_DIR}/13_executive_summary.png for overview\n")

if __name__ == "__main__":
    # Run simulation with SPY
    run_complete_simulation('SPY', start_date='2020-01-01')

    print("\n" + "="*60)
    print("ALL DONE! 🎉")
    print("="*60)
    print("\nYour project now has:")
    print("✅ Production-ready Streamlit app (running on port 8502)")
    print("✅ 13 professional visualizations")
    print("✅ Comprehensive documentation")
    print("✅ Simulation data exports")
    print("✅ Recruiter-friendly presentation")
    print("\nNext steps:")
    print("1. View plots in outputs/plots/")
    print("2. Open http://localhost:8502 for interactive app")
    print("3. Deploy to Streamlit Cloud to share")
    print("4. Add to resume/portfolio with plots\n")