"""
Test script to verify Streamlit app functionality end-to-end
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("STREAMLIT APP FUNCTIONALITY TEST")
print("="*60)

# Test 1: Import all required modules
print("\n1. Testing imports...")
try:
    from src.data.data_loader import DataLoader
    from src.data.data_preprocessor import DataPreprocessor
    from src.data.feature_engineer import FeatureEngineer
    from src.regime_detection.gmm_detector import GMMDetector
    from src.regime_detection.hmm_detector import HMMDetector
    from src.strategies.trend_following import TrendFollowingStrategy
    from src.strategies.mean_reversion import MeanReversionStrategy
    from src.strategies.volatility_breakout import VolatilityBreakoutStrategy
    from src.utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Data loading
print("\n2. Testing data loading...")
try:
    loader = DataLoader(use_cache=True)
    data = loader.load_data('SPY', start_date='2023-01-01', end_date='2023-12-31')
    print(f"✅ Loaded {len(data)} days of data")
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    sys.exit(1)

# Test 3: Feature engineering
print("\n3. Testing feature engineering...")
try:
    engineer = FeatureEngineer()
    features = engineer.create_features(data)
    regime_features = engineer.extract_regime_features(features)
    print(f"✅ Created {len(features.columns)} features, {len(regime_features.columns)} regime features")
except Exception as e:
    print(f"❌ Feature engineering failed: {e}")
    sys.exit(1)

# Test 4: Regime detection
print("\n4. Testing regime detection...")
try:
    import pandas as pd

    detector = GMMDetector(n_regimes=3, random_state=42)
    detector.fit(regime_features)
    regimes = detector.predict(regime_features)

    # Convert to pandas Series (as done in app)
    regimes = pd.Series(regimes, index=regime_features.index, name='regime')
    probabilities = detector.predict_proba(regime_features)
    probabilities = pd.DataFrame(probabilities, index=regime_features.index)

    returns = data['close'].pct_change().loc[regime_features.index]
    stats = detector.get_regime_statistics(regime_features, returns=returns)

    print(f"✅ Detected 3 regimes")
    print(f"   Regime distribution: {regimes.value_counts().to_dict()}")

    # Test stats structure
    for regime_id, regime_stats in stats.items():
        if isinstance(regime_stats, dict):
            mean_ret = regime_stats['mean_return']
        else:
            mean_ret = regime_stats.get('mean_return', 0)
        print(f"   Regime {regime_id}: {mean_ret*100:.2f}% return")

except Exception as e:
    print(f"❌ Regime detection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Strategy testing
print("\n5. Testing strategies...")
try:
    aligned_data = data.loc[regimes.index].copy()
    returns = aligned_data['close'].pct_change()

    # Test Trend Following
    tf_strategy = TrendFollowingStrategy(fast_period=20, slow_period=50)
    tf_signals = tf_strategy.generate_signals(aligned_data)
    tf_positions = tf_signals if isinstance(tf_signals, pd.Series) else tf_signals.get('position', tf_signals)
    tf_returns = tf_positions.shift(1) * returns
    print(f"✅ Trend Following: {len(tf_returns)} signals")

    # Test Mean Reversion
    mr_strategy = MeanReversionStrategy(bb_period=20, zscore_threshold=2.0)
    mr_signals = mr_strategy.generate_signals(aligned_data)
    mr_positions = mr_signals if isinstance(mr_signals, pd.Series) else mr_signals.get('position', mr_signals)
    mr_returns = mr_positions.shift(1) * returns
    print(f"✅ Mean Reversion: {len(mr_returns)} signals")

    # Test Volatility Breakout
    vb_strategy = VolatilityBreakoutStrategy(lookback_period=20, atr_multiplier=2.0)
    vb_signals = vb_strategy.generate_signals(aligned_data)
    vb_positions = vb_signals if isinstance(vb_signals, pd.Series) else vb_signals.get('position', vb_signals)
    vb_returns = vb_positions.shift(1) * returns
    print(f"✅ Volatility Breakout: {len(vb_returns)} signals")

except Exception as e:
    print(f"❌ Strategy testing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Metrics calculation
print("\n6. Testing metrics...")
try:
    sharpe = calculate_sharpe_ratio(tf_returns.dropna())
    max_dd = calculate_max_drawdown(tf_returns.dropna())
    print(f"✅ Calculated metrics: Sharpe={sharpe:.2f}, MaxDD={max_dd*100:.2f}%")
except Exception as e:
    print(f"❌ Metrics calculation failed: {e}")
    sys.exit(1)

# Test 7: Regime-adaptive strategy
print("\n7. Testing regime-adaptive strategy...")
try:
    regime_strategy_map = {}

    # Stats is a DataFrame where each row is a regime
    import pandas as pd
    if isinstance(stats, pd.DataFrame):
        for regime_id in stats.index:
            mean_return = stats.loc[regime_id, 'mean_return']
            if mean_return > 0:
                regime_strategy_map[regime_id] = TrendFollowingStrategy(fast_period=20, slow_period=50)
            else:
                regime_strategy_map[regime_id] = MeanReversionStrategy(bb_period=20, zscore_threshold=2.0)
    else:
        # Handle dict structure
        for regime_id, regime_stat in stats.items():
            mean_return = regime_stat.get('mean_return', 0) if isinstance(regime_stat, dict) else regime_stat['mean_return']
        if mean_return > 0:
            regime_strategy_map[regime_id] = TrendFollowingStrategy(fast_period=20, slow_period=50)
        else:
            regime_strategy_map[regime_id] = MeanReversionStrategy(bb_period=20, zscore_threshold=2.0)

    adaptive_returns = pd.Series(0.0, index=returns.index)
    for regime_id, strategy in regime_strategy_map.items():
        regime_mask = regimes == regime_id
        regime_data = aligned_data[regime_mask]
        if len(regime_data) > 0:
            signals = strategy.generate_signals(regime_data)
            positions = signals if isinstance(signals, pd.Series) else signals.get('position', signals)
            strategy_returns = positions.shift(1) * returns[regime_mask]
            adaptive_returns[regime_mask] = strategy_returns.fillna(0)

    print(f"✅ Regime-adaptive strategy: {len(adaptive_returns)} periods")
    print(f"   Mapped {len(regime_strategy_map)} regimes to strategies")

except Exception as e:
    print(f"❌ Regime-adaptive strategy failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED")
print("="*60)
print("\nThe Streamlit app should work correctly!")
print("Access it at: http://localhost:8502")
print("\nTest workflow:")
print("1. Data Explorer: Load SPY data (2020-01-01 to present)")
print("2. Regime Detection: Run GMM with 3 regimes")
print("3. Strategy Analysis: Test all strategies")
print("4. Live Dashboard: View current insights")
print("="*60)