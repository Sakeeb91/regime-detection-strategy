"""
Integration tests for specific pipeline components.

Tests specific integrations between modules and validates
data flow through the system.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.data_preprocessor import DataPreprocessor
from src.data.feature_engineer import FeatureEngineer
from src.regime_detection.gmm_detector import GMMDetector
from src.regime_detection.hmm_detector import HMMDetector
from src.regime_detection.dtw_clustering import DTWClustering
from src.strategies.strategy_selector import StrategySelector
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy


class TestDataToFeaturesPipeline:
    """Test data preprocessing to feature engineering pipeline."""

    @pytest.fixture
    def raw_data(self):
        """Create raw OHLCV data with some issues."""
        dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")
        n = len(dates)

        data = pd.DataFrame(
            {
                "open": np.random.randn(n).cumsum() + 100,
                "high": np.random.randn(n).cumsum() + 102,
                "low": np.random.randn(n).cumsum() + 98,
                "close": np.random.randn(n).cumsum() + 100,
                "volume": np.random.randint(100000, 1000000, n),
            },
            index=dates,
        )

        # Introduce some issues
        data.loc[data.index[10], "close"] = np.nan  # Missing value
        data.loc[data.index[50], "high"] = (
            data.loc[data.index[50], "close"] * 3
        )  # Outlier

        return data

    def test_preprocessing_to_features(self, raw_data):
        """Test that preprocessed data flows correctly to feature engineering."""
        # Preprocess
        preprocessor = DataPreprocessor()
        clean_data = preprocessor.clean_data(raw_data)

        # Should have removed/fixed issues
        assert not clean_data.isnull().any().any()
        assert len(clean_data) <= len(raw_data)

        # Engineer features
        engineer = FeatureEngineer()
        features = engineer.create_features(clean_data)

        # Should have many more columns
        assert len(features.columns) > len(clean_data.columns)

        # Should have returns
        assert "returns" in features.columns
        assert "log_returns" in features.columns

        # Should not have NaN in output
        assert not features.isnull().any().any()

    def test_feature_groups(self, raw_data):
        """Test creating specific feature groups."""
        preprocessor = DataPreprocessor()
        clean_data = preprocessor.clean_data(raw_data)

        engineer = FeatureEngineer()

        # Test individual feature groups
        for group in ["trend", "volatility", "momentum", "volume", "statistical"]:
            features = engineer.create_features(clean_data, feature_groups=[group])
            assert len(features) > 0
            assert not features.isnull().any().any()

    def test_regime_feature_extraction(self, raw_data):
        """Test extracting features suitable for regime detection."""
        preprocessor = DataPreprocessor()
        clean_data = preprocessor.clean_data(raw_data)

        engineer = FeatureEngineer()
        all_features = engineer.create_features(clean_data)

        # Extract regime-relevant features
        regime_features = engineer.extract_regime_features(all_features)

        # Should be a subset of features
        assert len(regime_features.columns) < len(all_features.columns)
        assert len(regime_features) == len(all_features)

        # Should not have NaN
        assert not regime_features.isnull().any().any()


class TestFeaturesToRegimesPipeline:
    """Test feature engineering to regime detection pipeline."""

    @pytest.fixture
    def feature_data(self):
        """Create feature data for regime detection."""
        dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")
        n = len(dates)

        # Create data with distinct regimes
        regime_1 = pd.DataFrame(
            {
                "returns": np.random.randn(n // 3) * 0.005,
                "volatility": np.random.randn(n // 3) * 0.01 + 0.01,
                "trend_strength": np.random.randn(n // 3) * 0.1 + 0.3,
            }
        )

        regime_2 = pd.DataFrame(
            {
                "returns": np.random.randn(n // 3) * 0.02,
                "volatility": np.random.randn(n // 3) * 0.02 + 0.03,
                "trend_strength": np.random.randn(n // 3) * 0.1 + 0.7,
            }
        )

        regime_3 = pd.DataFrame(
            {
                "returns": np.random.randn(n // 3 + n % 3) * 0.001,
                "volatility": np.random.randn(n // 3 + n % 3) * 0.005 + 0.005,
                "trend_strength": np.random.randn(n // 3 + n % 3) * 0.1 + 0.1,
            }
        )

        data = pd.concat([regime_1, regime_2, regime_3], ignore_index=True)
        data.index = dates

        return data

    def test_gmm_regime_detection(self, feature_data):
        """Test GMM regime detection on features."""
        detector = GMMDetector(n_regimes=3, random_state=42)
        detector.fit(feature_data)
        regimes = detector.predict(feature_data)

        assert len(regimes) == len(feature_data)
        assert len(np.unique(regimes)) == 3
        assert regimes.min() == 0
        assert regimes.max() == 2

        # Check statistics
        stats = detector.get_regime_statistics(feature_data)
        assert len(stats) == 3

    def test_hmm_regime_detection(self, feature_data):
        """Test HMM regime detection on features."""
        detector = HMMDetector(n_regimes=3, random_state=42)
        detector.fit(feature_data)
        regimes = detector.predict(feature_data)

        assert len(regimes) == len(feature_data)
        assert len(np.unique(regimes)) <= 3

        # Check transition probabilities
        trans_prob = detector.get_transition_matrix()
        assert trans_prob.shape == (3, 3)
        assert np.allclose(trans_prob.sum(axis=1), 1.0)

    def test_dtw_clustering(self, feature_data):
        """Test DTW clustering on features."""
        detector = DTWClustering(n_clusters=3, random_state=42)

        # Use returns as pandas Series for clustering
        returns_series = feature_data["returns"]
        detector.fit(returns_series)
        labels = detector.predict(returns_series)

        assert len(labels) == len(feature_data)
        assert len(np.unique(labels)) <= 3

    def test_regime_consistency(self, feature_data):
        """Test that detected regimes are consistent."""
        detector = GMMDetector(n_regimes=3, random_state=42)
        detector.fit(feature_data)
        regimes = detector.predict(feature_data)

        # Calculate regime persistence (average duration)
        # Convert to pandas Series for diff operation
        regimes_series = pd.Series(regimes)
        regime_changes = (regimes_series.diff() != 0).sum()
        avg_duration = (
            len(regimes) / regime_changes if regime_changes > 0 else len(regimes)
        )

        # Regimes should persist for multiple periods
        assert avg_duration > 1


class TestRegimesToStrategiesPipeline:
    """Test regime detection to strategy selection pipeline."""

    @pytest.fixture
    def price_data_with_regimes(self):
        """Create price data with regime labels."""
        dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")
        n = len(dates)

        data = pd.DataFrame(
            {
                "open": np.random.randn(n).cumsum() + 100,
                "high": np.random.randn(n).cumsum() + 102,
                "low": np.random.randn(n).cumsum() + 98,
                "close": np.random.randn(n).cumsum() + 100,
                "volume": np.random.randint(100000, 1000000, n),
            },
            index=dates,
        )

        # Create regime labels (0, 1, 2)
        regimes = pd.Series(np.random.choice([0, 1, 2], size=n), index=dates)

        return data, regimes

    def test_strategy_selector(self, price_data_with_regimes):
        """Test strategy selector with regime labels."""
        data, regimes = price_data_with_regimes

        # Create strategy mapping
        strategy_map = {
            0: TrendFollowingStrategy(),
            1: MeanReversionStrategy(),
            2: VolatilityBreakoutStrategy(),
        }

        selector = StrategySelector(strategy_map)

        # Get strategy for each regime
        for regime in [0, 1, 2]:
            strategy = selector.select_strategy(regime)
            assert strategy is not None
            assert strategy == strategy_map[regime]

    def test_adaptive_signals(self, price_data_with_regimes):
        """Test generating adaptive signals based on regimes."""
        data, regimes = price_data_with_regimes

        # Engineer features first
        engineer = FeatureEngineer()
        features = engineer.create_features(data)

        # Align regimes with features
        regimes_aligned = regimes.loc[features.index]

        # Create adaptive signals
        adaptive_signals = pd.Series(0, index=features.index)

        strategies = {
            0: TrendFollowingStrategy(),
            1: MeanReversionStrategy(),
            2: VolatilityBreakoutStrategy(),
        }

        for regime_id, strategy in strategies.items():
            regime_mask = regimes_aligned == regime_id
            if regime_mask.any():
                regime_data = features[regime_mask]
                regime_signals = strategy.generate_signals(regime_data)
                adaptive_signals[regime_mask] = regime_signals

        # Should have generated some signals
        assert adaptive_signals.abs().sum() > 0


class TestStrategiesToBacktestPipeline:
    """Test strategy signals to backtest results pipeline."""

    @pytest.fixture
    def strategy_setup(self):
        """Setup data and strategy for testing."""
        dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")
        n = len(dates)

        data = pd.DataFrame(
            {
                "open": np.random.randn(n).cumsum() + 100,
                "high": np.random.randn(n).cumsum() + 102,
                "low": np.random.randn(n).cumsum() + 98,
                "close": np.random.randn(n).cumsum() + 100,
                "volume": np.random.randint(100000, 1000000, n),
            },
            index=dates,
        )

        # Add required features
        engineer = FeatureEngineer()
        features = engineer.create_features(data)

        strategy = TrendFollowingStrategy()

        return features, strategy

    def test_signal_to_position_conversion(self, strategy_setup):
        """Test converting signals to positions."""
        features, strategy = strategy_setup

        signals = strategy.generate_signals(features)
        positions = strategy.get_positions(signals)

        # Positions should match signals
        assert len(positions) == len(signals)
        assert set(positions.unique()).issubset({-1, 0, 1})

    def test_position_to_returns(self, strategy_setup):
        """Test calculating returns from positions."""
        features, strategy = strategy_setup

        signals = strategy.generate_signals(features)
        positions = strategy.get_positions(signals)
        market_returns = features["close"].pct_change()
        strategy_returns = strategy.calculate_returns(positions, market_returns)

        assert len(strategy_returns) == len(positions)
        assert not strategy_returns.isnull().all()


class TestCrossModuleCompatibility:
    """Test compatibility between different module combinations."""

    def test_all_detectors_with_all_strategies(self):
        """Test that all detectors work with all strategies."""
        # Create sample data
        dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")
        n = len(dates)

        data = pd.DataFrame(
            {
                "open": np.random.randn(n).cumsum() + 100,
                "high": np.random.randn(n).cumsum() + 102,
                "low": np.random.randn(n).cumsum() + 98,
                "close": np.random.randn(n).cumsum() + 100,
                "volume": np.random.randint(100000, 1000000, n),
            },
            index=dates,
        )

        # Prepare features
        engineer = FeatureEngineer()
        features = engineer.create_features(data)
        regime_features = engineer.extract_regime_features(features)

        # Test all detector-strategy combinations
        detectors = [
            GMMDetector(n_regimes=3, random_state=42),
            HMMDetector(n_regimes=3, random_state=42),
        ]

        strategies = [
            TrendFollowingStrategy(),
            MeanReversionStrategy(),
            VolatilityBreakoutStrategy(),
        ]

        for detector in detectors:
            detector.fit(regime_features)
            detector.predict(regime_features)

            for strategy in strategies:
                # Should work without errors
                signals = strategy.generate_signals(features)
                assert len(signals) == len(features)
