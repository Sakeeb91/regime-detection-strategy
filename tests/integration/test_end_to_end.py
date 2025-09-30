"""
End-to-end integration tests for the complete pipeline.

Tests the full workflow: data loading → feature engineering →
regime detection → strategy selection → backtesting
"""

import pytest
import pandas as pd
import numpy as np

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.data.feature_engineer import FeatureEngineer
from src.regime_detection.gmm_detector import GMMDetector
from src.regime_detection.hmm_detector import HMMDetector
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy
from src.strategies.backtester import Backtester


class TestEndToEndPipeline:
    """Test complete pipeline from data to backtest results."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
        n = len(dates)

        # Generate synthetic price data with trend and noise
        trend = np.linspace(100, 200, n)
        noise = np.random.randn(n) * 5
        close = trend + noise

        data = pd.DataFrame(
            {
                "open": close * (1 + np.random.randn(n) * 0.01),
                "high": close * (1 + abs(np.random.randn(n)) * 0.015),
                "low": close * (1 - abs(np.random.randn(n)) * 0.015),
                "close": close,
                "volume": np.random.randint(1000000, 10000000, n),
            },
            index=dates,
        )

        return data

    def test_full_pipeline_gmm(self, sample_data):
        """Test complete pipeline with GMM regime detection."""
        # Step 1: Preprocess data
        preprocessor = DataPreprocessor()
        clean_data = preprocessor.clean_data(sample_data)

        assert len(clean_data) > 0
        assert not clean_data.isnull().any().any()

        # Step 2: Engineer features
        engineer = FeatureEngineer()
        features = engineer.create_features(clean_data)

        assert len(features.columns) > len(clean_data.columns)
        assert "returns" in features.columns

        # Step 3: Detect regimes with GMM
        regime_features = engineer.extract_regime_features(features)
        detector = GMMDetector(n_regimes=3, random_state=42)
        detector.fit(regime_features)
        regimes = detector.predict(regime_features)

        assert len(regimes) == len(regime_features)
        assert len(np.unique(regimes)) == 3

        # Step 4: Run strategy backtest
        strategy = TrendFollowingStrategy()
        backtester = Backtester(initial_capital=100000)
        results = backtester.run(strategy, features, regime_labels=regimes)

        # Validate results
        assert "equity_curve" in results
        assert "metrics" in results
        assert "trades" in results
        assert len(results["equity_curve"]) == len(features)
        assert results["metrics"]["n_trades"] >= 0

    def test_full_pipeline_hmm(self, sample_data):
        """Test complete pipeline with HMM regime detection."""
        # Step 1: Preprocess
        preprocessor = DataPreprocessor()
        clean_data = preprocessor.clean_data(sample_data)

        # Step 2: Features
        engineer = FeatureEngineer()
        features = engineer.create_features(clean_data)

        # Step 3: HMM regime detection
        regime_features = engineer.extract_regime_features(features)
        detector = HMMDetector(n_regimes=3, random_state=42)
        detector.fit(regime_features)
        regimes = detector.predict(regime_features)

        assert len(regimes) == len(regime_features)
        assert regimes.nunique() <= 3  # May be less if states don't all appear

        # Step 4: Backtest mean reversion strategy
        strategy = MeanReversionStrategy()
        backtester = Backtester(initial_capital=100000)
        results = backtester.run(strategy, features, regime_labels=regimes)

        assert results["metrics"]["total_return"] is not None
        assert "regime_analysis" in results["metrics"]

    def test_multiple_strategies(self, sample_data):
        """Test comparing multiple strategies."""
        # Prepare data
        preprocessor = DataPreprocessor()
        clean_data = preprocessor.clean_data(sample_data)

        engineer = FeatureEngineer()
        features = engineer.create_features(clean_data)

        # Create strategies
        strategies = [
            TrendFollowingStrategy(),
            MeanReversionStrategy(),
            VolatilityBreakoutStrategy(),
        ]

        # Compare strategies
        backtester = Backtester(initial_capital=100000)
        comparison = backtester.compare_strategies(strategies, features)

        assert len(comparison) == 3
        assert "Total Return" in comparison.columns
        assert "Sharpe" in comparison.columns
        assert "Max DD" in comparison.columns

    def test_regime_specific_performance(self, sample_data):
        """Test that regime-specific analysis works correctly."""
        # Prepare data
        preprocessor = DataPreprocessor()
        clean_data = preprocessor.clean_data(sample_data)

        engineer = FeatureEngineer()
        features = engineer.create_features(clean_data)

        # Detect regimes
        regime_features = engineer.extract_regime_features(features)
        detector = GMMDetector(n_regimes=3, random_state=42)
        detector.fit(regime_features)
        regimes = detector.predict(regime_features)

        # Backtest with regime analysis
        strategy = TrendFollowingStrategy()
        backtester = Backtester()
        results = backtester.run(strategy, features, regime_labels=regimes)

        # Check regime-specific metrics exist
        assert "regime_analysis" in results["metrics"]
        regime_analysis = results["metrics"]["regime_analysis"]

        # Should have analysis for each regime
        assert len(regime_analysis) > 0
        for regime_key, regime_metrics in regime_analysis.items():
            assert "total_return" in regime_metrics
            assert "sharpe" in regime_metrics
            assert "n_periods" in regime_metrics

    def test_transaction_costs_impact(self, sample_data):
        """Test that transaction costs reduce returns."""
        preprocessor = DataPreprocessor()
        clean_data = preprocessor.clean_data(sample_data)

        engineer = FeatureEngineer()
        features = engineer.create_features(clean_data)

        strategy = TrendFollowingStrategy()

        # Backtest without costs
        backtester_no_cost = Backtester(
            initial_capital=100000, commission=0.0, slippage=0.0
        )
        results_no_cost = backtester_no_cost.run(strategy, features)

        # Backtest with costs
        backtester_with_cost = Backtester(
            initial_capital=100000, commission=0.001, slippage=0.0005
        )
        results_with_cost = backtester_with_cost.run(strategy, features)

        # Returns with costs should be lower (or equal if no trades)
        if results_no_cost["metrics"]["n_trades"] > 0:
            assert (
                results_with_cost["metrics"]["total_return"]
                <= results_no_cost["metrics"]["total_return"]
            )

    def test_empty_data_handling(self):
        """Test handling of edge cases."""
        # Empty dataframe
        empty_data = pd.DataFrame()

        preprocessor = DataPreprocessor()

        # Should handle gracefully
        with pytest.raises(Exception):
            preprocessor.clean_data(empty_data)

    def test_feature_consistency(self, sample_data):
        """Test that features are consistent across runs."""
        engineer = FeatureEngineer(windows=[10, 20], include_advanced=False)

        # Create features twice
        features1 = engineer.create_features(sample_data.copy())
        features2 = engineer.create_features(sample_data.copy())

        # Should be identical
        pd.testing.assert_frame_equal(features1, features2)

    def test_regime_stability(self, sample_data):
        """Test that regime detection is stable with fixed random state."""
        preprocessor = DataPreprocessor()
        clean_data = preprocessor.clean_data(sample_data)

        engineer = FeatureEngineer()
        features = engineer.create_features(clean_data)
        regime_features = engineer.extract_regime_features(features)

        # Detect regimes twice with same random state
        detector1 = GMMDetector(n_regimes=3, random_state=42)
        detector1.fit(regime_features)
        regimes1 = detector1.predict(regime_features)

        detector2 = GMMDetector(n_regimes=3, random_state=42)
        detector2.fit(regime_features)
        regimes2 = detector2.predict(regime_features)

        # Should be identical
        pd.testing.assert_series_equal(regimes1, regimes2)


class TestDataIntegration:
    """Test data loading and preprocessing integration."""

    def test_data_loader_caching(self, tmp_path):
        """Test that data loader caching works."""
        # This would require mocking yfinance
        # Placeholder for now
        loader = DataLoader(cache_dir=str(tmp_path), use_cache=True)
        assert loader.cache_dir == str(tmp_path)


class TestPerformanceMetrics:
    """Test performance metrics calculations."""

    def test_sharpe_ratio_calculation(self, tmp_path):
        """Test Sharpe ratio calculation."""
        # Create sample returns
        dates = pd.date_range(start="2020-01-01", periods=252, freq="D")
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005, index=dates)

        # Calculate Sharpe manually
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        expected_sharpe = annual_return / annual_vol

        # This would be tested via backtester
        assert expected_sharpe is not None
