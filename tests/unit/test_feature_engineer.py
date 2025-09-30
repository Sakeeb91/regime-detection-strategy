"""
Unit tests for FeatureEngineer module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        data = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(200) * 0.5),
            'high': 102 + np.cumsum(np.random.randn(200) * 0.5),
            'low': 98 + np.cumsum(np.random.randn(200) * 0.5),
            'close': 100 + np.cumsum(np.random.randn(200) * 0.5),
            'volume': np.random.randint(1000000, 5000000, 200)
        }, index=dates)
        return data

    @pytest.fixture
    def engineer(self):
        """Create FeatureEngineer instance."""
        return FeatureEngineer(windows=[5, 10, 20])

    def test_initialization(self, engineer):
        """Test FeatureEngineer initialization."""
        assert engineer.windows == [5, 10, 20]
        assert engineer.include_advanced is True

    def test_create_features(self, engineer, sample_data):
        """Test feature creation."""
        features = engineer.create_features(sample_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) <= len(sample_data)
        assert 'returns' in features.columns
        assert 'volatility_20' in features.columns

    def test_trend_features(self, engineer, sample_data):
        """Test trend feature creation."""
        features = engineer.create_features(
            sample_data,
            feature_groups=['trend']
        )

        assert 'sma_5' in features.columns
        assert 'ema_10' in features.columns
        assert 'macd' in features.columns

    def test_volatility_features(self, engineer, sample_data):
        """Test volatility feature creation."""
        features = engineer.create_features(
            sample_data,
            feature_groups=['volatility']
        )

        assert 'volatility_20' in features.columns
        assert 'atr_14' in features.columns

    def test_momentum_features(self, engineer, sample_data):
        """Test momentum feature creation."""
        features = engineer.create_features(
            sample_data,
            feature_groups=['momentum']
        )

        assert 'rsi_14' in features.columns
        assert 'momentum_5' in features.columns

    def test_extract_regime_features(self, engineer, sample_data):
        """Test regime feature extraction."""
        all_features = engineer.create_features(sample_data)
        regime_features = engineer.extract_regime_features(all_features)

        assert isinstance(regime_features, pd.DataFrame)
        assert len(regime_features.columns) < len(all_features.columns)
        assert 'returns' in regime_features.columns

    def test_get_feature_names(self, engineer, sample_data):
        """Test feature name grouping."""
        features = engineer.create_features(sample_data)
        feature_groups = engineer.get_feature_names(features)

        assert isinstance(feature_groups, dict)
        assert 'trend' in feature_groups
        assert 'volatility' in feature_groups
        assert 'momentum' in feature_groups