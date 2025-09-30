"""
Unit tests for regime detection models.
"""

import pytest
import pandas as pd
import numpy as np
from src.regime_detection.gmm_detector import GMMDetector
from src.regime_detection.hmm_detector import HMMDetector


class TestGMMDetector:
    """Test suite for GMMDetector."""

    @pytest.fixture
    def sample_features(self):
        """Create sample features."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "volatility": np.random.randn(200) * 0.1 + 0.2,
                "returns": np.random.randn(200) * 0.02,
                "momentum": np.random.randn(200) * 0.5,
            }
        )

    @pytest.fixture
    def detector(self):
        """Create GMMDetector instance."""
        return GMMDetector(n_regimes=3, random_state=42)

    def test_initialization(self, detector):
        """Test GMMDetector initialization."""
        assert detector.n_regimes == 3
        assert detector.random_state == 42
        assert detector.is_fitted is False

    def test_fit(self, detector, sample_features):
        """Test model fitting."""
        detector.fit(sample_features)

        assert detector.is_fitted is True
        assert detector.gmm is not None

    def test_predict(self, detector, sample_features):
        """Test regime prediction."""
        detector.fit(sample_features)
        regimes = detector.predict(sample_features)

        assert len(regimes) == len(sample_features)
        assert regimes.min() >= 0
        assert regimes.max() < detector.n_regimes

    def test_predict_proba(self, detector, sample_features):
        """Test probability prediction."""
        detector.fit(sample_features)
        probas = detector.predict_proba(sample_features)

        assert probas.shape == (len(sample_features), detector.n_regimes)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_predict_not_fitted(self, detector, sample_features):
        """Test prediction without fitting."""
        with pytest.raises(ValueError):
            detector.predict(sample_features)


class TestHMMDetector:
    """Test suite for HMMDetector."""

    @pytest.fixture
    def sample_features(self):
        """Create sample features."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "volatility": np.random.randn(200) * 0.1 + 0.2,
                "returns": np.random.randn(200) * 0.02,
            }
        )

    @pytest.fixture
    def detector(self):
        """Create HMMDetector instance."""
        return HMMDetector(n_regimes=3, random_state=42)

    def test_initialization(self, detector):
        """Test HMMDetector initialization."""
        assert detector.n_regimes == 3
        assert detector.is_fitted is False

    def test_fit(self, detector, sample_features):
        """Test model fitting."""
        detector.fit(sample_features)

        assert detector.is_fitted is True
        assert detector.hmm is not None

    def test_predict(self, detector, sample_features):
        """Test regime prediction."""
        detector.fit(sample_features)
        regimes = detector.predict(sample_features)

        assert len(regimes) == len(sample_features)
        assert regimes.min() >= 0
        assert regimes.max() < detector.n_regimes

    def test_get_transition_matrix(self, detector, sample_features):
        """Test transition matrix retrieval."""
        detector.fit(sample_features)
        trans_matrix = detector.get_transition_matrix()

        assert trans_matrix.shape == (detector.n_regimes, detector.n_regimes)
        assert np.allclose(trans_matrix.sum(axis=1), 1.0)
