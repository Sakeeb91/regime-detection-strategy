"""
Hidden Markov Model (HMM) for regime detection.

Uses HMM to model market regimes as hidden states with transition
probabilities, allowing for temporal regime dynamics.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn import hmm
from loguru import logger
from sklearn.preprocessing import StandardScaler


class HMMDetector:
    """
    Detect market regimes using Hidden Markov Models.

    Models market regimes as hidden states with Gaussian emissions,
    capturing temporal dynamics and regime persistence.

    Attributes:
        n_regimes (int): Number of hidden states (regimes)
        covariance_type (str): Type of covariance ('full', 'diag', 'spherical')
        scaler: StandardScaler for feature normalization

    Examples:
        >>> detector = HMMDetector(n_regimes=3)
        >>> detector.fit(features)
        >>> regimes = detector.predict(features)
        >>> smooth_regimes = detector.predict_smooth(features)
    """

    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42
    ):
        """
        Initialize the HMM regime detector.

        Args:
            n_regimes: Number of hidden states (regimes)
            covariance_type: 'full', 'diag', or 'spherical'
            n_iter: Maximum number of EM iterations
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.hmm = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

        logger.info(
            f"HMMDetector initialized: n_regimes={n_regimes}, "
            f"covariance_type={covariance_type}"
        )

    def fit(
        self,
        features: pd.DataFrame,
        lengths: Optional[np.ndarray] = None
    ) -> 'HMMDetector':
        """
        Fit HMM to feature data.

        Args:
            features: DataFrame of features for regime detection
            lengths: Optional array of sequence lengths for multiple sequences

        Returns:
            Self for method chaining

        Examples:
            >>> detector = HMMDetector(n_regimes=3)
            >>> detector.fit(features)
        """
        self.feature_names = list(features.columns)

        # Scale features
        X_scaled = self.scaler.fit_transform(features)

        logger.info(
            f"Fitting HMM on {X_scaled.shape[0]} samples, "
            f"{X_scaled.shape[1]} features"
        )

        # Initialize HMM
        self.hmm = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False
        )

        # Fit model
        if lengths is not None:
            self.hmm.fit(X_scaled, lengths=lengths)
        else:
            self.hmm.fit(X_scaled)

        self.is_fitted = True

        logger.info("HMM fitting complete")

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict most likely regime sequence using Viterbi algorithm.

        Args:
            features: DataFrame of features

        Returns:
            Array of regime labels (0 to n_regimes-1)

        Examples:
            >>> regimes = detector.predict(test_features)
        """
        self._check_fitted()

        X_scaled = self.scaler.transform(features)
        regimes = self.hmm.predict(X_scaled)

        logger.debug(f"Predicted {len(regimes)} regime labels using Viterbi")

        return regimes

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities using forward-backward algorithm.

        Args:
            features: DataFrame of features

        Returns:
            Array of shape (n_samples, n_regimes) with probabilities

        Examples:
            >>> probs = detector.predict_proba(test_features)
        """
        self._check_fitted()

        X_scaled = self.scaler.transform(features)
        probas = self.hmm.predict_proba(X_scaled)

        return probas

    def predict_smooth(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict smoothed regime sequence (most likely at each time point).

        Uses forward-backward algorithm for smoothing, providing
        more stable regime assignments than Viterbi.

        Args:
            features: DataFrame of features

        Returns:
            Array of regime labels

        Examples:
            >>> smooth_regimes = detector.predict_smooth(test_features)
        """
        probas = self.predict_proba(features)
        smooth_regimes = probas.argmax(axis=1)

        return smooth_regimes

    def score(self, features: pd.DataFrame) -> float:
        """
        Calculate log-likelihood of observations under the model.

        Args:
            features: DataFrame of features

        Returns:
            Log-likelihood score

        Examples:
            >>> ll = detector.score(test_features)
            >>> print(f"Log-likelihood: {ll:.2f}")
        """
        self._check_fitted()

        X_scaled = self.scaler.transform(features)
        return self.hmm.score(X_scaled)

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get regime transition probability matrix.

        Returns:
            DataFrame with transition probabilities

        Examples:
            >>> trans = detector.get_transition_matrix()
            >>> print(trans)
        """
        self._check_fitted()

        return pd.DataFrame(
            self.hmm.transmat_,
            index=[f"From_{i}" for i in range(self.n_regimes)],
            columns=[f"To_{i}" for i in range(self.n_regimes)]
        )

    def get_stationary_distribution(self) -> np.ndarray:
        """
        Calculate stationary distribution of regimes.

        Returns equilibrium probabilities for each regime.

        Returns:
            Array of stationary probabilities

        Examples:
            >>> stationary = detector.get_stationary_distribution()
            >>> print(f"Long-run regime probabilities: {stationary}")
        """
        self._check_fitted()

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(self.hmm.transmat_.T)

        # Find eigenvector corresponding to eigenvalue of 1
        idx = np.argmax(np.abs(eigenvalues - 1.0) < 1e-8)
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()

        return stationary

    def get_regime_statistics(
        self,
        features: pd.DataFrame,
        returns: Optional[pd.Series] = None,
        use_smooth: bool = True
    ) -> pd.DataFrame:
        """
        Calculate statistics for each regime.

        Args:
            features: DataFrame of features
            returns: Optional return series for performance metrics
            use_smooth: Use smoothed predictions instead of Viterbi

        Returns:
            DataFrame with regime statistics

        Examples:
            >>> stats = detector.get_regime_statistics(features, returns)
        """
        self._check_fitted()

        if use_smooth:
            regimes = self.predict_smooth(features)
        else:
            regimes = self.predict(features)

        probas = self.predict_proba(features)

        stats_list = []

        for regime in range(self.n_regimes):
            regime_mask = regimes == regime

            stats = {
                'regime': regime,
                'count': regime_mask.sum(),
                'percentage': regime_mask.mean() * 100,
                'avg_confidence': probas[regime_mask, regime].mean(),
                'mean_duration': self._calculate_mean_duration(regimes, regime)
            }

            # Add feature statistics
            regime_features = features[regime_mask]
            for col in features.columns:
                stats[f'{col}_mean'] = regime_features[col].mean()
                stats[f'{col}_std'] = regime_features[col].std()

            # Add return statistics
            if returns is not None:
                regime_returns = returns[regime_mask]
                stats['mean_return'] = regime_returns.mean()
                stats['std_return'] = regime_returns.std()
                stats['sharpe'] = (
                    regime_returns.mean() / regime_returns.std() * np.sqrt(252)
                    if regime_returns.std() > 0 else 0
                )

            stats_list.append(stats)

        return pd.DataFrame(stats_list)

    def get_expected_duration(self) -> np.ndarray:
        """
        Calculate expected duration for each regime.

        Based on self-transition probabilities.

        Returns:
            Array of expected durations (in periods)

        Examples:
            >>> durations = detector.get_expected_duration()
            >>> for i, d in enumerate(durations):
            ...     print(f"Regime {i}: {d:.1f} periods")
        """
        self._check_fitted()

        # Expected duration = 1 / (1 - self_transition_prob)
        self_trans_probs = np.diag(self.hmm.transmat_)
        expected_durations = 1.0 / (1.0 - self_trans_probs)

        return expected_durations

    def simulate(
        self,
        n_samples: int,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate regime sequence and observations from the fitted model.

        Args:
            n_samples: Number of samples to generate
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (observations, regimes)

        Examples:
            >>> obs, regimes = detector.simulate(n_samples=1000)
        """
        self._check_fitted()

        if random_state is not None:
            np.random.seed(random_state)

        observations, regimes = self.hmm.sample(n_samples)

        # Inverse transform observations
        observations = self.scaler.inverse_transform(observations)

        return observations, regimes

    def _calculate_mean_duration(self, regimes: np.ndarray, regime: int) -> float:
        """
        Calculate mean duration of regime in periods.

        Args:
            regimes: Array of regime labels
            regime: Regime to calculate duration for

        Returns:
            Mean duration in periods
        """
        durations = []
        current_duration = 0

        for r in regimes:
            if r == regime:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        return np.mean(durations) if durations else 0.0

    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")