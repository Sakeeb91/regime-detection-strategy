"""
Gaussian Mixture Model (GMM) for regime detection.

Uses unsupervised clustering to identify distinct market regimes
based on feature distributions.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class GMMDetector:
    """
    Detect market regimes using Gaussian Mixture Models.

    Fits GMM to feature space and identifies distinct market regimes
    (e.g., high volatility, low volatility, trending, mean-reverting).

    Attributes:
        n_regimes (int): Number of regimes to detect
        covariance_type (str): Type of covariance parameters
        scaler: StandardScaler for feature normalization

    Examples:
        >>> detector = GMMDetector(n_regimes=3)
        >>> detector.fit(features)
        >>> regimes = detector.predict(features)
        >>> probs = detector.predict_proba(features)
    """

    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = "full",
        random_state: int = 42,
        max_iter: int = 100,
    ):
        """
        Initialize the GMM regime detector.

        Args:
            n_regimes: Number of distinct regimes to identify
            covariance_type: 'full', 'tied', 'diag', or 'spherical'
            random_state: Random seed for reproducibility
            max_iter: Maximum number of EM iterations
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.max_iter = max_iter

        self.gmm = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

        logger.info(
            f"GMMDetector initialized: n_regimes={n_regimes}, "
            f"covariance_type={covariance_type}"
        )

    def fit(
        self,
        features: pd.DataFrame,
        n_init: int = 10,
        optimize_n: bool = False,
        n_range: Optional[Tuple[int, int]] = None,
    ) -> "GMMDetector":
        """
        Fit GMM to feature data.

        Args:
            features: DataFrame of features for regime detection
            n_init: Number of initializations to try
            optimize_n: Whether to optimize number of regimes using BIC
            n_range: Range of n_regimes to test if optimize_n=True (min, max)

        Returns:
            Self for method chaining

        Examples:
            >>> detector = GMMDetector()
            >>> detector.fit(features, optimize_n=True, n_range=(2, 5))
        """
        self.feature_names = list(features.columns)

        # Scale features
        X_scaled = self.scaler.fit_transform(features)

        logger.info(
            f"Fitting GMM on {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features"
        )

        # Optimize number of regimes if requested
        if optimize_n:
            if n_range is None:
                n_range = (2, min(10, X_scaled.shape[0] // 50))

            optimal_n = self._optimize_n_regimes(X_scaled, n_range, n_init)
            logger.info(f"Optimal number of regimes: {optimal_n}")
            self.n_regimes = optimal_n

        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=n_init,
        )

        self.gmm.fit(X_scaled)
        self.is_fitted = True

        logger.info(f"GMM fitting complete. Converged: {self.gmm.converged_}")

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict regime labels for given features.

        Args:
            features: DataFrame of features

        Returns:
            Array of regime labels (0 to n_regimes-1)

        Raises:
            ValueError: If model not fitted

        Examples:
            >>> regimes = detector.predict(test_features)
        """
        self._check_fitted()

        X_scaled = self.scaler.transform(features)
        regimes = self.gmm.predict(X_scaled)

        logger.debug(f"Predicted {len(regimes)} regime labels")

        return regimes

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities for given features.

        Args:
            features: DataFrame of features

        Returns:
            Array of shape (n_samples, n_regimes) with probabilities

        Examples:
            >>> probs = detector.predict_proba(test_features)
            >>> confidence = probs.max(axis=1)  # Get confidence scores
        """
        self._check_fitted()

        X_scaled = self.scaler.transform(features)
        probas = self.gmm.predict_proba(X_scaled)

        return probas

    def get_regime_statistics(
        self, features: pd.DataFrame, returns: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Calculate statistics for each regime.

        Args:
            features: DataFrame of features
            returns: Optional return series for performance metrics

        Returns:
            DataFrame with regime statistics

        Examples:
            >>> stats = detector.get_regime_statistics(features, returns)
            >>> print(stats)
        """
        self._check_fitted()

        regimes = self.predict(features)
        probas = self.predict_proba(features)

        stats_list = []

        for regime in range(self.n_regimes):
            regime_mask = regimes == regime

            stats = {
                "regime": regime,
                "count": regime_mask.sum(),
                "percentage": regime_mask.mean() * 100,
                "avg_confidence": probas[regime_mask, regime].mean(),
                "mean_duration": self._calculate_mean_duration(regimes, regime),
            }

            # Add feature statistics
            regime_features = features[regime_mask]
            for col in features.columns:
                stats[f"{col}_mean"] = regime_features[col].mean()
                stats[f"{col}_std"] = regime_features[col].std()

            # Add return statistics if provided
            if returns is not None:
                regime_returns = returns[regime_mask]
                stats["mean_return"] = regime_returns.mean()
                stats["std_return"] = regime_returns.std()
                stats["sharpe"] = (
                    regime_returns.mean() / regime_returns.std() * np.sqrt(252)
                    if regime_returns.std() > 0
                    else 0
                )

            stats_list.append(stats)

        return pd.DataFrame(stats_list)

    def get_bic_aic(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate BIC and AIC scores for model selection.

        Args:
            features: DataFrame of features

        Returns:
            Dictionary with BIC and AIC scores

        Examples:
            >>> scores = detector.get_bic_aic(features)
            >>> print(f"BIC: {scores['bic']:.2f}, AIC: {scores['aic']:.2f}")
        """
        self._check_fitted()

        X_scaled = self.scaler.transform(features)

        return {"bic": self.gmm.bic(X_scaled), "aic": self.gmm.aic(X_scaled)}

    def get_regime_transitions(self, regimes: np.ndarray) -> pd.DataFrame:
        """
        Calculate regime transition matrix.

        Args:
            regimes: Array of regime labels

        Returns:
            DataFrame showing transition probabilities between regimes

        Examples:
            >>> transitions = detector.get_regime_transitions(regimes)
            >>> print(transitions)
        """
        transitions = np.zeros((self.n_regimes, self.n_regimes))

        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]
            transitions[from_regime, to_regime] += 1

        # Normalize to get probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_probs = transitions / row_sums

        return pd.DataFrame(
            transition_probs,
            index=[f"From_{i}" for i in range(self.n_regimes)],
            columns=[f"To_{i}" for i in range(self.n_regimes)],
        )

    def _optimize_n_regimes(
        self, X: np.ndarray, n_range: Tuple[int, int], n_init: int
    ) -> int:
        """
        Find optimal number of regimes using BIC.

        Args:
            X: Feature matrix
            n_range: Range of components to test (min, max)
            n_init: Number of initializations per model

        Returns:
            Optimal number of regimes
        """
        bic_scores = []
        n_components_range = range(n_range[0], n_range[1] + 1)

        logger.info(f"Testing n_regimes from {n_range[0]} to {n_range[1]}")

        for n in n_components_range:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                max_iter=self.max_iter,
                n_init=n_init,
            )
            gmm.fit(X)
            bic = gmm.bic(X)
            bic_scores.append(bic)
            logger.debug(f"n={n}, BIC={bic:.2f}")

        # Find n with lowest BIC
        optimal_idx = np.argmin(bic_scores)
        optimal_n = n_components_range[optimal_idx]

        return optimal_n

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
        """
        Check if model is fitted.

        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
