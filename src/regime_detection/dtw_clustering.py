"""
Dynamic Time Warping (DTW) based clustering for regime detection.

Uses DTW distance metric with K-means clustering to identify
similar market patterns across time.
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


class DTWClustering:
    """
    Detect market regimes using DTW-based clustering.

    Clusters time series windows using Dynamic Time Warping distance,
    identifying similar market patterns regardless of phase shifts.

    Examples:
        >>> clusterer = DTWClustering(n_clusters=3, window_size=20)
        >>> clusterer.fit(returns_series)
        >>> regimes = clusterer.predict(returns_series)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        window_size: int = 20,
        metric: str = "dtw",
        random_state: int = 42
    ):
        """
        Initialize DTW clustering detector.

        Args:
            n_clusters: Number of clusters (regimes)
            window_size: Size of rolling window for pattern extraction
            metric: Distance metric ('dtw' or 'softdtw')
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.window_size = window_size
        self.metric = metric
        self.random_state = random_state

        self.model = None
        self.scaler = TimeSeriesScalerMeanVariance()
        self.is_fitted = False

        logger.info(
            f"DTWClustering initialized: n_clusters={n_clusters}, "
            f"window_size={window_size}"
        )

    def fit(self, series: pd.Series) -> 'DTWClustering':
        """
        Fit DTW clustering model.

        Args:
            series: Time series data (e.g., returns or volatility)

        Returns:
            Self for method chaining
        """
        # Create rolling windows
        windows = self._create_windows(series)

        logger.info(f"Fitting DTW clustering on {len(windows)} windows")

        # Scale windows
        windows_scaled = self.scaler.fit_transform(windows)

        # Fit time series K-means with DTW
        self.model = TimeSeriesKMeans(
            n_clusters=self.n_clusters,
            metric=self.metric,
            random_state=self.random_state,
            verbose=False
        )

        self.model.fit(windows_scaled)
        self.is_fitted = True

        logger.info("DTW clustering complete")

        return self

    def predict(self, series: pd.Series) -> np.ndarray:
        """
        Predict regime labels for time series.

        Args:
            series: Time series data

        Returns:
            Array of regime labels
        """
        self._check_fitted()

        windows = self._create_windows(series)
        windows_scaled = self.scaler.transform(windows)

        labels = self.model.predict(windows_scaled)

        # Extend labels to match original series length
        full_labels = self._extend_labels(labels, len(series))

        return full_labels

    def _create_windows(self, series: pd.Series) -> np.ndarray:
        """Create rolling windows from time series."""
        values = series.values
        windows = []

        for i in range(len(values) - self.window_size + 1):
            window = values[i:i + self.window_size]
            windows.append(window)

        return np.array(windows).reshape(-1, self.window_size, 1)

    def _extend_labels(self, labels: np.ndarray, target_length: int) -> np.ndarray:
        """Extend window labels to match original series length."""
        extended = np.full(target_length, -1)

        # First window_size-1 points get label of first window
        extended[:self.window_size-1] = labels[0]

        # Rest get labels from windows
        for i, label in enumerate(labels):
            extended[i + self.window_size - 1] = label

        return extended

    def _check_fitted(self):
        """Check if model is fitted."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")