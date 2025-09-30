"""
Regime transition predictor using supervised learning.

Predicts regime transitions using Random Forest and XGBoost classifiers.
"""

from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class TransitionPredictor:
    """
    Predict regime transitions using machine learning.

    Trains classifiers to predict upcoming regime changes based on
    current features, enabling proactive strategy adjustments.

    Examples:
        >>> predictor = TransitionPredictor(model_type='xgboost')
        >>> predictor.fit(features, regime_labels)
        >>> transitions = predictor.predict(new_features)
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        lookahead: int = 1,
        random_state: int = 42,
    ):
        """
        Initialize transition predictor.

        Args:
            model_type: 'random_forest' or 'xgboost'
            lookahead: Number of periods ahead to predict
            random_state: Random seed
        """
        self.model_type = model_type
        self.lookahead = lookahead
        self.random_state = random_state

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

        logger.info(f"TransitionPredictor initialized: model_type={model_type}")

    def fit(
        self, features: pd.DataFrame, regimes: np.ndarray, test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Fit transition prediction model.

        Args:
            features: DataFrame of features
            regimes: Array of regime labels
            test_size: Fraction of data for testing

        Returns:
            Dictionary with training metrics
        """
        self.feature_names = list(features.columns)

        # Create target (future regime)
        y = np.roll(regimes, -self.lookahead)

        # Remove last lookahead samples (no future data)
        X = features.iloc[: -self.lookahead]
        y = y[: -self.lookahead]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, shuffle=False
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"Training {self.model_type} on {len(X_train)} samples")

        # Train model
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == "xgboost":
            self.model = XGBClassifier(
                n_estimators=100, max_depth=6, random_state=self.random_state, n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        logger.info(
            f"Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}"
        )

        return {"train_accuracy": train_score, "test_accuracy": test_score}

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict future regimes.

        Args:
            features: DataFrame of features

        Returns:
            Array of predicted regime labels
        """
        self._check_fitted()

        X_scaled = self.scaler.transform(features)
        predictions = self.model.predict(X_scaled)

        return predictions

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities.

        Args:
            features: DataFrame of features

        Returns:
            Array of probabilities for each regime
        """
        self._check_fitted()

        X_scaled = self.scaler.transform(features)
        probas = self.model.predict_proba(X_scaled)

        return probas

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            DataFrame with features ranked by importance
        """
        self._check_fitted()

        importances = self.model.feature_importances_
        feature_imp = pd.DataFrame(
            {"feature": self.feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        return feature_imp

    def _check_fitted(self):
        """Check if model is fitted."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
