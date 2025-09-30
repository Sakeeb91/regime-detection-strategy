# API Reference

## Data Module

### DataLoader

Load financial market data from various sources.

```python
from src.data import DataLoader

loader = DataLoader(cache_dir="data/raw", use_cache=True)
data = loader.load_data('SPY', start_date='2020-01-01', end_date='2023-12-31')
```

**Methods:**
- `load_data(symbols, start_date, end_date, interval='1d')` - Load market data
- `get_available_symbols()` - List cached symbols

### DataPreprocessor

Clean and preprocess market data.

```python
from src.data import DataPreprocessor

preprocessor = DataPreprocessor(fill_method='forward')
clean_data = preprocessor.clean_data(raw_data)
is_valid, msg = preprocessor.validate_data(clean_data)
```

**Methods:**
- `clean_data(df, remove_outliers=True)` - Clean and preprocess data
- `validate_data(df)` - Validate data quality
- `resample_data(df, frequency='1W')` - Resample to different frequency
- `calculate_returns(df, method='log')` - Calculate returns

### FeatureEngineer

Create technical indicators and features for regime detection.

```python
from src.data import FeatureEngineer

engineer = FeatureEngineer(windows=[5, 10, 20, 50])
features = engineer.create_features(price_data)
regime_features = engineer.extract_regime_features(features)
```

**Methods:**
- `create_features(df, feature_groups=None)` - Create comprehensive feature set
- `extract_regime_features(df, n_features=None)` - Extract regime-relevant features
- `get_feature_names(df)` - Get feature names by category

---

## Regime Detection Module

### GMMDetector

Detect market regimes using Gaussian Mixture Models.

```python
from src.regime_detection import GMMDetector

detector = GMMDetector(n_regimes=3, random_state=42)
detector.fit(features)
regimes = detector.predict(features)
probas = detector.predict_proba(features)
```

**Methods:**
- `fit(features, optimize_n=False)` - Fit GMM to features
- `predict(features)` - Predict regime labels
- `predict_proba(features)` - Get regime probabilities
- `get_regime_statistics(features, returns=None)` - Calculate regime statistics
- `get_bic_aic(features)` - Get model selection scores

### HMMDetector

Detect market regimes using Hidden Markov Models.

```python
from src.regime_detection import HMMDetector

detector = HMMDetector(n_regimes=3, random_state=42)
detector.fit(features)
regimes = detector.predict(features)  # Viterbi algorithm
smooth_regimes = detector.predict_smooth(features)  # Forward-backward
```

**Methods:**
- `fit(features)` - Fit HMM to features
- `predict(features)` - Predict using Viterbi algorithm
- `predict_proba(features)` - Get regime probabilities
- `predict_smooth(features)` - Smoothed regime predictions
- `get_transition_matrix()` - Get transition probability matrix
- `get_stationary_distribution()` - Get equilibrium regime probabilities

### DTWClustering

Cluster time series patterns using Dynamic Time Warping.

```python
from src.regime_detection import DTWClustering

clusterer = DTWClustering(n_clusters=3, window_size=20)
clusterer.fit(returns_series)
regimes = clusterer.predict(returns_series)
```

**Methods:**
- `fit(series)` - Fit DTW clustering model
- `predict(series)` - Predict regime labels

### TransitionPredictor

Predict regime transitions using machine learning.

```python
from src.regime_detection import TransitionPredictor

predictor = TransitionPredictor(model_type='xgboost', lookahead=1)
metrics = predictor.fit(features, regimes)
future_regimes = predictor.predict(new_features)
importance = predictor.get_feature_importance()
```

**Methods:**
- `fit(features, regimes, test_size=0.2)` - Train prediction model
- `predict(features)` - Predict future regimes
- `predict_proba(features)` - Get transition probabilities
- `get_feature_importance()` - Get feature importance rankings

---

## Strategy Module

### BaseStrategy

Abstract base class for all trading strategies.

```python
from src.strategies import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, data):
        # Implementation
        return signals
```

### StrategySelector

Select optimal strategy based on detected regime.

```python
from src.strategies import StrategySelector

selector = StrategySelector(
    regime_strategy_map={0: 'trend', 1: 'mean_reversion', 2: 'neutral'}
)
strategy = selector.select_strategy(current_regime=0)
```

**Methods:**
- `select_strategy(regime)` - Select strategy for regime
- `get_strategy_series(regimes)` - Get strategy assignments for regime sequence

---

## Utils Module

### Plotting

Visualization utilities for regimes and performance.

```python
from src.utils import plot_regimes, plot_equity_curve

fig = plot_regimes(prices, regimes, save_path='plots/regimes.png')
fig = plot_equity_curve(returns_dict, save_path='plots/equity.png')
```

### Metrics

Performance metrics for strategy evaluation.

```python
from src.utils import calculate_sharpe_ratio, calculate_max_drawdown

sharpe = calculate_sharpe_ratio(returns)
max_dd = calculate_max_drawdown(returns)
```

**Functions:**
- `calculate_sharpe_ratio(returns, risk_free_rate=0.0)` - Sharpe ratio
- `calculate_max_drawdown(returns)` - Maximum drawdown
- `calculate_sortino_ratio(returns, risk_free_rate=0.0)` - Sortino ratio
- `calculate_calmar_ratio(returns)` - Calmar ratio

---

*Last Updated: 2025-09-30*