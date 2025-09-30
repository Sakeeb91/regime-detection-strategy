# Testing Guide

## Overview

This project uses `pytest` for unit and integration testing. Our goal is to maintain **>80% test coverage** for all critical functionality.

## Running Tests

### Run All Tests
```bash
pytest
```

### Run with Coverage Report
```bash
pytest --cov=src --cov-report=html --cov-report=term
```

### Run Specific Test File
```bash
pytest tests/unit/test_data_loader.py
```

### Run Tests with Verbose Output
```bash
pytest -v
```

### Run Tests in Parallel (faster)
```bash
pytest -n auto
```

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual modules
│   ├── test_data_loader.py
│   ├── test_feature_engineer.py
│   ├── test_regime_detectors.py
│   └── test_strategies.py
└── integration/             # Integration tests for complete workflows
    ├── test_end_to_end.py
    └── test_pipeline.py
```

## Writing Tests

### Test Naming Convention
- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<functionality>`

### Example Test
```python
import pytest
from src.data.data_loader import DataLoader

class TestDataLoader:
    @pytest.fixture
    def loader(self):
        return DataLoader(use_cache=False)

    def test_load_data(self, loader):
        data = loader.load_data('SPY', start_date='2020-01-01')
        assert not data.empty
```

## Test Coverage Goals

| Module | Target Coverage | Current |
|--------|----------------|---------|
| data/ | 85% | TBD |
| regime_detection/ | 80% | TBD |
| strategies/ | 80% | TBD |
| utils/ | 90% | TBD |

## Continuous Integration

Tests are automatically run on every push and pull request via GitHub Actions.

See `.github/workflows/tests.yml` for CI configuration.

## Mocking External Dependencies

For tests that require external API calls (e.g., Yahoo Finance), use mocking:

```python
from unittest.mock import patch, MagicMock

def test_load_data_with_mock():
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_df
        # Test code here
```

## Performance Testing

For computationally intensive tests, use benchmarking:

```python
def test_model_performance(benchmark):
    result = benchmark(detector.fit, features)
```

## Test Data

Sample test data is located in `tests/fixtures/`. Do not commit large data files.

---

*Last Updated: 2025-09-30*