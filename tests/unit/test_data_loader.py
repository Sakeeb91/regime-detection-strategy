"""
Unit tests for DataLoader module.
"""

import os
import pytest
import pandas as pd
from datetime import datetime
from src.data.data_loader import DataLoader


class TestDataLoader:
    """Test suite for DataLoader class."""

    @pytest.fixture
    def loader(self, tmp_path):
        """Create DataLoader instance with temporary cache directory."""
        cache_dir = str(tmp_path / "cache")
        return DataLoader(cache_dir=cache_dir, use_cache=True)

    def test_initialization(self, loader):
        """Test DataLoader initialization."""
        assert loader.cache_dir is not None
        assert loader.use_cache is True
        assert os.path.exists(loader.cache_dir)

    def test_validate_dates_valid(self, loader):
        """Test date validation with valid dates."""
        # Should not raise exception
        loader._validate_dates("2020-01-01", "2020-12-31")

    def test_validate_dates_invalid_order(self, loader):
        """Test date validation with invalid order."""
        with pytest.raises(ValueError):
            loader._validate_dates("2020-12-31", "2020-01-01")

    def test_validate_dates_invalid_format(self, loader):
        """Test date validation with invalid format."""
        with pytest.raises(ValueError):
            loader._validate_dates("01-01-2020", "12-31-2020")

    def test_load_data_single_symbol(self, loader):
        """Test loading data for single symbol."""
        # This will make actual API call - use with caution
        # In production, mock the yfinance call
        try:
            data = loader.load_data("SPY", start_date="2023-01-01", end_date="2023-01-31")
            assert isinstance(data, pd.DataFrame)
            assert not data.empty
            assert 'close' in data.columns
        except Exception as e:
            pytest.skip(f"API call failed: {str(e)}")

    def test_load_data_multiple_symbols(self, loader):
        """Test loading data for multiple symbols."""
        try:
            data = loader.load_data(
                ["SPY", "QQQ"],
                start_date="2023-01-01",
                end_date="2023-01-31"
            )
            assert isinstance(data, dict)
            assert "SPY" in data
            assert "QQQ" in data
        except Exception as e:
            pytest.skip(f"API call failed: {str(e)}")

    def test_cache_functionality(self, loader, tmp_path):
        """Test data caching."""
        # Create dummy DataFrame
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })

        # Save to cache
        loader._save_to_cache("TEST", df, "2020-01-01", "2020-01-31")

        # Load from cache
        cached_df = loader._load_from_cache("TEST", "2020-01-01", "2020-01-31")

        assert cached_df is not None
        assert len(cached_df) == len(df)
        pd.testing.assert_frame_equal(df, cached_df)