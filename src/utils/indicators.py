"""
Technical indicators module providing fallback implementations.

Provides basic technical indicators when pandas_ta is not available.
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, length: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()

    rs = gain / loss
    rsi_values = 100 - (100 / (1 + rs))
    return rsi_values


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.Series:
    """Calculate Average True Range."""
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_values = true_range.rolling(window=length).mean()
    return atr_values


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.DataFrame:
    """Calculate Average Directional Index."""
    # Calculate directional movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

    # Calculate true range
    tr = atr(high, low, close, length=1)

    # Calculate directional indicators
    plus_di = 100 * (
        plus_dm.rolling(window=length).mean() / tr.rolling(window=length).mean()
    )
    minus_di = 100 * (
        minus_dm.rolling(window=length).mean() / tr.rolling(window=length).mean()
    )

    # Calculate ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_values = dx.rolling(window=length).mean()

    return pd.DataFrame({f"ADX_{length}": adx_values}, index=close.index)


def bbands(series: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    middle = sma(series, length)
    rolling_std = series.rolling(window=length).std()

    upper = middle + (rolling_std * std)
    lower = middle - (rolling_std * std)
    width = (upper - lower) / middle

    return pd.DataFrame(
        {
            f"BBU_{length}_{std}": upper,
            f"BBM_{length}_{std}": middle,
            f"BBL_{length}_{std}": lower,
            f"BBB_{length}_{std}": width,
        },
        index=series.index,
    )


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Calculate MACD."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line

    return pd.DataFrame(
        {
            f"MACD_{fast}_{slow}_{signal}": macd_line,
            f"MACDs_{fast}_{slow}_{signal}": signal_line,
            f"MACDh_{fast}_{slow}_{signal}": hist,
        },
        index=series.index,
    )


def roc(series: pd.Series, length: int) -> pd.Series:
    """Calculate Rate of Change."""
    return ((series - series.shift(length)) / series.shift(length)) * 100


def stoch(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3
) -> pd.DataFrame:
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(window=k).min()
    highest_high = high.rolling(window=k).max()

    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d).mean()

    return pd.DataFrame(
        {f"STOCHk_{k}_3_3": stoch_k, f"STOCHd_{k}_3_3": stoch_d}, index=close.index
    )


def cci(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20
) -> pd.Series:
    """Calculate Commodity Channel Index."""
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=length).mean()
    mean_deviation = (typical_price - sma_tp).abs().rolling(window=length).mean()

    cci_values = (typical_price - sma_tp) / (0.015 * mean_deviation)
    return cci_values


def willr(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.Series:
    """Calculate Williams %R."""
    highest_high = high.rolling(window=length).max()
    lowest_low = low.rolling(window=length).min()

    willr_values = -100 * (highest_high - close) / (highest_high - lowest_low)
    return willr_values


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume."""
    obv_values = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv_values
