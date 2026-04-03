"""Trading strategy definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Base class for all trading strategies."""

    name = "base_strategy"

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return a long/flat signal series aligned to the input data."""


class BuyAndHoldStrategy(Strategy):
    """Enter once after the first signalable bar and hold until the end."""

    name = "buy_and_hold"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(1, index=data.index, dtype=int, name="signal")
        if not signals.empty:
            signals.iloc[0] = 0
        return signals


class MovingAverageCrossoverStrategy(Strategy):
    """Long when the short moving average is above the long moving average."""

    name = "moving_average_crossover"

    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        short_ma = close.rolling(window=self.short_window, min_periods=self.short_window).mean()
        long_ma = close.rolling(window=self.long_window, min_periods=self.long_window).mean()

        signals = (short_ma > long_ma).astype(int).fillna(0)
        return pd.Series(signals, index=data.index, name="signal")


class RSIStrategy(Strategy):
    """Long when RSI is below the oversold threshold, otherwise flat."""

    name = "rsi_strategy"

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> None:
        if period <= 0:
            raise ValueError("period must be positive")
        if oversold >= overbought:
            raise ValueError("oversold must be smaller than overbought")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close_delta = data["close"].diff()
        gains = close_delta.clip(lower=0)
        losses = -close_delta.clip(upper=0)

        avg_gain = gains.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()

        relative_strength = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + relative_strength))
        rsi = rsi.fillna(50.0)

        signals = pd.Series(0, index=data.index, dtype=int, name="signal")
        in_position = False

        for index, value in rsi.items():
            if not in_position and value <= self.oversold:
                in_position = True
            elif in_position and value >= self.overbought:
                in_position = False
            signals.loc[index] = int(in_position)

        return signals


class PrecomputedSignalStrategy(Strategy):
    """Wrap a precomputed signal series so it can run through the backtester."""

    def __init__(self, signals: pd.Series, name: str = "precomputed_signal_strategy") -> None:
        self.signals = signals.astype(int)
        self.name = name

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if len(self.signals) != len(data):
            raise ValueError("Precomputed signals must match the input data length")
        return pd.Series(self.signals, index=data.index, name="signal")
