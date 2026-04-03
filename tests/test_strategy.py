from __future__ import annotations

import pandas as pd
import pytest

from backtesting.strategy import (
    BuyAndHoldStrategy,
    MovingAverageCrossoverStrategy,
    PrecomputedSignalStrategy,
)


def test_buy_and_hold_signals_enter_after_first_row() -> None:
    data = pd.DataFrame({"close": [100, 101, 102]})

    signals = BuyAndHoldStrategy().generate_signals(data)

    assert list(signals) == [0, 1, 1]


def test_moving_average_crossover_generates_expected_signal_shape() -> None:
    data = pd.DataFrame({"close": [10, 11, 12, 13, 12, 11]})

    strategy = MovingAverageCrossoverStrategy(short_window=2, long_window=3)
    signals = strategy.generate_signals(data)

    assert len(signals) == len(data)
    assert set(signals.unique()).issubset({0, 1})


def test_precomputed_signal_strategy_validates_length() -> None:
    strategy = PrecomputedSignalStrategy(pd.Series([1, 0]))
    data = pd.DataFrame({"close": [100, 101, 102]})

    with pytest.raises(ValueError, match="must match the input data length"):
        strategy.generate_signals(data)
