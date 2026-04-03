from __future__ import annotations

import pandas as pd
import pytest

from backtesting.backtester import Backtester
from backtesting.strategy import BuyAndHoldStrategy, PrecomputedSignalStrategy


def build_price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "open": [100.0, 100.0, 100.0, 110.0],
            "high": [101.0, 111.0, 121.0, 121.0],
            "low": [99.0, 99.0, 99.0, 109.0],
            "close": [100.0, 110.0, 120.0, 110.0],
            "volume": [1000, 1000, 1000, 1000],
            "returns": [0.0, 0.1, 0.0909090909, -0.0833333333],
        }
    )


def test_backtester_runs_buy_and_hold_and_generates_trades() -> None:
    result = Backtester(initial_capital=1000.0, transaction_cost=0.0).run(
        build_price_frame(),
        BuyAndHoldStrategy(),
        asset_name="demo_asset",
    )

    assert result.asset_name == "demo_asset"
    assert result.strategy_name == "buy_and_hold"
    assert len(result.history) == 4
    assert len(result.trades) == 2
    assert result.trades["action"].tolist() == ["BUY", "SELL_END"]
    assert result.metrics["final_equity"] == pytest.approx(1100.0)
    assert result.metrics["num_trades"] == 2
    assert result.metrics["win_rate"] == pytest.approx(1.0)


def test_backtester_handles_no_trade_strategy() -> None:
    signals = pd.Series([0, 0, 0, 0], dtype=int)
    strategy = PrecomputedSignalStrategy(signals, name="flat_strategy")

    result = Backtester(initial_capital=1000.0, transaction_cost=0.0).run(
        build_price_frame(),
        strategy,
    )

    assert result.metrics["final_equity"] == pytest.approx(1000.0)
    assert result.metrics["num_trades"] == 0
    assert result.metrics["exposure_rate"] == pytest.approx(0.0)
    assert result.metrics["win_rate"] == pytest.approx(0.0)


def test_backtester_rejects_signal_length_mismatch() -> None:
    bad_strategy = PrecomputedSignalStrategy(pd.Series([1, 0]), name="bad")

    with pytest.raises(ValueError, match="must match the input data length"):
        Backtester().run(build_price_frame(), bad_strategy)
