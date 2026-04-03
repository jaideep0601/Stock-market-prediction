"""Reusable service helpers for running backtests from scripts or APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .backtester import Backtester, BacktestResult
from .data_loader import load_price_data
from .ml import generate_ml_signals
from .strategy import (
    BuyAndHoldStrategy,
    MovingAverageCrossoverStrategy,
    PrecomputedSignalStrategy,
    RSIStrategy,
)


def list_datasets(data_dir: str | Path = "data") -> list[dict[str, str]]:
    """Return available CSV datasets under the data directory."""
    base_dir = Path(data_dir)
    if not base_dir.exists():
        return []

    datasets: list[dict[str, str]] = []
    for csv_path in sorted(base_dir.glob("*.csv")):
        datasets.append(
            {
                "name": csv_path.stem,
                "path": str(csv_path),
            }
        )
    return datasets


def run_backtest_service(
    csv_path: str | Path,
    strategy_name: str,
    *,
    compare_buy_hold: bool = True,
    capital: float = 10_000.0,
    cost: float = 0.001,
    short_window: int = 20,
    long_window: int = 50,
    rsi_period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    ml_model: str = "logistic",
    train_split: float = 0.7,
    threshold: float = 0.5,
    validation_mode: str = "walk_forward",
    min_train_size: int = 252,
    step_size: int = 21,
) -> dict[str, object]:
    """Run a backtest and return a JSON-serializable payload."""
    data_path = Path(csv_path)
    data = load_price_data(data_path)
    asset_name = data_path.stem
    backtester = Backtester(initial_capital=capital, transaction_cost=cost)

    results: list[BacktestResult] = []
    strategy, ml_metadata = _build_strategy_and_metadata(
        data=data,
        strategy_name=strategy_name,
        short_window=short_window,
        long_window=long_window,
        rsi_period=rsi_period,
        oversold=oversold,
        overbought=overbought,
        ml_model=ml_model,
        train_split=train_split,
        threshold=threshold,
        validation_mode=validation_mode,
        min_train_size=min_train_size,
        step_size=step_size,
    )

    primary_result = backtester.run(data, strategy, asset_name=asset_name)
    if ml_metadata is not None:
        primary_result.metrics["ml_accuracy"] = ml_metadata["accuracy"]
        primary_result.metrics["ml_validation_mode"] = ml_metadata["validation_mode"]
        primary_result.metrics["ml_num_predictions"] = ml_metadata["num_predictions"]
    results.append(primary_result)

    if compare_buy_hold and strategy.name != "buy_and_hold":
        results.append(backtester.run(data, BuyAndHoldStrategy(), asset_name=asset_name))

    return {
        "asset": asset_name,
        "strategy": strategy.name,
        "ml_metadata": ml_metadata,
        "results": [_serialize_result(result) for result in results],
    }


def run_multi_backtest_service(
    csv_path: str | Path,
    strategies: list[str],
    *,
    capital: float = 10_000.0,
    cost: float = 0.001,
    short_window: int = 20,
    long_window: int = 50,
    rsi_period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    ml_model: str = "logistic",
    train_split: float = 0.7,
    threshold: float = 0.5,
    validation_mode: str = "walk_forward",
    min_train_size: int = 252,
    step_size: int = 21,
) -> dict[str, object]:
    """Run multiple strategies on the same dataset for comparison."""
    if not strategies:
        raise ValueError("At least one strategy must be provided")

    data_path = Path(csv_path)
    data = load_price_data(data_path)
    asset_name = data_path.stem
    backtester = Backtester(initial_capital=capital, transaction_cost=cost)

    results: list[BacktestResult] = []
    ml_metadata: list[dict[str, Any]] = []
    seen_strategy_names: set[str] = set()

    for requested_name in strategies:
        strategy, strategy_ml_metadata = _build_strategy_and_metadata(
            data=data,
            strategy_name=requested_name,
            short_window=short_window,
            long_window=long_window,
            rsi_period=rsi_period,
            oversold=oversold,
            overbought=overbought,
            ml_model=ml_model,
            train_split=train_split,
            threshold=threshold,
            validation_mode=validation_mode,
            min_train_size=min_train_size,
            step_size=step_size,
        )

        if strategy.name in seen_strategy_names:
            continue
        seen_strategy_names.add(strategy.name)

        result = backtester.run(data, strategy, asset_name=asset_name)
        if strategy_ml_metadata is not None:
            result.metrics["ml_accuracy"] = strategy_ml_metadata["accuracy"]
            result.metrics["ml_validation_mode"] = strategy_ml_metadata["validation_mode"]
            result.metrics["ml_num_predictions"] = strategy_ml_metadata["num_predictions"]
            ml_metadata.append(strategy_ml_metadata)
        results.append(result)

    return {
        "asset": asset_name,
        "strategies": [result.strategy_name for result in results],
        "ml_metadata": ml_metadata,
        "results": [_serialize_result(result) for result in results],
    }


def _build_strategy_and_metadata(
    *,
    data,
    strategy_name: str,
    short_window: int,
    long_window: int,
    rsi_period: int,
    oversold: float,
    overbought: float,
    ml_model: str,
    train_split: float,
    threshold: float,
    validation_mode: str,
    min_train_size: int,
    step_size: int,
):
    """Build a strategy object and optional ML metadata from request parameters."""
    ml_metadata = None

    if strategy_name == "buy_hold":
        strategy = BuyAndHoldStrategy()
    elif strategy_name == "ma":
        strategy = MovingAverageCrossoverStrategy(short_window=short_window, long_window=long_window)
    elif strategy_name == "rsi":
        strategy = RSIStrategy(period=rsi_period, oversold=oversold, overbought=overbought)
    elif strategy_name == "ml":
        ml_result = generate_ml_signals(
            data,
            model_type=ml_model,
            train_split=train_split,
            probability_threshold=threshold,
            validation_mode=validation_mode,
            min_train_size=min_train_size,
            step_size=step_size,
        )
        strategy = PrecomputedSignalStrategy(ml_result.signals, name=ml_result.model_name)
        ml_metadata = {
            "requested_name": strategy_name,
            "model_name": ml_result.model_name,
            "accuracy": ml_result.accuracy,
            "validation_mode": ml_result.validation_mode,
            "test_start_index": ml_result.test_start_index,
            "num_predictions": ml_result.num_predictions,
        }
    else:
        raise ValueError(f"Unsupported strategy: {strategy_name}")

    return strategy, ml_metadata


def _serialize_result(result: BacktestResult) -> dict[str, object]:
    """Convert a backtest result into a JSON-serializable dictionary."""
    history = result.history.copy()
    trades = result.trades.copy()

    history["date"] = history["date"].dt.strftime("%Y-%m-%d")
    if not trades.empty and "date" in trades.columns:
        trades["date"] = trades["date"].dt.strftime("%Y-%m-%d")

    history_points = history[["date", "equity", "drawdown", "position"]].to_dict(orient="records")
    trade_points = trades.to_dict(orient="records")

    return {
        "asset_name": result.asset_name,
        "strategy_name": result.strategy_name,
        "metrics": result.metrics,
        "history": history_points,
        "trades": trade_points,
    }
