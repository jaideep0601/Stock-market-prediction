"""Core package for the stock backtesting project."""

from .backtester import BacktestResult, Backtester
from .data_loader import load_price_data
from .ml import MLSignalResult, build_feature_frame, generate_ml_signals
from .service import list_datasets, run_backtest_service
from .strategy import (
    BuyAndHoldStrategy,
    MovingAverageCrossoverStrategy,
    PrecomputedSignalStrategy,
    RSIStrategy,
    Strategy,
)

__all__ = [
    "BacktestResult",
    "Backtester",
    "BuyAndHoldStrategy",
    "build_feature_frame",
    "list_datasets",
    "load_price_data",
    "MLSignalResult",
    "MovingAverageCrossoverStrategy",
    "generate_ml_signals",
    "PrecomputedSignalStrategy",
    "RSIStrategy",
    "run_backtest_service",
    "Strategy",
]
