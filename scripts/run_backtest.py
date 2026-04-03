"""Command-line runner for the Week 1 backtesting engine."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from backtesting.backtester import Backtester
from backtesting.data_loader import load_price_data
from backtesting.ml import generate_ml_signals
from backtesting.reporting import (
    save_analysis_report,
    save_backtest_result,
    save_comparison,
    save_summary,
)
from backtesting.strategy import (
    BuyAndHoldStrategy,
    MovingAverageCrossoverStrategy,
    PrecomputedSignalStrategy,
    RSIStrategy,
)
from backtesting.visualization import plot_backtest


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run a stock strategy backtest.")
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="One or more paths to OHLCV CSV files.",
    )
    parser.add_argument(
        "--strategy",
        choices=["buy_hold", "ma", "ml", "rsi"],
        default="ma",
        help="Strategy to run.",
    )
    parser.add_argument(
        "--compare-buy-hold",
        action="store_true",
        help="Also run buy-and-hold as a benchmark for each asset.",
    )
    parser.add_argument("--capital", type=float, default=10_000.0, help="Initial capital.")
    parser.add_argument("--cost", type=float, default=0.001, help="Transaction cost as a decimal.")
    parser.add_argument("--short-window", type=int, default=20, help="Short MA window.")
    parser.add_argument("--long-window", type=int, default=50, help="Long MA window.")
    parser.add_argument("--rsi-period", type=int, default=14, help="RSI lookback period.")
    parser.add_argument("--oversold", type=float, default=30.0, help="RSI oversold threshold.")
    parser.add_argument("--overbought", type=float, default=70.0, help="RSI overbought threshold.")
    parser.add_argument(
        "--ml-model",
        choices=["logistic", "random_forest"],
        default="logistic",
        help="ML model used for signal generation.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Fraction of engineered samples used for training in the ML workflow.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to convert ML predictions into long/flat signals.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory for saved results.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting the result.")
    return parser


def build_strategy(args: argparse.Namespace):
    """Instantiate the requested strategy from CLI arguments."""
    if args.strategy == "buy_hold":
        return BuyAndHoldStrategy()

    if args.strategy == "ma":
        return MovingAverageCrossoverStrategy(
            short_window=args.short_window,
            long_window=args.long_window,
        )

    if args.strategy == "ml":
        return None

    return RSIStrategy(
        period=args.rsi_period,
        oversold=args.oversold,
        overbought=args.overbought,
    )


def print_result(asset_name: str, strategy_name: str, result) -> None:
    """Print a readable summary for one backtest result."""
    print(f"Asset: {asset_name}")
    print(f"Strategy: {strategy_name}")
    print(f"Rows: {len(result.history)}")
    print(f"Trades: {len(result.trades)}")
    for metric_name, metric_value in result.metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name}: {metric_value:.6f}")
        else:
            print(f"{metric_name}: {metric_value}")
    print()


def build_asset_labels(csv_paths: list[Path]) -> list[str]:
    """Create unique, readable asset labels for CLI input paths."""
    base_names = [path.stem for path in csv_paths]
    counts = Counter(base_names)
    seen: Counter[str] = Counter()
    labels: list[str] = []

    for path, base_name in zip(csv_paths, base_names):
        seen[base_name] += 1
        if counts[base_name] == 1:
            labels.append(base_name)
        else:
            labels.append(f"{base_name}_{seen[base_name]}")

    return labels


def main() -> None:
    """Load data, run the backtest, and print summary metrics."""
    parser = build_parser()
    args = parser.parse_args()
    backtester = Backtester(initial_capital=args.capital, transaction_cost=args.cost)
    primary_strategy = build_strategy(args)
    strategies = [primary_strategy] if primary_strategy is not None else []
    if args.compare_buy_hold and args.strategy != "buy_hold":
        strategies.append(BuyAndHoldStrategy())

    summary_rows: list[dict[str, object]] = []
    plotted = False
    csv_paths = [Path(csv_value) for csv_value in args.csv]

    for csv_path in csv_paths:
        if not csv_path.exists():
            parser.error(
                f"CSV file not found: {csv_path}. "
                "Pass a real file path, for example: data/sample_prices.csv"
            )

    asset_labels = build_asset_labels(csv_paths)

    for csv_path, asset_name in zip(csv_paths, asset_labels):
        data = load_price_data(csv_path)
        run_strategies = list(strategies)

        if args.strategy == "ml":
            ml_result = generate_ml_signals(
                data,
                model_type=args.ml_model,
                train_split=args.train_split,
                probability_threshold=args.threshold,
            )
            ml_strategy = PrecomputedSignalStrategy(
                ml_result.signals,
                name=ml_result.model_name,
            )
            run_strategies.insert(0, ml_strategy)
            print(f"Asset: {asset_name}")
            print(f"ML model: {ml_result.model_name}")
            print(f"ML accuracy: {ml_result.accuracy:.6f}")
            print(f"ML test start index: {ml_result.test_start_index}")
            print()

        for strategy in run_strategies:
            result = backtester.run(data, strategy, asset_name=asset_name)
            if args.strategy == "ml" and strategy.name == ml_result.model_name:
                result.metrics["ml_accuracy"] = ml_result.accuracy
                result.metrics["ml_train_split"] = args.train_split
                result.metrics["ml_probability_threshold"] = args.threshold
            print_result(asset_name, strategy.name, result)
            summary_rows.append(save_backtest_result(result, args.output_dir))

            if not args.no_plot and not plotted:
                plot_backtest(result)
                plotted = True

    summary_path = save_summary(summary_rows, args.output_dir)
    comparison_path = save_comparison(summary_rows, args.output_dir)
    analysis_path = save_analysis_report(summary_rows, args.output_dir)
    print(f"Saved summary metrics to: {summary_path}")
    print(f"Saved comparison metrics to: {comparison_path}")
    print(f"Saved analysis report to: {analysis_path}")


if __name__ == "__main__":
    main()
