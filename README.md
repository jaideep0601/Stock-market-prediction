# Stock Market Prediction

A modular backtesting project for evaluating rule-based and ML-based stock trading signals with realistic execution assumptions.

## Project Summary

This project was built to demonstrate internship-ready skills across:

- data cleaning and preprocessing for OHLCV market data
- object-oriented strategy design
- realistic backtesting with next-candle execution
- risk-adjusted evaluation using Sharpe ratio and drawdown
- benchmark comparison against buy-and-hold
- ML signal generation with time-based validation

## Implemented Features

- CSV-based historical data loading with return calculation
- Rule-based strategies:
  `MovingAverageCrossoverStrategy`, `RSIStrategy`, `BuyAndHoldStrategy`
- ML-driven strategy workflow using:
  lagged returns, moving-average ratio, RSI, volatility
- Logistic Regression and Random Forest signal generation
- Long/flat portfolio simulation with transaction costs
- Multi-asset CLI execution
- Exported outputs for history, trades, metrics, plots, and analysis

## Architecture

```text
src/backtesting/
  data_loader.py      -> load and validate OHLCV data
  strategy.py         -> rule-based and precomputed-signal strategies
  ml.py               -> feature engineering and model training
  backtester.py       -> execution engine and portfolio tracking
  metrics.py          -> return, Sharpe ratio, drawdown
  reporting.py        -> CSV, JSON, PNG, and Markdown outputs
  visualization.py    -> equity, drawdown, and comparison plots
scripts/
  run_backtest.py     -> CLI entrypoint
data/
  sample_prices.csv   -> starter dataset for testing
```

## Expected CSV Format

Required columns:

- `Date`
- `Close`

Recommended columns:

- `Open`
- `High`
- `Low`
- `Volume`

Column matching is case-insensitive.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a rule-based strategy:

```bash
python scripts/run_backtest.py --csv data/sample_prices.csv --strategy ma --compare-buy-hold --no-plot
```

Run an ML-based strategy:

```bash
python scripts/run_backtest.py --csv data/sample_prices.csv --strategy ml --ml-model logistic --compare-buy-hold --no-plot
```

Run tests:

```bash
pytest
```

## Output Artifacts

Each run saves files under `outputs/` by default:

- `*_history.csv`
- `*_trades.csv`
- `*_metrics.json`
- `*_plot.png`
- `summary_metrics.csv`
- `comparison_metrics.csv`
- `comparison_plot.png`
- `analysis_report.md`

## Backtesting Rules

- Signals are created using information available at time `t`.
- Trades are executed on the next candle using `Open` when available.
- Only one position is allowed at a time: `long` or `flat`.
- Transaction costs are applied to entries and exits.
- ML validation uses a time-based split with no shuffling.

## Resume Bullet

Built a modular stock backtesting engine in Python to evaluate rule-based and ML-based trading strategies using next-candle execution, transaction-cost-aware simulation, and risk-adjusted metrics such as Sharpe ratio and max drawdown.
