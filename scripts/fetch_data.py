"""Download historical stock data and save it as a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Download OHLCV data from Yahoo Finance.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, for example AAPL.")
    parser.add_argument("--start", default="2020-01-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", default=None, help="Optional end date in YYYY-MM-DD format.")
    parser.add_argument("--interval", default="1d", help="Data interval, for example 1d.")
    parser.add_argument("--output-dir", default="data", help="Directory where the CSV will be saved.")
    return parser


def main() -> None:
    """Download ticker data and save it in the format used by the backtester."""
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = yf.download(
        tickers=args.ticker,
        start=args.start,
        end=args.end,
        interval=args.interval,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        parser.error(f"No data returned for ticker {args.ticker}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    if "Adj Close" in data.columns:
        data = data.drop(columns=["Adj Close"])

    csv_path = output_dir / f"{args.ticker.lower()}_{args.interval}.csv"
    data.to_csv(csv_path, index=False)
    print(f"Saved data to: {csv_path}")


if __name__ == "__main__":
    main()
