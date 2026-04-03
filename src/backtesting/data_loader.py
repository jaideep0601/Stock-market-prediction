"""Utilities for loading and validating historical market data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"date", "close"}
OPTIONAL_COLUMNS = {"open", "high", "low", "volume"}


def load_price_data(csv_path: str | Path) -> pd.DataFrame:
    """Load OHLCV data from CSV, normalize columns, and compute returns."""
    data = pd.read_csv(csv_path)
    data.columns = [column.strip().lower() for column in data.columns]

    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "close"]).copy()
    data = data.sort_values("date").reset_index(drop=True)

    for column in OPTIONAL_COLUMNS:
        if column not in data.columns:
            data[column] = data["close"]

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["open", "high", "low", "close"]).copy()
    data["returns"] = data["close"].pct_change().fillna(0.0)

    ordered_columns = ["date", "open", "high", "low", "close", "volume", "returns"]
    return data[ordered_columns]
