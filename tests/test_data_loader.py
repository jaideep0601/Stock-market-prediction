from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from backtesting.data_loader import load_price_data


def make_test_csv(contents: str) -> Path:
    runtime_dir = Path("tests_runtime")
    runtime_dir.mkdir(exist_ok=True)
    csv_path = runtime_dir / f"{uuid4().hex}.csv"
    csv_path.write_text(contents, encoding="utf-8")
    return csv_path


def test_load_price_data_sorts_rows_and_computes_returns() -> None:
    csv_path = make_test_csv(
        "\n".join(
            [
                "Date,Close",
                "2024-01-03,102",
                "2024-01-01,100",
                "2024-01-02,101",
            ]
        )
    )

    result = load_price_data(csv_path)

    assert list(result["close"]) == [100, 101, 102]
    assert list(result["open"]) == [100, 101, 102]
    assert result["returns"].iloc[0] == 0.0
    assert result["returns"].iloc[1] == pytest.approx(0.01)


def test_load_price_data_requires_date_and_close_columns() -> None:
    csv_path = make_test_csv("Date,Open\n2024-01-01,100\n")

    with pytest.raises(ValueError, match="Missing required columns"):
        load_price_data(csv_path)


def test_load_price_data_drops_invalid_dates_and_prices() -> None:
    csv_path = make_test_csv(
        "\n".join(
            [
                "Date,Open,High,Low,Close,Volume",
                "not-a-date,100,101,99,100,1000",
                "2024-01-01,100,101,99,abc,1000",
                "2024-01-02,101,102,100,101,1000",
            ]
        )
    )

    result = load_price_data(csv_path)

    assert len(result) == 1
    assert pd.Timestamp("2024-01-02") == result["date"].iloc[0]
