"""Core backtesting engine for long/flat strategies."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import summarize_performance
from .strategy import Strategy


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    asset_name: str
    strategy_name: str
    history: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


class Backtester:
    """Simple next-candle execution engine for long/flat strategies."""

    def __init__(self, initial_capital: float = 10_000.0, transaction_cost: float = 0.001) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if transaction_cost < 0:
            raise ValueError("transaction_cost cannot be negative")

        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

    def run(self, data: pd.DataFrame, strategy: Strategy, asset_name: str = "asset") -> BacktestResult:
        """Run a backtest for a strategy over the provided market data."""
        if data.empty:
            raise ValueError("Input data is empty")

        history = data.copy()
        raw_signals = strategy.generate_signals(history)
        if len(raw_signals) != len(history):
            raise ValueError("Strategy signal length must match input data length")

        history["signal"] = (
            pd.Series(raw_signals, index=history.index)
            .fillna(0)
            .clip(lower=0, upper=1)
            .astype(int)
        )
        history["target_position"] = history["signal"].shift(1).fillna(0).astype(int)

        cash = self.initial_capital
        shares = 0.0
        position = 0
        records: list[dict[str, float | int | str | pd.Timestamp]] = []
        trades: list[dict[str, float | str | pd.Timestamp]] = []

        for row in history.itertuples(index=False):
            target_position = int(row.target_position)
            execution_price = float(row.open) if pd.notna(row.open) else float(row.close)

            if target_position != position:
                if target_position == 1 and position == 0:
                    shares = cash / (execution_price * (1 + self.transaction_cost))
                    cash = 0.0
                    position = 1
                    trades.append(
                        {
                            "date": row.date,
                            "action": "BUY",
                            "price": execution_price,
                            "shares": shares,
                        }
                    )
                elif target_position == 0 and position == 1:
                    cash = shares * execution_price * (1 - self.transaction_cost)
                    trades.append(
                        {
                            "date": row.date,
                            "action": "SELL",
                            "price": execution_price,
                            "shares": shares,
                        }
                    )
                    shares = 0.0
                    position = 0

            holdings_value = shares * float(row.close)
            equity = cash + holdings_value

            records.append(
                {
                    "date": row.date,
                    "open": float(row.open),
                    "close": float(row.close),
                    "signal": int(row.signal),
                    "target_position": target_position,
                    "position": position,
                    "cash": cash,
                    "shares": shares,
                    "holdings": holdings_value,
                    "equity": equity,
                }
            )

        history_df = pd.DataFrame(records)
        history_df["daily_return"] = history_df["equity"].pct_change().fillna(0.0)
        history_df["rolling_peak"] = history_df["equity"].cummax()
        history_df["drawdown"] = history_df["equity"] / history_df["rolling_peak"] - 1

        if position == 1 and not history_df.empty:
            last_close = float(history_df.iloc[-1]["close"])
            final_cash = shares * last_close * (1 - self.transaction_cost)
            history_df.loc[history_df.index[-1], "equity"] = final_cash
            history_df.loc[history_df.index[-1], "cash"] = final_cash
            history_df.loc[history_df.index[-1], "shares"] = 0.0
            history_df.loc[history_df.index[-1], "holdings"] = 0.0
            history_df.loc[history_df.index[-1], "position"] = 0
            history_df["daily_return"] = history_df["equity"].pct_change().fillna(0.0)
            history_df["rolling_peak"] = history_df["equity"].cummax()
            history_df["drawdown"] = history_df["equity"] / history_df["rolling_peak"] - 1
            trades.append(
                {
                    "date": history_df.iloc[-1]["date"],
                    "action": "SELL_END",
                    "price": last_close,
                    "shares": shares,
                }
            )

        trades_df = pd.DataFrame(trades)
        metrics = summarize_performance(history_df["equity"])
        metrics.update(
            {
                "initial_capital": float(self.initial_capital),
                "final_equity": float(history_df["equity"].iloc[-1]),
                "num_trades": int(len(trades_df)),
                "exposure_rate": float(history_df["position"].mean()),
                "win_rate": _calculate_win_rate(trades_df),
            }
        )

        return BacktestResult(
            asset_name=asset_name,
            strategy_name=strategy.name,
            history=history_df,
            trades=trades_df,
            metrics=metrics,
        )


def _calculate_win_rate(trades: pd.DataFrame) -> float:
    """Estimate win rate by pairing sequential buy and sell trades."""
    if trades.empty or "action" not in trades.columns:
        return 0.0

    buy_price: float | None = None
    wins = 0
    completed_round_trips = 0

    for trade in trades.itertuples(index=False):
        action = str(trade.action)
        price = float(trade.price)

        if action == "BUY":
            buy_price = price
        elif action in {"SELL", "SELL_END"} and buy_price is not None:
            completed_round_trips += 1
            if price > buy_price:
                wins += 1
            buy_price = None

    if completed_round_trips == 0:
        return 0.0

    return wins / completed_round_trips
