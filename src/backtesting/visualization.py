"""Plotting helpers for backtest outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .backtester import BacktestResult


def plot_backtest(result: BacktestResult) -> None:
    """Plot the equity curve and drawdown curve for a backtest result."""
    history = result.history

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(history["date"], history["equity"], label="Equity Curve", color="navy")
    axes[0].set_title(f"Equity Curve: {result.strategy_name}")
    axes[0].set_ylabel("Portfolio Value")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].fill_between(history["date"], history["drawdown"], 0, color="crimson", alpha=0.35)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def save_backtest_plot(result: BacktestResult, output_path: str | Path) -> Path:
    """Save the equity curve and drawdown curve for a single strategy."""
    history = result.history
    output_file = Path(output_path)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(history["date"], history["equity"], label="Equity Curve", color="navy")
    axes[0].set_title(f"Equity Curve: {result.asset_name} - {result.strategy_name}")
    axes[0].set_ylabel("Portfolio Value")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].fill_between(history["date"], history["drawdown"], 0, color="crimson", alpha=0.35)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_file


def save_comparison_plot(summary_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a bar chart comparing total return and Sharpe ratio across strategies."""
    output_file = Path(output_path)
    if summary_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No results available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_file

    labels = summary_df["asset"] + " | " + summary_df["strategy"]
    x_positions = range(len(summary_df))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].bar(x_positions, summary_df["total_return"], color="#1f77b4")
    axes[0].set_title("Total Return by Strategy")
    axes[0].set_ylabel("Total Return")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x_positions, summary_df["sharpe_ratio"], color="#2ca02c")
    axes[1].set_title("Sharpe Ratio by Strategy")
    axes[1].set_ylabel("Sharpe Ratio")
    axes[1].set_xticks(list(x_positions))
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_file
