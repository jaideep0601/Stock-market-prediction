"""Helpers for saving backtest outputs to disk."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .backtester import BacktestResult
from .visualization import save_backtest_plot, save_comparison_plot


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create the output directory if it does not already exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_backtest_result(
    result: BacktestResult,
    output_dir: str | Path,
) -> dict[str, object]:
    """Persist history, trades, and metrics for one backtest run."""
    base_dir = ensure_output_dir(output_dir)
    run_name = f"{result.asset_name}_{result.strategy_name}"

    history_path = base_dir / f"{run_name}_history.csv"
    trades_path = base_dir / f"{run_name}_trades.csv"
    metrics_path = base_dir / f"{run_name}_metrics.json"
    plot_path = base_dir / f"{run_name}_plot.png"

    result.history.to_csv(history_path, index=False)
    result.trades.to_csv(trades_path, index=False)
    metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    save_backtest_plot(result, plot_path)

    return {
        "asset": result.asset_name,
        "strategy": result.strategy_name,
        **result.metrics,
        "history_path": str(history_path),
        "trades_path": str(trades_path),
        "metrics_path": str(metrics_path),
        "plot_path": str(plot_path),
    }


def save_summary(summary_rows: list[dict[str, object]], output_dir: str | Path) -> Path:
    """Save the aggregate metrics table for all executed backtests."""
    base_dir = ensure_output_dir(output_dir)
    summary_path = base_dir / "summary_metrics.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return summary_path


def save_comparison(summary_rows: list[dict[str, object]], output_dir: str | Path) -> Path:
    """Save a strategy-vs-benchmark comparison table by asset."""
    base_dir = ensure_output_dir(output_dir)
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        comparison_df = pd.DataFrame()
    else:
        benchmark_rows = (
            summary_df[summary_df["strategy"] == "buy_and_hold"]
            .set_index("asset")
            .add_prefix("benchmark_")
        )
        comparison_df = summary_df.join(benchmark_rows, on="asset")
        comparison_df["total_return_vs_benchmark"] = (
            comparison_df["total_return"] - comparison_df["benchmark_total_return"]
        )
        comparison_df["sharpe_vs_benchmark"] = (
            comparison_df["sharpe_ratio"] - comparison_df["benchmark_sharpe_ratio"]
        )
        comparison_df["max_drawdown_vs_benchmark"] = (
            comparison_df["max_drawdown"] - comparison_df["benchmark_max_drawdown"]
        )

    comparison_path = base_dir / "comparison_metrics.csv"
    comparison_df.to_csv(comparison_path, index=False)
    return comparison_path


def save_analysis_report(summary_rows: list[dict[str, object]], output_dir: str | Path) -> Path:
    """Create a short Markdown report highlighting key strategy insights."""
    base_dir = ensure_output_dir(output_dir)
    summary_df = pd.DataFrame(summary_rows)
    report_path = base_dir / "analysis_report.md"

    if summary_df.empty:
        report_path.write_text("# Analysis Report\n\nNo backtest results were generated.\n", encoding="utf-8")
        return report_path

    best_return = summary_df.loc[summary_df["total_return"].idxmax()]
    best_sharpe = summary_df.loc[summary_df["sharpe_ratio"].idxmax()]
    lowest_drawdown = summary_df.loc[summary_df["max_drawdown"].idxmax()]
    comparison_plot_path = save_comparison_plot(summary_df, base_dir / "comparison_plot.png")

    lines = [
        "# Analysis Report",
        "",
        "## Highlights",
        "",
        f"- Best total return: `{best_return['strategy']}` on `{best_return['asset']}` with `{best_return['total_return']:.4f}`.",
        f"- Best Sharpe ratio: `{best_sharpe['strategy']}` on `{best_sharpe['asset']}` with `{best_sharpe['sharpe_ratio']:.4f}`.",
        f"- Least severe drawdown: `{lowest_drawdown['strategy']}` on `{lowest_drawdown['asset']}` with `{lowest_drawdown['max_drawdown']:.4f}`.",
        "",
        "## Interpretation",
        "",
        "- Compare `total_return` to judge raw growth over the test period.",
        "- Compare `sharpe_ratio` to judge return quality after volatility.",
        "- Compare `max_drawdown` to understand downside risk during bad periods.",
        "- Use `comparison_metrics.csv` to see how each strategy stacked up against buy-and-hold.",
        "",
        "## Output Files",
        "",
        f"- Summary metrics: `{base_dir / 'summary_metrics.csv'}`",
        f"- Comparison metrics: `{base_dir / 'comparison_metrics.csv'}`",
        f"- Comparison plot: `{comparison_plot_path}`",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
