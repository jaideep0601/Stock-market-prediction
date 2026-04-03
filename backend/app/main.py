"""FastAPI backend for the stock backtesting dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from backtesting.service import list_datasets, run_backtest_service, run_multi_backtest_service


class BacktestRequest(BaseModel):
    dataset: str = Field(..., description="CSV filename stem or path under the data directory")
    strategy: str = Field(default="ma")
    strategies: list[str] | None = None
    compare_buy_hold: bool = True
    capital: float = 10_000.0
    cost: float = 0.001
    short_window: int = 20
    long_window: int = 50
    rsi_period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    ml_model: str = "logistic"
    train_split: float = 0.7
    threshold: float = 0.5
    validation_mode: str = "walk_forward"
    min_train_size: int = 252
    step_size: int = 21


app = FastAPI(title="Stock Backtesting API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    """Return a lightweight health response."""
    return {"status": "ok"}


@app.get("/api/datasets")
def get_datasets() -> dict[str, object]:
    """Return local CSV datasets that can be used for backtesting."""
    return {"datasets": list_datasets(PROJECT_ROOT / "data")}


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, object]:
    """Upload a CSV dataset into the local data directory."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported.")

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in Path(file.filename).stem)
    safe_name = safe_name.strip("_") or "uploaded_dataset"
    output_path = data_dir / f"{safe_name}.csv"

    contents = await file.read()
    output_path.write_bytes(contents)

    return {
        "dataset": {
            "name": output_path.stem,
            "path": str(output_path),
        },
        "datasets": list_datasets(data_dir),
    }


@app.post("/api/backtest")
def run_backtest(request: BacktestRequest) -> dict[str, object]:
    """Run a backtest job and return the computed metrics and time series."""
    dataset_path = _resolve_dataset_path(request.dataset)
    try:
        if request.strategies:
            return run_multi_backtest_service(
                csv_path=dataset_path,
                strategies=request.strategies,
                capital=request.capital,
                cost=request.cost,
                short_window=request.short_window,
                long_window=request.long_window,
                rsi_period=request.rsi_period,
                oversold=request.oversold,
                overbought=request.overbought,
                ml_model=request.ml_model,
                train_split=request.train_split,
                threshold=request.threshold,
                validation_mode=request.validation_mode,
                min_train_size=request.min_train_size,
                step_size=request.step_size,
            )

        return run_backtest_service(
            csv_path=dataset_path,
            strategy_name=request.strategy,
            compare_buy_hold=request.compare_buy_hold,
            capital=request.capital,
            cost=request.cost,
            short_window=request.short_window,
            long_window=request.long_window,
            rsi_period=request.rsi_period,
            oversold=request.oversold,
            overbought=request.overbought,
            ml_model=request.ml_model,
            train_split=request.train_split,
            threshold=request.threshold,
            validation_mode=request.validation_mode,
            min_train_size=request.min_train_size,
            step_size=request.step_size,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _resolve_dataset_path(dataset: str) -> Path:
    """Resolve a dataset name or path inside the repository data directory."""
    data_dir = PROJECT_ROOT / "data"
    raw_path = Path(dataset)

    if raw_path.exists():
        return raw_path

    candidate = data_dir / f"{dataset}.csv"
    if candidate.exists():
        return candidate

    fallback = data_dir / dataset
    if fallback.exists():
        return fallback

    raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")
