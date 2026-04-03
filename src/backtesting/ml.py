"""ML feature engineering and model-based signal generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class MLSignalResult:
    """Container for ML-generated strategy signals and model metadata."""

    signals: pd.Series
    feature_frame: pd.DataFrame
    test_start_index: int
    num_predictions: int
    model_name: str
    accuracy: float
    validation_mode: str


FEATURE_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "sma_ratio_5_10",
    "sma_ratio_10_20",
    "volatility_5",
    "volatility_10",
    "rsi_14",
    "momentum_10",
]


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI values using an EMA-based smoothing approach."""
    close_delta = close.diff()
    gains = close_delta.clip(lower=0)
    losses = -close_delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    relative_strength = avg_gain / avg_loss.replace(0, pd.NA)

    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.fillna(50.0)


def build_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Create a supervised-learning feature table from price history."""
    frame = data.copy()
    frame["row_index"] = frame.index
    frame["return_1d"] = frame["close"].pct_change()
    frame["return_3d"] = frame["close"].pct_change(3)
    frame["return_5d"] = frame["close"].pct_change(5)
    frame["return_10d"] = frame["close"].pct_change(10)
    frame["sma_5"] = frame["close"].rolling(window=5, min_periods=5).mean()
    frame["sma_10"] = frame["close"].rolling(window=10, min_periods=10).mean()
    frame["sma_20"] = frame["close"].rolling(window=20, min_periods=20).mean()
    frame["sma_ratio_5_10"] = frame["sma_5"] / frame["sma_10"] - 1
    frame["sma_ratio_10_20"] = frame["sma_10"] / frame["sma_20"] - 1
    frame["volatility_5"] = frame["returns"].rolling(window=5, min_periods=5).std()
    frame["volatility_10"] = frame["returns"].rolling(window=10, min_periods=10).std()
    frame["rsi_14"] = compute_rsi(frame["close"], period=14)
    frame["momentum_10"] = frame["close"] / frame["close"].shift(10) - 1
    frame["label"] = (frame["close"].shift(-1) > frame["close"]).astype(int)

    supervised = frame[["row_index", "date", "close", "open"] + FEATURE_COLUMNS + ["label"]].copy()
    supervised = supervised.dropna().reset_index(drop=True)
    return supervised


def build_model(model_type: str):
    """Create the requested sklearn model."""
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
        ), "ml_random_forest"

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    ), "ml_logistic_regression"


def generate_ml_signals(
    data: pd.DataFrame,
    model_type: str = "logistic",
    train_split: float = 0.7,
    probability_threshold: float = 0.5,
    validation_mode: str = "walk_forward",
    min_train_size: int = 252,
    step_size: int = 21,
) -> MLSignalResult:
    """Train a model and generate out-of-sample long/flat signals."""
    if not 0.5 <= train_split < 1.0:
        raise ValueError("train_split must be between 0.5 and 1.0")
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability_threshold must be between 0 and 1")
    if min_train_size <= 0:
        raise ValueError("min_train_size must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    feature_frame = build_feature_frame(data)
    if len(feature_frame) < max(20, min_train_size + 1):
        raise ValueError("Not enough rows after feature engineering to train an ML model")

    if validation_mode == "single_split":
        return _generate_single_split_signals(
            data=data,
            feature_frame=feature_frame,
            model_type=model_type,
            train_split=train_split,
            probability_threshold=probability_threshold,
        )

    if validation_mode != "walk_forward":
        raise ValueError("validation_mode must be 'single_split' or 'walk_forward'")

    return _generate_walk_forward_signals(
        data=data,
        feature_frame=feature_frame,
        model_type=model_type,
        probability_threshold=probability_threshold,
        min_train_size=min_train_size,
        step_size=step_size,
    )


def _generate_single_split_signals(
    data: pd.DataFrame,
    feature_frame: pd.DataFrame,
    model_type: str,
    train_split: float,
    probability_threshold: float,
) -> MLSignalResult:
    """Train once on the initial split and predict on the remaining rows."""
    feature_columns = FEATURE_COLUMNS

    split_index = max(int(len(feature_frame) * train_split), 1)
    if split_index >= len(feature_frame):
        split_index = len(feature_frame) - 1
    if split_index <= 0:
        raise ValueError("Training split leaves no rows for evaluation")

    train_frame = feature_frame.iloc[:split_index].copy()
    test_frame = feature_frame.iloc[split_index:].copy()
    if test_frame.empty:
        raise ValueError("Test split is empty; adjust the train_split parameter")

    x_train = train_frame[feature_columns]
    y_train = train_frame["label"]
    x_test = test_frame[feature_columns]
    y_test = test_frame["label"]

    model, model_name = build_model(model_type)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= probability_threshold).astype(int)
    accuracy = float(accuracy_score(y_test, predictions))

    signals = pd.Series(0, index=data.index, dtype=int, name="signal")
    signal_index = feature_frame.loc[split_index:, "row_index"].astype(int)
    signals.loc[signal_index] = predictions

    return MLSignalResult(
        signals=signals,
        feature_frame=feature_frame,
        test_start_index=int(signal_index.iloc[0]),
        num_predictions=int(len(predictions)),
        model_name=model_name,
        accuracy=accuracy,
        validation_mode="single_split",
    )


def _generate_walk_forward_signals(
    data: pd.DataFrame,
    feature_frame: pd.DataFrame,
    model_type: str,
    probability_threshold: float,
    min_train_size: int,
    step_size: int,
) -> MLSignalResult:
    """Use an expanding window to repeatedly retrain and predict future chunks."""
    feature_columns = FEATURE_COLUMNS
    signals = pd.Series(0, index=data.index, dtype=int, name="signal")
    all_predictions: list[int] = []
    all_truths: list[int] = []
    predicted_row_indices: list[int] = []

    start_index = min_train_size
    if start_index >= len(feature_frame):
        raise ValueError("min_train_size is too large for the available data")

    model_name = ""
    for test_start in range(start_index, len(feature_frame), step_size):
        test_end = min(test_start + step_size, len(feature_frame))
        train_frame = feature_frame.iloc[:test_start].copy()
        test_frame = feature_frame.iloc[test_start:test_end].copy()
        if test_frame.empty:
            continue

        x_train = train_frame[feature_columns]
        y_train = train_frame["label"]
        x_test = test_frame[feature_columns]
        y_test = test_frame["label"]

        model, model_name = build_model(model_type)
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= probability_threshold).astype(int)

        row_indices = test_frame["row_index"].astype(int).tolist()
        signals.loc[row_indices] = predictions
        predicted_row_indices.extend(row_indices)
        all_predictions.extend(predictions.tolist())
        all_truths.extend(y_test.tolist())

    if not predicted_row_indices:
        raise ValueError("Walk-forward validation produced no predictions")

    accuracy = float(accuracy_score(all_truths, all_predictions))
    return MLSignalResult(
        signals=signals,
        feature_frame=feature_frame,
        test_start_index=int(predicted_row_indices[0]),
        num_predictions=int(len(all_predictions)),
        model_name=model_name,
        accuracy=accuracy,
        validation_mode="walk_forward",
    )
