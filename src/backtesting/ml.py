"""ML feature engineering and model-based signal generation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


@dataclass
class MLSignalResult:
    """Container for ML-generated strategy signals and model metadata."""

    signals: pd.Series
    feature_frame: pd.DataFrame
    test_start_index: int
    model_name: str
    accuracy: float


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
    frame["sma_5"] = frame["close"].rolling(window=5, min_periods=5).mean()
    frame["sma_10"] = frame["close"].rolling(window=10, min_periods=10).mean()
    frame["sma_ratio_5_10"] = frame["sma_5"] / frame["sma_10"] - 1
    frame["volatility_5"] = frame["returns"].rolling(window=5, min_periods=5).std()
    frame["rsi_14"] = compute_rsi(frame["close"], period=14)
    frame["label"] = (frame["close"].shift(-1) > frame["close"]).astype(int)

    feature_columns = [
        "return_1d",
        "return_3d",
        "return_5d",
        "sma_ratio_5_10",
        "volatility_5",
        "rsi_14",
    ]

    supervised = frame[["row_index", "date", "close", "open"] + feature_columns + ["label"]].copy()
    supervised = supervised.dropna().reset_index(drop=True)
    return supervised


def generate_ml_signals(
    data: pd.DataFrame,
    model_type: str = "logistic",
    train_split: float = 0.7,
    probability_threshold: float = 0.5,
) -> MLSignalResult:
    """Train a model on the first split and generate out-of-sample long/flat signals."""
    if not 0.5 <= train_split < 1.0:
        raise ValueError("train_split must be between 0.5 and 1.0")
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability_threshold must be between 0 and 1")

    feature_frame = build_feature_frame(data)
    if len(feature_frame) < 20:
        raise ValueError("Not enough rows after feature engineering to train an ML model")

    feature_columns = [
        "return_1d",
        "return_3d",
        "return_5d",
        "sma_ratio_5_10",
        "volatility_5",
        "rsi_14",
    ]

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

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
        )
        model_name = "ml_random_forest"
    else:
        model = LogisticRegression(max_iter=1000)
        model_name = "ml_logistic_regression"

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
        model_name=model_name,
        accuracy=accuracy,
    )
