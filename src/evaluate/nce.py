"""NCE of a model on one day: build_features with that day's stats, predict, compare with the clicks."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.common.io import read_clean_day, read_stats_day
from src.features.build import build_features
from src.train.model import load_model
from ta_kit import RAW_COLUMNS, nce


def load_scored_day(day: int) -> tuple[pd.DataFrame, np.ndarray]:
    """The raw columns of a day to score, without `click` (what the service will receive), and the clicks."""
    # The clicks are read on their own: dropping `click` from a full read would copy the whole day in memory.
    raw = read_clean_day(day=day, columns=[column for column in RAW_COLUMNS if column != "click"])
    return raw, read_clean_day(day=day, columns=["click"])["click"].to_numpy()


def check_predictions(predictions: np.ndarray | list[float], n_rows: int) -> np.ndarray:
    """The predictions as a float array, after checking there is one probability per row."""
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if predictions.shape != (n_rows,):
        raise ValueError(f"expected {n_rows} predictions, got {predictions.shape[0]}")
    if not np.all((predictions >= 0) & (predictions <= 1)):
        raise ValueError("predictions must be probabilities in [0, 1], with no NaN")
    return predictions


def evaluate(model_dir: Path, day: int) -> float:
    """NCE on every row of `day`, of the model saved in model_dir, loaded the way the registry will serve it."""
    model, feature_params = load_model(model_dir)
    raw, y = load_scored_day(day)
    features = build_features(raw=raw, stats=read_stats_day(day), params=feature_params)
    return nce(y, check_predictions(predictions=model.predict(features), n_rows=len(y)))


def score(python_model: mlflow.pyfunc.PythonModel, features: pd.DataFrame, y: np.ndarray) -> float:
    """NCE of a model that is not saved yet, for tuning: build the features once, then score each candidate."""
    predictions = python_model.predict(context=None, model_input=features)
    return nce(y, check_predictions(predictions=predictions, n_rows=len(y)))
