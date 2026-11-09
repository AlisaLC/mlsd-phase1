"""DVC stage `baseline`: tune the backoff CTR on tune_day, save it to models/baseline, report NCE on eval_day."""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.common.params import load_params
from src.evaluate.nce import evaluate
from src.train.model import MODELS_ROOT, save_model, write_train_info

MODEL_DIR = MODELS_ROOT / "baseline"
METRICS_PATH = Path("metrics/p1.json")


class BaselineModel(mlflow.pyfunc.PythonModel):
    """Backoff CTR: the smoothed CTR of the publisher, else of the device model, else the global CTR (see k)."""

    def __init__(self, k: int):
        # A publisher or device model needs at least k impressions in the stats for its own CTR to be used.
        self.k = k

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        """model_input is the output of build_features; returns one click probability per row."""
        # build_features must therefore also return what this needs, such as the impression counts compared with k.
        raise NotImplementedError("TODO(student): src/train/baseline.py BaselineModel.predict")


def tune(params: dict) -> tuple[BaselineModel, dict]:
    """The model and feature parameters (see build_features) with the lowest NCE on params.baseline.tune_day."""
    # Read the day with load_scored_day(tune_day) of src/evaluate/nce.py, build the features once per alpha, then
    # score() each k.
    # Never read eval_day here: it is the day the baseline is reported on.
    raise NotImplementedError("TODO(student): src/train/baseline.py tune")


def main() -> None:
    """Tunes, saves and scores the baseline, and writes metrics/p1.json."""
    params = load_params()
    model, feature_params = tune(params)
    save_model(model_dir=MODEL_DIR, python_model=model, feature_params=feature_params)
    # Both days are scored through the saved model, the way the registry will serve it.
    metrics = {
        "nce_tune_day": evaluate(model_dir=MODEL_DIR, day=params["baseline"]["tune_day"]),
        "nce_eval_day": evaluate(model_dir=MODEL_DIR, day=params["baseline"]["eval_day"]),
    }
    write_train_info(model_dir=MODEL_DIR, params={"k": model.k, **feature_params}, metrics=metrics)
    # The handout fixes this key: the NCE on day 4, which is eval_day.
    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps({"baseline_nce_day4": metrics["nce_eval_day"]}, indent=2, sort_keys=True) + "\n")
    print(f"baseline: k={model.k} {feature_params} {metrics}")


if __name__ == "__main__":
    main()
