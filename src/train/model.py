"""The model directory: an MLflow pyfunc model plus feature_params.json and train_info.json."""

import json
import shutil
from pathlib import Path

import mlflow

MODELS_ROOT = Path("models")
FEATURE_PARAMS_FILE = "feature_params.json"
TRAIN_INFO_FILE = "train_info.json"


def save_model(model_dir: Path, python_model: mlflow.pyfunc.PythonModel, feature_params: dict) -> None:
    """Saves the model and its feature parameters to model_dir, replacing what was there."""
    shutil.rmtree(model_dir, ignore_errors=True)
    # No pip requirements: the model runs only inside this repository's images, which pin every package.
    # No signature: the input columns are whatever build_features returns, and MLflow must not reject or
    # convert them in serving.
    mlflow.pyfunc.save_model(path=model_dir, python_model=python_model, pip_requirements=[], signature=False)
    (model_dir / FEATURE_PARAMS_FILE).write_text(json.dumps(feature_params, indent=2, sort_keys=True) + "\n")


def write_train_info(model_dir: Path, params: dict, metrics: dict) -> None:
    """Writes the parameters and metrics that register.py logs to the model's MLflow run."""
    info = {"params": params, "metrics": metrics}
    (model_dir / TRAIN_INFO_FILE).write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")


def load_model(model_dir: Path) -> tuple[mlflow.pyfunc.PyFuncModel, dict]:
    """The model and its feature parameters."""
    return mlflow.pyfunc.load_model(str(model_dir)), json.loads((model_dir / FEATURE_PARAMS_FILE).read_text())
