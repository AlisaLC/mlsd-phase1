"""Runs after `dvc repro`, outside DVC: registers every model of params.registry and moves the champion alias."""

import json
import sys

import mlflow

from src.common.params import PLACEHOLDER_STUDENT_ID, load_params, model_name
from src.train.model import MODELS_ROOT, TRAIN_INFO_FILE

EXPERIMENT = "mini-ctr"
CHAMPION_ALIAS = "champion"


def read_git_commit() -> str:
    """HEAD's sha, with a "-dirty" suffix when files other than dvc.lock and metrics/ have uncommitted changes."""
    # Untracked files that git does not ignore count as changes. dvc.lock and metrics/ do not, because `make train`
    # itself rewrites them on a clean clone.
    raise NotImplementedError("TODO(student): src/train/register.py read_git_commit")


def read_data_hash() -> str:
    """The md5 of the ingest stage's data/clean output in dvc.lock, with its .dir suffix: the training data's hash."""
    raise NotImplementedError("TODO(student): src/train/register.py read_data_hash")


def log_and_register(model_folder: str, registered_name: str, lineage: dict[str, str]) -> str:
    """Logs models/<model_folder> as an MLflow run and registers it; returns the new registry version."""
    model_dir = MODELS_ROOT / model_folder
    info = json.loads((model_dir / TRAIN_INFO_FILE).read_text())
    with mlflow.start_run(run_name=model_folder) as run:
        mlflow.log_params(info["params"])
        mlflow.log_metrics(info["metrics"])
        mlflow.set_tags({"model": model_folder, **lineage})
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
    return mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=registered_name).version


def main() -> None:
    """Registers params.registry.models and points the champion alias at params.registry.champion."""
    params = load_params()
    if params["student_id"] == PLACEHOLDER_STUDENT_ID:
        sys.exit("params.yaml: set student_id to yours; the model is registered as ctr-<student_id>")
    if not isinstance(params["student_id"], str):
        sys.exit('params.yaml: quote student_id ("400123456"), or YAML reads it as a number')
    registry = params["registry"]
    champion = registry["champion"]
    if champion not in registry["models"]:
        sys.exit(f"params.yaml: champion {champion!r} is not in registry.models")
    registered_name = model_name(params)
    # Every model registered by one `make train` run has the same commit and data, so both are read once.
    lineage = {"git_commit": read_git_commit(), "data_hash": read_data_hash()}
    mlflow.set_experiment(EXPERIMENT)
    # Model folder -> its new registry version.
    versions: dict[str, str] = {
        model_folder: log_and_register(model_folder=model_folder, registered_name=registered_name, lineage=lineage)
        for model_folder in registry["models"]
    }
    mlflow.MlflowClient().set_registered_model_alias(
        name=registered_name, alias=CHAMPION_ALIAS, version=versions[champion]
    )
    print(f"registered {registered_name}: {versions}; {CHAMPION_ALIAS} -> {champion} (version {versions[champion]})")


if __name__ == "__main__":
    main()
