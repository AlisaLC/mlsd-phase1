import numpy as np
import pytest

from src.common.io import read_stats_day
from src.evaluate.nce import check_predictions, evaluate, load_scored_day
from src.features.build import build_features
from src.train.model import load_model, save_model


def test_saved_baseline_beats_the_average_on_the_eval_day(in_workdir, tuned, params, tmp_path):
    model, feature_params = tuned
    model_dir = tmp_path / "baseline"
    save_model(model_dir=model_dir, python_model=model, feature_params=feature_params)
    assert evaluate(model_dir=model_dir, day=params["baseline"]["eval_day"]) < 1


def test_loaded_baseline_scores_a_single_row(in_workdir, tuned, params, tmp_path):
    model, feature_params = tuned
    model_dir = tmp_path / "baseline"
    save_model(model_dir=model_dir, python_model=model, feature_params=feature_params)
    # Loaded with its saved feature_params.json, the way the service will load it.
    loaded, saved_feature_params = load_model(model_dir)
    day = params["baseline"]["eval_day"]
    raw, _ = load_scored_day(day)
    # One row, as the service will send it. check_predictions is what evaluate() runs in `make train`.
    row = raw.head(1)
    features = build_features(raw=row, stats=read_stats_day(day), params=saved_feature_params)
    check_predictions(predictions=loaded.predict(features), n_rows=1)


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        pytest.param([0.1, 0.2], "expected 3 predictions", id="too few"),
        pytest.param([0.1, 0.2, 1.5], "probabilities", id="above 1"),
        pytest.param([0.1, 0.2, -0.1], "probabilities", id="below 0"),
        pytest.param([0.1, 0.2, np.nan], "probabilities", id="NaN"),
    ],
)
def test_check_predictions_rejects_bad_output(predictions, message):
    with pytest.raises(ValueError, match=message):
        check_predictions(predictions=predictions, n_rows=3)
