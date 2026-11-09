import pandas as pd
import pytest

from src.common.io import read_stats_day
from src.common.params import publisher_column
from src.evaluate.nce import load_scored_day
from src.features.build import build_features


@pytest.fixture
def raw(in_workdir, params) -> pd.DataFrame:
    """200 raw rows of the eval day, without `click`."""
    day_raw, _ = load_scored_day(params["baseline"]["eval_day"])
    return day_raw.head(200).reset_index(drop=True)


@pytest.fixture
def stats(in_workdir, params) -> pd.DataFrame:
    """The stats of the eval day."""
    return read_stats_day(params["baseline"]["eval_day"])


def test_one_row_per_input_row_and_no_missing_values(raw, stats, feature_params):
    features = build_features(raw=raw, stats=stats, params=feature_params)
    assert len(features) == len(raw)
    assert not features.isna().any().any()


def test_a_row_alone_gets_the_same_features_as_in_a_batch(raw, stats, feature_params):
    # The service scores one request at a time; Phase 2 compares its predictions with the offline ones.
    batch = build_features(raw=raw, stats=stats, params=feature_params)
    for i in range(5):
        alone = build_features(raw=raw.iloc[[i]].reset_index(drop=True), stats=stats, params=feature_params)
        # The values must match; a dtype may differ between one row and many (int64 against float64).
        pd.testing.assert_frame_equal(alone, batch.iloc[[i]].reset_index(drop=True), check_dtype=False)


def test_missing_and_malformed_fields_give_no_error_and_no_nan(raw, stats, feature_params, params):
    # Object columns, as a JSON request gives them, so any field can hold a value of the wrong type.
    bad = raw.head(3).astype(object)
    bad.loc[0, publisher_column(params)] = None
    bad.loc[1, "device_model"] = "never-seen-before"
    bad.loc[2, "hour"] = "not-an-hour"
    bad.loc[2, "C14"] = "abc"
    bad = bad.drop(columns=["banner_pos"])  # a field absent from every request in the batch
    features = build_features(raw=bad, stats=stats, params=feature_params)
    assert len(features) == 3
    assert not features.isna().any().any()


def test_is_deterministic_and_leaves_its_input_alone(raw, stats, feature_params):
    before = raw.copy()
    first = build_features(raw=raw, stats=stats, params=feature_params)
    pd.testing.assert_frame_equal(first, build_features(raw=raw, stats=stats, params=feature_params))
    pd.testing.assert_frame_equal(raw, before)
