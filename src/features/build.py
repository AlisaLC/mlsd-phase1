"""The one feature module. Training, evaluation and (in Phase 2) the service all call build_features."""

import pandas as pd


def build_features(raw: pd.DataFrame, stats: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Turns raw impressions into model input: one row per input row, in the same order."""
    # raw: the Avazu columns without `click`. In serving the values come from JSON, so any of them may be missing,
    # null or of the wrong type; such a field gets a default, never an error, and the output has no NaN.
    # stats: the stats table of the day the rows are scored on (read_stats_day of src/common/io.py).
    # params: the model's feature parameters, saved with it as feature_params.json. They hold at least
    # "publisher_col" (site_id or app_id) and "alpha" (the smoothing pseudo-count). You may add keys, but give each
    # a default when it is missing: the public tests pass only these two.
    raise NotImplementedError("TODO(student): src/features/build.py build_features")
