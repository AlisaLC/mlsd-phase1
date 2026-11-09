"""The public tests run on a small synthetic drop (ta_kit/synth.py), built once per session."""

import shutil
import typing
from pathlib import Path

import pytest

from src.common.io import CLEAN_ROOT, STATS_ROOT
from src.common.params import PARAMS_PATH, publisher_column
from src.ingest.run import ingest
from src.stats.build import build_stats
from ta_kit.synth import write_drop
from tests.synthetic import DAYS, PARAMS, REPO, ROWS_PER_DAY

if typing.TYPE_CHECKING:
    from src.train.baseline import BaselineModel


@pytest.fixture(scope="session")
def params() -> dict:
    """The repository's params.yaml."""
    return PARAMS


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory) -> Path:
    """A synthetic drop of the slice in params.yaml."""
    path = tmp_path_factory.mktemp("raw")
    write_drop(out_dir=path, days=DAYS, rows_per_day=ROWS_PER_DAY, traffic_slice=PARAMS["slice"])
    return path


@pytest.fixture(scope="session")
def workdir(tmp_path_factory, raw_dir) -> Path:
    """A folder laid out like the repository root, with params.yaml, and data/clean and stats/ built by your code."""
    root = tmp_path_factory.mktemp("work")
    shutil.copy(src=REPO / PARAMS_PATH, dst=root / PARAMS_PATH)
    # The default paths of src/common/io.py are relative, so inside root they point at the synthetic data.
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        ingest(raw_root=raw_dir, clean_root=CLEAN_ROOT)
        build_stats(clean_root=CLEAN_ROOT, stats_root=STATS_ROOT, params=PARAMS)
    return root


@pytest.fixture
def in_workdir(workdir, monkeypatch) -> Path:
    """Runs the test inside workdir, so the default paths of src/common/io.py point at the synthetic data."""
    monkeypatch.chdir(workdir)
    return workdir


@pytest.fixture(scope="session")
def feature_params() -> dict:
    """Feature parameters for testing build_features before tune is written."""
    return {"publisher_col": publisher_column(PARAMS), "alpha": PARAMS["baseline"]["alpha_grid"][0]}


@pytest.fixture(scope="session")
def tuned(workdir) -> "tuple[BaselineModel, dict]":
    """(model, feature_params) from your tune(), run inside workdir."""
    # Imported here, not at the top, so that a baseline.py that fails to import breaks only the tests that use it.
    from src.train.baseline import tune

    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(workdir)
        return tune(PARAMS)
