"""Properties every correct stats table has. Passing them does not prove the table equals ours."""

import shutil

import pandas as pd
import polars as pl
import pytest

from src.common.io import CLEAN_ROOT, STATS_SCHEMA, read_clean_day, read_stats_day, write_stats
from src.common.params import publisher_column
from src.stats.build import build_stats
from tests.synthetic import DAYS


@pytest.fixture
def clean(in_workdir) -> pd.DataFrame:
    """Every synthetic day of clean data."""
    return pd.concat([read_clean_day(day) for day in DAYS])


def keys(stats: pd.DataFrame, key_type: str) -> set[str]:
    """The keys of one key type."""
    return set(stats.loc[stats["key_type"] == key_type, "key"])


def test_required_key_types_are_present(in_workdir, params):
    for day in params["stats"]["days"]:
        assert {"global", "publisher", "device_model", "hour_of_day"} <= set(read_stats_day(day)["key_type"]), day


def test_global_counts_every_day_before(clean, params):
    for day in params["stats"]["days"]:
        before = clean[clean["day"] < day]
        row = read_stats_day(day).query("key_type == 'global'")
        assert row["key"].tolist() == ["all"], day
        assert (row["imps"].item(), row["clicks"].item()) == (len(before), before["click"].sum()), day


def test_a_new_publisher_appears_only_from_the_next_day(clean, params):
    column = publisher_column(params)
    days = params["stats"]["days"]
    for day, next_day in zip(days, [*days[1:], None]):
        new_publishers = set(clean.loc[clean["day"] == day, column]) - set(clean.loc[clean["day"] < day, column])
        assert len(new_publishers) > 0, f"the synthetic data has no new publisher on {day}"
        assert new_publishers.isdisjoint(keys(read_stats_day(day), "publisher")), (
            f"stats for {day} count publishers first seen on {day}"
        )
        if next_day is not None:
            assert new_publishers <= keys(read_stats_day(next_day), "publisher"), day


def test_counts_are_valid_and_never_shrink(in_workdir, params):
    days = params["stats"]["days"]
    for day in days:
        stats = read_stats_day(day)
        assert (stats["imps"] > 0).all() and (stats["clicks"] <= stats["imps"]).all(), day
        assert keys(stats, "hour_of_day") <= {str(hour) for hour in range(24)}, day
    for day, next_day in zip(days, days[1:]):
        # A left join, so a key that is missing on the next day shows up instead of being dropped.
        merged = read_stats_day(day).merge(
            read_stats_day(next_day), on=["key_type", "key"], how="left", suffixes=("", "_next"), indicator=True
        )
        assert (merged["_merge"] == "both").all(), f"keys of {day} are missing on {next_day}"
        for column in ["imps", "clicks"]:
            assert (merged[f"{column}_next"] >= merged[column]).all(), f"{column} shrink from {day} to {next_day}"


def stats_row(**changes) -> dict:
    """One valid stats row for 20141022, with `changes` applied."""
    return {"day": 20141022, "key_type": "global", "key": "all", "imps": 10, "clicks": 2, **changes}


@pytest.mark.parametrize(
    ("rows", "schema", "message"),
    [
        pytest.param([], STATS_SCHEMA, "no rows", id="empty"),
        pytest.param([stats_row(key=None)], STATS_SCHEMA, "null", id="a null key"),
        pytest.param([stats_row(day=20141023)], STATS_SCHEMA, "other days", id="another day"),
        pytest.param([stats_row(), stats_row()], STATS_SCHEMA, "not unique", id="a duplicate key"),
        pytest.param([stats_row()], {**STATS_SCHEMA, "imps": pl.Int32}, "expected columns and types", id="Int32 imps"),
    ],
)
def test_write_stats_rejects_a_bad_table(tmp_path, rows, schema, message):
    with pytest.raises(ValueError, match=message):
        write_stats(day=20141022, stats=pl.DataFrame(rows, schema=schema), root=tmp_path)


def test_build_stats_stops_on_the_other_slice(in_workdir, params, tmp_path):
    other_slice = "app" if params["slice"] == "site" else "site"
    with pytest.raises(SystemExit, match="is `slice` in params.yaml"):
        build_stats(clean_root=CLEAN_ROOT, stats_root=tmp_path / "stats", params={**params, "slice": other_slice})


@pytest.mark.parametrize(
    ("missing_day", "message"),
    [
        pytest.param(DAYS[0], "no clean day before", id="the history day"),
        pytest.param(DAYS[2], str(DAYS[2]), id="a stats day"),
    ],
)
def test_build_stats_stops_on_a_missing_day(in_workdir, params, tmp_path, missing_day, message):
    clean_root = tmp_path / "clean"
    shutil.copytree(src=CLEAN_ROOT, dst=clean_root)
    shutil.rmtree(clean_root / f"day={missing_day}")
    with pytest.raises(SystemExit, match=message):
        build_stats(clean_root=clean_root, stats_root=tmp_path / "stats", params=params)
