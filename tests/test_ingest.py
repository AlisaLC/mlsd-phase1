"""Tests of src/ingest: the validation of a raw day, add_time_columns and the ingest stage."""

import gzip
import shutil

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from src.common.params import publisher_column
from src.ingest.run import add_time_columns, ingest
from src.ingest.schema import VALIDATION_ERRORS, validate
from ta_kit import read_day
from tests.synthetic import DAYS

FIRST_DAY = DAYS[0]


@pytest.fixture
def raw(raw_dir) -> pl.DataFrame:
    """The first synthetic day, as read from its file."""
    return read_day(raw_dir / f"{FIRST_DAY}.csv.gz")


def set_first_row(raw: pl.DataFrame, column: str, value: str | None) -> pl.DataFrame:
    """raw with `value` in the first row of `column`."""
    first = pl.int_range(pl.len()) == 0
    return raw.with_columns(pl.when(first).then(pl.lit(value, dtype=pl.String)).otherwise(pl.col(column)).alias(column))


def test_clean_day_passes(raw):
    typed = validate(raw=raw, day=FIRST_DAY)
    assert typed.height == raw.height
    assert (typed["id"].dtype, typed["hour"].dtype, typed["C14"].dtype) == (pl.String, pl.Int64, pl.Int64)


# `hour` is YYMMDDHH, and FIRST_DAY % 1_000_000 is FIRST_DAY as YYMMDD.
@pytest.mark.parametrize(
    ("column", "value"),
    [
        pytest.param("C14", "abc", id="not an integer"),
        pytest.param("C14", "1.5", id="a float"),
        pytest.param("click", "2", id="click 2"),
        pytest.param("hour", str((FIRST_DAY % 1_000_000 + 1) * 100 + 13), id="hour on the next day"),
        pytest.param("hour", str(FIRST_DAY % 1_000_000 * 100 + 24), id="hour 24"),
        pytest.param("hour", str(FIRST_DAY % 1_000_000), id="hour without HH"),
        pytest.param("id", None, id="null id"),
        pytest.param("hour", None, id="null hour"),
        pytest.param("publisher", None, id="null publisher"),
    ],
)
def test_bad_value_fails(raw, params, column, value):
    column = publisher_column(params) if column == "publisher" else column
    with pytest.raises(VALIDATION_ERRORS):
        validate(raw=set_first_row(raw=raw, column=column, value=value), day=FIRST_DAY)


@pytest.mark.parametrize(
    "break_columns",
    [
        pytest.param(lambda raw: raw.drop("C21"), id="missing column"),
        pytest.param(lambda raw: raw.with_columns(extra=pl.lit("x")), id="extra column"),
    ],
)
def test_wrong_columns_fail(raw, break_columns):
    with pytest.raises(VALIDATION_ERRORS):
        validate(raw=break_columns(raw), day=FIRST_DAY)


def test_add_time_columns_parses_hour():
    typed = pl.DataFrame({"hour": [14102100, 14102113, 14103123]})
    expected = typed.with_columns(
        day=pl.Series([20141021, 20141021, 20141031], dtype=pl.Int32),
        hour_of_day=pl.Series([0, 13, 23], dtype=pl.Int8),
    )
    assert_frame_equal(add_time_columns(typed), expected, check_column_order=False)


def wrong_type(raw: pl.DataFrame) -> bytes:
    """The day's file with "abc" in C14 of the first row."""
    return gzip.compress(set_first_row(raw=raw, column="C14", value="abc").write_csv().encode())


def extra_field(raw: pl.DataFrame) -> bytes:
    """The day's file with one more field on the first row than in the header."""
    lines = raw.write_csv().split("\n")
    lines[1] += ",x"
    return gzip.compress("\n".join(lines).encode())


def truncated(raw: pl.DataFrame) -> bytes:
    """The day's file with its gzip stream cut short."""
    return gzip.compress(raw.write_csv().encode())[:-100]


@pytest.mark.parametrize("corrupt", [wrong_type, extra_field, truncated])
def test_a_bad_file_stops_ingest_and_writes_nothing(raw_dir, tmp_path, corrupt):
    bad_raw_dir = tmp_path / "raw"
    shutil.copytree(raw_dir, bad_raw_dir)
    # The last day, so the days before it have been written to the staging folder when it fails.
    last = bad_raw_dir / f"{DAYS[-1]}.csv.gz"
    last.write_bytes(corrupt(read_day(last)))
    with pytest.raises(SystemExit, match=last.name):
        ingest(raw_root=bad_raw_dir, clean_root=tmp_path / "clean")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["raw"]
