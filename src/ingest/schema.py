"""Validation of one raw day. A file that fails any check stops the pipeline; no row is dropped silently."""

import pandera.polars as pa
import polars as pl

from ta_kit import INT_COLUMNS, RAW_COLUMNS

# What validate() raises when a file is bad. ValueError is what parse_types and check_day raise, so a ValueError from
# a bug in check_day is reported as a validation failure too, without a traceback.
VALIDATION_ERRORS = (pa.errors.SchemaError, pa.errors.SchemaErrors, ValueError)


class RawDay(pa.DataFrameModel):
    """Every Avazu column with its type, after parse_types. Fields are non-nullable by default."""

    id: pl.String
    click: pl.Int64 = pa.Field(isin=[0, 1])
    hour: pl.Int64
    C1: pl.Int64
    banner_pos: pl.Int64
    site_id: pl.String
    site_domain: pl.String
    site_category: pl.String
    app_id: pl.String
    app_domain: pl.String
    app_category: pl.String
    device_id: pl.String
    device_ip: pl.String
    device_model: pl.String
    device_type: pl.Int64
    device_conn_type: pl.Int64
    C14: pl.Int64
    C15: pl.Int64
    C16: pl.Int64
    C17: pl.Int64
    C18: pl.Int64
    C19: pl.Int64
    C20: pl.Int64
    C21: pl.Int64


def parse_types(raw: pl.DataFrame) -> pl.DataFrame:
    """Parses the integer columns of a day read as strings; a value that is not an integer is an error."""
    missing = set(RAW_COLUMNS) - set(raw.columns)
    unexpected = set(raw.columns) - set(RAW_COLUMNS)
    if len(missing) > 0 or len(unexpected) > 0:
        raise ValueError(f"missing columns {sorted(missing)}, unexpected columns {sorted(unexpected)}")
    typed = raw.with_columns(pl.col(column).cast(pl.Int64, strict=False) for column in INT_COLUMNS)
    for column in INT_COLUMNS:
        # A null after the cast where the file had a value: that value is not an integer.
        not_integers = raw.filter(typed[column].is_null() & raw[column].is_not_null())[column]
        if len(not_integers) > 0:
            raise ValueError(f"column {column}: {len(not_integers)} values are not integers, e.g. {not_integers[0]!r}")
    return typed


def check_day(typed: pl.DataFrame, day: int) -> None:
    """Raises ValueError saying what failed unless every `hour` is a valid YYMMDDHH on `day` (YYYYMMDD)."""
    raise NotImplementedError("TODO(student): src/ingest/schema.py check_day")


def validate(raw: pl.DataFrame, day: int) -> pl.DataFrame:
    """Returns the day with every column parsed to its type, or raises one of VALIDATION_ERRORS."""
    typed = RawDay.validate(parse_types(raw), lazy=True)
    check_day(typed=typed, day=day)
    return typed
