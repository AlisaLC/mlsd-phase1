"""The raw drops: one gzipped Avazu CSV per day, named YYYYMMDD.csv.gz."""

import datetime
import re
import typing
from pathlib import Path

import polars as pl

# Avazu's columns, in file order.
RAW_COLUMNS = (
    "id", "click", "hour", "C1", "banner_pos",
    "site_id", "site_domain", "site_category", "app_id", "app_domain", "app_category",
    "device_id", "device_ip", "device_model", "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
)
# A variant's traffic: site traffic (variants A and B) or app traffic (C and D).
TrafficSlice = typing.Literal["site", "app"]
# Avazu's site_id on app traffic: a row is app traffic when it has this value, site traffic otherwise.
APP_SITE_ID = "85f751fd"
# The days (YYYYMMDD) of each drop.
DROP_DAYS = {1: [20141021, 20141022, 20141023, 20141024], 2: [20141025, 20141026, 20141027]}
# The integer columns. `id` is a number too, but it does not fit in int64, so it stays a string.
INT_COLUMNS = (
    "click", "hour", "C1", "banner_pos", "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
)


def list_day_files(raw_dir: Path) -> dict[int, Path]:
    """Maps each day (YYYYMMDD as int) to its file in raw_dir; other names ending in .csv.gz are an error."""
    day_files: dict[int, Path] = {}
    for path in sorted(raw_dir.glob("*.csv.gz")):
        name_match = re.fullmatch(r"(\d{8})\.csv\.gz", path.name)
        if name_match is None:
            raise ValueError(f"{path}: raw files must be named YYYYMMDD.csv.gz")
        try:
            datetime.datetime.strptime(name_match[1], "%Y%m%d")
        except ValueError:
            raise ValueError(f"{path}: {name_match[1]} is not a date") from None
        day_files[int(name_match[1])] = path
    if len(day_files) == 0:
        raise ValueError(f"{raw_dir}: no YYYYMMDD.csv.gz files")
    return day_files


def read_day(path: Path) -> pl.DataFrame:
    """One day's CSV with every column as a string; src/ingest parses and checks the types."""
    return pl.read_csv(path, infer_schema=False)
