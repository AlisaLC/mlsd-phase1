"""DVC stage `ingest`: data/raw/YYYYMMDD.csv.gz -> data/clean/day=YYYYMMDD/part.parquet."""

import shutil
import sys
from pathlib import Path

import polars as pl

from src.common.io import CLEAN_ROOT, RAW_ROOT, day_dir
from src.ingest.schema import VALIDATION_ERRORS, validate
from ta_kit import list_day_files, read_day


def add_time_columns(typed: pl.DataFrame) -> pl.DataFrame:
    """Adds `day` (Int32, YYYYMMDD, e.g. 20141021) and `hour_of_day` (Int8, 0-23), parsed from `hour` (YYMMDDHH)."""
    raise NotImplementedError("TODO(student): src/ingest/run.py add_time_columns")


def ingest(raw_root: Path, clean_root: Path) -> None:
    """Validates and writes every day; a bad file stops it with a message and leaves clean_root as it was."""
    try:
        day_files = list_day_files(raw_root)
    except ValueError as error:
        sys.exit(str(error))
    # Days are written to a temporary folder that becomes clean_root only once every day has passed.
    staging = clean_root.with_name(f".{clean_root.name}.tmp")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        for day, path in day_files.items():
            try:
                raw = read_day(path)
            except (pl.exceptions.PolarsError, OSError) as error:
                # PolarsError: a row with more fields than the header. OSError: a truncated or corrupt gzip file.
                sys.exit(f"{path.name}: could not read the file\n{error}")
            # Only validation errors are caught, so most bugs in your own code still show their traceback.
            try:
                typed = validate(raw=raw, day=day)
            except VALIDATION_ERRORS as error:
                sys.exit(f"{path.name}: validation failed\n{error}")
            clean = add_time_columns(typed)
            # A mistake in add_time_columns stops here, instead of showing up later as wrong stats.
            if (clean.schema.get("day"), clean.schema.get("hour_of_day")) != (pl.Int32, pl.Int8):
                raise TypeError(f"{path.name}: add_time_columns must add day as Int32 and hour_of_day as Int8")
            if clean.height != typed.height:
                raise ValueError(f"{path.name}: add_time_columns changed the number of rows")
            # A null day or hour must fail too, but any() and all() skip nulls: hence ne_missing and fill_null.
            if clean["day"].ne_missing(day).any():
                raise ValueError(f"{path.name}: add_time_columns gave a day other than {day}, or a null")
            if not clean["hour_of_day"].is_between(0, 23).fill_null(False).all():
                raise ValueError(f"{path.name}: add_time_columns gave an hour_of_day outside 0-23, or a null")
            folder = day_dir(root=staging, day=day)
            folder.mkdir(parents=True)
            clean.write_parquet(folder / "part.parquet")
            print(f"{path.name}: {clean.height} rows")
        shutil.rmtree(clean_root, ignore_errors=True)
        staging.rename(clean_root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    ingest(raw_root=RAW_ROOT, clean_root=CLEAN_ROOT)
