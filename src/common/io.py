"""Where the pipeline's data lives, and how it is read and written. Paths are relative to the repository root."""

from pathlib import Path

import pandas as pd
import polars as pl

RAW_ROOT = Path("data/raw")
CLEAN_ROOT = Path("data/clean")
STATS_ROOT = Path("stats")

# The stats layout of the handout.
STATS_SCHEMA = {"day": pl.Int32, "key_type": pl.String, "key": pl.String, "imps": pl.Int64, "clicks": pl.Int64}


def day_dir(root: Path, day: int) -> Path:
    """root/day=YYYYMMDD, the folder of one day."""
    return root / f"day={day}"


def read_clean_day(day: int, root: Path = CLEAN_ROOT, columns: list[str] | None = None) -> pd.DataFrame:
    """One day of clean data, with only `columns` when given."""
    return pd.read_parquet(day_dir(root=root, day=day) / "part.parquet", columns=columns)


def read_stats_day(day: int, root: Path = STATS_ROOT) -> pd.DataFrame:
    """The stats table used for scoring `day`, counted over the days before it."""
    return pd.read_parquet(day_dir(root=root, day=day) / "part.parquet")


def write_stats(day: int, stats: pl.DataFrame, root: Path) -> None:
    """Writes the stats for `day`, after checking STATS_SCHEMA, the day column, nulls and duplicate keys."""
    if dict(stats.schema) != STATS_SCHEMA:  # the same columns and types, in any order
        raise ValueError(f"stats for {day}: expected columns and types {STATS_SCHEMA}, got {dict(stats.schema)}")
    if stats.is_empty():
        raise ValueError(f"stats for {day}: no rows")
    if stats.null_count().sum_horizontal().item() > 0:
        raise ValueError(f"stats for {day}: null values")
    if (stats["day"] != day).any():
        raise ValueError(f"stats for {day}: the day column has other days")
    if stats.select(pl.struct("key_type", "key").is_duplicated().any()).item():
        raise ValueError(f"stats for {day}: (key_type, key) is not unique")
    folder = day_dir(root=root, day=day)
    folder.mkdir(parents=True, exist_ok=True)
    stats.select(STATS_SCHEMA.keys()).sort("key_type", "key").write_parquet(folder / "part.parquet")
