"""DVC stage `stats`: the CTR-stats table, stats/day=YYYYMMDD/part.parquet, for each day in params.stats.days."""

import datetime
import shutil
import sys
from pathlib import Path

import polars as pl

from src.common.io import CLEAN_ROOT, STATS_ROOT, read_clean_day, write_stats
from src.common.params import load_params, publisher_column


def compute_stats(clean_root: Path, day: int, key_types: list[str], publisher_col: str) -> pl.DataFrame:
    """The stats used for scoring `day`: one row per (key_type, key), counted over every clean day before `day`."""
    # Columns and types are STATS_SCHEMA of src/common/io.py. Key types: global (key "all"), publisher (the value
    # of the column publisher_col), device_model, hour_of_day (keys "0" to "23"), and any extras you add to
    # params.yaml. Clean days are in clean_root/day=YYYYMMDD/part.parquet. Several full days do not fit in memory as
    # pandas frames, so read only the columns you need, or use polars or duckdb, e.g.
    # pl.scan_parquet(clean_root / "*" / "part.parquet", hive_partitioning=False) filtered on its `day` column
    # (hive_partitioning=False, or polars adds a second `day` column from the folder names).
    raise NotImplementedError("TODO(student): src/stats/build.py compute_stats")


def days_before(first: int, end: int) -> list[int]:
    """Every calendar day from `first` up to, not including, `end`, all YYYYMMDD."""
    first_date, end_date = (datetime.datetime.strptime(str(day), "%Y%m%d") for day in (first, end))
    dates = [first_date + datetime.timedelta(days=n) for n in range((end_date - first_date).days)]
    return [int(date.strftime("%Y%m%d")) for date in dates]


def build_stats(clean_root: Path, stats_root: Path, params: dict) -> None:
    """Writes the stats for every day in params.stats.days, replacing stats_root."""
    stats_days = params["stats"]["days"]
    # Each stats day counts every day before it, so the clean days must start before the first stats day and have no
    # gap up to the last one; otherwise the counts come out short.
    clean_days = sorted(int(path.name.removeprefix("day=")) for path in clean_root.glob("day=*"))
    if len(clean_days) == 0 or clean_days[0] >= stats_days[0]:
        sys.exit(f"{clean_root}: no clean day before {stats_days[0]}, the first day in params.stats.days")
    missing = sorted(set(days_before(first=clean_days[0], end=max(stats_days))) - set(clean_days))
    if len(missing) > 0:
        sys.exit(f"{clean_root}: clean days {missing} are missing; is every day of the drop in RAW_DIR?")
    first_day = stats_days[0]
    publisher_col = publisher_column(params)
    # A drop of the other slice has one placeholder value in the publisher column: `slice` is wrong in params.yaml.
    if read_clean_day(day=first_day, root=clean_root, columns=[publisher_col])[publisher_col].nunique() == 1:
        sys.exit(f"{publisher_col} has one value on {first_day}: is `slice` in params.yaml your variant's slice?")
    shutil.rmtree(stats_root, ignore_errors=True)
    for day in stats_days:
        stats = compute_stats(
            clean_root=clean_root,
            day=day,
            key_types=params["stats"]["key_types"],
            publisher_col=publisher_col,
        )
        write_stats(day=day, stats=stats, root=stats_root)
        print(f"stats for {day}: {stats.height} keys")


if __name__ == "__main__":
    build_stats(clean_root=CLEAN_ROOT, stats_root=STATS_ROOT, params=load_params())
