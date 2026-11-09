"""Writes a small synthetic drop shaped like the Avazu files, for trying the pipeline without the real data.

    python -m ta_kit.synth --out synth-raw --slice site --drop 1    # the days of drop 1

Each publisher and device model has its own click rate, and some publishers appear only from a later day,
so a correct baseline beats the average CTR and leakage in the stats shows up in the tests.
The numbers mean nothing beyond that: do not tune anything on this data.
"""

import argparse
import dataclasses
import gzip
import typing
from pathlib import Path

import numpy as np
import polars as pl

from ta_kit.drops import APP_SITE_ID, DROP_DAYS, RAW_COLUMNS, TrafficSlice

DEFAULT_ROWS_PER_DAY = 50_000
DEFAULT_SEED = 1405
N_PUBLISHERS = 300
N_PUBLISHER_CATEGORIES = 20
N_DEVICE_MODELS = 150
BASE_CTR = 0.17  # about Avazu's average CTR
SHARE_OF_PUBLISHERS_FROM_DAY_ONE = 0.7  # the rest first appear on a random day, often a later one
# Spread of the per-publisher and per-device effects on the logit of the click rate.
PUBLISHER_EFFECT_STD = 0.8
DEVICE_EFFECT_STD = 0.5
HOUR_EFFECT = 0.2  # amplitude of a daily cycle in the logit
# The values Avazu uses in the columns of the other slice: app_* on site traffic, site_* on app traffic.
SITE_PLACEHOLDER = {"site_id": APP_SITE_ID, "site_domain": "c4e18dd6", "site_category": "50e219e0"}
APP_PLACEHOLDER = {"app_id": "ecad2386", "app_domain": "7801e8d9", "app_category": "07d7df22"}
# Avazu's most common device_id, and the share of rows that carry it.
COMMON_DEVICE_ID = "a99f214a"
COMMON_DEVICE_ID_SHARE = 0.8


@dataclasses.dataclass(frozen=True)
class Universe:
    """The publishers and device models of a drop, with their click effects; the same on every day."""

    publisher_id: np.ndarray
    publisher_domain: np.ndarray
    publisher_category: np.ndarray
    publisher_effect: np.ndarray
    publisher_weight: np.ndarray
    publisher_first_day: np.ndarray
    device_model: np.ndarray
    device_effect: np.ndarray
    device_weight: np.ndarray


def make_hex_ids(rng: np.random.Generator, n: int) -> np.ndarray:
    """n random 8-character hex strings, like Avazu's hashed IDs."""
    return np.char.mod("%08x", rng.integers(low=0, high=2**32, size=n))


def make_universe(seed: int, days: list[int]) -> Universe:
    """Draws the publishers and device models; a publisher's first day is days[0] or a random one of days."""
    rng = np.random.default_rng(seed)
    from_day_one = rng.random(N_PUBLISHERS) < SHARE_OF_PUBLISHERS_FROM_DAY_ONE
    # The weights are Zipf-like popularity: a few publishers and device models get most of the traffic.
    return Universe(
        publisher_id=make_hex_ids(rng=rng, n=N_PUBLISHERS),
        publisher_domain=make_hex_ids(rng=rng, n=N_PUBLISHERS),
        publisher_category=rng.choice(make_hex_ids(rng=rng, n=N_PUBLISHER_CATEGORIES), size=N_PUBLISHERS),
        publisher_effect=rng.normal(loc=0, scale=PUBLISHER_EFFECT_STD, size=N_PUBLISHERS),
        publisher_weight=rng.permutation(1 / np.arange(1, N_PUBLISHERS + 1)),
        publisher_first_day=np.where(from_day_one, days[0], rng.choice(days, size=N_PUBLISHERS)),
        device_model=make_hex_ids(rng=rng, n=N_DEVICE_MODELS),
        device_effect=rng.normal(loc=0, scale=DEVICE_EFFECT_STD, size=N_DEVICE_MODELS),
        device_weight=rng.permutation(1 / np.arange(1, N_DEVICE_MODELS + 1)),
    )


def make_day(day: int, rows: int, traffic_slice: TrafficSlice, universe: Universe, seed: int) -> pl.DataFrame:
    """One day of impressions, every column as Avazu has it."""
    rng = np.random.default_rng([seed, day])
    live_weight = np.where(universe.publisher_first_day <= day, universe.publisher_weight, 0)
    publisher_index = rng.choice(N_PUBLISHERS, size=rows, p=live_weight / live_weight.sum())
    device_index = rng.choice(N_DEVICE_MODELS, size=rows, p=universe.device_weight / universe.device_weight.sum())
    hour_of_day = rng.integers(low=0, high=24, size=rows)
    logit = (
        np.log(BASE_CTR / (1 - BASE_CTR))
        + universe.publisher_effect[publisher_index]
        + universe.device_effect[device_index]
        + HOUR_EFFECT * np.sin(hour_of_day / 24 * 2 * np.pi)
    )
    click = (rng.random(rows) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    yymmdd = day % 1_000_000  # 20141021 -> 141021
    columns: dict[str, np.ndarray] = {
        # Avazu IDs do not fit in int64.
        "id": rng.integers(low=2**63, high=2**64 - 1, size=rows, dtype=np.uint64).astype(str),
        "click": click,
        "hour": yymmdd * 100 + hour_of_day,
        # The click signal is in the hour, the publisher and the device model.
        f"{traffic_slice}_id": universe.publisher_id[publisher_index],
        f"{traffic_slice}_domain": universe.publisher_domain[publisher_index],
        f"{traffic_slice}_category": universe.publisher_category[publisher_index],
        "device_model": universe.device_model[device_index],
        # The other columns carry no click signal.
        "C1": rng.choice([1005, 1002, 1010, 1001, 1007], size=rows, p=[0.9, 0.05, 0.03, 0.01, 0.01]),
        "banner_pos": rng.choice([0, 1, 2, 7], size=rows, p=[0.7, 0.27, 0.02, 0.01]),
        "device_id": np.where(
            rng.random(rows) < COMMON_DEVICE_ID_SHARE, COMMON_DEVICE_ID, make_hex_ids(rng=rng, n=rows)
        ),
        "device_ip": make_hex_ids(rng=rng, n=rows),
        "device_type": rng.choice([1, 0, 4, 5], size=rows, p=[0.9, 0.05, 0.04, 0.01]),
        "device_conn_type": rng.choice([0, 2, 3, 5], size=rows, p=[0.85, 0.08, 0.06, 0.01]),
        "C14": rng.integers(low=15000, high=24000, size=rows),
        "C15": rng.choice([320, 300, 216, 728], size=rows, p=[0.93, 0.05, 0.01, 0.01]),
        "C16": rng.choice([50, 250, 36, 480], size=rows, p=[0.94, 0.04, 0.01, 0.01]),
        "C17": rng.integers(low=1000, high=2800, size=rows),
        "C18": rng.choice([0, 1, 2, 3], size=rows, p=[0.42, 0.06, 0.18, 0.34]),
        "C19": rng.choice([35, 39, 161, 1063], size=rows),
        # -1 is Avazu's missing value in C20.
        "C20": np.where(rng.random(rows) < 0.47, -1, rng.integers(low=100000, high=100250, size=rows)),
        "C21": rng.integers(low=1, high=200, size=rows),
    }
    placeholder = APP_PLACEHOLDER if traffic_slice == "site" else SITE_PLACEHOLDER
    columns.update({name: np.full(rows, value) for name, value in placeholder.items()})
    return pl.DataFrame({name: columns[name] for name in RAW_COLUMNS})


def write_drop(
    out_dir: Path, days: list[int], rows_per_day: int, traffic_slice: TrafficSlice, seed: int = DEFAULT_SEED
) -> None:
    """Writes out_dir/YYYYMMDD.csv.gz for each day; the same arguments always give the same bytes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    universe = make_universe(seed=seed, days=sorted(days))
    for day in days:
        day_rows = make_day(day=day, rows=rows_per_day, traffic_slice=traffic_slice, universe=universe, seed=seed)
        (out_dir / f"{day}.csv.gz").write_bytes(gzip.compress(day_rows.write_csv().encode(), mtime=0))


def main() -> None:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--drop", type=int, choices=sorted(DROP_DAYS), required=True, help="writes this drop's days")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS_PER_DAY, help="rows per day")
    parser.add_argument(
        "--slice", choices=typing.get_args(TrafficSlice), required=True, help="the slice of your variant"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    days = DROP_DAYS[args.drop]
    write_drop(out_dir=args.out, days=days, rows_per_day=args.rows, traffic_slice=args.slice, seed=args.seed)
    print(f"wrote {len(days)} days of {args.rows} {args.slice} rows to {args.out}/")


if __name__ == "__main__":
    main()
