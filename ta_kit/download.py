"""Downloads Avazu and writes your slice of one drop: OUT/YYYYMMDD.csv.gz, one file per day.

    python -m ta_kit.download --slice site --drop 1 --out drops    # what `make download` runs

The source is reczoo's copy of Kaggle's Avazu training file on Hugging Face (Avazu_x4.zip, 1.3 GB), pinned to one
revision and checked by sha256. Its three CSVs (train, valid, test) hold every row of Kaggle's file with the original
columns, in a random split; each day is rebuilt from all three and sorted by (hour, id). The download resumes if
interrupted and is kept in the cache folder for the next drop. Set the environment variable AVAZU_URL (which
`make download` passes into the container) to download the same file from a mirror, such as
https://hf-mirror.com/datasets/reczoo/Avazu_x4/resolve/<REVISION>/Avazu_x4.zip.

The same arguments always give the same bytes, and the row counts are checked against Kaggle's file, so your files
are the ones we grade with.
"""

import argparse
import gzip
import hashlib
import io
import os
import sys
import typing
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path

import polars as pl

from ta_kit.drops import APP_SITE_ID, DROP_DAYS, RAW_COLUMNS, TrafficSlice

REVISION = "a436154d73738b1ea35acf3d1cc56f72997266d9"
DEFAULT_URL = f"https://huggingface.co/datasets/reczoo/Avazu_x4/resolve/{REVISION}/Avazu_x4.zip"
SHA256 = "4ba8bf46eada70abbe068e4e5bfb3b3d879e6fbd2f2fcf51760cd1388b789b65"
MEMBERS = ("train.csv", "valid.csv", "test.csv")
HEADER = ",".join(RAW_COLUMNS)
HOUR_INDEX, SITE_ID_INDEX = RAW_COLUMNS.index("hour"), RAW_COLUMNS.index("site_id")
# Rows per (slice, day) in Kaggle's train.gz, counted by the TAs.
EXPECTED_ROWS = {
    ("site", 20141021): 2590484, ("site", 20141022): 3227169, ("site", 20141023): 2644413,
    ("site", 20141024): 2238704, ("site", 20141025): 2300820, ("site", 20141026): 2662376,
    ("site", 20141027): 2126284,
    ("app", 20141021): 1532511, ("app", 20141022): 2109957, ("app", 20141023): 1226339,
    ("app", 20141024): 1096598, ("app", 20141025): 1062302, ("app", 20141026): 1173516,
    ("app", 20141027): 1098726,
}


def download(url: str, path: Path) -> None:
    """Downloads url to path, resuming a partial path.part, and checks its sha256."""
    if path.exists():
        return
    partial = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_bytes = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={downloaded_bytes}-"} if downloaded_bytes > 0 else {}
    print(f"downloading {url}" + (f" from byte {downloaded_bytes}" if downloaded_bytes > 0 else ""))
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as error:
        # 416: the range starts at the end of the file, so the .part is complete (the last run stopped while hashing).
        if error.code != 416:
            raise
    else:
        with response, open(partial, "ab" if response.status == 206 else "wb") as file:
            if response.status != 206:
                downloaded_bytes = 0  # no range was asked for, or the server ignored it: start over
            total_bytes = downloaded_bytes + int(response.headers.get("Content-Length", 0))
            next_report = downloaded_bytes
            while chunk := response.read(1 << 20):
                file.write(chunk)
                downloaded_bytes += len(chunk)
                if downloaded_bytes >= next_report:
                    print(f"  {downloaded_bytes >> 20} of {total_bytes >> 20} MB")
                    next_report += max(total_bytes // 20, 1)
    digest = hashlib.sha256()
    with open(partial, "rb") as file:
        while chunk := file.read(1 << 24):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        partial.unlink()
        sys.exit(f"{url}: the download is corrupt (sha256 {digest.hexdigest()}); run the command again")
    partial.rename(path)


def read_rows(archive: zipfile.ZipFile, member: str) -> Iterator[tuple[int, bool, str]]:
    """(day, is_app, line) for each row of one CSV in the archive."""
    with io.TextIOWrapper(archive.open(member), encoding="ascii", newline="") as lines:
        header = lines.readline().rstrip("\r\n")
        if header != HEADER:
            sys.exit(f"{member}: unexpected header {header!r}")
        for line in lines:
            # Only the fields up to site_id are needed.
            fields = line.split(",", maxsplit=SITE_ID_INDEX + 1)
            # hour is YYMMDDHH, so hour // 100 is YYMMDD.
            day = 20_000_000 + int(fields[HOUR_INDEX]) // 100
            yield day, fields[SITE_ID_INDEX] == APP_SITE_ID, line if line.endswith("\n") else line + "\n"


def write_drop(archive_path: Path, traffic_slice: TrafficSlice, days: list[int], out_dir: Path) -> dict[int, int]:
    """Writes out_dir/YYYYMMDD.csv.gz for each day of the slice, sorted by (hour, id); returns the rows per day."""
    out_dir.mkdir(parents=True, exist_ok=True)
    row_counts = dict.fromkeys(days, 0)
    # Pass 1: each day's rows, in the order read, go to an uncompressed .part file.
    unsorted: dict[int, typing.TextIO] = {
        day: open(out_dir / f"{day}.csv.part", "w", encoding="ascii") for day in days
    }
    for file in unsorted.values():
        file.write(f"{HEADER}\n")
    with zipfile.ZipFile(archive_path) as archive:
        for member in MEMBERS:
            print(f"reading {member}")
            for day, is_app, line in read_rows(archive=archive, member=member):
                if day in unsorted and is_app == (traffic_slice == "app"):
                    unsorted[day].write(line)
                    row_counts[day] += 1
    for file in unsorted.values():
        file.close()
    if 0 in row_counts.values():
        sys.exit(f"{archive_path}: no {traffic_slice} rows for {[day for day in days if row_counts[day] == 0]}")
    # Pass 2: each day sorted and gzipped, one day in memory at a time.
    for day in days:
        print(f"writing {out_dir / f'{day}.csv.gz'}")
        # Every column is read as a string, so every field is written back exactly as read. ids are unique and fit
        # in UInt64.
        day_rows = pl.read_csv(out_dir / f"{day}.csv.part", infer_schema=False).sort(
            pl.col("hour").cast(pl.Int64), pl.col("id").cast(pl.UInt64)
        )
        with open(out_dir / f"{day}.csv.gz.part", "wb") as file:
            # No file name and mtime 0 in the gzip header, so the bytes depend only on the rows.
            with gzip.GzipFile(fileobj=file, mode="wb", compresslevel=6, mtime=0, filename="") as output:
                day_rows.write_csv(output)
        del day_rows  # frees this day before the next one is read
        (out_dir / f"{day}.csv.gz.part").rename(out_dir / f"{day}.csv.gz")
        (out_dir / f"{day}.csv.part").unlink()
    return row_counts


def main() -> None:
    """Downloads the archive if needed, writes the drop and checks its row counts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slice", choices=typing.get_args(TrafficSlice), required=True)
    parser.add_argument("--drop", type=int, choices=sorted(DROP_DAYS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache"), help="where the 1.3 GB archive is kept")
    args = parser.parse_args()

    archive_path = args.cache / "Avazu_x4.zip"
    download(url=os.environ.get("AVAZU_URL") or DEFAULT_URL, path=archive_path)
    row_counts = write_drop(
        archive_path=archive_path, traffic_slice=args.slice, days=DROP_DAYS[args.drop], out_dir=args.out
    )
    for day, count in row_counts.items():
        print(f"{day}.csv.gz: {count} rows")
    wrong: dict[int, int] = {
        day: count for day, count in row_counts.items() if count != EXPECTED_ROWS[args.slice, day]
    }
    if len(wrong) > 0:
        sys.exit(f"wrong row counts {wrong}: delete {archive_path} and run make download again")


if __name__ == "__main__":
    main()
