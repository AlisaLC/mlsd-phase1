"""Tests of the TAs' code: the metric, the drop reader, the synthetic data and the downloader."""

import gzip
import hashlib
import urllib.error
import zipfile

import pytest

from ta_kit import download, list_day_files, nce, read_day
from ta_kit.drops import APP_SITE_ID
from ta_kit.synth import write_drop


def test_nce_of_the_mean_is_one():
    assert nce([0, 1, 0, 0], [0.25] * 4) == pytest.approx(1)


def test_nce_of_a_perfect_predictor_is_near_zero():
    assert nce([0, 1, 1], [0, 1, 1]) < 1e-12


@pytest.mark.parametrize(
    ("y", "p", "message"),
    [
        pytest.param([0, 0], [0.1, 0.2], "every label is the same", id="constant labels"),
        pytest.param([0, 1], [0.1], "same length", id="lengths differ"),
        pytest.param([], [], "non-empty", id="empty"),
        pytest.param([[0, 1]], [[0.1, 0.2]], "1-D", id="2-D"),
        pytest.param([0, 2], [0.1, 0.2], "0 or 1", id="label 2"),
        pytest.param([0, 1], [0.1, float("nan")], "NaN", id="NaN prediction"),
    ],
)
def test_nce_rejects_bad_input(y, p, message):
    with pytest.raises(ValueError, match=message):
        nce(y, p)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        pytest.param("2014-10-21.csv.gz", "must be named", id="dashes"),
        pytest.param("day1.csv.gz", "must be named", id="not eight digits"),
        pytest.param("20141321.csv.gz", "is not a date", id="month 13"),
    ],
)
def test_list_day_files_rejects_bad_names(tmp_path, name, message):
    (tmp_path / name).touch()
    with pytest.raises(ValueError, match=message):
        list_day_files(tmp_path)


def test_list_day_files_rejects_an_empty_folder(tmp_path):
    with pytest.raises(ValueError, match="no YYYYMMDD.csv.gz files"):
        list_day_files(tmp_path)


def test_list_day_files_maps_days_and_ignores_other_files(tmp_path):
    (tmp_path / "20141022.csv.gz").touch()
    (tmp_path / "20141021.csv.gz").touch()
    (tmp_path / ".gitkeep").touch()
    assert list_day_files(tmp_path) == {
        20141021: tmp_path / "20141021.csv.gz",
        20141022: tmp_path / "20141022.csv.gz",
    }


@pytest.mark.parametrize("traffic_slice", ["site", "app"])
def test_synth_is_deterministic(tmp_path, traffic_slice):
    for folder in ("a", "b"):
        write_drop(out_dir=tmp_path / folder, days=[20141021], rows_per_day=100, traffic_slice=traffic_slice)
    assert (tmp_path / "a/20141021.csv.gz").read_bytes() == (tmp_path / "b/20141021.csv.gz").read_bytes()


def make_row(row_id: int, hour: int, site_id: str) -> str:
    """A raw row with the given id, hour and site_id, and 0 in every other column."""
    fields = ["0"] * len(download.RAW_COLUMNS)
    fields[0], fields[download.HOUR_INDEX], fields[download.SITE_ID_INDEX] = str(row_id), str(hour), site_id
    return ",".join(fields)


@pytest.mark.parametrize(
    ("traffic_slice", "expected_ids"),
    [
        pytest.param("site", {20141021: ["1", "3", "4", "6"], 20141022: ["5", "2"]}, id="site"),
        pytest.param("app", {20141021: ["9"], 20141022: ["8"]}, id="app"),
    ],
)
def test_write_drop_merges_the_members_into_one_sorted_file_per_day(tmp_path, traffic_slice, expected_ids):
    site = "1fbe01fe"
    # Like Avazu_x4's train.csv, a member is not sorted by hour: row 5 (day 22) comes before row 6 (day 21).
    members = {
        "train.csv": [
            make_row(row_id=3, hour=14102100, site_id=site), make_row(row_id=9, hour=14102100, site_id=APP_SITE_ID),
            make_row(row_id=5, hour=14102200, site_id=site), make_row(row_id=6, hour=14102101, site_id=site),
        ],
        "valid.csv": [
            make_row(row_id=1, hour=14102100, site_id=site), make_row(row_id=2, hour=14102223, site_id=site),
            make_row(row_id=7, hour=14102500, site_id=site), make_row(row_id=8, hour=14102201, site_id=APP_SITE_ID),
        ],
        "test.csv": [make_row(row_id=4, hour=14102101, site_id=site)],
    }
    with zipfile.ZipFile(tmp_path / "Avazu_x4.zip", mode="w") as archive:
        for name, lines in members.items():
            archive.writestr(name, data="\n".join([download.HEADER, *lines]) + "\n")
    out_dir = tmp_path / "out"
    row_counts = download.write_drop(
        archive_path=tmp_path / "Avazu_x4.zip", traffic_slice=traffic_slice, days=[20141021, 20141022], out_dir=out_dir
    )
    assert row_counts == {day: len(ids) for day, ids in expected_ids.items()}
    # No .part file is left behind.
    assert sorted(path.name for path in out_dir.iterdir()) == ["20141021.csv.gz", "20141022.csv.gz"]
    assert {day: read_day(out_dir / f"{day}.csv.gz")["id"].to_list() for day in expected_ids} == expected_ids
    assert gzip.decompress((out_dir / "20141022.csv.gz").read_bytes()).startswith(download.HEADER.encode())


def test_download_finishes_a_complete_part_file(tmp_path, monkeypatch):
    # A run stopped while hashing leaves a complete .part; the server then answers 416 to the range request.
    content = b"the whole archive"
    (tmp_path / "Avazu_x4.zip.part").write_bytes(content)
    monkeypatch.setattr(download, "SHA256", hashlib.sha256(content).hexdigest())

    def answer_416(request, timeout):
        """What the server answers to a range that starts at the end of the file."""
        raise urllib.error.HTTPError(url=request.full_url, code=416, msg="Range Not Satisfiable", hdrs=None, fp=None)

    monkeypatch.setattr(download.urllib.request, "urlopen", answer_416)
    download.download(url="https://example.invalid/Avazu_x4.zip", path=tmp_path / "Avazu_x4.zip")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["Avazu_x4.zip"]
