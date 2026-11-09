"""Tests of the lineage that src/train/register.py logs with every run."""

import subprocess

from src.train.register import read_data_hash, read_git_commit

# The part of a dvc.lock that read_data_hash reads, as `dvc repro` writes it.
DVC_LOCK = """schema: '2.0'
stages:
  ingest:
    cmd: python -m src.ingest.run
    outs:
    - path: data/clean
      hash: md5
      md5: 0123456789abcdef0123456789abcdef.dir
      size: 1000
      nfiles: 4
  stats:
    cmd: python -m src.stats.build
    outs:
    - path: stats
      hash: md5
      md5: fedcba9876543210fedcba9876543210.dir
      size: 100
      nfiles: 3
"""


def git(*args: str) -> str:
    """Runs git in the current folder and returns its output."""
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def test_read_data_hash_reads_the_ingest_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dvc.lock").write_text(DVC_LOCK)
    assert read_data_hash() == "0123456789abcdef0123456789abcdef.dir"


def test_read_git_commit_is_dirty_only_for_other_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    git("init", "-q")
    (tmp_path / "metrics").mkdir()
    for name in ("dvc.lock", "metrics/p1.json", "params.yaml"):
        (tmp_path / name).write_text("{}\n")
    git("add", "-A")
    git("-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-qm", "first")
    head = git("rev-parse", "HEAD")
    assert read_git_commit() == head
    # make train rewrites these itself, so they do not make the commit dirty.
    (tmp_path / "dvc.lock").write_text("changed\n")
    (tmp_path / "metrics/p1.json").write_text("changed\n")
    assert read_git_commit() == head
    (tmp_path / "notes.txt").write_text("an untracked file\n")
    assert read_git_commit() == f"{head}-dirty"
