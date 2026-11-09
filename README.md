# Mini-CTR

- Name: TODO
- Student ID: TODO (also in `params.yaml`)
- Variant: TODO (A, B, C or D)

## How to run

Everything runs in Docker. The host needs Docker with Compose v2, GNU make and git (Linux, macOS, or Windows
through WSL2, with the repository inside the WSL file system). Nothing else is installed on the host. On a full
day, the pipeline container needs about 4 GB of memory and MLflow about 0.5 GB, so give Docker at least 6 GB (Docker
Desktop: Settings > Resources; WSL2: `memory=` in `%UserProfile%\.wslconfig`).

```bash
make download    # your slice of drop 1 into drops/, one YYYYMMDD.csv.gz per day
make train
```

`make download` reads your slice from `params.yaml`, so set it first (Setup, step 2). It downloads Avazu once
(1.3 GB, about 7 minutes at 3 MB/s, resumed if interrupted) into `.cache/`, then writes the days, which takes a
few more minutes. If Hugging Face is blocked, download from its mirror:

```bash
export AVAZU_URL=https://hf-mirror.com/datasets/reczoo/Avazu_x4/resolve/a436154d73738b1ea35acf3d1cc56f72997266d9/Avazu_x4.zip
make download
```

To keep the drops elsewhere, set `RAW_DIR` for both commands. When Phase 2 opens, `make download DROP=2` adds days
5-7 to the same folder. Delete `.cache/` once you have both drops.

| Target | What it does |
|---|---|
| `make download` | writes your slice of drop `DROP` (1 by default) to `RAW_DIR` (`drops/` by default) |
| `make data` | ingests and validates every day in `RAW_DIR` (`dvc repro ingest`, `RAW_DIR` mounted at `data/raw`) |
| `make train` | `make data`, starts MLflow, `dvc repro`, then `src/train/register.py`, which moves `champion` |
| `make test` | the public tests, on synthetic data (no `RAW_DIR` needed) |
| `make synth` | writes a small synthetic drop of your slice to `synth-raw/` (see below the table) |
| `make shell` | a shell in the pipeline container; `RAW_DIR` is not mounted there, so run stages with `make` |
| `make lock` | re-pins `requirements.txt` after you add a package to `requirements.in` |
| `make push` | `dvc push` to a local remote, `../mini-ctr-dvc-remote` by default (`DVC_REMOTE=...`) |
| `make mlflow` | starts MLflow alone, for the UI |
| `make down` | stops MLflow |
| `make serve`, `make bench` | added in Phase 2 |

`RAW_DIR=synth-raw make train` tries the whole pipeline without the real data, but it rewrites `dvc.lock`,
`metrics/p1.json` and the `champion` alias: run `make train` on your real drops again before you commit.

The MLflow UI is at <http://localhost:5000> while MLflow runs. If port 5000 is taken (macOS uses it for AirPlay),
set another one, such as `export MLFLOW_PORT=5050`.

## Design

TODO: at most one page, answering the four questions of the handout's Design section.

---

The rest of this file describes the starter repository. Keep it: the Phase 2 pull updates it.

## Setup

1. Create an empty private repository, then clone the starter and point it at yours. The starter stays as the
   `upstream` remote, since the Phase 2 parts are added to it when Phase 2 opens:
   ```bash
   git clone <starter repository URL> mini-ctr && cd mini-ctr
   git remote rename origin upstream
   git remote add origin <your private repository URL>
   git push -u origin main
   ```
   When Phase 2 opens, `git pull --no-rebase upstream main` merges its parts into your history. Phase 2 adds files
   and changes the Makefile, the compose file, the requirements and this README, so the pull can conflict where you
   changed those too. Keep your parts and take the new ones.
2. In `params.yaml`, set `student_id` and `slice` (`site` for variants A and B, `app` for C and D). Registration,
   the last step of `make train`, refuses the placeholder `student_id`.
3. Run `make test`. The first run builds the image `mini-ctr-pipeline:local`, which takes a few minutes. Later runs
   reuse it; it is rebuilt only when `requirements.txt`, `docker/pipeline.Dockerfile` or `BASE_IMAGE` changes. On the
   untouched starter, every test that reaches a `TODO(student)` fails with its `NotImplementedError`, and the others
   pass. If the build fails on a blocked network, set up the mirrors below first. On Linux, add yourself to the
   `docker` group rather than running `sudo make`, which writes files owned by root.

If Docker Hub, Debian or PyPI is blocked or slow on your network, configure a registry mirror for Docker
(`registry-mirrors` in `/etc/docker/daemon.json` on Linux, Settings > Docker Engine in Docker Desktop), and export
the mirrors in your shell profile, so that every build uses them, including the automatic rebuild after
`requirements.txt` changes:

```bash
export APT_MIRROR=https://repo.hmirror.ir/debian PIP_INDEX_URL=https://repo.hmirror.ir/python/simple/
```

These worked from Iran in November 2026; any Debian and PyPI mirror does.

`APT_MIRROR` replaces Debian's sources with the mirror's main and updates suites, and needs a Debian base image.
Set `BASE_IMAGE` only to pull the same Python image from another registry: the service in Phase 2 runs Python
3.11 and must load your models. Changing a mirror does not rebuild an image that is up to date.

## What you write

Every part you write raises `NotImplementedError("TODO(student): ...")` until you do, so `make test` and
`make train` name the file and function of the first missing part they reach. Most tests build the synthetic data
through ingest and stats first, so write the parts in the order of the table.

| Where | What | Graded in |
|---|---|---|
| `src/ingest/schema.py` `check_day` | `hour` is a valid YYMMDDHH on the file's day | Ingestion and validation |
| `src/ingest/run.py` `add_time_columns` | `day` and `hour_of_day` from `hour` | Ingestion and validation |
| `src/stats/build.py` `compute_stats` | the CTR-stats table for one day, from earlier days | Stats table |
| `src/features/build.py` `build_features` | the feature module | Registered baseline, Reproducibility |
| `src/train/baseline.py` `BaselineModel.predict`, `tune` | the baseline | Registered baseline, Reproducibility |
| `src/train/register.py` `read_git_commit`, `read_data_hash` | the lineage of every run | Registered baseline |
| `README.md` | the Design section above | Design section |

Column presence and types, `click` in {0, 1} and non-null fields are already checked by `src/ingest/schema.py`.
The feature tests use `{"publisher_col": ..., "alpha": ...}` as feature parameters, so they run before `tune` is
written.

You may change any given file outside `ta_kit` as long as the commands above and the handout's contracts still
hold. To add a package, add it to `requirements.in`, run `make lock` and commit both files; the next target
rebuilds the image.

## How the pieces fit

```
RAW_DIR ──(mounted at data/raw)──> ingest ──> data/clean/day=*/ ──> stats ──> stats/day=*/
                                                    └──────────────┬─────────────┘
                                                                baseline ──> models/baseline/, metrics/p1.json
                                       make train, after dvc repro: register.py ──> MLflow: ctr-<student_id>@champion
```

- The DVC stages are in `dvc.yaml`; the parameters they read are in `params.yaml`, so changing a parameter reruns
  only the stages that use it.
- Registration is not a DVC stage: `make train` runs `src/train/register.py` after `dvc repro` every time, so each
  run adds new model versions; the service follows the `champion` alias.
- A model directory under `models/` is an MLflow pyfunc model plus `feature_params.json` (the parameters
  `build_features` needs) and `train_info.json` (the parameters and metrics `register.py` logs). Use
  `src/train/model.py` to save one.
- `ta_kit` is the TAs' code: `nce`, the drop reader and downloader, and the synthetic data. Do not change it;
  grading uses ours.
- MLflow keeps its database in `mlflow.db` and its artifacts in `mlartifacts/`, served through the MLflow server,
  so other containers can load models from `models:/ctr-<student_id>@champion`.
- Commit `dvc.lock` and `metrics/p1.json` after `make train` on your real drops, not on `synth-raw`.

## Before you tag

Check that a clean machine can rebuild everything from what you pushed:

```bash
git push origin main
make down                     # the check starts its own MLflow on the same port
git clone <your repository> ../mini-ctr-check && cd ../mini-ctr-check
RAW_DIR=../mini-ctr/drops make train    # your drops, from the original clone
```

A fresh clone has no DVC cache, no `mlflow.db` and no outputs, which is the state the grading machine starts from.
Its `metrics/p1.json` should match the one you committed to about 1e-4. Then run `make down` there and delete the
folder. After `git tag p1 && git push origin p1`, repeat the check with `git checkout p1` after the `cd`: grading
checks out only the tag.
