# Every target runs in Docker; the host needs only Docker Compose v2, make and git.
SHELL := /bin/bash

# Containers run as you, so the files they write are yours.
export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)
# The host port of the MLflow UI; change it if 5000 is taken (macOS uses it for AirPlay).
export MLFLOW_PORT ?= 5000
export PIPELINE_IMAGE := mini-ctr-pipeline:local

BASE_IMAGE ?= python:3.11.18-slim
APT_MIRROR ?=
PIP_INDEX_URL ?= https://pypi.org/simple
DVC_REMOTE ?= ../mini-ctr-dvc-remote
# The folder with your drops: make download fills it, make data reads it.
RAW_DIR ?= drops
DROP ?= 1
SLICE = $(shell sed -n 's/^slice:[^a-z]*\([a-z]*\).*/\1/p' params.yaml)

# An exported CDPATH makes `cd` print the folder, which would break the -v arguments below.
unexport CDPATH

COMPOSE := docker compose
RUN := $(COMPOSE) run --rm pipeline
# The path is resolved by the shell, so a RAW_DIR with spaces works.
RUN_RAW = $(COMPOSE) run --rm -v "$$(cd "$(RAW_DIR)" && pwd):/work/data/raw:ro" pipeline

.PHONY: download data train serve bench test synth shell push mlflow down image lock check-raw

# Your slice of drop DROP from Avazu (a 1.3 GB download, kept in .cache/ for the next drop) into RAW_DIR.
download: image
	mkdir -p "$(RAW_DIR)"
	$(COMPOSE) run --rm -v "$$(cd "$(RAW_DIR)" && pwd):/drops" -e AVAZU_URL pipeline \
		python -m ta_kit.download --slice $(SLICE) --drop $(DROP) --out /drops

data: check-raw image
	$(RUN_RAW) dvc repro ingest

train: data mlflow
	$(RUN_RAW) dvc repro
	$(RUN) python -m src.train.register

serve bench:
	@echo "make $@ is added to the starter repository when Phase 2 opens."
	@exit 1

test: image
	$(RUN) pytest

# A synthetic drop with drop 1's days in your slice (params.yaml), for trying make train without the real data.
synth: image
	$(RUN) python -m ta_kit.synth --out synth-raw --drop 1 --slice $(SLICE)

shell: image
	$(RUN) bash

push: image
	mkdir -p "$(DVC_REMOTE)"
	$(COMPOSE) run --rm -v "$$(cd "$(DVC_REMOTE)" && pwd):/dvc-remote" pipeline dvc push

mlflow: image
	$(COMPOSE) up -d --wait mlflow

down:
	$(COMPOSE) down

check-raw:
	@test -d "$(RAW_DIR)" || { echo "RAW_DIR=$(RAW_DIR) is not a folder: run make download first."; exit 1; }
	@ls "$(RAW_DIR)"/*.csv.gz >/dev/null 2>&1 || \
		{ echo "RAW_DIR=$(RAW_DIR) has no .csv.gz files: run make download first."; exit 1; }

# Builds the image when it is missing or was built from another base image, Dockerfile or lock file. The image
# carries a hash of the three as a label, so a fresh clone reuses an image that is up to date. The mirrors are not
# part of the hash: they change where packages come from, not which ones. git hashes it, since sha256sum is not
# on every host.
IMAGE_INPUTS := $(shell (echo $(BASE_IMAGE); cat docker/pipeline.Dockerfile requirements.txt) | git hash-object --stdin)

image:
	@test -n "$(IMAGE_INPUTS)" || { echo "Could not hash the image inputs; is git installed?"; exit 1; }
	@if [ "$$(docker image inspect -f '{{index .Config.Labels "mini-ctr.inputs"}}' $(PIPELINE_IMAGE) 2>/dev/null)" \
		!= "$(IMAGE_INPUTS)" ]; then \
		docker build -f docker/pipeline.Dockerfile -t $(PIPELINE_IMAGE) --label mini-ctr.inputs=$(IMAGE_INPUTS) \
			--build-arg BASE_IMAGE=$(BASE_IMAGE) --build-arg APT_MIRROR=$(APT_MIRROR) \
			--build-arg PIP_INDEX_URL=$(PIP_INDEX_URL) . ; \
	fi

# Re-pins requirements.txt from requirements.in; run it after adding a package there. The pins are versions only
# (no hashes), resolved for Linux on x86_64; an arm64 image installs the same versions.
lock:
	docker run --rm -u $(HOST_UID):$(HOST_GID) -e HOME=/tmp -v "$(CURDIR):/work" -w /work $(BASE_IMAGE) sh -c \
		"pip install --quiet --user --index-url $(PIP_INDEX_URL) uv==0.13.7 && python -m uv pip compile \
		--python-version 3.11 --python-platform linux --index-url $(PIP_INDEX_URL) \
		--custom-compile-command 'make lock' --no-annotate requirements.in -o requirements.txt"
