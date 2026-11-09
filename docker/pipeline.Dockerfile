# The one image of Phase 1: the pipeline (DVC stages, tests) and the MLflow
# server run from it, so the MLflow client and server versions always match.
# The repository is mounted at /work at run time, not copied.
ARG BASE_IMAGE=python:3.11.18-slim
FROM ${BASE_IMAGE}

# Where Docker Hub, Debian or PyPI is blocked, set mirrors (see the README). APT_MIRROR replaces the Debian
# sources with that mirror's main and updates suites.
ARG APT_MIRROR=
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN if [ -n "${APT_MIRROR}" ]; then \
        . /etc/os-release \
        && printf 'Types: deb\nURIs: %s\nSuites: %s %s-updates\nComponents: main\nSigned-By: %s\n' \
            "${APT_MIRROR}" "${VERSION_CODENAME}" "${VERSION_CODENAME}" /usr/share/keyrings/debian-archive-keyring.gpg \
            > /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    # git refuses a repository owned by another uid, and the mount's owner can differ from the container's user
    # (Docker Desktop, rootless Docker), so every path is trusted.
    && git config --system safe.directory '*'

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -r /tmp/requirements.txt

# Containers run as the host user, which has no passwd entry or home directory in the image;
# USER and LOGNAME stop getpass.getuser() (used by DVC) from looking the user up.
ENV HOME=/tmp \
    USER=mini-ctr \
    LOGNAME=mini-ctr \
    MLFLOW_DISABLE_TELEMETRY=true \
    DVC_NO_ANALYTICS=1 \
    PYTHONPATH=/work \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /work
