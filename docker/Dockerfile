# Whisper dev image.
#
# redback and whisper_labia are installed *editable* inside the running container
# (from mounted source), so this image only carries Python + a build toolchain +
# the pinned third-party libraries. Per project rule, we never delete anything
# (e.g. we do NOT purge apt lists).
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build toolchain for redback's compiled dependencies
# (afterglowpy, extinction, sncosmo, lxml, ...).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        git \
        pkg-config \
        libxml2-dev \
        libxslt1-dev \
        ca-certificates

RUN python -m pip install --upgrade pip setuptools wheel Cython

# Pin numpy/scipy to match redback, then the two beta samplers, so the whole
# environment resolves against numpy 1.26 up front.
RUN pip install "numpy==1.26.0" "scipy<1.14.0" emcee dynesty

WORKDIR /workspace
CMD ["sleep", "infinity"]
