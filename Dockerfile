# F1 Race Strategy Intelligence — backend API
#
# Code-only image: `artifacts/` and `data/` are NOT baked in here. They are
# produced by `python scripts/build_all.py` and mounted at runtime as bind
# volumes (see docker-compose.yml), matching how ./run.sh uses them from a
# host checkout. Baking either into the image would mean a stale, one-time
# copy of files the platform is designed to regenerate.
#
# Base image matches this repo's actual interpreter, not just pyproject.toml's
# ">=3.10" floor: requirements.txt pins Keras/torch because TensorFlow ships
# no wheel for Python 3.14, and the tracked .venv here runs CPython 3.14.6.
FROM python:3.14-slim

WORKDIR /app

# System deps: curl for the compose healthcheck; build-essential because a
# couple of pinned scientific wheels (this project's numpy/scipy/torch stack)
# still fall back to source builds on fresh Python releases like 3.14.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before copying the rest of the source so an
# application-code change doesn't invalidate the (slow) dependency layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Now the source needed to `pip install -e .` and run the API. `artifacts/`,
# `data/`, and `.venv` are excluded via .dockerignore — they are host-mounted,
# not shipped in the image.
COPY pyproject.toml ./
COPY app ./app
COPY scripts ./scripts
RUN pip install --no-cache-dir -e .

# artifacts/ and data/ are bind-mounted here at runtime (see
# docker-compose.yml); app.core.paths resolves everything relative to
# REPO_ROOT = /app, so these directories just need to exist as mount points.
RUN mkdir -p /app/artifacts /app/data

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
