# Container image for the Deep Research Agent API (Stage 10).
#
# uv variant: copy the uv binary from its official image, then install from the
# committed uv.lock for a reproducible build. `--frozen` makes the build fail
# loudly if the lockfile is out of date with pyproject.toml.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /bin/uv

WORKDIR /app

# Copy dependency manifests first so this layer is cached and only re-runs when
# the dependencies change — not on every source edit.
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# `--no-dev` skips the dev group (pytest); installing the project also puts the
# `research_agent` package on the import path, so no PYTHONPATH is needed.
RUN uv sync --frozen --no-dev

EXPOSE 8000

# 0.0.0.0 so the server is reachable from outside the container.
CMD ["uv", "run", "uvicorn", "research_agent.api:api", "--host", "0.0.0.0", "--port", "8000"]
