# Use uv's official Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0

COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY app/ ./app/
COPY docs/ ./docs/
COPY .streamlit/ ./.streamlit/
COPY streamlit_app.py ./

RUN uv sync --locked --no-install-project
RUN uv sync --locked --no-editable

FROM python:3.12-slim-bookworm AS production

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser
WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/app/ ./app/
COPY --from=builder --chown=appuser:appuser /app/docs/ ./docs/
COPY --from=builder --chown=appuser:appuser /app/.streamlit/ ./.streamlit/
COPY --from=builder --chown=appuser:appuser /app/streamlit_app.py ./

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

USER appuser
EXPOSE 8501
WORKDIR /app
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["/app/.venv/bin/streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
