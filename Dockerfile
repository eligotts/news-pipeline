# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy all source files needed for install
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install dependencies (non-editable for production)
RUN uv pip install --system --no-cache .

# Set Python path
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Run the pipeline
CMD ["python", "scripts/run_pipeline.py"]
