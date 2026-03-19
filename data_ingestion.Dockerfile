# ============================================================================
# Data Ingestion Pipeline — ECS Task Image
# ============================================================================
# Runs: run_pipeline.py  (lists S3 docs → chunk → embed → upsert to Pinecone)
#
# Build:
#   docker build -t rag-ingestion:latest .
#
# Run locally (for testing):
#   docker run --rm \
#     -e OPENAI_API_KEY=sk-... \
#     -e PINECONE_API_KEY=... \
#     -e PINECONE_HOST=... \
#     -e S3_BUCKET_NAME=my-bucket \
#     -e AWS_REGION=us-east-1 \
#     -e AWS_ACCESS_KEY_ID=... \
#     -e AWS_SECRET_ACCESS_KEY=... \
#     rag-ingestion:latest
#
# In ECS: supply secrets via task definition environment / AWS Secrets Manager.
#         AWS credentials are injected automatically via the ECS task IAM role.
# ============================================================================

FROM python:3.10-slim





# --------------------------------------------------------------------------
# Runtime environment
# --------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Make src/ importable as a package from anywhere inside /app
    PYTHONPATH=/app \
    # Sane pip defaults
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1






# --------------------------------------------------------------------------
# System dependencies
# --------------------------------------------------------------------------
# gcc / g++ are required to compile some Python packages (e.g. tiktoken, numpy).
# libgomp1 is required by numpy/scipy at runtime on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*






# --------------------------------------------------------------------------
# Working directory
# --------------------------------------------------------------------------
WORKDIR /app





# --------------------------------------------------------------------------
# Install Python dependencies
# (Copy requirements first so Docker can cache this layer)
# --------------------------------------------------------------------------
COPY requirements-ingestion.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements-ingestion.txt






# --------------------------------------------------------------------------
# Copy application source
# --------------------------------------------------------------------------
# Only what the pipeline needs — tests, notebooks, .env, .venv are excluded
# via .dockerignore
COPY src/ ./src/




# --------------------------------------------------------------------------
# Output directory (used when SAVE_EXTRACTED_DOCS=true)
# --------------------------------------------------------------------------
RUN mkdir -p /app/data/extracted_documents





# --------------------------------------------------------------------------
# Security: run as a non-root user
# --------------------------------------------------------------------------
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser





# --------------------------------------------------------------------------
# Health-check (ECS can use this to detect stuck containers)
# --------------------------------------------------------------------------
# A simple file-existence check: the entrypoint must be present.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=1 \
    CMD python -c "import src.data_ingestion.pipeline" || exit 1





# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------
# ENTRYPOINT ensures the script always runs.
# CMD is the default argument — override in ECS task definition if needed.
ENTRYPOINT ["python", "-u", "-m", "src.data_ingestion.run_pipeline"]
