#!/usr/bin/env python3
"""
ECS Entrypoint for the Data Ingestion Pipeline.

Lists all HTML documents from the configured S3 bucket and runs the
complete ingestion pipeline (load → parse → chunk → embed → upsert to Pinecone).

All configuration is driven by environment variables (set in ECS task definition):
    Required:
        OPENAI_API_KEY       (or GITHUB_OPENAI_API_KEY)
        PINECONE_API_KEY
        PINECONE_HOST
        S3_BUCKET_NAME
        AWS_REGION           (default: us-east-1)

    Optional (have defaults in config.py):
        S3_DOCUMENT_PREFIX   PINECONE_INDEX_NAME   EMBEDDING_MODEL
        CHUNK_SIZE            CHUNK_OVERLAP          BATCH_SIZE
        LOG_LEVEL             LOG_FORMAT             ENABLE_CLOUDWATCH
        OUTPUT_DIR            SAVE_EXTRACTED_DOCS

Note: In ECS, AWS credentials are supplied automatically via the task IAM role —
      you do NOT need AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars.
"""

import json
import sys

from src.data_ingestion.config import config
from src.data_ingestion.document_loader import S3DocumentLoader
from src.data_ingestion.logging_config import create_logger
from src.data_ingestion.pipeline import create_and_run_pipeline


def main() -> None:
    logger = create_logger(
        name="PipelineRunner",
        level=config.LOG_LEVEL,
        log_format=config.LOG_FORMAT,
        enable_cloudwatch=config.ENABLE_CLOUDWATCH,
    )

    logger.info(
        "ECS Data Ingestion Pipeline starting",
        {
            "s3_bucket": config.S3_BUCKET_NAME,
            "s3_prefix": config.S3_DOCUMENT_PREFIX,
            "pinecone_index": config.PINECONE_INDEX_NAME,
            "embedding_model": config.EMBEDDING_MODEL,
            "log_level": config.LOG_LEVEL,
        },
    )

    # ------------------------------------------------------------------
    # Validate essential configuration before doing any real work
    # ------------------------------------------------------------------
    if not config.S3_BUCKET_NAME:
        logger.error("S3_BUCKET_NAME is not set — cannot discover documents. Exiting.")
        sys.exit(1)

    if not config.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set — cannot generate embeddings. Exiting.")
        sys.exit(1)

    if not config.PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY is not set — cannot upload vectors. Exiting.")
        sys.exit(1)

    if not config.PINECONE_HOST:
        logger.error("PINECONE_HOST is not set — cannot connect to Pinecone. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Discover documents from S3
    # ------------------------------------------------------------------
    logger.info(
        "Listing HTML documents from S3",
        {"bucket": config.S3_BUCKET_NAME, "prefix": config.S3_DOCUMENT_PREFIX},
    )

    loader = S3DocumentLoader()
    document_sources = loader.list_documents(file_extension=".html")

    if not document_sources:
        logger.error(
            "No HTML documents found in S3 bucket — nothing to ingest. Exiting.",
            {"bucket": config.S3_BUCKET_NAME, "prefix": config.S3_DOCUMENT_PREFIX},
        )
        sys.exit(1)

    logger.info(
        f"Discovered {len(document_sources)} documents — starting pipeline",
        {"document_count": len(document_sources)},
    )

    # ------------------------------------------------------------------
    # Run the complete pipeline
    # ------------------------------------------------------------------
    results = create_and_run_pipeline(
        document_sources=document_sources,
        from_s3=True,
        enable_cloudwatch=config.ENABLE_CLOUDWATCH,
    )

    # ------------------------------------------------------------------
    # Log final summary and exit
    # ------------------------------------------------------------------
    logger.info(
        "Pipeline finished",
        {
            "success": results.get("overall_success"),
            "documents_ingested": results.get("total_documents_ingested"),
            "chunks_created": results.get("total_chunks_created"),
            "embeddings_generated": results.get("total_embeddings"),
            "vectors_in_pinecone": results.get("vectors_in_pinecone"),
            "duration_seconds": results.get("duration_seconds"),
            "errors": results.get("errors", []),
        },
    )

    # Print JSON summary — visible in CloudWatch / ECS logs
    print(json.dumps(results, indent=2, default=str))

    sys.exit(0 if results.get("overall_success") else 1)


if __name__ == "__main__":
    main()
