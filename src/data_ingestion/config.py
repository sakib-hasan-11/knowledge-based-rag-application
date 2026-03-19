"""
Configuration Management Module

Handles all environment variables and configuration for the data ingestion pipeline.
Loads from .env file using python-dotenv.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class Config:
    """Configuration class for data ingestion pipeline"""

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration from environment variables.

        Args:
            env_file: Optional path to .env file (defaults to project root .env)
        """
        # Load .env file
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to find .env in project root
            project_root = Path(__file__).parent.parent.parent
            env_path = project_root / ".env"
            if env_path.exists():
                load_dotenv(env_path)
            else:
                load_dotenv()

        self._load_configuration()

    def _load_configuration(self) -> None:
        """Load all configuration from environment variables"""

        # ============================================================================
        # AWS Configuration
        # ============================================================================
        self.AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
        self.AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        self.AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

        # ============================================================================
        # S3 Configuration
        # ============================================================================
        self.S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
        self.S3_DOCUMENT_PREFIX = os.getenv("S3_DOCUMENT_PREFIX", "raw_data/aapl-20250927.html")

        # ============================================================================
        # OpenAI Configuration
        # ============================================================================
        self.OPENAI_API_KEY = os.getenv(
            "GITHUB_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")
        )
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        # Safe type conversion with error handling
        try:
            self.EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid EMBEDDING_DIMENSION: must be an integer, got '{os.getenv('EMBEDDING_DIMENSION')}'"
            )

        # ============================================================================
        # Pinecone Configuration
        # ============================================================================
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "apple-rag-project")
        self.PINECONE_HOST = os.getenv(
            "PINECONE_HOST"
        )  # Required for actual data upsert
        self.PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")

        # ============================================================================
        # Pipeline Configuration
        # ============================================================================
        try:
            self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
            if self.CHUNK_SIZE <= 0 or self.CHUNK_SIZE > 10000:
                raise ValueError(
                    f"CHUNK_SIZE must be between 1 and 10000, got {self.CHUNK_SIZE}"
                )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid CHUNK_SIZE: {str(e)}")

        try:
            self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
            if self.CHUNK_OVERLAP < 0 or self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
                raise ValueError(
                    f"CHUNK_OVERLAP must be between 0 and CHUNK_SIZE-1, got {self.CHUNK_OVERLAP}"
                )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid CHUNK_OVERLAP: {str(e)}")

        try:
            self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
            if not (0.0 <= self.SIMILARITY_THRESHOLD <= 1.0):
                raise ValueError(
                    f"SIMILARITY_THRESHOLD must be between 0.0 and 1.0, got {self.SIMILARITY_THRESHOLD}"
                )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid SIMILARITY_THRESHOLD: {str(e)}")

        try:
            self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
            if self.BATCH_SIZE <= 0 or self.BATCH_SIZE > 10000:
                raise ValueError(
                    f"BATCH_SIZE must be between 1 and 10000, got {self.BATCH_SIZE}"
                )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid BATCH_SIZE: {str(e)}")

        try:
            self.UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "100"))
            if self.UPSERT_BATCH_SIZE <= 0 or self.UPSERT_BATCH_SIZE > 10000:
                raise ValueError(
                    f"UPSERT_BATCH_SIZE must be between 1 and 10000, got {self.UPSERT_BATCH_SIZE}"
                )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid UPSERT_BATCH_SIZE: {str(e)}")

        # ============================================================================
        # Output Configuration
        # ============================================================================
        self.OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./data/extracted_documents")
        self.SAVE_EXTRACTED_DOCS = (
            os.getenv("SAVE_EXTRACTED_DOCS", "true").lower() == "true"
        )

        # ============================================================================
        # Logging Configuration
        # ============================================================================
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FORMAT = os.getenv("LOG_FORMAT", "json")  # 'json' or 'plain'
        self.ENABLE_CLOUDWATCH = (
            os.getenv("ENABLE_CLOUDWATCH", "false").lower() == "true"
        )
        self.CLOUDWATCH_LOG_GROUP = os.getenv(
            "CLOUDWATCH_LOG_GROUP", "/aws/lambda/rag-ingestion"
        )
        self.CLOUDWATCH_LOG_STREAM = os.getenv(
            "CLOUDWATCH_LOG_STREAM", "data-ingestion"
        )

    def validate(self, strict: bool = False) -> bool:
        """
        Validate required configuration parameters.

        Args:
            strict: If True, requires all production credentials.
                   If False (default), only validates format/bounds but allows missing credentials
                   (suitable for testing with mocks).

        Returns:
            True if configuration is valid, False otherwise
        """
        issues = []

        # Always validate numeric parameters (even if credentials missing)
        if self.CHUNK_SIZE <= 0 or self.CHUNK_SIZE > 10000:
            issues.append(
                f"CHUNK_SIZE must be between 1 and 10000, got {self.CHUNK_SIZE}"
            )
        if self.CHUNK_OVERLAP < 0 or self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            issues.append(
                f"CHUNK_OVERLAP must be between 0 and CHUNK_SIZE-1, got {self.CHUNK_OVERLAP}"
            )
        if not (0.0 <= self.SIMILARITY_THRESHOLD <= 1.0):
            issues.append(
                f"SIMILARITY_THRESHOLD must be between 0.0 and 1.0, got {self.SIMILARITY_THRESHOLD}"
            )
        if self.BATCH_SIZE <= 0 or self.BATCH_SIZE > 10000:
            issues.append(
                f"BATCH_SIZE must be between 1 and 10000, got {self.BATCH_SIZE}"
            )
        if self.UPSERT_BATCH_SIZE <= 0 or self.UPSERT_BATCH_SIZE > 10000:
            issues.append(
                f"UPSERT_BATCH_SIZE must be between 1 and 10000, got {self.UPSERT_BATCH_SIZE}"
            )

        # Only in strict mode (production), require credentials
        if strict:
            # Validate AWS Configuration (required for production)
            if not self.AWS_ACCESS_KEY_ID:
                issues.append("AWS_ACCESS_KEY_ID is required but not set")
            if not self.AWS_SECRET_ACCESS_KEY:
                issues.append("AWS_SECRET_ACCESS_KEY is required but not set")
            if not self.S3_BUCKET_NAME:
                issues.append("S3_BUCKET_NAME is required but not set")

            # Validate OpenAI Configuration (required for Stage 2+)
            if not self.OPENAI_API_KEY:
                issues.append(
                    "OPENAI_API_KEY is required but not set (needed for retrieval and argumentation stages)"
                )

            # Validate Pinecone Configuration (required for Stage 2+)
            if not self.PINECONE_API_KEY:
                issues.append(
                    "PINECONE_API_KEY is required but not set (needed for vector database operations)"
                )
            if not self.PINECONE_HOST:
                issues.append(
                    "PINECONE_HOST is required but not set (needed for vector database operations)"
                )

        if issues:
            for issue in issues:
                print(f"[ERROR] {issue}")
            return False

        return True

    def __repr__(self) -> str:
        """String representation of configuration (without sensitive data)"""
        safe_attrs = [
            "AWS_REGION",
            "S3_BUCKET_NAME",
            "EMBEDDING_MODEL",
            "EMBEDDING_DIMENSION",
            "PINECONE_INDEX_NAME",
            "CHUNK_SIZE",
            "BATCH_SIZE",
            "LOG_LEVEL",
            "OUTPUT_DIR",
        ]

        config_items = {k: getattr(self, k, "NOT SET") for k in safe_attrs}
        return str(config_items)


# Global configuration instance
config = Config()
