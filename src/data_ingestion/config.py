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
        self.S3_DOCUMENT_PREFIX = os.getenv("S3_DOCUMENT_PREFIX", "")

        # ============================================================================
        # OpenAI Configuration
        # ============================================================================
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 1536))

        # ============================================================================
        # Pinecone Configuration
        # ============================================================================
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-documents")
        self.PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")

        # ============================================================================
        # Pipeline Configuration
        # ============================================================================
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
        self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.85))
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))
        self.UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", 100))

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

    def validate(self) -> bool:
        """
        Validate required configuration parameters.

        Returns:
            True if all required configs are present, False otherwise
        """
        required_configs = [
            ("OPENAI_API_KEY", self.OPENAI_API_KEY),
            ("PINECONE_API_KEY", self.PINECONE_API_KEY),
        ]

        missing = []
        for config_name, config_value in required_configs:
            if not config_value:
                missing.append(config_name)

        if missing:
            print(f"⚠ Missing required configuration: {', '.join(missing)}")
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
