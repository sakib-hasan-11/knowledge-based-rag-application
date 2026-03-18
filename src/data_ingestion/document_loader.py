"""
Document Loader Module (Phase 2.5)

Loads HTML documents from AWS S3 bucket or local filesystem.
Includes error handling and logging for production use.
"""

from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from .config import config
from .logging_config import create_logger


class S3DocumentLoader:
    """
    Loads HTML documents from AWS S3 bucket.
    Handles both individual files and batch operations with error handling.
    """

    def __init__(
        self,
        s3_client: Optional[object] = None,
        bucket_name: Optional[str] = None,
        prefix: Optional[str] = None,
        logger_name: str = "S3DocumentLoader",
    ):
        """
        Initialize S3 document loader.

        Args:
            s3_client: boto3 S3 client (auto-created if None)
            bucket_name: S3 bucket name (from config if None)
            prefix: S3 prefix for filtering objects (from config if None)
            logger_name: Logger identifier
        """
        self.logger = create_logger(logger_name)
        self.bucket_name = bucket_name or config.S3_BUCKET_NAME
        self.prefix = prefix or config.S3_DOCUMENT_PREFIX

        # Initialize S3 client with error handling
        self.s3_client = s3_client
        if not self.s3_client:
            try:
                self.s3_client = boto3.client(
                    "s3",
                    region_name=config.AWS_REGION,
                    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                )

                # Test connection
                self.s3_client.head_bucket(Bucket=self.bucket_name)
                self.logger.info(
                    f"Successfully connected to S3 bucket: {self.bucket_name}",
                    {"region": config.AWS_REGION},
                )
            except ClientError as e:
                self.logger.error(
                    f"Failed to connect to S3 bucket: {self.bucket_name}",
                    {
                        "error": str(e),
                        "error_code": e.response.get("Error", {}).get(
                            "Code", "UNKNOWN"
                        ),
                    },
                )
                self.s3_client = None
            except Exception as e:
                self.logger.error(
                    f"Unexpected error initializing S3 client: {str(e)}",
                    {"exception_type": type(e).__name__},
                )
                self.s3_client = None

    def list_documents(self, file_extension: str = ".html") -> List[str]:
        """
        List all documents in S3 bucket matching criteria.

        Args:
            file_extension: File extension to filter (e.g., '.html', '.pdf')

        Returns:
            List of S3 object keys, empty list if error occurs
        """
        if not self.s3_client:
            self.logger.error("S3 client not available")
            return []

        objects = []
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)

            for page in pages:
                if "Contents" not in page:
                    continue
                for obj in page["Contents"]:
                    if obj["Key"].endswith(file_extension):
                        objects.append(obj["Key"])

            objects = sorted(objects)
            self.logger.info(
                f"Listed {len(objects)} documents",
                {"file_extension": file_extension, "bucket": self.bucket_name},
            )
            return objects

        except ClientError as e:
            self.logger.error(
                f"Error listing S3 objects: {str(e)}",
                {
                    "bucket": self.bucket_name,
                    "prefix": self.prefix,
                    "error_code": e.response.get("Error", {}).get("Code", "UNKNOWN"),
                },
            )
            return []
        except Exception as e:
            self.logger.error(
                f"Unexpected error listing S3 documents: {str(e)}",
                {"exception_type": type(e).__name__},
            )
            return []

    def load_document(self, s3_key: str) -> Optional[str]:
        """
        Load a single document from S3 with error handling.

        Args:
            s3_key: Full S3 object key

        Returns:
            Document content as string, or None if error occurs
        """
        if not self.s3_client:
            self.logger.error("S3 client not available")
            return None

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            content = response["Body"].read().decode("utf-8")

            self.logger.info(
                f"Loaded document from S3",
                {
                    "s3_key": s3_key,
                    "size_bytes": len(content),
                    "bucket": self.bucket_name,
                },
            )
            return content

        except ClientError as e:
            self.logger.error(
                f"Error loading document from S3: {str(e)}",
                {
                    "s3_key": s3_key,
                    "bucket": self.bucket_name,
                    "error_code": e.response.get("Error", {}).get("Code", "UNKNOWN"),
                },
            )
            return None
        except UnicodeDecodeError as e:
            self.logger.error(
                f"Failed to decode document content: {str(e)}",
                {"s3_key": s3_key, "exception_type": "UnicodeDecodeError"},
            )
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error loading S3 document: {str(e)}",
                {"s3_key": s3_key, "exception_type": type(e).__name__},
            )
            return None

    def load_local_html(self, file_path: str) -> Optional[str]:
        """
        Load HTML document from local filesystem (for testing/demo).

        Args:
            file_path: Path to local HTML file

        Returns:
            Document content as string, or None if error occurs
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.logger.info(
                f"Loaded local HTML document",
                {"file_path": file_path, "size_bytes": len(content)},
            )
            return content

        except FileNotFoundError:
            self.logger.error(
                f"Local file not found: {file_path}",
                {"file_path": file_path, "exception_type": "FileNotFoundError"},
            )
            return None
        except PermissionError as e:
            self.logger.error(
                f"Permission denied reading file: {file_path}",
                {"file_path": file_path, "exception_type": "PermissionError"},
            )
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error loading local document: {str(e)}",
                {"file_path": file_path, "exception_type": type(e).__name__},
            )
            return None

    def get_file_metadata(self, s3_key: str) -> Dict:
        """
        Extract metadata from S3 object key.
        Assumes naming convention: company/fiscal_year/filing_type/document.html

        Args:
            s3_key: S3 object key

        Returns:
            Dict with extracted metadata
        """
        try:
            parts = s3_key.split("/")

            metadata = {
                "source_file": s3_key,
                "company": parts[0] if len(parts) > 0 else "UNKNOWN",
                "fiscal_year": parts[1] if len(parts) > 1 else None,
                "filing_type": parts[2] if len(parts) > 2 else None,
            }

            self.logger.debug(
                f"Extracted metadata from S3 key",
                {"s3_key": s3_key, "metadata": metadata},
            )
            return metadata

        except Exception as e:
            self.logger.error(
                f"Error extracting metadata from S3 key: {str(e)}",
                {"s3_key": s3_key, "exception_type": type(e).__name__},
            )
            return {"source_file": s3_key, "company": "UNKNOWN", "error": str(e)}
