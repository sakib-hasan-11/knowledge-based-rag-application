"""
Pinecone Uploader Module (Phase 6)

Manages Pinecone index initialization and hybrid vector uploading
with comprehensive error handling and production logging.
"""

import time
from typing import Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec

from .config import config
from .logging_config import create_logger


class PineconeUploader:
    """
    Manages Pinecone vector database operations:
    - Index creation and management
    - Hybrid vector (dense + sparse) upsert
    - Error handling and retry logic
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        environment: Optional[str] = None,
        logger_name: str = "PineconeUploader",
    ):
        """
        Initialize Pinecone uploader.

        Args:
            api_key: Pinecone API key
            index_name: Pinecone index name
            environment: Pinecone environment
            logger_name: Logger identifier
        """
        self.logger = create_logger(logger_name)

        self.api_key = api_key or config.PINECONE_API_KEY
        self.index_name = index_name or config.PINECONE_INDEX_NAME
        self.environment = environment or config.PINECONE_ENVIRONMENT
        self.metric = config.PINECONE_METRIC

        self.pc: Optional[Pinecone] = None
        self.index = None

        self._initialize_pinecone()

    def _initialize_pinecone(self) -> bool:
        """
        Initialize Pinecone client with error handling.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.pc = Pinecone(api_key=self.api_key)

            self.logger.info(
                "Pinecone client initialized",
                {
                    "api_key_provided": bool(self.api_key),
                    "index_name": self.index_name,
                    "environment": self.environment,
                },
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Error initializing Pinecone client: {str(e)}",
                {"exception_type": type(e).__name__},
            )
            return False

    def create_index_if_needed(self, dimension: int = 1536, timeout: int = 30) -> bool:
        """
        Create Pinecone index if it doesn't exist.

        Args:
            dimension: Vector dimension (default 1536 for OpenAI's text-embedding-3-small)
            timeout: Timeout in seconds for index creation

        Returns:
            True if index created or already exists, False if error
        """
        if not self.pc:
            self.logger.error("Pinecone client not initialized")
            return False

        try:
            # List existing indexes
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            self.logger.info(
                "Listed existing Pinecone indexes", {"indexes": existing_indexes}
            )

            # Create index if doesn't exist
            if self.index_name not in existing_indexes:
                self.logger.info(
                    f"Creating Pinecone index: {self.index_name}",
                    {
                        "dimension": dimension,
                        "metric": self.metric,
                        "environment": self.environment,
                    },
                )

                # Parse environment (e.g., "us-east-1-aws" -> "us-east-1")
                region = self.environment.replace("-aws", "").replace("-gcp", "")

                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud="aws", region=region),
                )

                self.logger.info(f"Index creation request sent")

                # Wait for index to be ready
                time.sleep(5)

                # Verify creation
                max_retries = 10
                for attempt in range(max_retries):
                    try:
                        indexes = [idx.name for idx in self.pc.list_indexes()]
                        if self.index_name in indexes:
                            self.logger.info(
                                f"Index {self.index_name} is ready",
                                {"ready_attempts": attempt + 1},
                            )
                            return True
                        time.sleep(2)
                    except Exception as e:
                        self.logger.debug(f"Waiting for index to be ready: {str(e)}")

                self.logger.warning(
                    f"Index creation timeout after {max_retries} attempts"
                )
                return False
            else:
                self.logger.info(f"Index already exists: {self.index_name}")
                return True

        except Exception as e:
            self.logger.error(
                f"Error creating Pinecone index: {str(e)}",
                {"index_name": self.index_name, "exception_type": type(e).__name__},
            )
            return False

    def get_index_reference(self) -> bool:
        """
        Get reference to Pinecone index.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.pc:
                self.logger.error("Pinecone client not initialized")
                return False

            self.index = self.pc.Index(self.index_name)

            # Get index statistics
            stats = self.index.describe_index_stats()

            self.logger.info(
                "Got Pinecone index reference",
                {
                    "index_name": self.index_name,
                    "dimension": stats.dimension,
                    "total_vectors": stats.total_vector_count,
                },
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Error getting index reference: {str(e)}",
                {"index_name": self.index_name, "exception_type": type(e).__name__},
            )
            return False

    def upsert_hybrid_vectors(
        self, vectors: List[Dict], batch_size: Optional[int] = None
    ) -> Dict:
        """
        Upsert hybrid vectors (dense + sparse) to Pinecone.

        Args:
            vectors: List of vector dicts with id, values, sparse_values, metadata
            batch_size: Batch size for upserting

        Returns:
            Dictionary with upsert statistics
        """
        if not self.index:
            self.logger.error("Pinecone index reference not available")
            return {"success": False, "error": "Index not initialized"}

        if not vectors:
            self.logger.warning("No vectors provided for upsert")
            return {
                "success": True,
                "total_vectors": 0,
                "upserted_count": 0,
                "failed_count": 0,
            }

        batch_size = batch_size or config.UPSERT_BATCH_SIZE
        upserted_count = 0
        failed_count = 0
        failed_batches = []

        try:
            self.logger.info(
                f"Starting hybrid vector upsert",
                {"total_vectors": len(vectors), "batch_size": batch_size},
            )

            # Upsert in batches
            for batch_idx in range(0, len(vectors), batch_size):
                batch = vectors[batch_idx : batch_idx + batch_size]

                try:
                    # Upsert batch to Pinecone
                    upsert_response = self.index.upsert(vectors=batch)
                    upserted_count += len(batch)

                    # Log progress
                    if (batch_idx + batch_size) % (batch_size * 2) == 0 or (
                        batch_idx + batch_size
                    ) >= len(vectors):
                        progress = min(batch_idx + batch_size, len(vectors))
                        self.logger.info(
                            f"Upsert progress",
                            {
                                "processed": progress,
                                "total": len(vectors),
                                "upserted": upserted_count,
                                "failed": failed_count,
                            },
                        )

                except Exception as batch_error:
                    self.logger.error(
                        f"Error upserting batch: {str(batch_error)}",
                        {
                            "batch_index": batch_idx,
                            "batch_size": len(batch),
                            "exception_type": type(batch_error).__name__,
                        },
                    )
                    failed_count += len(batch)
                    failed_batches.append(batch_idx)
                    continue

            # Get updated index statistics
            try:
                index_stats = self.index.describe_index_stats()
                total_vectors_in_index = index_stats.total_vector_count
            except Exception as e:
                self.logger.warning(f"Could not fetch updated index stats: {str(e)}")
                total_vectors_in_index = None

            result = {
                "success": failed_count == 0,
                "total_vectors": len(vectors),
                "upserted_count": upserted_count,
                "failed_count": failed_count,
                "failed_batches": failed_batches if failed_batches else None,
                "total_vectors_in_index": total_vectors_in_index,
            }

            self.logger.info(
                f"Hybrid vector upsert completed",
                {
                    "success": result["success"],
                    "upserted": upserted_count,
                    "failed": failed_count,
                    "total_in_index": total_vectors_in_index,
                },
            )
            return result

        except Exception as e:
            self.logger.error(
                f"Error in upsert_hybrid_vectors: {str(e)}",
                {"total_vectors": len(vectors), "exception_type": type(e).__name__},
            )
            return {
                "success": False,
                "total_vectors": len(vectors),
                "upserted_count": upserted_count,
                "failed_count": failed_count
                + (len(vectors) - upserted_count - failed_count),
                "error": str(e),
            }

    def get_index_statistics(self) -> Optional[Dict]:
        """
        Get current index statistics.

        Returns:
            Dictionary with index statistics or None if error
        """
        try:
            if not self.index:
                self.logger.error("Pinecone index reference not available")
                return None

            stats = self.index.describe_index_stats()

            result = {
                "index_name": self.index_name,
                "dimension": stats.dimension,
                "total_vectors": stats.total_vector_count,
                "metric": self.metric,
            }

            self.logger.info("Retrieved index statistics", result)
            return result

        except Exception as e:
            self.logger.error(
                f"Error getting index statistics: {str(e)}",
                {"exception_type": type(e).__name__},
            )
            return None
