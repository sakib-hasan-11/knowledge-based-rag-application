"""
Embeddings Generator Module (Phase 4)

Generates vector embeddings using OpenAI's embedding models with
proper error handling, batching, and production logging.
"""

from typing import Dict, List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import config
from .logging_config import create_logger


class EmbeddingsGenerator:
    """
    Generates dense vector embeddings for documents using OpenAI's API.
    Handles batching, error handling, and comprehensive logging.
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: Optional[int] = None,
        logger_name: str = "EmbeddingsGenerator",
    ):
        """
        Initialize embeddings generator.

        Args:
            embedding_model: OpenAI embedding model name
            dimensions: Vector dimension size
            batch_size: Batch size for processing
            logger_name: Logger identifier
        """
        self.logger = create_logger(logger_name)

        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.dimensions = dimensions or config.EMBEDDING_DIMENSION
        self.batch_size = batch_size or config.BATCH_SIZE

        # Initialize OpenAI embeddings
        self.embeddings_client = None
        self._initialize_embeddings()

    def _initialize_embeddings(self) -> bool:
        """
        Initialize OpenAI embeddings client with error handling.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.embeddings_client = OpenAIEmbeddings(
                model=self.embedding_model, dimensions=self.dimensions
            )

            self.logger.info(
                "OpenAI embeddings client initialized",
                {"model": self.embedding_model, "dimensions": self.dimensions},
            )
            return True

        except ValueError as e:
            self.logger.error(
                f"Invalid OpenAI configuration: {str(e)}",
                {"exception_type": type(e).__name__},
            )
            return False
        except Exception as e:
            self.logger.error(
                f"Error initializing OpenAI embeddings: {str(e)}",
                {"exception_type": type(e).__name__},
            )
            return False

    def generate_embeddings(self, documents: List[Document]) -> List[Dict]:
        """
        Generate embeddings for all documents with batching.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of dicts with id, embedding vector, metadata, and text preview
        """
        if not documents:
            self.logger.warning("No documents provided for embedding generation")
            return []

        if not self.embeddings_client:
            self.logger.error("Embeddings client not initialized")
            return []

        embeddings_data = []
        successful_count = 0
        failed_count = 0

        try:
            # Prepare documents for embedding
            documents_for_embedding = []
            for i, doc in enumerate(documents):
                doc_id = (
                    f"{doc.metadata.get('section', 'unknown')}_"
                    f"{doc.metadata.get('chunk_id', i)}"
                )
                documents_for_embedding.append(
                    {"id": doc_id, "text": doc.page_content, "metadata": doc.metadata}
                )

            self.logger.info(
                f"Starting embedding generation",
                {
                    "total_documents": len(documents_for_embedding),
                    "batch_size": self.batch_size,
                },
            )

            # Generate embeddings in batches
            for i in range(0, len(documents_for_embedding), self.batch_size):
                batch = documents_for_embedding[i : i + self.batch_size]
                batch_texts = [doc["text"] for doc in batch]

                try:
                    # Generate embeddings for batch using LangChain
                    batch_embeddings = self.embeddings_client.embed_documents(
                        batch_texts
                    )

                    # Store embeddings with metadata
                    for doc, embedding in zip(batch, batch_embeddings):
                        embeddings_data.append(
                            {
                                "id": doc["id"],
                                "values": embedding,
                                "metadata": doc["metadata"],
                                "text_preview": (
                                    doc["text"][:100] + "..."
                                    if len(doc["text"]) > 100
                                    else doc["text"]
                                ),
                            }
                        )
                        successful_count += 1

                    progress = min(i + self.batch_size, len(documents_for_embedding))
                    if progress % (self.batch_size * 2) == 0 or progress >= len(
                        documents_for_embedding
                    ):
                        self.logger.info(
                            f"Embedding generation progress",
                            {
                                "processed": progress,
                                "total": len(documents_for_embedding),
                                "successful": successful_count,
                                "failed": failed_count,
                            },
                        )

                except Exception as batch_error:
                    self.logger.error(
                        f"Error processing embedding batch: {str(batch_error)}",
                        {
                            "batch_start": i,
                            "batch_end": i + self.batch_size,
                            "batch_size": len(batch),
                            "exception_type": type(batch_error).__name__,
                        },
                    )
                    failed_count += len(batch)
                    continue

            self.logger.info(
                f"Embedding generation completed",
                {
                    "total_embeddings": len(embeddings_data),
                    "successful": successful_count,
                    "failed": failed_count,
                    "vector_dimension": self.dimensions if embeddings_data else 0,
                },
            )
            return embeddings_data

        except Exception as e:
            self.logger.error(
                f"Error in generate_embeddings: {str(e)}",
                {"total_documents": len(documents), "exception_type": type(e).__name__},
            )
            return embeddings_data  # Return partial results

    def generate_query_embedding(self, query_text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single query text.

        Args:
            query_text: Query text to embed

        Returns:
            Embedding vector or None if error occurs
        """
        if not self.embeddings_client:
            self.logger.error("Embeddings client not initialized")
            return None

        try:
            embedding = self.embeddings_client.embed_query(query_text)

            self.logger.debug(
                f"Generated query embedding",
                {"query_length": len(query_text), "vector_dimension": len(embedding)},
            )
            return embedding

        except Exception as e:
            self.logger.error(
                f"Error generating query embedding: {str(e)}",
                {"query_length": len(query_text), "exception_type": type(e).__name__},
            )
            return None

    def get_embedding_statistics(self, embeddings_data: List[Dict]) -> dict:
        """
        Calculate statistics about generated embeddings.

        Args:
            embeddings_data: List of embedding dicts

        Returns:
            Dictionary with embedding statistics
        """
        try:
            if not embeddings_data:
                return {"total_embeddings": 0, "vector_dimension": 0}

            vectors = [e["values"] for e in embeddings_data]
            vectors_array = np.array(vectors)

            stats = {
                "total_embeddings": len(embeddings_data),
                "vector_dimension": len(vectors[0]) if vectors else 0,
                "mean_vector": np.mean(vectors_array, axis=0).tolist()[
                    :3
                ],  # First 3 dims
                "std_vector": np.std(vectors_array, axis=0).tolist()[
                    :3
                ],  # First 3 dims
                "min_l2_norm": float(np.min([np.linalg.norm(v) for v in vectors])),
                "max_l2_norm": float(np.max([np.linalg.norm(v) for v in vectors])),
                "mean_l2_norm": float(np.mean([np.linalg.norm(v) for v in vectors])),
            }

            self.logger.info(
                "Calculated embedding statistics",
                {
                    "total_embeddings": stats["total_embeddings"],
                    "vector_dimension": stats["vector_dimension"],
                    "mean_l2_norm": stats["mean_l2_norm"],
                },
            )
            return stats

        except Exception as e:
            self.logger.error(
                f"Error calculating embedding statistics: {str(e)}",
                {"exception_type": type(e).__name__},
            )
            return {"error": str(e), "total_embeddings": len(embeddings_data)}
