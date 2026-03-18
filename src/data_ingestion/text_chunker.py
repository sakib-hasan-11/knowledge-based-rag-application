"""
Text Chunking Module (Phase 3)

Chunks documents using semantic approach with proper error handling
and logging for production deployment.
"""

from typing import List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import config
from .logging_config import create_logger


class SemanticTextChunker:
    """
    Chunks documents using a semantic approach:
    1. Split with RecursiveCharacterTextSplitter
    2. Add metadata for chunk tracking
    3. Preserve semantic boundaries
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        logger_name: str = "SemanticTextChunker",
    ):
        """
        Initialize semantic chunker.

        Args:
            chunk_size: Size of chunks in characters
            chunk_overlap: Overlap between chunks in characters
            logger_name: Logger identifier
        """
        self.logger = create_logger(logger_name)

        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

        try:
            # Initialize recursive text splitter for chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )

            self.logger.info(
                "SemanticTextChunker initialized",
                {"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap},
            )
        except Exception as e:
            self.logger.error(
                f"Error initializing text splitter: {str(e)}",
                {"exception_type": type(e).__name__},
            )
            raise

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks with semantic preservation.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of semantically chunked documents
        """
        if not documents:
            self.logger.warning("No documents provided for chunking")
            return []

        chunked_docs = []
        total_chunks = 0

        try:
            for doc_idx, doc in enumerate(documents):
                try:
                    # Split document text into chunks
                    initial_chunks = self.text_splitter.split_text(doc.page_content)

                    if not initial_chunks:
                        self.logger.warning(
                            f"No chunks created for document {doc_idx}",
                            {"document_length": len(doc.page_content)},
                        )
                        continue

                    # Create Document objects for chunks with enhanced metadata
                    for chunk_id, chunk_text in enumerate(initial_chunks):
                        try:
                            chunk_doc = Document(
                                page_content=chunk_text,
                                metadata={
                                    **doc.metadata,
                                    "chunk_id": chunk_id,
                                    "chunk_count": len(initial_chunks),
                                    "original_length": len(doc.page_content),
                                    "chunk_length": len(chunk_text),
                                    "document_index": doc_idx,
                                },
                            )
                            chunked_docs.append(chunk_doc)
                            total_chunks += 1

                        except Exception as e:
                            self.logger.error(
                                f"Error creating chunk {chunk_id}: {str(e)}",
                                {
                                    "document_index": doc_idx,
                                    "chunk_id": chunk_id,
                                    "exception_type": type(e).__name__,
                                },
                            )
                            continue

                except Exception as e:
                    self.logger.error(
                        f"Error chunking document {doc_idx}: {str(e)}",
                        {"document_index": doc_idx, "exception_type": type(e).__name__},
                    )
                    continue

            self.logger.info(
                "Document chunking completed",
                {
                    "input_documents": len(documents),
                    "output_chunks": total_chunks,
                    "avg_chunks_per_document": total_chunks / max(len(documents), 1),
                },
            )
            return chunked_docs

        except Exception as e:
            self.logger.error(
                f"Error in chunk_documents: {str(e)}",
                {"num_documents": len(documents), "exception_type": type(e).__name__},
            )
            return chunked_docs  # Return partial results

    def get_chunk_statistics(self, chunked_documents: List[Document]) -> dict:
        """
        Calculate statistics about chunks.

        Args:
            chunked_documents: List of chunked documents

        Returns:
            Dictionary with chunk statistics
        """
        try:
            if not chunked_documents:
                return {
                    "total_chunks": 0,
                    "avg_chunk_size": 0,
                    "min_chunk_size": 0,
                    "max_chunk_size": 0,
                    "total_content_size": 0,
                }

            chunk_sizes = [len(doc.page_content) for doc in chunked_documents]
            total_size = sum(chunk_sizes)

            stats = {
                "total_chunks": len(chunked_documents),
                "avg_chunk_size": total_size // max(len(chunk_sizes), 1),
                "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
                "total_content_size": total_size,
                "median_chunk_size": sorted(chunk_sizes)[len(chunk_sizes) // 2]
                if chunk_sizes
                else 0,
            }

            self.logger.info("Calculated chunk statistics", stats)
            return stats

        except Exception as e:
            self.logger.error(
                f"Error calculating chunk statistics: {str(e)}",
                {"exception_type": type(e).__name__},
            )
            return {
                "error": str(e),
                "total_chunks": len(chunked_documents) if chunked_documents else 0,
            }
