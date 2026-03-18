"""
Sparse Vector Generator Module (Phase 5)

Generates sparse vectors using BM25 algorithm for keyword-based matching.
Includes comprehensive error handling and production logging.
"""

import string
from typing import Dict, List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .config import config
from .logging_config import create_logger


class BM25SparseVectorGenerator:
    """
    Generates sparse vectors using BM25 algorithm.
    Sparse vectors contain non-zero values only for relevant terms,
    enabling efficient hybrid search with dense embeddings.
    """

    def __init__(self, logger_name: str = "BM25SparseVectorGenerator"):
        """
        Initialize BM25 sparse vector generator.

        Args:
            logger_name: Logger identifier
        """
        self.logger = create_logger(logger_name)
        self.bm25_model: Optional[BM25Okapi] = None
        self.corpus_tokens: List[List[str]] = []
        self.vocabulary: Dict[str, int] = {}
        self.token_to_idx: Dict[str, int] = {}

        # Common English stopwords
        self.stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "is",
            "was",
            "are",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
        }

        self.logger.info("BM25SparseVectorGenerator initialized")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text with stopword removal.

        Args:
            text: Text to tokenize

        Returns:
            List of cleaned tokens
        """
        try:
            # Lowercase
            text = text.lower()

            # Remove punctuation
            text = text.translate(str.maketrans("", "", string.punctuation))

            # Split into tokens
            tokens = text.split()

            # Filter: remove short tokens and stopwords
            tokens = [t for t in tokens if len(t) > 2 and t not in self.stopwords]

            return tokens

        except Exception as e:
            self.logger.error(
                f"Error tokenizing text: {str(e)}",
                {"text_length": len(text), "exception_type": type(e).__name__},
            )
            return []

    def build_corpus(self, documents: List[Document]) -> bool:
        """
        Build BM25 corpus from documents.

        Args:
            documents: List of LangChain Document objects

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Building BM25 corpus", {"num_documents": len(documents)})

            # Tokenize all documents and build vocabulary
            for doc_idx, doc in enumerate(documents):
                try:
                    tokens = self._tokenize(doc.page_content)
                    self.corpus_tokens.append(tokens)

                    # Build vocabulary
                    for token in tokens:
                        if token not in self.vocabulary:
                            self.token_to_idx[token] = len(self.token_to_idx)
                            self.vocabulary[token] = 0
                        self.vocabulary[token] += 1

                except Exception as e:
                    self.logger.error(
                        f"Error tokenizing document {doc_idx}: {str(e)}",
                        {"document_index": doc_idx, "exception_type": type(e).__name__},
                    )
                    self.corpus_tokens.append([])
                    continue

            # Create BM25 model
            if self.corpus_tokens:
                self.bm25_model = BM25Okapi(self.corpus_tokens)

                self.logger.info(
                    f"BM25 corpus built successfully",
                    {
                        "total_documents": len(self.corpus_tokens),
                        "vocabulary_size": len(self.vocabulary),
                        "avg_tokens_per_doc": (
                            sum(len(tokens) for tokens in self.corpus_tokens)
                            / max(len(self.corpus_tokens), 1)
                        ),
                    },
                )
                return True
            else:
                self.logger.error("No valid tokens found in corpus")
                return False

        except Exception as e:
            self.logger.error(
                f"Error building BM25 corpus: {str(e)}",
                {"num_documents": len(documents), "exception_type": type(e).__name__},
            )
            return False

    def get_sparse_vector(self, text: str) -> Dict[int, float]:
        """
        Generate sparse vector (as dict of token_idx: score) for text.

        Args:
            text: Text to vectorize

        Returns:
            Dict mapping token indices to normalized scores
        """
        try:
            tokens = self._tokenize(text)

            if not tokens:
                return {}

            sparse_vector = {}

            for token in tokens:
                if token in self.token_to_idx:
                    token_idx = self.token_to_idx[token]

                    # Score based on term frequency in this document
                    freq = tokens.count(token)

                    # Normalize by document length and cap at 1.0
                    score = freq / (len(tokens) + 1e-8)
                    score = min(score, 1.0)

                    sparse_vector[token_idx] = score

            return sparse_vector

        except Exception as e:
            self.logger.error(
                f"Error generating sparse vector: {str(e)}",
                {"text_length": len(text), "exception_type": type(e).__name__},
            )
            return {}

    def generate_all_sparse_vectors(
        self, documents: List[Document]
    ) -> List[Dict[int, float]]:
        """
        Generate sparse vectors for all documents.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of sparse vectors (List of Dict[int, float])
        """
        if not documents:
            self.logger.warning("No documents provided for sparse vector generation")
            return []

        sparse_vectors = []
        successful_count = 0
        failed_count = 0

        try:
            self.logger.info(
                f"Generating sparse vectors", {"total_documents": len(documents)}
            )

            for i, doc in enumerate(documents):
                try:
                    sparse_vec = self.get_sparse_vector(doc.page_content)
                    sparse_vectors.append(sparse_vec)
                    successful_count += 1

                    if (i + 1) % 100 == 0 or (i + 1) == len(documents):
                        self.logger.info(
                            f"Sparse vector generation progress",
                            {
                                "processed": i + 1,
                                "total": len(documents),
                                "successful": successful_count,
                                "failed": failed_count,
                            },
                        )

                except Exception as e:
                    self.logger.error(
                        f"Error generating sparse vector {i}: {str(e)}",
                        {"document_index": i, "exception_type": type(e).__name__},
                    )
                    sparse_vectors.append({})
                    failed_count += 1
                    continue

            self.logger.info(
                f"Sparse vector generation completed",
                {
                    "total_vectors": len(sparse_vectors),
                    "successful": successful_count,
                    "failed": failed_count,
                },
            )
            return sparse_vectors

        except Exception as e:
            self.logger.error(
                f"Error in generate_all_sparse_vectors: {str(e)}",
                {"num_documents": len(documents), "exception_type": type(e).__name__},
            )
            return sparse_vectors

    def get_sparse_vector_statistics(
        self, sparse_vectors: List[Dict[int, float]]
    ) -> dict:
        """
        Calculate statistics about sparse vectors.

        Args:
            sparse_vectors: List of sparse vectors

        Returns:
            Dictionary with sparse vector statistics
        """
        try:
            if not sparse_vectors:
                return {
                    "total_vectors": 0,
                    "avg_non_zero_elements": 0,
                    "min_non_zero_elements": 0,
                    "max_non_zero_elements": 0,
                    "sparsity": 0.0,
                }

            non_zero_counts = [len(sv) for sv in sparse_vectors]

            total_possible_elements = len(self.vocabulary) * len(sparse_vectors)
            total_non_zero = sum(non_zero_counts)
            sparsity = 1.0 - (total_non_zero / max(total_possible_elements, 1))

            stats = {
                "total_vectors": len(sparse_vectors),
                "vocabulary_size": len(self.vocabulary),
                "avg_non_zero_elements": sum(non_zero_counts)
                / max(len(non_zero_counts), 1),
                "min_non_zero_elements": min(non_zero_counts) if non_zero_counts else 0,
                "max_non_zero_elements": max(non_zero_counts) if non_zero_counts else 0,
                "total_non_zero_elements": total_non_zero,
                "sparsity_percentage": sparsity * 100,
            }

            self.logger.info(
                f"Calculated sparse vector statistics",
                {
                    "total_vectors": stats["total_vectors"],
                    "avg_non_zero": stats["avg_non_zero_elements"],
                    "sparsity_pct": stats["sparsity_percentage"],
                },
            )
            return stats

        except Exception as e:
            self.logger.error(
                f"Error calculating sparse vector statistics: {str(e)}",
                {"exception_type": type(e).__name__},
            )
            return {"error": str(e), "total_vectors": len(sparse_vectors)}
