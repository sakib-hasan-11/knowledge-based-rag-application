"""
During-Retrieval Pipeline Module (Phase 8)

Handles retrieval operations with multiple reranking strategies:
- Hybrid retrieval (dense + sparse vectors)
- Maximal Marginal Relevance (MMR) for diversity
- Cross-encoder reranking for fine-tuning
"""

import os
from typing import Dict, List, Optional

import numpy as np
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_openai import OpenAIEmbeddings

from src.data_ingestion.config import config
from src.data_ingestion.logging_config import create_logger


class HybridRetriever:
    """Combines dense and sparse vector retrieval with alpha-weighted scoring."""

    def __init__(
        self,
        embeddings_model: OpenAIEmbeddings,
        sparse_generator,
        index,
        alpha: float = 0.5,
        top_k: int = 10,
        logger_name: str = "HybridRetriever",
    ):
        """Initialize hybrid retriever."""
        self.logger = create_logger(logger_name)
        self.embeddings_model = embeddings_model
        self.sparse_generator = sparse_generator
        self.index = index
        self.alpha = alpha
        self.top_k = top_k

        self.logger.info(
            "HybridRetriever initialized", {"alpha": alpha, "top_k": top_k}
        )

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        try:
            if not scores:
                return scores

            min_score = min(scores)
            max_score = max(scores)

            if max_score - min_score == 0:
                return [0.5] * len(scores)

            normalized = [(s - min_score) / (max_score - min_score) for s in scores]
            return normalized
        except Exception as e:
            self.logger.error(f"Error normalizing scores: {str(e)}")
            return [0.5] * len(scores)

    def retrieve(self, query: str) -> List[Dict]:
        """Hybrid retrieval combining dense and sparse search."""
        try:
            dense_results = self._dense_retrieve(query)
            sparse_results = self._sparse_retrieve(query)

            merged = self._merge_results(dense_results, sparse_results)

            self.logger.info(
                "Hybrid retrieval completed",
                {
                    "dense_results": len(dense_results),
                    "sparse_results": len(sparse_results),
                    "merged_results": len(merged),
                    "query": query[:100],
                },
            )
            return merged
        except Exception as e:
            self.logger.error(f"Error in hybrid retrieval: {str(e)}")
            return []

    def _dense_retrieve(self, query: str) -> List[Dict]:
        """Dense retrieval using embeddings."""
        try:
            embedding = self.embeddings_model.embed_query(query)
            results = self.index.query(
                vector=embedding, top_k=self.top_k * 2, include_metadata=True
            )

            return [
                {
                    "id": r["id"],
                    "score": r["score"],
                    "metadata": r.get("metadata", {}),
                    "source": "dense",
                }
                for r in results.get("matches", [])
            ]
        except Exception as e:
            self.logger.error(f"Error in dense retrieval: {str(e)}")
            return []

    def _sparse_retrieve(self, query: str) -> List[Dict]:
        """Sparse retrieval using BM25."""
        try:
            tokens = self.sparse_generator._tokenize(query)
            sparse_scores = {}

            for token in tokens:
                if token in self.sparse_generator.token_to_idx:
                    token_idx = self.sparse_generator.token_to_idx[token]
                    freq = tokens.count(token)
                    score = freq / (len(tokens) + 1e-8)
                    sparse_scores[token_idx] = min(score, 1.0)

            return [
                {
                    "id": None,
                    "score": sum(sparse_scores.values()) / max(len(sparse_scores), 1),
                    "sparse_scores": sparse_scores,
                    "source": "sparse",
                }
            ]
        except Exception as e:
            self.logger.error(f"Error in sparse retrieval: {str(e)}")
            return []

    def _merge_results(
        self, dense_results: List[Dict], sparse_results: List[Dict]
    ) -> List[Dict]:
        """Merge dense and sparse results with weighting."""
        try:
            merged = {}

            dense_scores = [r["score"] for r in dense_results]
            normalized_dense = self._normalize_scores(dense_scores)

            for result, norm_score in zip(dense_results, normalized_dense):
                doc_id = result["id"]
                merged[doc_id] = {
                    **result,
                    "hybrid_score": self.alpha * norm_score,
                    "dense_score": norm_score,
                    "sparse_score": 0.0,
                }

            sorted_results = sorted(
                merged.values(), key=lambda x: x["hybrid_score"], reverse=True
            )

            return sorted_results[: self.top_k]
        except Exception as e:
            self.logger.error(f"Error merging retrieval results: {str(e)}")
            return dense_results[: self.top_k]


class MMRReranker:
    """Removes redundant results using Maximal Marginal Relevance."""

    def __init__(
        self,
        embeddings_model: Optional[OpenAIEmbeddings] = None,
        diversity_factor: float = 0.5,
        lambda_param: Optional[float] = None,
        logger_name: str = "MMRReranker",
    ):
        """
        Initialize MMR reranker.

        Args:
            embeddings_model: OpenAI embeddings model. If None, creates default model.
            diversity_factor: Weight for diversity vs relevance (0.0-1.0). Default 0.5.
            lambda_param: Deprecated. Use diversity_factor instead.
            logger_name: Logger name for this instance.
        """
        self.logger = create_logger(logger_name)

        # Validate API key before creating embeddings
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            self.logger.warning(
                "OPENAI_API_KEY not set. MMRReranker will fail when used in production. "
                "Set OPENAI_API_KEY environment variable before using this class."
            )

        # Initialize embeddings model if not provided
        if embeddings_model is None:
            try:
                self.embeddings_model = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAIEmbeddings: {str(e)}")
                raise
        else:
            self.embeddings_model = embeddings_model

        # Support both diversity_factor and lambda_param for backwards compatibility
        if lambda_param is not None:
            self.diversity_factor = lambda_param
        else:
            self.diversity_factor = diversity_factor

        if not 0.0 <= self.diversity_factor <= 1.0:
            raise ValueError(
                f"diversity_factor must be between 0.0 and 1.0, got {self.diversity_factor}"
            )

        self.logger.info(
            "MMRReranker initialized", {"diversity_factor": self.diversity_factor}
        )

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity."""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

            if norm_product == 0:
                return 0.0

            return float(np.dot(vec1, vec2) / norm_product)
        except Exception as e:
            self.logger.debug(f"Error calculating similarity: {str(e)}")
            return 0.0

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """Apply MMR for diversity."""
        try:
            if not documents:
                return []

            query_embedding = self.embeddings_model.embed_query(query)
            selected = []
            remaining = list(documents)
            doc_embeddings = {}

            while len(selected) < top_k and remaining:
                mmr_scores = []

                for doc in remaining:
                    doc_id = doc.get("id")

                    if doc_id and doc_id not in doc_embeddings:
                        try:
                            content = doc.get("metadata", {}).get("text_preview", "")
                            if content:
                                doc_embeddings[doc_id] = (
                                    self.embeddings_model.embed_query(content)
                                )
                        except Exception as e:
                            self.logger.debug(f"Error embedding document: {str(e)}")
                            continue

                    relevance = doc.get("hybrid_score", 0.0)

                    diversity = 1.0
                    if selected and doc_id in doc_embeddings:
                        selected_sims = [
                            self._cosine_similarity(
                                doc_embeddings[doc_id], doc_embeddings[s.get("id")]
                            )
                            for s in selected
                            if s.get("id") in doc_embeddings
                        ]
                        if selected_sims:
                            diversity = 1.0 - max(selected_sims)

                    mmr_score = (
                        self.diversity_factor * relevance
                        + (1 - self.diversity_factor) * diversity
                    )
                    mmr_scores.append((mmr_score, doc))

                if mmr_scores:
                    best_doc = max(mmr_scores, key=lambda x: x[0])[1]
                    selected.append({**best_doc, "mmr_score": mmr_scores[0][0]})
                    remaining.remove(best_doc)
                else:
                    break

            self.logger.info(
                "MMR reranking completed",
                {"input_docs": len(documents), "output_docs": len(selected)},
            )
            return selected
        except Exception as e:
            self.logger.error(f"Error in MMR reranking: {str(e)}")
            return documents[:top_k]


class CrossEncoderReranker:
    """Reranks documents using cross-encoder model."""

    def __init__(
        self, reranker_type: str = "bge", logger_name: str = "CrossEncoderReranker"
    ):
        """Initialize cross-encoder reranker."""
        self.logger = create_logger(logger_name)
        self.reranker_type = reranker_type
        self.reranker = None

        self._init_reranker()

    def _init_reranker(self):
        """Initialize the reranker model with error handling."""
        try:
            if self.reranker_type == "bge":
                from langchain_community.cross_encoders import HuggingFaceCrossEncoder

                model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
                self.reranker = model
                self.logger.info("BGE cross-encoder initialized successfully")
            else:
                self.logger.warning(f"Unknown reranker type: {self.reranker_type}")
        except ImportError:
            self.logger.error(
                "BGE cross-encoder requires package installation",
                {
                    "package": "sentence-transformers",
                    "command": "pip install sentence-transformers",
                },
            )
            self.reranker = None
        except Exception as e:
            self.logger.error(f"Error initializing cross-encoder: {str(e)}")
            self.reranker = None

    def rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """Rerank documents with error handling."""
        try:
            if not documents or not self.reranker:
                self.logger.warning("Reranker unavailable, returning original order")
                return documents[:top_k]

            doc_texts = [
                d.get("metadata", {}).get("text_preview", "") for d in documents
            ]

            if not all(doc_texts):
                self.logger.warning(
                    "Some documents missing text, returning original order"
                )
                return documents[:top_k]

            scores = self.reranker.predict([[query, doc] for doc in doc_texts])

            scored_docs = []
            for doc, score in zip(documents, scores):
                doc_copy = dict(doc)
                doc_copy["reranker_score"] = float(score)
                scored_docs.append(doc_copy)

            sorted_docs = sorted(
                scored_docs, key=lambda x: x.get("reranker_score", 0), reverse=True
            )

            self.logger.info(
                "Cross-encoder reranking completed",
                {"input_docs": len(documents), "top_k": top_k},
            )
            return sorted_docs[:top_k]
        except Exception as e:
            self.logger.error(f"Error in cross-encoder reranking: {str(e)}")
            return documents[:top_k]
