"""
Main Data Ingestion Pipeline Orchestrator

Orchestrates all phases of the data ingestion pipeline:
- Phase 2: Document Loading & Parsing
- Phase 3: Semantic Chunking
- Phase 4: Dense Embeddings (OpenAI)
- Phase 5: Sparse Vectors (BM25)
- Phase 6: Pinecone Upload (Hybrid Search)

Production-grade pipeline with comprehensive error handling, logging, and monitoring.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import config
from .document_loader import S3DocumentLoader
from .document_parser import DocumentIngestionPipeline, HTMLDocumentParser
from .embeddings_generator import EmbeddingsGenerator
from .logging_config import create_logger
from .pinecone_uploader import PineconeUploader
from .sparse_vector_generator import BM25SparseVectorGenerator
from .text_chunker import SemanticTextChunker


class DataIngestionPipeline:
    """
    Production-grade end-to-end data ingestion pipeline for RAG applications.
    Handles all phases from document loading through Pinecone vector upload.
    """

    def __init__(
        self,
        enable_cloudwatch: bool = False,
        logger_name: str = "DataIngestionPipeline",
    ):
        """
        Initialize the complete data ingestion pipeline.

        Args:
            enable_cloudwatch: Enable CloudWatch logging
            logger_name: Logger identifier
        """
        self.logger = create_logger(
            logger_name,
            level=config.LOG_LEVEL,
            log_format=config.LOG_FORMAT,
            enable_cloudwatch=enable_cloudwatch,
        )

        # Initialize components - Phase 2, 3, 4, 5, 6
        self.s3_loader = S3DocumentLoader()
        self.html_parser = HTMLDocumentParser()
        self.document_pipeline = DocumentIngestionPipeline(
            self.s3_loader, self.html_parser
        )
        self.text_chunker = SemanticTextChunker()
        self.embeddings_generator = EmbeddingsGenerator()
        self.sparse_generator = BM25SparseVectorGenerator()
        self.pinecone_uploader = PineconeUploader()

        # Pipeline state
        self.extracted_documents = []
        self.chunked_documents = []
        self.embeddings_data = []
        self.sparse_vectors = []

        self.logger.info("DataIngestionPipeline initialized successfully")

    def validate_configuration(self) -> bool:
        """
        Validate all required configurations.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            self.logger.info("Validating pipeline configuration")

            if not config.validate():
                self.logger.error("Configuration validation failed")
                return False

            self.logger.info(f"Configuration: {config}")
            return True

        except Exception as e:
            self.logger.error(
                f"Error validating configuration: {str(e)}",
                {"exception_type": type(e).__name__},
            )
            return False

    def run_phase_2_document_ingestion(
        self, document_sources: List[str], from_s3: bool = False
    ) -> Tuple[bool, List]:
        """
        Phase 2: Load and parse documents from S3 or local filesystem.

        Args:
            document_sources: List of S3 keys or local file paths
            from_s3: Whether to load from S3 (True) or local (False)

        Returns:
            Tuple of (success: bool, extracted_documents: List)
        """
        try:
            self.logger.info(
                "Starting Phase 2: Document Ingestion",
                {"num_documents": len(document_sources), "from_s3": from_s3},
            )

            # Process documents
            results = self.document_pipeline.process_batch(
                document_sources, from_s3=from_s3
            )

            # Collect all extracted documents
            self.extracted_documents = (
                self.document_pipeline.get_all_extracted_documents()
            )

            self.logger.info(
                "Phase 2 completed",
                {"total_documents_extracted": len(self.extracted_documents)},
            )

            # Optionally save extracted documents
            if config.SAVE_EXTRACTED_DOCS:
                output_path = config.OUTPUT_DIR + "/extracted_documents.jsonl"
                self.document_pipeline.save_extracted_documents(
                    output_path, format="jsonl"
                )

            return len(self.extracted_documents) > 0, self.extracted_documents

        except Exception as e:
            self.logger.error(
                f"Error in Phase 2: {str(e)}", {"exception_type": type(e).__name__}
            )
            return False, []

    def run_phase_3_semantic_chunking(self) -> Tuple[bool, List]:
        """
        Phase 3: Perform semantic text chunking on extracted documents.

        Returns:
            Tuple of (success: bool, chunked_documents: List)
        """
        try:
            if not self.extracted_documents:
                self.logger.error("No extracted documents for chunking")
                return False, []

            self.logger.info(
                "Starting Phase 3: Semantic Chunking",
                {"num_documents": len(self.extracted_documents)},
            )

            # Perform chunking
            self.chunked_documents = self.text_chunker.chunk_documents(
                self.extracted_documents
            )

            # Get statistics
            stats = self.text_chunker.get_chunk_statistics(self.chunked_documents)

            self.logger.info(
                "Phase 3 completed",
                {
                    "total_chunks": stats.get("total_chunks"),
                    "avg_chunk_size": stats.get("avg_chunk_size"),
                    "min_chunk_size": stats.get("min_chunk_size"),
                    "max_chunk_size": stats.get("max_chunk_size"),
                },
            )

            return len(self.chunked_documents) > 0, self.chunked_documents

        except Exception as e:
            self.logger.error(
                f"Error in Phase 3: {str(e)}", {"exception_type": type(e).__name__}
            )
            return False, []

    def run_phase_4_embeddings_generation(self) -> Tuple[bool, List]:
        """
        Phase 4: Generate dense embeddings using OpenAI.

        Returns:
            Tuple of (success: bool, embeddings_data: List)
        """
        try:
            if not self.chunked_documents:
                self.logger.error("No chunked documents for embedding")
                return False, []

            self.logger.info(
                "Starting Phase 4: Dense Embeddings Generation",
                {"num_chunks": len(self.chunked_documents)},
            )

            # Generate embeddings
            self.embeddings_data = self.embeddings_generator.generate_embeddings(
                self.chunked_documents
            )

            # Get statistics
            stats = self.embeddings_generator.get_embedding_statistics(
                self.embeddings_data
            )

            self.logger.info(
                "Phase 4 completed",
                {
                    "total_embeddings": stats.get("total_embeddings"),
                    "vector_dimension": stats.get("vector_dimension"),
                    "mean_l2_norm": stats.get("mean_l2_norm"),
                },
            )

            return len(self.embeddings_data) > 0, self.embeddings_data

        except Exception as e:
            self.logger.error(
                f"Error in Phase 4: {str(e)}", {"exception_type": type(e).__name__}
            )
            return False, []

    def run_phase_5_sparse_vectors_generation(self) -> Tuple[bool, List]:
        """
        Phase 5: Generate sparse vectors using BM25.

        Returns:
            Tuple of (success: bool, sparse_vectors: List)
        """
        try:
            if not self.chunked_documents:
                self.logger.error("No chunked documents for sparse vector generation")
                return False, []

            self.logger.info(
                "Starting Phase 5: Sparse Vectors Generation (BM25)",
                {"num_chunks": len(self.chunked_documents)},
            )

            # Build BM25 corpus
            if not self.sparse_generator.build_corpus(self.chunked_documents):
                self.logger.error("Failed to build BM25 corpus")
                return False, []

            # Generate sparse vectors
            self.sparse_vectors = self.sparse_generator.generate_all_sparse_vectors(
                self.chunked_documents
            )

            # Get statistics
            stats = self.sparse_generator.get_sparse_vector_statistics(
                self.sparse_vectors
            )

            self.logger.info(
                "Phase 5 completed",
                {
                    "total_vectors": stats.get("total_vectors"),
                    "vocabulary_size": stats.get("vocabulary_size"),
                    "avg_non_zero": stats.get("avg_non_zero_elements"),
                    "sparsity_pct": stats.get("sparsity_percentage"),
                },
            )

            return len(self.sparse_vectors) > 0, self.sparse_vectors

        except Exception as e:
            self.logger.error(
                f"Error in Phase 5: {str(e)}", {"exception_type": type(e).__name__}
            )
            return False, []

    def run_phase_6_pinecone_upload(self) -> Tuple[bool, Dict]:
        """
        Phase 6: Create Pinecone index and upload hybrid vectors.

        Returns:
            Tuple of (success: bool, upsert_result: Dict)
        """
        try:
            if not self.embeddings_data or len(self.embeddings_data) != len(
                self.sparse_vectors
            ):
                self.logger.error(
                    "Mismatch between dense and sparse vectors",
                    {
                        "embeddings": len(self.embeddings_data),
                        "sparse": len(self.sparse_vectors),
                    },
                )
                return False, {}

            self.logger.info(
                "Starting Phase 6: Pinecone Index Creation & Hybrid Vector Upload",
                {
                    "total_vectors": len(self.embeddings_data),
                    "vector_dimension": config.EMBEDDING_DIMENSION,
                },
            )

            # Create index if needed
            if not self.pinecone_uploader.create_index_if_needed(
                dimension=config.EMBEDDING_DIMENSION
            ):
                self.logger.error("Failed to create Pinecone index")
                return False, {}

            # Get index reference
            if not self.pinecone_uploader.get_index_reference():
                self.logger.error("Failed to get Pinecone index reference")
                return False, {}

            # Prepare hybrid vectors
            vectors_to_upsert = []

            for idx, (embed_data, sparse_vec) in enumerate(
                zip(self.embeddings_data, self.sparse_vectors)
            ):
                try:
                    # Format sparse vector for Pinecone
                    if sparse_vec:
                        sparse_indices = list(sparse_vec.keys())
                        sparse_values = list(sparse_vec.values())
                        sparse_values_dict = {
                            "indices": sparse_indices,
                            "values": sparse_values,
                        }
                    else:
                        sparse_values_dict = {"indices": [], "values": []}

                    # Combine metadata
                    metadata = {
                        **embed_data["metadata"],
                        "has_sparse": len(sparse_vec) > 0,
                        "sparse_dim": len(sparse_vec),
                        "uploaded_at": datetime.now().isoformat(),
                    }

                    # Create hybrid vector record
                    vector_dict = {
                        "id": embed_data["id"],
                        "values": embed_data["values"],  # Dense 1536D
                        "sparse_values": sparse_values_dict,  # Sparse BM25
                        "metadata": metadata,
                    }

                    vectors_to_upsert.append(vector_dict)

                except Exception as e:
                    self.logger.error(
                        f"Error preparing vector {idx}: {str(e)}",
                        {"vector_index": idx, "exception_type": type(e).__name__},
                    )
                    continue

            self.logger.info(
                f"Prepared {len(vectors_to_upsert)} hybrid vectors for upload",
                {"total_prepared": len(vectors_to_upsert)},
            )

            # Upsert to Pinecone
            upsert_result = self.pinecone_uploader.upsert_hybrid_vectors(
                vectors_to_upsert
            )

            self.logger.info(
                "Phase 6 completed",
                {
                    "success": upsert_result.get("success"),
                    "upserted": upsert_result.get("upserted_count"),
                    "failed": upsert_result.get("failed_count"),
                    "total_in_index": upsert_result.get("total_vectors_in_index"),
                },
            )

            return upsert_result.get("success", False), upsert_result

        except Exception as e:
            self.logger.error(
                f"Error in Phase 6: {str(e)}", {"exception_type": type(e).__name__}
            )
            return False, {"error": str(e)}

    def run_complete_pipeline(
        self,
        document_sources: List[str],
        from_s3: bool = False,
        enable_cloudwatch: bool = False,
    ) -> Dict:
        """
        Execute the complete data ingestion pipeline (all phases).

        Args:
            document_sources: List of documents to ingest
            from_s3: Load from S3 or local filesystem
            enable_cloudwatch: Enable CloudWatch logging

        Returns:
            Dictionary with pipeline execution results
        """
        pipeline_start_time = datetime.now()

        self.logger.info(
            "=" * 80, {"message": "STARTING COMPLETE DATA INGESTION PIPELINE"}
        )

        # Validate configuration
        if not self.validate_configuration():
            return {
                "success": False,
                "error": "Configuration validation failed",
                "timestamp": datetime.now().isoformat(),
            }

        results = {
            "pipeline_name": "DataIngestionPipeline",
            "start_time": pipeline_start_time.isoformat(),
            "phases": {},
            "overall_success": False,
            "total_documents_ingested": 0,
            "total_chunks_created": 0,
            "total_embeddings": 0,
            "vectors_in_pinecone": 0,
            "errors": [],
        }

        try:
            # Phase 2: Document Ingestion
            self.logger.info("\n" + "=" * 80)
            phase2_success, phase2_docs = self.run_phase_2_document_ingestion(
                document_sources, from_s3=from_s3
            )
            results["phases"]["phase_2_ingestion"] = {
                "success": phase2_success,
                "documents_extracted": len(phase2_docs) if phase2_success else 0,
            }

            if not phase2_success:
                results["errors"].append("Phase 2 (Document Ingestion) failed")
                raise Exception("Document ingestion failed - stopping pipeline")

            results["total_documents_ingested"] = len(self.extracted_documents)

            # Phase 3: Semantic Chunking
            self.logger.info("\n" + "=" * 80)
            phase3_success, phase3_chunks = self.run_phase_3_semantic_chunking()
            results["phases"]["phase_3_chunking"] = {
                "success": phase3_success,
                "chunks_created": len(phase3_chunks) if phase3_success else 0,
            }

            if not phase3_success:
                results["errors"].append("Phase 3 (Semantic Chunking) failed")
                raise Exception("Semantic chunking failed - stopping pipeline")

            results["total_chunks_created"] = len(self.chunked_documents)

            # Phase 4: Embeddings Generation
            self.logger.info("\n" + "=" * 80)
            phase4_success, phase4_embeddings = self.run_phase_4_embeddings_generation()
            results["phases"]["phase_4_embeddings"] = {
                "success": phase4_success,
                "embeddings_generated": len(phase4_embeddings) if phase4_success else 0,
            }

            if not phase4_success:
                results["errors"].append("Phase 4 (Embeddings Generation) failed")
                raise Exception("Embeddings generation failed - stopping pipeline")

            results["total_embeddings"] = len(self.embeddings_data)

            # Phase 5: Sparse Vectors Generation
            self.logger.info("\n" + "=" * 80)
            phase5_success, phase5_sparse = self.run_phase_5_sparse_vectors_generation()
            results["phases"]["phase_5_sparse_vectors"] = {
                "success": phase5_success,
                "sparse_vectors_generated": len(phase5_sparse) if phase5_success else 0,
            }

            if not phase5_success:
                results["errors"].append("Phase 5 (Sparse Vectors) failed")
                raise Exception("Sparse vector generation failed - stopping pipeline")

            # Phase 6: Pinecone Upload
            self.logger.info("\n" + "=" * 80)
            phase6_success, phase6_result = self.run_phase_6_pinecone_upload()
            results["phases"]["phase_6_pinecone"] = {
                "success": phase6_success,
                "upsert_result": phase6_result,
            }

            if not phase6_success:
                results["errors"].append(
                    f"Phase 6 (Pinecone Upload) failed: {phase6_result.get('error', '')}"
                )

            results["vectors_in_pinecone"] = phase6_result.get(
                "total_vectors_in_index", 0
            )
            results["overall_success"] = (
                phase2_success
                and phase3_success
                and phase4_success
                and phase5_success
                and phase6_success
            )

        except Exception as e:
            self.logger.error(
                f"Pipeline execution error: {str(e)}",
                {"exception_type": type(e).__name__},
            )
            results["errors"].append(f"Pipeline execution error: {str(e)}")
            results["overall_success"] = False

        finally:
            # Timing
            pipeline_end_time = datetime.now()
            results["end_time"] = pipeline_end_time.isoformat()
            results["duration_seconds"] = (
                pipeline_end_time - pipeline_start_time
            ).total_seconds()

            self.logger.info(
                "\n" + "=" * 80,
                {
                    "message": "PIPELINE EXECUTION COMPLETED",
                    "success": results["overall_success"],
                    "duration": results["duration_seconds"],
                },
            )

            self.logger.info(
                "Pipeline Summary",
                {
                    "overall_success": results["overall_success"],
                    "documents_ingested": results["total_documents_ingested"],
                    "chunks_created": results["total_chunks_created"],
                    "embeddings": results["total_embeddings"],
                    "vectors_in_pinecone": results["vectors_in_pinecone"],
                    "errors_count": len(results["errors"]),
                },
            )

        return results


def create_and_run_pipeline(
    document_sources: List[str], from_s3: bool = False, enable_cloudwatch: bool = False
) -> Dict:
    """
    Factory function to create and execute the complete data ingestion pipeline.

    Args:
        document_sources: List of document paths/S3 keys
        from_s3: Load from S3 (True) or local (False)
        enable_cloudwatch: Enable CloudWatch logging

    Returns:
        Pipeline execution results dictionary
    """
    pipeline = DataIngestionPipeline(enable_cloudwatch=enable_cloudwatch)
    return pipeline.run_complete_pipeline(document_sources, from_s3=from_s3)
