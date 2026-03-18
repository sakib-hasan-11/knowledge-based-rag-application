"""
Main Retrieval Pipeline Orchestrator

Orchestrates all phases of the retrieval pipeline:
- Phase 7: Pre-Retrieval query optimization
- Phase 8: During-Retrieval hybrid search with reranking
- Phase 9: Post-Retrieval response generation

Production-grade pipeline with comprehensive error handling and monitoring.
"""

from datetime import datetime
from typing import Dict, List, Optional

from src.data_ingestion.config import config
from src.data_ingestion.logging_config import create_logger
from src.retrieval.during_retrieval import (
    CrossEncoderReranker,
    HybridRetriever,
    MMRReranker,
)
from src.retrieval.post_retrieval import (
    ChainOfThoughtReasoner,
    ContextualCompressor,
    ConversationMemoryManager,
    PromptTemplateBuilder,
    TokenCounter,
)
from src.retrieval.pre_retrieval import (
    DomainRouter,
    HyDEGenerator,
    MultiQueryGenerator,
    QueryRewriter,
)


class RetrievalPipeline:
    """
    Production-grade end-to-end retrieval pipeline for RAG applications.
    Handles query optimization, multi-strategy retrieval, and response generation.
    """

    def __init__(
        self,
        embeddings_model,
        sparse_generator,
        index,
        s3_client=None,
        enable_cloudwatch: bool = False,
        logger_name: str = "RetrievalPipeline",
    ):
        """
        Initialize complete retrieval pipeline.

        Args:
            embeddings_model: OpenAI embeddings model
            sparse_generator: BM25 sparse vector generator
            index: Pinecone index reference
            s3_client: boto3 S3 client for memory persistence
            enable_cloudwatch: Enable CloudWatch logging
            logger_name: Logger identifier
        """
        self.logger = create_logger(
            logger_name,
            level=config.LOG_LEVEL,
            log_format=config.LOG_FORMAT,
            enable_cloudwatch=enable_cloudwatch,
        )

        # Store components
        self.embeddings_model = embeddings_model
        self.sparse_generator = sparse_generator
        self.index = index
        self.s3_client = s3_client

        # Initialize Phase 7: Pre-Retrieval
        self.query_rewriter = QueryRewriter()
        self.multi_query_gen = MultiQueryGenerator()
        self.hyde_generator = HyDEGenerator(embedding_model=embeddings_model)
        self.domain_router = DomainRouter()

        # Initialize Phase 8: During-Retrieval
        self.hybrid_retriever = HybridRetriever(
            embeddings_model, sparse_generator, index
        )
        self.mmr_reranker = MMRReranker(embeddings_model)
        self.cross_encoder = CrossEncoderReranker()

        # Initialize Phase 9: Post-Retrieval
        self.token_counter = TokenCounter()
        self.contextual_compressor = ContextualCompressor()
        self.prompt_builder = PromptTemplateBuilder()
        self.memory_manager = ConversationMemoryManager(s3_client)
        self.reasoner = ChainOfThoughtReasoner()

        self.logger.info("RetrievalPipeline initialized successfully")

    def run_phase_7_pre_retrieval(
        self,
        query: str,
        use_hyde: bool = True,
        use_multi_query: bool = True,
        num_queries: int = 3,
    ) -> Dict:
        """
        Phase 7: Pre-Retrieval query optimization and routing.

        Args:
            query: User query
            use_hyde: Enable HyDE generation
            use_multi_query: Enable multi-query generation
            num_queries: Number of query variations

        Returns:
            Dictionary with pre-retrieval results
        """
        try:
            self.logger.info(
                "Starting Phase 7: Pre-Retrieval",
                {
                    "query": query[:100],
                    "use_hyde": use_hyde,
                    "use_multi_query": use_multi_query,
                },
            )

            phase_output = {"original_query": query, "stages": {}}

            # Stage 1: Query Rewriting
            rewritten = self.query_rewriter.rewrite(query)
            phase_output["stages"]["query_rewriting"] = {
                "original": query,
                "rewritten": rewritten,
            }

            query_for_retrieval = rewritten

            # Stage 2: Domain Routing
            routing_config = self.domain_router.get_routing_config(query_for_retrieval)
            phase_output["stages"]["domain_routing"] = routing_config

            # Stage 3: Multi-Query or HyDE
            queries_to_search = [query_for_retrieval]

            if use_multi_query:
                queries_to_search = self.multi_query_gen.generate_queries(
                    query_for_retrieval
                )
                phase_output["stages"]["multi_query"] = {
                    "variations": queries_to_search,
                    "count": len(queries_to_search),
                }

            if use_hyde:
                hyde_embedding = self.hyde_generator.get_hyde_embedding(
                    query_for_retrieval
                )
                phase_output["stages"]["hyde"] = {
                    "embedding_available": hyde_embedding is not None,
                    "dimension": len(hyde_embedding) if hyde_embedding else 0,
                }

            phase_output["queries_for_retrieval"] = queries_to_search

            self.logger.info(
                "Phase 7 completed",
                {
                    "domain": routing_config.get("domain"),
                    "queries_generated": len(queries_to_search),
                },
            )
            return phase_output

        except Exception as e:
            self.logger.error(f"Error in Phase 7: {str(e)}")
            return {
                "original_query": query,
                "error": str(e),
                "queries_for_retrieval": [query],
            }

    def run_phase_8_during_retrieval(
        self,
        queries: List[str],
        use_hybrid: bool = True,
        use_mmr: bool = True,
        use_reranking: bool = True,
        top_k: int = 5,
    ) -> Dict:
        """
        Phase 8: Multi-strategy retrieval with reranking.

        Args:
            queries: List of queries to search
            use_hybrid: Use hybrid retrieval
            use_mmr: Use MMR reranking
            use_reranking: Use cross-encoder reranking
            top_k: Final number of results

        Returns:
            Dictionary with retrieval results
        """
        try:
            self.logger.info(
                "Starting Phase 8: During-Retrieval",
                {
                    "num_queries": len(queries),
                    "use_hybrid": use_hybrid,
                    "use_mmr": use_mmr,
                    "use_reranking": use_reranking,
                },
            )

            phase_output = {"stages": {}}

            # Stage 1: Hybrid Retrieval
            all_results = []
            for query in queries:
                try:
                    results = self.hybrid_retriever.retrieve(query)
                    all_results.append(results)
                except Exception as e:
                    self.logger.warning(
                        f"Error retrieving for query '{query}': {str(e)}"
                    )
                    continue

            if not all_results:
                self.logger.error("No retrieval results obtained")
                return {"error": "No retrieval results", "final_results": []}

            # Merge multi-query results
            merged_results = (
                self.multi_query_gen.merge_results(all_results)
                if len(all_results) > 1
                else all_results[0]
            )

            phase_output["stages"]["hybrid_retrieval"] = {
                "results_count": len(merged_results)
            }

            current_results = merged_results

            # Stage 2: MMR Reranking
            if use_mmr and len(current_results) > 0:
                mmr_results = self.mmr_reranker.rerank(
                    queries[0],  # Use first query for MMR
                    current_results,
                    top_k=min(len(current_results), 10),
                )
                phase_output["stages"]["mmr_reranking"] = {
                    "results_count": len(mmr_results)
                }
                current_results = mmr_results

            # Stage 3: Cross-Encoder Reranking
            if use_reranking and len(current_results) > 0:
                final_results = self.cross_encoder.rerank(
                    queries[0], current_results, top_k=top_k
                )
                phase_output["stages"]["cross_encoder_reranking"] = {
                    "results_count": len(final_results)
                }
                current_results = final_results

            phase_output["final_results"] = current_results

            self.logger.info(
                "Phase 8 completed", {"final_results_count": len(current_results)}
            )
            return phase_output

        except Exception as e:
            self.logger.error(f"Error in Phase 8: {str(e)}")
            return {"error": str(e), "final_results": []}

    def run_phase_9_post_retrieval(
        self,
        query: str,
        retrieved_documents: List[Dict],
        use_compression: bool = True,
        use_cot: bool = True,
        session_id: Optional[str] = None,
    ) -> Dict:
        """
        Phase 9: Response generation and context management.

        Args:
            query: Original user query
            retrieved_documents: Documents from Phase 8
            use_compression: Compress documents
            use_cot: Use chain-of-thought reasoning
            session_id: Session ID for memory retrieval

        Returns:
            Dictionary with response and metadata
        """
        try:
            self.logger.info(
                "Starting Phase 9: Post-Retrieval",
                {
                    "query": query[:100],
                    "num_docs": len(retrieved_documents),
                    "use_compression": use_compression,
                    "use_cot": use_cot,
                },
            )

            phase_output = {"query": query, "stages": {}}

            # Load session if provided
            if session_id:
                self.memory_manager.load_session_from_s3(session_id)

            # Stage 1: Token Counting
            token_count = sum(
                self.token_counter.estimate_doc_tokens(doc)
                for doc in retrieved_documents
            )
            phase_output["stages"]["token_counting"] = {"total_tokens": token_count}

            # Stage 2: Contextual Compression
            docs_to_use = retrieved_documents
            if use_compression:
                compressed = self.contextual_compressor.compress_documents(
                    query, retrieved_documents
                )
                phase_output["stages"]["compression"] = {
                    "original_docs": len(retrieved_documents),
                    "compressed_ratio": sum(
                        d.get("compression_ratio", 0) for d in compressed
                    )
                    / max(len(compressed), 1),
                }
                docs_to_use = compressed

            # Stage 3: Format Context
            context = self._format_context(docs_to_use)
            phase_output["stages"]["context_formatting"] = {
                "context_length": len(context)
            }

            # Stage 4: Analyze Complexity
            complexity = self.reasoner.analyze_query_complexity(query)
            phase_output["stages"]["complexity_analysis"] = complexity

            # Stage 5: Add to Memory
            self.memory_manager.add_interaction(
                query,
                "",  # Response will be added later
                [doc.get("metadata", {}).get("source") for doc in docs_to_use[:3]],
            )

            # Stage 6: Build Prompts
            system_prompt = self.prompt_builder.build_system_prompt()
            memory_str = self.memory_manager.get_memory_string()
            user_prompt = self.prompt_builder.build_user_prompt(
                query, context, memory_str
            )

            phase_output["stages"]["prompt_building"] = {
                "system_prompt_length": len(system_prompt),
                "user_prompt_length": len(user_prompt),
                "include_memory": bool(memory_str),
            }

            phase_output["prompts"] = {"system": system_prompt, "user": user_prompt}
            phase_output["final_documents"] = docs_to_use
            phase_output["session_id"] = self.memory_manager.session_id

            self.logger.info(
                "Phase 9 completed",
                {
                    "complexity_score": complexity["complexity_score"],
                    "uses_cot": use_cot and complexity["requires_cot"],
                },
            )
            return phase_output

        except Exception as e:
            self.logger.error(f"Error in Phase 9: {str(e)}")
            return {
                "error": str(e),
                "query": query,
                "final_documents": retrieved_documents,
            }

    def _format_context(self, documents: List[Dict]) -> str:
        """Format documents into context string."""
        try:
            context_parts = []

            for i, doc in enumerate(documents[:5], 1):
                title = doc.get("metadata", {}).get("section", "Document")
                source = doc.get("metadata", {}).get("source", "Unknown")
                content = doc.get("compressed_content", "")

                if not content:
                    content = doc.get("metadata", {}).get("text_preview", "")

                if content:
                    context_parts.append(f"[Doc {i}: {title} ({source})]\n{content}\n")

            return (
                "\n".join(context_parts)
                if context_parts
                else "No relevant documents found."
            )
        except Exception as e:
            self.logger.error(f"Error formatting context: {str(e)}")
            return "Error formatting context"

    def run_complete_pipeline(
        self,
        query: str,
        enable_hyde: bool = True,
        enable_multi_query: bool = True,
        enable_mmr: bool = True,
        enable_compression: bool = True,
        session_id: Optional[str] = None,
    ) -> Dict:
        """
        Execute complete retrieval pipeline (all phases).

        Args:
            query: User query
            enable_hyde: Enable HyDE in Phase 7
            enable_multi_query: Enable multi-query in Phase 7
            enable_mmr: Enable MMR in Phase 8
            enable_compression: Enable compression in Phase 9
            session_id: Session ID for memory

        Returns:
            Complete pipeline execution results
        """
        pipeline_start_time = datetime.now()

        self.logger.info("=" * 80, {"message": "STARTING COMPLETE RETRIEVAL PIPELINE"})

        results = {
            "pipeline_name": "RetrievalPipeline",
            "start_time": pipeline_start_time.isoformat(),
            "query": query,
            "phases": {},
            "overall_success": False,
            "errors": [],
        }

        try:
            # Phase 7: Pre-Retrieval
            self.logger.info("\n" + "=" * 80)
            phase7_results = self.run_phase_7_pre_retrieval(
                query, use_hyde=enable_hyde, use_multi_query=enable_multi_query
            )
            results["phases"]["phase_7_pre_retrieval"] = phase7_results

            if "error" in phase7_results:
                results["errors"].append(f"Phase 7 error: {phase7_results['error']}")
                queries = [query]
            else:
                queries = phase7_results.get("queries_for_retrieval", [query])

            # Phase 8: During-Retrieval
            self.logger.info("\n" + "=" * 80)
            phase8_results = self.run_phase_8_during_retrieval(
                queries, use_hybrid=True, use_mmr=enable_mmr, use_reranking=True
            )
            results["phases"]["phase_8_during_retrieval"] = phase8_results

            if "error" in phase8_results:
                results["errors"].append(f"Phase 8 error: {phase8_results['error']}")
                final_documents = []
            else:
                final_documents = phase8_results.get("final_results", [])

            # Phase 9: Post-Retrieval
            self.logger.info("\n" + "=" * 80)
            phase9_results = self.run_phase_9_post_retrieval(
                query,
                final_documents,
                use_compression=enable_compression,
                session_id=session_id,
            )
            results["phases"]["phase_9_post_retrieval"] = phase9_results

            if "error" in phase9_results:
                results["errors"].append(f"Phase 9 error: {phase9_results['error']}")

            results["final_documents"] = final_documents
            results["overall_success"] = (
                len(final_documents) > 0 and len(results["errors"]) == 0
            )

        except Exception as e:
            self.logger.error(f"Pipeline execution error: {str(e)}")
            results["errors"].append(f"Pipeline error: {str(e)}")
            results["overall_success"] = False

        finally:
            pipeline_end_time = datetime.now()
            results["end_time"] = pipeline_end_time.isoformat()
            results["duration_seconds"] = (
                pipeline_end_time - pipeline_start_time
            ).total_seconds()

            self.logger.info(
                "\n" + "=" * 80,
                {
                    "message": "RETRIEVAL PIPELINE COMPLETED",
                    "success": results["overall_success"],
                    "duration": results["duration_seconds"],
                    "documents_retrieved": len(results.get("final_documents", [])),
                },
            )

        return results


def create_and_run_retrieval_pipeline(
    query: str,
    embeddings_model,
    sparse_generator,
    index,
    s3_client=None,
    enable_cloudwatch: bool = False,
) -> Dict:
    """
    Factory function to create and execute retrieval pipeline.

    Args:
        query: User query
        embeddings_model: Embeddings model
        sparse_generator: Sparse vector generator
        index: Pinecone index
        s3_client: S3 client for memory
        enable_cloudwatch: Enable CloudWatch logging

    Returns:
        Pipeline execution results
    """
    pipeline = RetrievalPipeline(
        embeddings_model,
        sparse_generator,
        index,
        s3_client,
        enable_cloudwatch=enable_cloudwatch,
    )

    return pipeline.run_complete_pipeline(query)
