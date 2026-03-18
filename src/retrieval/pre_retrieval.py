"""
Pre-Retrieval Pipeline Module (Phase 7)

Handles query optimization and routing before retrieval:
- Query rewriting for clarity
- Multi-query generation for comprehensive search
- HyDE (Hypothetical Document Embeddings) generation
- Domain-aware routing for section selection
"""

import os
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.data_ingestion.config import config
from src.data_ingestion.logging_config import create_logger


class QueryRewriter:
    """Improves query clarity and intent using LLM with error handling."""

    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
        temperature: float = 0.3,
        logger_name: str = "QueryRewriter",
    ):
        """Initialize query rewriter with error handling."""
        self.logger = create_logger(logger_name)
        self.model_name = model_name
        self.temperature = temperature

        # Validate API key is available
        if not config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not configured. Set OPENAI_API_KEY in .env file or environment variables."
            )

        try:
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=256,
                api_key=config.OPENAI_API_KEY,
            )

            self.rewrite_prompt = PromptTemplate(
                input_variables=["original_query"],
                template="""Rewrite the user's query to improve clarity and specificity for document retrieval.
Keep the core meaning but make it more explicit.

Original Query: {original_query}

Rewritten Query:""",
            )

            self.logger.info("QueryRewriter initialized", {"model": model_name})
        except ValueError as e:
            self.logger.error(
                f"Configuration error initializing QueryRewriter: {str(e)}"
            )
            raise
        except Exception as e:
            self.logger.error(f"Error initializing QueryRewriter: {str(e)}")
            raise

    def rewrite(self, query: str) -> str:
        """Rewrite a single query with error handling."""
        try:
            prompt_text = self.rewrite_prompt.format(original_query=query)
            message = HumanMessage(content=prompt_text)
            response = self.llm.invoke([message])

            rewritten = response.content.strip()
            self.logger.debug(
                "Query rewritten",
                {
                    "original": query[:100],
                    "rewritten": rewritten[:100],
                    "model": self.model_name,
                },
            )
            return rewritten
        except Exception as e:
            self.logger.error(
                f"Error rewriting query: {str(e)}",
                {"query": query[:100], "exception_type": type(e).__name__},
            )
            return query  # Fallback to original query

    def batch_rewrite(self, queries: List[str]) -> List[str]:
        """Rewrite multiple queries with error recovery."""
        results = []
        for query in queries:
            try:
                results.append(self.rewrite(query))
            except Exception as e:
                self.logger.warning(
                    f"Failed to rewrite query, using original: {str(e)}"
                )
                results.append(query)
        return results


class MultiQueryGenerator:
    """Generates query variations for comprehensive multi-perspective search."""

    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
        num_queries: int = 4,
        temperature: float = 0.7,
        logger_name: str = "MultiQueryGenerator",
    ):
        """Initialize multi-query generator."""
        self.logger = create_logger(logger_name)
        self.model_name = model_name
        self.num_queries = num_queries
        self.temperature = temperature

        # Validate API key is available
        if not config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not configured. Set OPENAI_API_KEY in .env file or environment variables."
            )

        try:
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=512,
                api_key=config.OPENAI_API_KEY,
            )

            self.generation_prompt = PromptTemplate(
                input_variables=["original_query", "num_queries"],
                template="""Generate {num_queries} different query variations capturing different aspects.
Each variation on a new line, numbered. Keep concise and focused.

Original Query: {original_query}

Query Variations:""",
            )

            self.logger.info(
                "MultiQueryGenerator initialized",
                {"model": model_name, "num_queries": num_queries},
            )
        except Exception as e:
            self.logger.error(f"Error initializing MultiQueryGenerator: {str(e)}")
            raise

    def generate_queries(self, query: str) -> List[str]:
        """Generate query variations with error handling."""
        try:
            prompt_text = self.generation_prompt.format(
                original_query=query, num_queries=self.num_queries
            )
            message = HumanMessage(content=prompt_text)
            response = self.llm.invoke([message])

            lines = response.content.split("\n")
            variations = []

            for line in lines:
                cleaned = line.lstrip("0123456789.-) ").strip()
                if cleaned and len(cleaned) > 5:
                    variations.append(cleaned)

            result = [query] + variations[: self.num_queries - 1]

            self.logger.info(
                "Generated query variations",
                {"original": query[:100], "variations_count": len(result)},
            )
            return result
        except Exception as e:
            self.logger.error(
                f"Error generating queries: {str(e)}",
                {"query": query[:100], "exception_type": type(e).__name__},
            )
            return [query]  # Fallback to original

    def merge_results(self, search_results: List[List[Dict]]) -> List[Dict]:
        """Merge and deduplicate results from multiple queries."""
        try:
            merged = {}

            for query_results in search_results:
                for result in query_results:
                    vec_id = result.get("id")
                    if vec_id not in merged:
                        merged[vec_id] = {
                            **result,
                            "occurrence_count": 1,
                            "combined_score": result.get("score", 0),
                        }
                    else:
                        merged[vec_id]["occurrence_count"] += 1
                        merged[vec_id]["combined_score"] += result.get("score", 0)

            sorted_results = sorted(
                merged.values(),
                key=lambda x: (x["occurrence_count"], x["combined_score"]),
                reverse=True,
            )

            self.logger.info(
                "Merged search results",
                {
                    "input_result_sets": len(search_results),
                    "output_unique_docs": len(sorted_results),
                },
            )
            return sorted_results
        except Exception as e:
            self.logger.error(
                f"Error merging results: {str(e)}", {"exception_type": type(e).__name__}
            )
            return []  # Return empty on merge failure


class HyDEGenerator:
    """Generates hypothetical documents for improved semantic search."""

    def __init__(
        self,
        llm_model: str = "gpt-4-turbo",
        embedding_model: Optional[OpenAIEmbeddings] = None,
        temperature: float = 0.7,
        logger_name: str = "HyDEGenerator",
    ):
        """Initialize HyDE generator."""
        self.logger = create_logger(logger_name)
        self.llm_model = llm_model

        # Validate API keys are available
        if not config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not configured. Set OPENAI_API_KEY in .env file or environment variables."
            )

        try:
            self.llm = ChatOpenAI(
                model_name=llm_model,
                temperature=temperature,
                max_tokens=512,
                api_key=config.OPENAI_API_KEY,
            )

            self.embedding_model = embedding_model or OpenAIEmbeddings(
                model=config.EMBEDDING_MODEL, dimensions=config.EMBEDDING_DIMENSION
            )

            self.hyde_prompt = PromptTemplate(
                input_variables=["query"],
                template="""Imagine you are writing a comprehensive financial document that answers the following query.
Write a natural, detailed response with relevant terminology and context. Keep it focused and 150-250 words.

Query: {query}

Hypothetical Document:""",
            )

            self.logger.info("HyDEGenerator initialized", {"model": llm_model})
        except ValueError as e:
            self.logger.error(
                f"Configuration error initializing HyDEGenerator: {str(e)}"
            )
            raise
        except Exception as e:
            self.logger.error(f"Error initializing HyDEGenerator: {str(e)}")
            raise

    def generate_hypothetical_doc(self, query: str) -> Optional[str]:
        """Generate hypothetical document with error handling."""
        try:
            prompt_text = self.hyde_prompt.format(query=query)
            message = HumanMessage(content=prompt_text)
            response = self.llm.invoke([message])

            hypo_doc = response.content.strip()
            self.logger.debug(
                "Generated hypothetical document",
                {"query": query[:100], "doc_length": len(hypo_doc)},
            )
            return hypo_doc
        except Exception as e:
            self.logger.error(
                f"Error generating hypothetical document: {str(e)}",
                {"query": query[:100], "exception_type": type(e).__name__},
            )
            return None

    def get_hyde_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for hypothetical document."""
        try:
            hypo_doc = self.generate_hypothetical_doc(query)
            if not hypo_doc:
                return None

            embedding = self.embedding_model.embed_query(hypo_doc)
            self.logger.debug(
                "Generated HyDE embedding",
                {"dimension": len(embedding), "query": query[:100]},
            )
            return embedding
        except Exception as e:
            self.logger.error(
                f"Error embedding hypothetical document: {str(e)}",
                {"exception_type": type(e).__name__},
            )
            return None

    def batch_hyde_embeddings(self, queries: List[str]) -> List[Optional[List[float]]]:
        """Generate HyDE embeddings for multiple queries."""
        embeddings = []
        for query in queries:
            try:
                embedding = self.get_hyde_embedding(query)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.warning(f"Failed to generate HyDE for query: {str(e)}")
                embeddings.append(None)
        return embeddings


class DomainRouter:
    """Classifies queries and routes to appropriate retrieval strategy."""

    DOMAINS = {
        "finance": {
            "keywords": [
                "revenue",
                "earnings",
                "financial",
                "profit",
                "cash flow",
                "balance sheet",
            ],
            "sections": ["Item 7", "Item 8"],
            "retriever_type": "hybrid",
        },
        "operations": {
            "keywords": [
                "business",
                "operations",
                "segment",
                "product",
                "service",
                "market",
            ],
            "sections": ["Item 1", "Item 1A"],
            "retriever_type": "semantic",
        },
        "risk": {
            "keywords": [
                "risk",
                "uncertainty",
                "liability",
                "threat",
                "challenge",
                "competition",
            ],
            "sections": ["Item 1A"],
            "retriever_type": "semantic",
        },
    }

    def __init__(
        self,
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        logger_name: str = "DomainRouter",
    ):
        """Initialize domain router."""
        self.logger = create_logger(logger_name)

        # Validate API key is available
        if not config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not configured. Set OPENAI_API_KEY in .env file or environment variables."
            )

        try:
            self.llm = ChatOpenAI(
                model_name=llm_model,
                temperature=temperature,
                max_tokens=256,
                api_key=config.OPENAI_API_KEY,
            )

            self.classification_prompt = PromptTemplate(
                input_variables=["query"],
                template="""Classify this financial query into ONE domain: 'finance', 'operations', or 'risk'.
Return only the domain name.

Query: {query}

Domain:""",
            )

            self.logger.info("DomainRouter initialized", {"model": llm_model})
        except Exception as e:
            self.logger.error(f"Error initializing DomainRouter: {str(e)}")
            raise

    def classify_domain(self, query: str) -> str:
        """Classify query domain with fallback."""
        try:
            prompt_text = self.classification_prompt.format(query=query)
            message = HumanMessage(content=prompt_text)
            response = self.llm.invoke([message])

            domain = response.content.lower().strip()

            if domain not in self.DOMAINS:
                domain = self._fallback_classify(query)

            self.logger.debug(
                "Classified query domain", {"query": query[:100], "domain": domain}
            )
            return domain
        except Exception as e:
            self.logger.error(
                f"Error classifying domain: {str(e)}",
                {"query": query[:100], "exception_type": type(e).__name__},
            )
            return self._fallback_classify(query)

    def _fallback_classify(self, query: str) -> str:
        """Keyword-based fallback classification."""
        query_lower = query.lower()
        scores = {}

        for domain, config_dict in self.DOMAINS.items():
            score = sum(1 for kw in config_dict["keywords"] if kw in query_lower)
            scores[domain] = score

        result = max(scores, key=scores.get) if max(scores.values()) > 0 else "finance"
        self.logger.debug("Using fallback keyword classification", {"domain": result})
        return result

    def get_section_filters(self, domain: str) -> List[str]:
        """Get relevant sections for domain."""
        return self.DOMAINS.get(domain, {}).get("sections", [])

    def get_routing_config(self, query: str) -> Dict:
        """Get complete routing configuration."""
        try:
            domain = self.classify_domain(query)
            config = self.DOMAINS.get(domain, self.DOMAINS["finance"])

            return {
                "domain": domain,
                "sections": config["sections"],
                "retriever_type": config["retriever_type"],
            }
        except Exception as e:
            self.logger.error(f"Error getting routing config: {str(e)}")
            return {
                "domain": "finance",
                "sections": ["Item 7", "Item 8"],
                "retriever_type": "hybrid",
            }  # Safe default
