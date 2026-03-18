"""
Argumentation & Generation Pipeline Orchestrator (Phase 10)

Main orchestrator for complete response generation pipeline combining
prompt building, memory management, reasoning, and LLM invocation.
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation_components import (
    ChainOfThoughtReasoner,
    ConversationMemoryManager,
    PromptTemplateBuilder,
)

from data_ingestion.config import Config
from data_ingestion.logging_config import create_logger

try:
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage
except ImportError:
    try:
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI
    except ImportError:
        ChatOpenAI = None
        HumanMessage = None

logger = create_logger(__name__)


class ArgumentationPipeline:
    """
    Complete pipeline: query → memory → prompt → LLM → response → memory save.

    Orchestrates all components of Phase 10 (argumentation & generation) including
    prompt building, conversation management, and response generation.
    """

    def __init__(
        self,
        llm_client,
        s3_client,
        bucket_name: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.3,
        max_tokens: int = 500,
    ):
        """
        Initialize argumentation pipeline.

        Args:
            llm_client: OpenAI or Claude client
            s3_client: boto3 S3 client
            bucket_name: S3 bucket for chat history
            model_name: LLM model name
            temperature: LLM temperature setting
            max_tokens: Max tokens in response
        """
        self.llm_client = llm_client
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize components
        self.prompt_builder = PromptTemplateBuilder()
        self.memory_manager = ConversationMemoryManager(s3_client, bucket_name)
        self.reasoner = ChainOfThoughtReasoner()

        logger.info(
            f"ArgumentationPipeline initialized",
            extra={
                "model": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "session_id": self.memory_manager.session_id,
            },
        )

    def generate_response(
        self,
        query: str,
        retrieved_documents: List[Dict],
        session_id: str = None,
        use_cot: bool = True,
        verbose: bool = False,
    ) -> Dict:
        """
        Execute full generation pipeline.

        Args:
            query: User query
            retrieved_documents: List of retrieved context documents
            session_id: Optional session ID to load previous context
            use_cot: Whether to use chain-of-thought reasoning
            verbose: Whether to log detailed steps

        Returns:
            Dict with response, citations, and metadata
        """
        start_time = time.time()

        try:
            logger.info(
                f"Phase 10 - Generation started",
                extra={
                    "query": query[:100],
                    "num_docs": len(retrieved_documents),
                    "use_cot": use_cot,
                },
            )

            # Load previous session if provided
            if session_id:
                try:
                    self.memory_manager.load_session_from_s3(session_id)
                    logger.debug(f"Session loaded: {session_id}")
                except Exception as e:
                    logger.warning(f"Could not load session {session_id}: {str(e)}")

            # Step 1: Format context from retrieved documents
            context = self._format_context(retrieved_documents, verbose)
            if verbose:
                logger.info(f"Context formatted: {len(context)} chars")

            # Step 2: Get conversation memory
            memory_summary = self.memory_manager.get_memory_string()
            if verbose:
                logger.info(f"Memory retrieved: {len(memory_summary)} chars")

            # Step 3: Analyze query complexity
            complexity = self.reasoner.analyze_query_complexity(query)
            if verbose:
                logger.info(
                    f"Query complexity: {complexity['complexity_score']}, requires CoT: {complexity['requires_cot']}"
                )

            # Step 4: Build system prompt
            system_prompt = self.prompt_builder.build_system_prompt()

            # Step 5: Build user prompt with context and memory
            user_prompt = self.prompt_builder.build_user_prompt(
                query, context, memory_summary
            )

            # Step 6: Add chain-of-thought if needed
            if use_cot and complexity["requires_cot"]:
                cot_prompt = self.reasoner.build_cot_prompt(query, context)
                user_prompt = cot_prompt + "\n\n" + user_prompt
                if verbose:
                    logger.info("Chain-of-thought reasoning added")

            # Step 7: Add few-shot examples
            user_prompt = self.prompt_builder.add_few_shot_to_prompt(
                user_prompt, complexity["indicators"]
            )
            if verbose:
                logger.info("Few-shot examples added")

            # Step 8: Call LLM
            response_text = self._call_llm(system_prompt, user_prompt, verbose)
            if verbose:
                logger.info(f"LLM response generated: {len(response_text)} chars")

            # Step 9: Extract citations
            citations = self._extract_citations(response_text, retrieved_documents)

            # Step 10: Extract reasoning steps if CoT was used
            reasoning_steps = {}
            if use_cot and complexity["requires_cot"]:
                reasoning_steps = self.reasoner.extract_reasoning_steps(response_text)

            # Step 11: Store in memory
            sources = [
                doc.get("metadata", {}).get("source", "Unknown")
                for doc in retrieved_documents[:3]
            ]
            self.memory_manager.add_interaction(
                query,
                response_text,
                sources,
                reasoning=" | ".join(reasoning_steps.get("reasoning_steps", [])),
            )

            # Step 12: Save session to S3
            self.memory_manager.save_to_s3()

            elapsed_time = time.time() - start_time

            result = {
                "query": query,
                "response": response_text,
                "citations": citations,
                "session_id": self.memory_manager.session_id,
                "used_cot": (use_cot and complexity["requires_cot"]),
                "complexity_score": complexity["complexity_score"],
                "reasoning_steps": reasoning_steps,
                "sources_used": len(retrieved_documents),
                "execution_time_seconds": elapsed_time,
            }

            logger.info(
                f"Phase 10 - Generation completed",
                extra={
                    "response_length": len(response_text),
                    "citations_count": len(citations),
                    "execution_time": elapsed_time,
                },
            )

            return result

        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}", exc_info=True)
            return {
                "query": query,
                "response": f"Error generating response: {str(e)}",
                "citations": [],
                "session_id": self.memory_manager.session_id,
                "used_cot": False,
                "complexity_score": 0,
                "error": str(e),
                "execution_time_seconds": time.time() - start_time,
            }

    def _format_context(self, documents: List[Dict], verbose: bool = False) -> str:
        """
        Format retrieved documents into context string.

        Args:
            documents: List of retrieved documents
            verbose: Whether to log details

        Returns:
            Formatted context string
        """
        try:
            context_parts = []

            for i, doc in enumerate(documents[:5], 1):
                title = doc.get("metadata", {}).get("title", "Document")
                source = doc.get("metadata", {}).get("source", "Unknown")
                content = doc.get("metadata", {}).get("text_preview", "")

                if not content:
                    content = doc.get("page_content", "")

                context_parts.append(
                    f"[Doc {i}: {title} (Source: {source})]\n{content}\n"
                )

            context_str = (
                "\n".join(context_parts)
                if context_parts
                else "No relevant documents found."
            )

            if verbose:
                logger.debug(
                    f"Context formatted from {len(documents)} documents",
                    extra={
                        "context_length": len(context_str),
                        "doc_count": len(context_parts),
                    },
                )

            return context_str
        except Exception as e:
            logger.error(f"Error formatting context: {str(e)}", exc_info=True)
            return "Error formatting context documents."

    def _call_llm(
        self, system_prompt: str, user_prompt: str, verbose: bool = False
    ) -> str:
        """
        Call LLM with prompts.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt with context
            verbose: Whether to log details

        Returns:
            LLM response or error message
        """
        try:
            if verbose:
                logger.debug(f"Calling LLM: {self.model_name}")

            completion = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            response = completion.choices[0].message.content

            if verbose:
                logger.debug(
                    f"LLM response received",
                    extra={
                        "response_length": len(response),
                        "finish_reason": completion.choices[0].finish_reason,
                    },
                )

            return response
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}", exc_info=True)
            return f"Error generating response: {str(e)}"

    def _extract_citations(self, response: str, documents: List[Dict]) -> List[Dict]:
        """
        Extract document citations from response.

        Args:
            response: LLM response
            documents: Retrieved documents

        Returns:
            List of citations
        """
        try:
            citations = []

            for doc in documents[:3]:
                title = doc.get("metadata", {}).get("title", "Document")
                source = doc.get("metadata", {}).get("source", "Unknown")
                relevance_score = doc.get("score", 0.0)

                citations.append(
                    {"title": title, "source": source, "relevance": relevance_score}
                )

            logger.debug(f"Citations extracted: {len(citations)}")
            return citations
        except Exception as e:
            logger.warning(f"Error extracting citations: {str(e)}")
            return []

    def update_session_summary(self, summary_text: str):
        """
        Update conversation summary.

        Args:
            summary_text: New summary text
        """
        try:
            self.memory_manager.update_summary(summary_text)
            self.memory_manager.save_to_s3()
            logger.info(f"Session summary updated")
        except Exception as e:
            logger.error(f"Error updating session summary: {str(e)}", exc_info=True)

    def get_session_memory(self) -> str:
        """
        Get current session memory.

        Returns:
            Formatted memory string
        """
        try:
            return self.memory_manager.get_memory_string()
        except Exception as e:
            logger.error(f"Error getting session memory: {str(e)}", exc_info=True)
            return ""

    def clear_session(self):
        """Clear current session and create new one."""
        try:
            self.memory_manager.session_id = str(__import__("uuid").uuid4())
            self.memory_manager.conversation_buffer = []
            self.memory_manager.summary = ""
            logger.info(
                f"Session cleared, new session created: {self.memory_manager.session_id}"
            )
        except Exception as e:
            logger.error(f"Error clearing session: {str(e)}", exc_info=True)


# Statistics collector for batch operations
class GenerationStatistics:
    """Collects statistics for generation pipeline execution."""

    def __init__(self):
        """Initialize statistics collector."""
        self.total_queries = 0
        self.successful_generations = 0
        self.failed_generations = 0
        self.total_time = 0.0
        self.total_tokens_generated = 0
        self.cot_used_count = 0
        self.average_complexity = 0.0

        logger.debug("GenerationStatistics initialized")

    def record_generation(self, result: Dict):
        """Record generation result."""
        try:
            self.total_queries += 1

            if "error" not in result:
                self.successful_generations += 1
            else:
                self.failed_generations += 1

            self.total_time += result.get("execution_time_seconds", 0)
            self.total_tokens_generated += len(result.get("response", "").split())

            if result.get("used_cot"):
                self.cot_used_count += 1

            self.average_complexity = (
                self.average_complexity * (self.total_queries - 1)
                + result.get("complexity_score", 0)
            ) / self.total_queries
        except Exception as e:
            logger.warning(f"Error recording generation stats: {str(e)}")

    def get_summary(self) -> Dict:
        """Get statistics summary."""
        try:
            avg_time = (
                self.total_time / self.total_queries if self.total_queries > 0 else 0
            )
            success_rate = (
                (self.successful_generations / self.total_queries * 100)
                if self.total_queries > 0
                else 0
            )

            summary = {
                "total_queries": self.total_queries,
                "successful": self.successful_generations,
                "failed": self.failed_generations,
                "success_rate_percent": success_rate,
                "total_execution_time_seconds": self.total_time,
                "average_execution_time_seconds": avg_time,
                "total_tokens_generated": self.total_tokens_generated,
                "cot_used_count": self.cot_used_count,
                "average_complexity_score": self.average_complexity,
            }

            logger.info(f"Generation statistics summary", extra=summary)
            return summary
        except Exception as e:
            logger.error(f"Error getting statistics summary: {str(e)}", exc_info=True)
            return {}
