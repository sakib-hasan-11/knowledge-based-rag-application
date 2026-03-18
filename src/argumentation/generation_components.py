"""
Generation Components for Argumentation & Response Pipeline (Phase 10)

Implements core components for response generation including prompt building,
conversation memory management, and chain-of-thought reasoning.
"""

import json
import os
import re

# Add parent directory to path for imports
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_ingestion.config import Config
from data_ingestion.logging_config import PipelineLogger

logger = PipelineLogger.create_logger(__name__)


@dataclass
class InteractionRecord:
    """Represents a single conversation interaction."""

    timestamp: str
    query: str
    response: str
    sources: List[str]
    reasoning: Optional[str] = None


class PromptTemplateBuilder:
    """
    Constructs system and user prompts with anti-hallucination constraints.

    Provides system prompts, user prompts with context injection, and
    few-shot examples for in-context learning.
    """

    SYSTEM_PROMPT = """You are a financial Q&A assistant powered by knowledge base retrieval.

STRICT RULES:
1. Only answer based on provided context documents
2. If context doesn't contain answer, respond: "I don't have this information in my knowledge base"
3. Always cite sources with [Doc X: Title] format
4. For multi-part questions, address each part separately
5. Keep responses concise and factual
6. Do not speculate, infer, or provide external knowledge
7. If query is out-of-scope or harmful, decline politely"""

    FEW_SHOT_EXAMPLES = [
        {
            "query": "What was Apple's revenue in 2023?",
            "reasoning": "1. Search for Apple financial data 2023\n2. Locate revenue line item\n3. Extract exact figure with source",
            "answer": "Apple reported total revenue of $383.3 billion for fiscal year 2023, ending September 30, 2023. [Doc 1: Apple 10-K 2023]",
        },
        {
            "query": "How does Tesla compare to Ford in market cap?",
            "reasoning": "1. Find Tesla market cap\n2. Find Ford market cap\n3. Calculate comparison ratio\n4. Note data timeliness",
            "answer": "Based on latest available data: [Doc 5: Market Comparisons Q4 2023]. For current market cap, please check real-time financial sources.",
        },
        {
            "query": "What are the main revenue drivers?",
            "reasoning": "1. Identify revenue categories\n2. Extract contribution percentages\n3. Rank by importance\n4. Cite each source",
            "answer": "Main revenue drivers: 1) iPhone sales (50%), 2) Services (25%), 3) Other hardware (15%), 4) Wearables (10%). [Doc 1: Apple 10-K 2023]",
        },
    ]

    def __init__(self, custom_system_prompt: Optional[str] = None):
        """
        Initialize prompt builder.

        Args:
            custom_system_prompt: Override default system prompt
        """
        self.system_prompt = custom_system_prompt or self.SYSTEM_PROMPT
        logger.info(
            "PromptTemplateBuilder initialized",
            extra={"custom_prompt": custom_system_prompt is not None},
        )

    def build_system_prompt(self) -> str:
        """Get system prompt."""
        return self.system_prompt

    def build_user_prompt(
        self, query: str, context: str, memory_summary: str = ""
    ) -> str:
        """
        Build complete user prompt with context and memory.

        Args:
            query: User query
            context: Retrieved context documents
            memory_summary: Optional conversation history summary

        Returns:
            Formatted user prompt
        """
        try:
            memory_section = (
                memory_summary if memory_summary else "No previous conversation"
            )

            prompt = f"""Conversation History Summary:
{memory_section}

Retrieved Context:
{context}

Current Question: {query}

Step-by-step reasoning:
1. Review context documents
2. Identify relevant information
3. Check if all parts are covered in context
4. Formulate answer with citations

Answer:"""

            logger.debug(
                f"User prompt built",
                extra={"query_length": len(query), "context_length": len(context)},
            )
            return prompt
        except Exception as e:
            logger.error(f"Error building user prompt: {str(e)}", exc_info=True)
            return f"Current Question: {query}\n\nAnswer:"

    @staticmethod
    def get_few_shot_examples() -> List[Dict]:
        """Get few-shot examples for in-context learning."""
        return PromptTemplateBuilder.FEW_SHOT_EXAMPLES

    def add_few_shot_to_prompt(self, user_prompt: str, indicators: Dict) -> str:
        """
        Add relevant few-shot examples based on query indicators.

        Args:
            user_prompt: Base user prompt
            indicators: Query complexity indicators

        Returns:
            Prompt with few-shot examples appended
        """
        try:
            examples = self.get_few_shot_examples()

            if indicators.get("comparison") and len(examples) > 1:
                example = examples[1]
                few_shot = f"\nExample for comparison queries:\nQ: {example['query']}\nA: {example['answer']}"
            elif indicators.get("calculation") and len(examples) > 2:
                example = examples[2]
                few_shot = f"\nExample for calculation queries:\nQ: {example['query']}\nA: {example['answer']}"
            elif len(examples) > 0:
                example = examples[0]
                few_shot = (
                    f"\nExample format:\nQ: {example['query']}\nA: {example['answer']}"
                )
            else:
                few_shot = ""

            return user_prompt + few_shot
        except Exception as e:
            logger.warning(f"Error adding few-shot examples: {str(e)}")
            return user_prompt


class ConversationMemoryManager:
    """
    Manages conversation history with S3 persistence.

    Maintains conversation buffer, session management, and persistence
    to S3 for multi-session retrieval.
    """

    def __init__(self, s3_client, bucket_name: str, max_window: int = 5):
        """
        Initialize memory manager.

        Args:
            s3_client: boto3 S3 client
            bucket_name: S3 bucket for chat history
            max_window: Number of recent interactions to keep in memory
        """
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.max_window = max_window
        self.session_id = str(uuid.uuid4())
        self.conversation_buffer: List[Dict] = []
        self.summary = ""

        logger.info(
            f"ConversationMemoryManager initialized",
            extra={
                "session_id": self.session_id,
                "max_window": max_window,
                "bucket_name": bucket_name,
            },
        )

    def add_interaction(
        self,
        query: str,
        response: str,
        sources: List[str] = None,
        reasoning: str = None,
    ):
        """
        Add query-response pair to buffer.

        Args:
            query: User query
            response: Generated response
            sources: List of source documents used
            reasoning: Optional reasoning steps
        """
        try:
            interaction = {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "response": response,
                "sources": sources or [],
                "reasoning": reasoning,
            }
            self.conversation_buffer.append(interaction)

            if len(self.conversation_buffer) > self.max_window:
                removed = self.conversation_buffer.pop(0)
                logger.debug(f"Removed oldest interaction from buffer")

            logger.debug(
                f"Interaction added to buffer",
                extra={"buffer_size": len(self.conversation_buffer)},
            )
        except Exception as e:
            logger.error(f"Error adding interaction: {str(e)}", exc_info=True)

    def update_summary(self, summary_text: str):
        """Update conversation summary."""
        try:
            self.summary = summary_text
            logger.debug(
                f"Conversation summary updated",
                extra={"summary_length": len(summary_text)},
            )
        except Exception as e:
            logger.error(f"Error updating summary: {str(e)}", exc_info=True)

    def get_memory_string(self) -> str:
        """
        Get formatted memory for prompt injection.

        Returns:
            Formatted memory string or empty string if no history
        """
        try:
            if not self.summary and not self.conversation_buffer:
                return ""

            memory_parts = []

            if self.summary:
                memory_parts.append(f"Previous Topics: {self.summary}")

            if self.conversation_buffer:
                recent = "Recent Questions:\n"
                for i, interaction in enumerate(self.conversation_buffer[-3:], 1):
                    query_preview = interaction["query"][:100] + (
                        "..." if len(interaction["query"]) > 100 else ""
                    )
                    recent += f"{i}. Q: {query_preview}\n"
                memory_parts.append(recent)

            return "\n".join(memory_parts)
        except Exception as e:
            logger.warning(f"Error getting memory string: {str(e)}")
            return ""

    def save_to_s3(self) -> bool:
        """
        Save full session to S3.

        Returns:
            True if successful, False otherwise
        """
        try:
            session_data = {
                "session_id": self.session_id,
                "created_at": datetime.utcnow().isoformat(),
                "summary": self.summary,
                "interactions": self.conversation_buffer,
            }

            key = f"chat-history/{self.session_id}/session.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(session_data, indent=2),
                ContentType="application/json",
            )

            logger.info(
                f"Session saved to S3",
                extra={
                    "session_id": self.session_id,
                    "s3_key": key,
                    "interactions_count": len(self.conversation_buffer),
                },
            )
            return True
        except Exception as e:
            logger.error(f"Error saving session to S3: {str(e)}", exc_info=True)
            return False

    def load_session_from_s3(self, session_id: str) -> bool:
        """
        Load previous session from S3.

        Args:
            session_id: UUID of session to load

        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"chat-history/{session_id}/session.json"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            session_data = json.loads(response["Body"].read().decode("utf-8"))

            self.session_id = session_data["session_id"]
            self.summary = session_data.get("summary", "")
            self.conversation_buffer = session_data.get("interactions", [])[
                -self.max_window :
            ]

            logger.info(
                f"Session loaded from S3",
                extra={
                    "session_id": session_id,
                    "interactions_loaded": len(self.conversation_buffer),
                },
            )
            return True
        except Exception as e:
            logger.warning(f"Error loading session from S3: {str(e)}")
            return False


class ChainOfThoughtReasoner:
    """
    Generates step-by-step reasoning for complex queries.

    Analyzes query complexity and generates structured reasoning
    prompts for LLM step-by-step generation.
    """

    # Complexity indicators keywords
    MULTI_PART_KEYWORDS = ["?", "and", ","]
    COMPARISON_KEYWORDS = ["compare", "vs", "versus", "difference", "similar"]
    CALCULATION_KEYWORDS = ["calculate", "how much", "percent", "ratio", "total"]
    EXPLANATION_KEYWORDS = ["explain", "why", "how does", "mechanism", "process"]
    HISTORICAL_KEYWORDS = ["trend", "change", "history", "evolution", "development"]

    @staticmethod
    def analyze_query_complexity(query: str) -> Dict:
        """
        Determine if query requires chain-of-thought reasoning.

        Args:
            query: User query

        Returns:
            Dict with complexity indicators and score
        """
        try:
            query_lower = query.lower()

            complexity_indicators = {
                "multi_part": len([q for q in query.split("?") if q.strip()]) > 1,
                "comparison": any(
                    word in query_lower
                    for word in ChainOfThoughtReasoner.COMPARISON_KEYWORDS
                ),
                "calculation": any(
                    word in query_lower
                    for word in ChainOfThoughtReasoner.CALCULATION_KEYWORDS
                ),
                "explanation": any(
                    word in query_lower
                    for word in ChainOfThoughtReasoner.EXPLANATION_KEYWORDS
                ),
                "historical": any(
                    word in query_lower
                    for word in ChainOfThoughtReasoner.HISTORICAL_KEYWORDS
                ),
            }

            requires_cot = any(complexity_indicators.values())
            complexity_score = sum(complexity_indicators.values())

            logger.debug(
                f"Query complexity analyzed",
                extra={
                    "requires_cot": requires_cot,
                    "complexity_score": complexity_score,
                    "indicators": complexity_indicators,
                },
            )

            return {
                "requires_cot": requires_cot,
                "indicators": complexity_indicators,
                "complexity_score": complexity_score,
            }
        except Exception as e:
            logger.error(f"Error analyzing query complexity: {str(e)}", exc_info=True)
            return {"requires_cot": False, "indicators": {}, "complexity_score": 0}

    @staticmethod
    def build_cot_prompt(query: str, context: str) -> str:
        """
        Build chain-of-thought prompt for complex reasoning.

        Args:
            query: User query
            context: Retrieved documents

        Returns:
            Chain-of-thought reasoning template
        """
        try:
            cot_template = """Reasoning Steps:

1. Extract Key Information:
   - What are the main entities/concepts in the query?
   - What information is explicitly provided in context?

2. Identify Relationships:
   - How do the concepts relate to each other?
   - What connections exist in the provided context?

3. Synthesize Answer:
   - Combine information from multiple parts if needed
   - Build logical chain from retrieved facts

4. Validate Against Context:
   - Is every claim backed by context?
   - Are all sources cited?

Based on above reasoning, provide your answer:"""

            logger.debug(f"Chain-of-thought prompt created")
            return cot_template
        except Exception as e:
            logger.error(f"Error building CoT prompt: {str(e)}", exc_info=True)
            return "Provide a detailed, step-by-step answer:"

    @staticmethod
    def extract_reasoning_steps(llm_response: str) -> Dict:
        """
        Parse reasoning steps from LLM response.

        Args:
            llm_response: LLM generated response

        Returns:
            Dict with extracted reasoning steps and count
        """
        try:
            steps = []
            lines = llm_response.split("\n")

            for line in lines:
                stripped = line.strip()
                if stripped and stripped[0] in ["-", "•", "*", "1", "2", "3", "4", "5"]:
                    steps.append(stripped)

            logger.debug(f"Reasoning steps extracted", extra={"step_count": len(steps)})

            return {"reasoning_steps": steps, "step_count": len(steps)}
        except Exception as e:
            logger.warning(f"Error extracting reasoning steps: {str(e)}")
            return {"reasoning_steps": [], "step_count": 0}
