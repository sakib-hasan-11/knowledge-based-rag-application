"""
Post-Retrieval Pipeline Module (Phase 9)

Handles response generation and output:
- Token counting for budget management
- Contextual compression for relevance
- Prompt template construction
- Conversation memory management
- Chain-of-thought reasoning
- Response generation with citations
"""

import json
import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.data_ingestion.config import config
from src.data_ingestion.logging_config import create_logger

try:
    import tiktoken
except ImportError:
    tiktoken = None


class TokenCounter:
    """Estimates and counts tokens for budget management."""

    def __init__(self, model: str = "gpt-4", logger_name: str = "TokenCounter"):
        """Initialize token counter."""
        self.logger = create_logger(logger_name)
        self.model = model

        try:
            if tiktoken:
                self.encoding = tiktoken.encoding_for_model(model)
            else:
                self.encoding = None
                self.logger.warning(
                    "tiktoken not available, using approximate counting"
                )
        except Exception as e:
            self.logger.warning(
                f"Error initializing tokenizer: {str(e)}, using approximate counting"
            )
            self.encoding = None

    def count_text(self, text: str) -> int:
        """Count tokens in text."""
        try:
            if self.encoding:
                return len(self.encoding.encode(text))
            else:
                # Approximate: 1 token ≈ 4 characters
                return len(text) // 4
        except Exception as e:
            self.logger.debug(f"Error counting tokens: {str(e)}")
            return len(text) // 4

    def count_messages(self, messages: List[Dict]) -> int:
        """Count tokens in message list."""
        try:
            total = 0
            for msg in messages:
                total += 4  # Message overhead
                for key, value in msg.items():
                    total += self.count_text(str(value))
            return total
        except Exception as e:
            self.logger.debug(f"Error counting message tokens: {str(e)}")
            return 0

    def estimate_doc_tokens(self, doc: Dict) -> int:
        """Estimate tokens in document."""
        try:
            content = doc.get("metadata", {}).get("text_preview", "")
            if not content:
                content = str(doc.get("page_content", ""))
            return self.count_text(content)
        except Exception as e:
            self.logger.debug(f"Error estimating document tokens: {str(e)}")
            return 0


class ContextualCompressor:
    """Extracts query-relevant segments using LLM."""

    def __init__(
        self,
        llm_model: str = "gpt-3.5-turbo",
        compression_ratio: float = 0.5,
        logger_name: str = "ContextualCompressor",
    ):
        """Initialize contextual compressor."""
        self.logger = create_logger(logger_name)
        self.compression_ratio = compression_ratio

        try:
            self.llm = ChatOpenAI(
                model_name=llm_model,
                temperature=0.0,
                max_tokens=512,
                api_key=config.OPENAI_API_KEY,
            )

            self.compression_prompt = PromptTemplate(
                input_variables=["document", "query"],
                template="""Extract only the sentences that directly answer or relate to the query.
Keep extracted sentences coherent and in order. Remove filler and irrelevant details.

Query: {query}

Document:
{document}

Extracted Relevant Sentences:""",
            )

            self.logger.info("ContextualCompressor initialized")
        except Exception as e:
            self.logger.error(f"Error initializing ContextualCompressor: {str(e)}")
            raise

    def compress(self, query: str, document: str) -> str:
        """Extract query-relevant content."""
        try:
            if len(document) < 100:
                return document

            prompt_text = self.compression_prompt.format(query=query, document=document)
            message = HumanMessage(content=prompt_text)
            response = self.llm.invoke([message])
            compressed = response.content.strip()

            if not compressed or len(compressed) < 20:
                sentences = re.split(r"[.!?]+", document)
                ratio = max(1, int(len(sentences) * self.compression_ratio))
                compressed = ". ".join(sentences[:ratio])

            self.logger.debug(
                "Compressed document",
                {
                    "original_length": len(document),
                    "compressed_length": len(compressed),
                    "ratio": len(compressed) / len(document),
                },
            )
            return compressed
        except Exception as e:
            self.logger.error(f"Error compressing document: {str(e)}")
            return document  # Fallback to original

    def compress_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Compress multiple documents."""
        compressed = []

        for i, doc in enumerate(documents):
            try:
                content = doc.get("metadata", {}).get("text_preview", "")
                if not content:
                    content = doc.get("page_content", "")

                compressed_content = (
                    self.compress(query, content) if content else content
                )

                doc_copy = dict(doc)
                doc_copy["compressed_content"] = compressed_content
                doc_copy["original_length"] = len(content)
                doc_copy["compressed_length"] = len(compressed_content)
                doc_copy["compression_ratio"] = (
                    len(compressed_content) / len(content) if content else 0.0
                )
                compressed.append(doc_copy)
            except Exception as e:
                self.logger.warning(f"Error compressing document {i}: {str(e)}")
                compressed.append(doc)

        return compressed


class PromptTemplateBuilder:
    """Constructs prompts with anti-hallucination constraints."""

    SYSTEM_PROMPT = """You are a financial Q&A assistant powered by knowledge base retrieval.

STRICT RULES:
1. Only answer based on provided context documents
2. If context doesn't contain answer, respond: "I don't have this information in my knowledge base"
3. Always cite sources with [Doc X: Title] format
4. For multi-part questions, address each part separately
5. Keep responses concise and factual
6. Do not speculate, infer, or provide external knowledge
7. If query is out-of-scope or harmful, decline politely"""

    @staticmethod
    def build_system_prompt() -> str:
        """Get system prompt."""
        return PromptTemplateBuilder.SYSTEM_PROMPT

    @staticmethod
    def build_user_prompt(query: str, context: str, memory_summary: str = "") -> str:
        """Build complete user prompt with context."""
        prompt = f"""Conversation History Summary:
{memory_summary if memory_summary else "No previous conversation"}

Retrieved Context:
{context}

Current Question: {query}

Answer:"""
        return prompt

    @staticmethod
    def get_few_shot_examples() -> List[Dict]:
        """Get few-shot examples."""
        return [
            {
                "query": "What was Apple's revenue in 2023?",
                "answer": "Apple reported total revenue of $383.3 billion for fiscal year 2023. [Doc 1: Apple 10-K 2023]",
            }
        ]


class ConversationMemoryManager:
    """Manages conversation history with persistence."""

    def __init__(
        self,
        s3_client=None,
        bucket_name: Optional[str] = None,
        max_window: int = 5,
        logger_name: str = "ConversationMemoryManager",
    ):
        """Initialize memory manager."""
        self.logger = create_logger(logger_name)
        self.s3_client = s3_client
        self.bucket_name = bucket_name or config.S3_BUCKET_NAME
        self.max_window = max_window
        self.session_id = str(uuid.uuid4())
        self.conversation_buffer = []
        self.summary = ""

        self.logger.info(
            "ConversationMemoryManager initialized",
            {"session_id": self.session_id, "max_window": max_window},
        )

    def add_interaction(
        self, query: str, response: str, sources: Optional[List[str]] = None
    ) -> None:
        """Add query-response pair to buffer."""
        try:
            interaction = {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "response": response,
                "sources": sources or [],
            }
            self.conversation_buffer.append(interaction)

            if len(self.conversation_buffer) > self.max_window:
                self.conversation_buffer.pop(0)

            self.logger.debug(f"Added interaction to memory buffer")
        except Exception as e:
            self.logger.error(f"Error adding interaction: {str(e)}")

    def update_summary(self, summary_text: str) -> None:
        """Update conversation summary."""
        self.summary = summary_text
        self.logger.debug("Updated conversation summary")

    def get_memory_string(self) -> str:
        """Get formatted memory for prompt injection."""
        try:
            if not self.summary and not self.conversation_buffer:
                return ""

            memory_parts = []

            if self.summary:
                memory_parts.append(f"Previous Topics: {self.summary}")

            if self.conversation_buffer:
                recent = "Recent Questions:\n"
                for i, interaction in enumerate(self.conversation_buffer[-3:], 1):
                    recent += f"{i}. Q: {interaction['query'][:100]}...\n"
                memory_parts.append(recent)

            return "\n".join(memory_parts)
        except Exception as e:
            self.logger.error(f"Error formatting memory string: {str(e)}")
            return ""

    def save_to_s3(self) -> bool:
        """Save session to S3 with error handling."""
        if not self.s3_client:
            self.logger.warning("S3 client not available, skipping save")
            return False

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

            self.logger.info(f"Saved session to S3", {"key": key})
            return True
        except Exception as e:
            self.logger.error(f"Error saving to S3: {str(e)}")
            return False


class ChainOfThoughtReasoner:
    """Generates step-by-step reasoning for complex queries."""

    def __init__(self, logger_name: str = "ChainOfThoughtReasoner"):
        """Initialize reasoner."""
        self.logger = create_logger(logger_name)

    @staticmethod
    def analyze_query_complexity(query: str) -> Dict:
        """Determine query complexity."""
        complexity_indicators = {
            "multi_part": len([q for q in query.split("?") if q.strip()]) > 1,
            "comparison": any(w in query.lower() for w in ["compare", "vs", "versus"]),
            "calculation": any(
                w in query.lower() for w in ["calculate", "how much", "percent"]
            ),
            "explanation": any(
                w in query.lower() for w in ["explain", "why", "how does"]
            ),
        }

        return {
            "requires_cot": any(complexity_indicators.values()),
            "indicators": complexity_indicators,
            "complexity_score": sum(complexity_indicators.values()),
        }

    @staticmethod
    def build_cot_prompt(query: str, context: str) -> str:
        """Build chain-of-thought prompt."""
        return """Reasoning Steps:

1. Extract Key Information:
   - What are the main entities in the query?
   - What information is in the context?

2. Identify Relationships:
   - How do concepts relate?

3. Synthesize Answer:
   - Combine information logically

4. Validate Against Context:
   - Is every claim backed by context?

Based on above reasoning, provide your answer:"""
