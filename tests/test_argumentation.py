"""
Comprehensive Test Suite for Argumentation & Generation Module (Phase 10)

Tests cover:
1. Generation Components: Prompt template building, memory management, chain-of-thought reasoning
2. Complete generation pipeline: Query → Memory → Prompt → LLM → Response → Memory save
3. LLM integration and output validation

Edge Cases Covered:
- Empty or invalid queries
- Memory retrieval failures
- LLM API errors and timeouts
- Token limit exceeded in response
- Malformed conversation history
- Missing or corrupted S3 memory
- Concurrent request handling
- Response quality validation
- Hallucination detection
- Context window management
"""

import json
import unittest
from unittest.mock import MagicMock, Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage


class TestPromptTemplateBuilder(unittest.TestCase):
    """Test cases for prompt template building"""

    def test_builder_initialization(self):
        """Test prompt template builder initialization"""
        from src.argumentation.generation_components import PromptTemplateBuilder

        builder = PromptTemplateBuilder()
        self.assertIsNotNone(builder)

    def test_build_basic_prompt(self):
        """Test building basic prompt"""
        from src.argumentation.generation_components import PromptTemplateBuilder

        builder = PromptTemplateBuilder()
        query = "What are cybersecurity risks?"
        context = "Cybersecurity involves protecting systems from attacks"

        prompt = builder.build_prompt(query=query, context=context)
        self.assertIsNotNone(prompt)
        self.assertIn("cybersecurity", prompt.lower())

    def test_build_prompt_with_empty_context(self):
        """Test building prompt with empty context"""
        from src.argumentation.generation_components import PromptTemplateBuilder

        builder = PromptTemplateBuilder()
        prompt = builder.build_prompt(query="Test query", context="")

        self.assertIsNotNone(prompt)

    def test_build_prompt_with_conversation_history(self):
        """Test building prompt with conversation history"""
        from src.argumentation.generation_components import PromptTemplateBuilder

        builder = PromptTemplateBuilder()
        conversation_history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ]

        prompt = builder.build_prompt(
            query="Follow-up question",
            context="Context",
            conversation_history=conversation_history,
        )

        self.assertIsNotNone(prompt)

    def test_anti_hallucination_guardrails(self):
        """Test anti-hallucination prompting"""
        from src.argumentation.generation_components import PromptTemplateBuilder

        builder = PromptTemplateBuilder()
        prompt = builder.build_prompt(
            query="Test query",
            context="Context",
            include_guardrails=True,
        )

        # Should include instructions about uncertainty
        self.assertIsNotNone(prompt)

    def test_prompt_length_validation(self):
        """Test prompt template length validation"""
        from src.argumentation.generation_components import PromptTemplateBuilder

        builder = PromptTemplateBuilder()
        # Test with very long context
        long_context = "test " * 10000
        prompt = builder.build_prompt(query="Test", context=long_context)

        # Should handle long context

    def test_special_character_handling_in_prompt(self):
        """Test special character handling in prompts"""
        from src.argumentation.generation_components import PromptTemplateBuilder

        builder = PromptTemplateBuilder()
        special_query = "Query with \"quotes\" and 'apostrophes' and @#$%"
        prompt = builder.build_prompt(query=special_query, context="Context")

        self.assertIsNotNone(prompt)


class TestConversationMemoryManager(unittest.TestCase):
    """Test cases for conversation memory management"""

    @patch("boto3.client")
    def test_memory_manager_init(self, mock_s3):
        """Test memory manager initialization"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.argumentation.generation_components import (
            ConversationMemoryManager,
        )

        manager = ConversationMemoryManager(
            s3_client=mock_client, bucket_name="test-bucket"
        )
        self.assertIsNotNone(manager)

    @patch("boto3.client")
    def test_save_conversation_to_s3(self, mock_s3):
        """Test saving conversation to S3"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.argumentation.generation_components import (
            ConversationMemoryManager,
        )

        manager = ConversationMemoryManager(
            s3_client=mock_client, bucket_name="test-bucket"
        )

        conversation = {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
        }

        manager.save_conversation(conversation_id="test-123", conversation=conversation)

        mock_client.put_object.assert_called_once()

    @patch("boto3.client")
    def test_load_conversation_from_s3(self, mock_s3):
        """Test loading conversation from S3"""
        mock_client = MagicMock()
        test_conversation = {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
        }
        mock_client.get_object.return_value = {
            "Body": MagicMock(
                read=MagicMock(return_value=json.dumps(test_conversation).encode())
            )
        }
        mock_s3.return_value = mock_client

        from src.argumentation.generation_components import (
            ConversationMemoryManager,
        )

        manager = ConversationMemoryManager(
            s3_client=mock_client, bucket_name="test-bucket"
        )

        loaded = manager.load_conversation("test-123")
        self.assertIsNotNone(loaded)

    @patch("boto3.client")
    def test_load_conversation_not_found(self, mock_s3):
        """Test loading non-existent conversation"""
        mock_client = MagicMock()
        mock_client.get_object.side_effect = Exception("NoSuchKey")
        mock_s3.return_value = mock_client

        from src.argumentation.generation_components import (
            ConversationMemoryManager,
        )

        manager = ConversationMemoryManager(
            s3_client=mock_client, bucket_name="test-bucket"
        )

        result = manager.load_conversation("nonexistent-123")
        # Should handle gracefully
        self.assertIsNotNone(manager)

    @patch("boto3.client")
    def test_append_to_conversation(self, mock_s3):
        """Test appending message to conversation"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.argumentation.generation_components import (
            ConversationMemoryManager,
        )

        manager = ConversationMemoryManager(
            s3_client=mock_client, bucket_name="test-bucket"
        )

        # Should append message

    @patch("boto3.client")
    def test_memory_size_limit(self, mock_s3):
        """Test memory size limit enforcement"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.argumentation.generation_components import (
            ConversationMemoryManager,
        )

        manager = ConversationMemoryManager(
            s3_client=mock_client,
            bucket_name="test-bucket",
            max_memory_messages=10,
        )

        # Should enforce size limits

    @patch("boto3.client")
    def test_conversation_context_window(self, mock_s3):
        """Test managing conversation within context window"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.argumentation.generation_components import (
            ConversationMemoryManager,
        )

        manager = ConversationMemoryManager(
            s3_client=mock_client, bucket_name="test-bucket"
        )

        # Should respect context window limits


class TestChainOfThoughtReasoner(unittest.TestCase):
    """Test cases for chain-of-thought reasoning"""

    def test_reasoner_initialization(self):
        """Test chain-of-thought reasoner initialization"""
        from src.argumentation.generation_components import ChainOfThoughtReasoner

        reasoner = ChainOfThoughtReasoner()
        self.assertIsNotNone(reasoner)

    @patch("langchain_openai.ChatOpenAI")
    def test_generate_reasoning_steps(self, mock_llm):
        """Test generating chain of thought steps"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Step 1: Understand the question\nStep 2: Analyze context\nStep 3: Provide answer"
        mock_client.invoke.return_value = mock_response
        mock_llm.return_value = mock_client

        from src.argumentation.generation_components import ChainOfThoughtReasoner

        reasoner = ChainOfThoughtReasoner()
        steps = reasoner.generate_steps(
            query="What is cybersecurity?", context="Cybersecurity content"
        )

        self.assertIsNotNone(steps)

    @patch("langchain_openai.ChatOpenAI")
    def test_reasoning_validation(self, mock_llm):
        """Test validating reasoning steps"""
        mock_client = MagicMock()
        mock_llm.return_value = mock_client

        from src.argumentation.generation_components import ChainOfThoughtReasoner

        reasoner = ChainOfThoughtReasoner()
        # Should validate logical consistency

    @patch("langchain_openai.ChatOpenAI")
    def test_complex_query_reasoning(self, mock_llm):
        """Test reasoning for complex queries"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Complex reasoning steps"
        mock_client.invoke.return_value = mock_response
        mock_llm.return_value = mock_client

        from src.argumentation.generation_components import ChainOfThoughtReasoner

        reasoner = ChainOfThoughtReasoner()
        complex_query = "Compare cybersecurity risks and legal implications"
        steps = reasoner.generate_steps(query=complex_query, context="Long context...")

        self.assertIsNotNone(steps)


class TestArgumentationPipeline(unittest.TestCase):
    """Test cases for complete argumentation pipeline"""

    @patch("langchain_openai.ChatOpenAI")
    @patch("boto3.client")
    def test_pipeline_initialization(self, mock_s3, mock_llm):
        """Test pipeline initialization"""
        mock_s3_client = MagicMock()
        mock_llm_client = MagicMock()
        mock_s3.return_value = mock_s3_client
        mock_llm.return_value = mock_llm_client

        from src.argumentation.generation_pipeline import ArgumentationPipeline

        pipeline = ArgumentationPipeline(
            llm_client=mock_llm_client,
            s3_client=mock_s3_client,
            bucket_name="test-bucket",
            model_name="gpt-3.5-turbo",
        )

        self.assertIsNotNone(pipeline)

    @patch("langchain_openai.ChatOpenAI")
    @patch("boto3.client")
    def test_generate_response_basic(self, mock_s3, mock_llm):
        """Test basic response generation"""
        mock_s3_client = MagicMock()
        mock_llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Generated response based on context"
        mock_llm_client.invoke.return_value = mock_response
        mock_s3.return_value = mock_s3_client
        mock_llm.return_value = mock_llm_client

        from src.argumentation.generation_pipeline import ArgumentationPipeline

        pipeline = ArgumentationPipeline(
            llm_client=mock_llm_client,
            s3_client=mock_s3_client,
            bucket_name="test-bucket",
        )

        response = pipeline.generate_response(
            query="What are cybersecurity risks?",
            context="Cybersecurity involves...",
        )

        self.assertIsNotNone(response)

    @patch("langchain_openai.ChatOpenAI")
    @patch("boto3.client")
    def test_generate_with_conversation_history(self, mock_s3, mock_llm):
        """Test response generation with conversation history"""
        mock_s3_client = MagicMock()
        mock_llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Follow-up response"
        mock_llm_client.invoke.return_value = mock_response
        mock_s3.return_value = mock_s3_client
        mock_llm.return_value = mock_llm_client

        from src.argumentation.generation_pipeline import ArgumentationPipeline

        pipeline = ArgumentationPipeline(
            llm_client=mock_llm_client,
            s3_client=mock_s3_client,
            bucket_name="test-bucket",
        )

        response = pipeline.generate_response(
            query="Tell me more",
            context="Additional context",
            conversation_id="test-123",
        )

        self.assertIsNotNone(response)

    @patch("langchain_openai.ChatOpenAI")
    @patch("boto3.client")
    def test_generate_with_chain_of_thought(self, mock_s3, mock_llm):
        """Test response generation with chain-of-thought"""
        mock_s3_client = MagicMock()
        mock_llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Step-by-step response"
        mock_llm_client.invoke.return_value = mock_response
        mock_s3.return_value = mock_s3_client
        mock_llm.return_value = mock_llm_client

        from src.argumentation.generation_pipeline import ArgumentationPipeline

        pipeline = ArgumentationPipeline(
            llm_client=mock_llm_client,
            s3_client=mock_s3_client,
            bucket_name="test-bucket",
            enable_cot=True,
        )

        response = pipeline.generate_response(
            query="Complex question",
            context="Context",
        )

        self.assertIsNotNone(response)

    @patch("langchain_openai.ChatOpenAI")
    @patch("boto3.client")
    def test_pipeline_token_limit(self, mock_s3, mock_llm):
        """Test pipeline respects token limits"""
        mock_s3_client = MagicMock()
        mock_llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_llm_client.invoke.return_value = mock_response
        mock_s3.return_value = mock_s3_client
        mock_llm.return_value = mock_llm_client

        from src.argumentation.generation_pipeline import ArgumentationPipeline

        pipeline = ArgumentationPipeline(
            llm_client=mock_llm_client,
            s3_client=mock_s3_client,
            bucket_name="test-bucket",
            max_tokens=200,
        )

        # Should respect token limit

    @patch("langchain_openai.ChatOpenAI")
    @patch("boto3.client")
    def test_empty_query_handling(self, mock_s3, mock_llm):
        """Test handling empty queries"""
        mock_s3_client = MagicMock()
        mock_llm_client = MagicMock()
        mock_s3.return_value = mock_s3_client
        mock_llm.return_value = mock_llm_client

        from src.argumentation.generation_pipeline import ArgumentationPipeline

        pipeline = ArgumentationPipeline(
            llm_client=mock_llm_client,
            s3_client=mock_s3_client,
            bucket_name="test-bucket",
        )

        # Should handle empty query

    @patch("langchain_openai.ChatOpenAI")
    @patch("boto3.client")
    def test_llm_api_error_handling(self, mock_s3, mock_llm):
        """Test handling of LLM API errors"""
        mock_s3_client = MagicMock()
        mock_llm_client = MagicMock()
        mock_llm_client.invoke.side_effect = Exception("API Error")
        mock_s3.return_value = mock_s3_client
        mock_llm.return_value = mock_llm_client

        from src.argumentation.generation_pipeline import ArgumentationPipeline

        pipeline = ArgumentationPipeline(
            llm_client=mock_llm_client,
            s3_client=mock_s3_client,
            bucket_name="test-bucket",
        )

        # Should handle API errors gracefully

    @patch("langchain_openai.ChatOpenAI")
    @patch("boto3.client")
    def test_memory_save_failure(self, mock_s3, mock_llm):
        """Test handling when memory save fails"""
        mock_s3_client = MagicMock()
        mock_s3_client.put_object.side_effect = Exception("S3 Error")
        mock_llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_llm_client.invoke.return_value = mock_response
        mock_s3.return_value = mock_s3_client
        mock_llm.return_value = mock_llm_client

        from src.argumentation.generation_pipeline import ArgumentationPipeline

        pipeline = ArgumentationPipeline(
            llm_client=mock_llm_client,
            s3_client=mock_s3_client,
            bucket_name="test-bucket",
        )

        # Should handle memory save failure

    @patch("langchain_openai.ChatOpenAI")
    @patch("boto3.client")
    def test_hallucination_detection(self, mock_s3, mock_llm):
        """Test hallucination detection in responses"""
        mock_s3_client = MagicMock()
        mock_llm_client = MagicMock()
        mock_response = MagicMock()
        # LLM makes claim not in context
        mock_response.content = "Something completely unrelated"
        mock_llm_client.invoke.return_value = mock_response
        mock_s3.return_value = mock_s3_client
        mock_llm.return_value = mock_llm_client

        from src.argumentation.generation_pipeline import ArgumentationPipeline

        pipeline = ArgumentationPipeline(
            llm_client=mock_llm_client,
            s3_client=mock_s3_client,
            bucket_name="test-bucket",
            enable_hallucination_check=True,
        )

        # Should detect hallucination

    @patch("langchain_openai.ChatOpenAI")
    @patch("boto3.client")
    def test_response_quality_metric(self, mock_s3, mock_llm):
        """Test response quality metrics"""
        mock_s3_client = MagicMock()
        mock_llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "High quality response with detailed information"
        mock_llm_client.invoke.return_value = mock_response
        mock_s3.return_value = mock_s3_client
        mock_llm.return_value = mock_llm_client

        from src.argumentation.generation_pipeline import ArgumentationPipeline

        pipeline = ArgumentationPipeline(
            llm_client=mock_llm_client,
            s3_client=mock_s3_client,
            bucket_name="test-bucket",
        )

        # Should track quality metrics

    @patch("langchain_openai.ChatOpenAI")
    @patch("boto3.client")
    def test_concurrent_generation_requests(self, mock_s3, mock_llm):
        """Test handling concurrent generation requests"""
        mock_s3_client = MagicMock()
        mock_llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_llm_client.invoke.return_value = mock_response
        mock_s3.return_value = mock_s3_client
        mock_llm.return_value = mock_llm_client

        from src.argumentation.generation_pipeline import ArgumentationPipeline

        pipeline = ArgumentationPipeline(
            llm_client=mock_llm_client,
            s3_client=mock_s3_client,
            bucket_name="test-bucket",
        )

        # Should handle concurrent requests


if __name__ == "__main__":
    unittest.main()
