"""
Argumentation & Generation Pipeline - Phase 10

This module implements the Response Generation phase of the RAG pipeline.
Includes prompt templating, conversation memory management, chain-of-thought reasoning,
and end-to-end response generation orchestration.

Components:
    - generation_components: Core generation classes
    - generation_pipeline: Main orchestrator
"""

__version__ = "1.0.0"
__author__ = "ML Engineering Team"
__all__ = [
    "PromptTemplateBuilder",
    "ConversationMemoryManager",
    "ChainOfThoughtReasoner",
    "ArgumentationPipeline",
]
