"""
Evaluation & Tracing Pipeline - Phase 11

This module implements the Evaluation & Tracing phase of the RAG pipeline.
Includes RAGAS metrics, LangSmith tracing, regression testing, and comprehensive reporting.

Components:
    - evaluation_components: Core evaluation classes
    - evaluation_pipeline: Main orchestrator
"""

__version__ = "1.0.0"
__author__ = "ML Engineering Team"
__all__ = [
    "RAGEvaluator",
    "RAGASMetricsEvaluator",
    "RegressionTester",
    "EvaluationReporter",
    "EvaluationPipeline",
]


