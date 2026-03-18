"""
Data Ingestion Pipeline Module for Knowledge-Based RAG Application

This module provides production-grade data ingestion pipeline with:
- Modular phases for document loading, parsing, chunking, embedding, and Pinecone upload
- Comprehensive error handling with try-except blocks
- CloudWatch-compatible logging
- Configuration management

Author: ML Engineering Team
"""

__version__ = "1.0.0"
