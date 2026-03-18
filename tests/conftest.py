"""
pytest configuration and fixtures for RAG application tests.

This module:
- Loads environment variables from .env before any tests run
- Provides common fixtures for test modules
- Initializes logging configuration
"""

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Ensure project root is in Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """
    pytest hook that runs before test collection.

    Loads .env file to ensure environment variables are available
    to all tests.
    """
    # Load .env file from project root
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"\n[OK] Loaded environment variables from {env_path}")
    else:
        print(f"\n[WARN] .env file not found at {env_path}")

    # Verify critical environment variables are loaded
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"\n[WARN] Missing environment variables: {', '.join(missing_vars)}")
    else:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        pinecone_key = os.getenv("PINECONE_API_KEY", "")
        print(f"[OK] All required environment variables are set:")
        print(
            f"  - OPENAI_API_KEY: {'*' * 10}...{openai_key[-4:] if len(openai_key) > 4 else ''}"
        )
        print(
            f"  - PINECONE_API_KEY: {'*' * 10}...{pinecone_key[-4:] if len(pinecone_key) > 4 else ''}"
        )


@pytest.fixture(scope="session", autouse=True)
def setup_env():
    """
    Session-scoped fixture that runs once before all tests.

    Ensures environment variables are properly loaded.
    Note: Stage 1 (data_ingestion) only needs AWS credentials.
    Stage 2+ (retrieval, argumentation) need OpenAI and Pinecone keys.
    """
    # Verify AWS credentials are loaded (needed for all stages)
    assert os.getenv("AWS_ACCESS_KEY_ID"), "AWS_ACCESS_KEY_ID not set in environment"
    assert os.getenv("AWS_SECRET_ACCESS_KEY"), (
        "AWS_SECRET_ACCESS_KEY not set in environment"
    )

    # Import config after environment is set up
    from src.data_ingestion.config import config

    # Just log warnings if OpenAI/Pinecone keys are missing - don't fail yet
    # Individual tests will use these and fail appropriately if needed
    if not config.OPENAI_API_KEY:
        print("\n[WARN] OPENAI_API_KEY not set - Stage 2+ tests may fail")
    if not config.PINECONE_API_KEY:
        print("\n[WARN] PINECONE_API_KEY not set - Stage 2+ tests may fail")

    print(f"\n[OK] Configuration initialized successfully")

    yield

    # Cleanup after all tests (if needed)


@pytest.fixture
def mock_openai_config(monkeypatch):
    """
    Fixture to ensure OpenAI API key is available for each test.

    This fixture ensures that the OpenAI API key is properly set
    for tests that require it, even if mocking is used.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        pytest.skip("OPENAI_API_KEY not found in environment")

    monkeypatch.setenv("OPENAI_API_KEY", openai_key)
    return openai_key


@pytest.fixture
def mock_pinecone_config(monkeypatch):
    """
    Fixture to ensure Pinecone API key is available for each test.

    This fixture ensures that the Pinecone API key is properly set
    for tests that require it, even if mocking is used.
    """
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        pytest.skip("PINECONE_API_KEY not found in environment")

    monkeypatch.setenv("PINECONE_API_KEY", pinecone_key)
    return pinecone_key
