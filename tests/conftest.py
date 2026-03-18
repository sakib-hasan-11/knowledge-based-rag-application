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

    Loads environment variables from:
    1. ENV_FILE environment variable (for GitHub Actions CI)
    2. .env file in project root (for local development)
    3. System environment variables (for containers/Docker)
    """
    # Check if ENV_FILE is set (GitHub Actions passes this)
    env_file = os.getenv("ENV_FILE")

    if env_file:
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path, override=True)
            print(f"\n[OK] Loaded environment variables from {env_path}")
        else:
            print(f"\n[WARN] ENV_FILE path not found: {env_path}")
    else:
        # Try to load .env file from project root (for local development)
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
            print(f"\n[OK] Loaded environment variables from {env_path}")
        else:
            print(f"\n[INFO] No .env file found at {env_path}")
            print(
                f"       Relying on system environment variables (GitHub secrets, etc.)"
            )

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
    Works in multiple environments:
    - Local development: Uses .env file
    - GitHub Actions CI: Uses GitHub secrets injected as env vars
    - Docker/containers: Uses env var injection

    Note: Stage 1 (data_ingestion) tests use mocks and don't need real credentials.
    Stage 2+ (retrieval, argumentation) need OpenAI and Pinecone keys.
    """
    # Import config after environment is set up
    from src.data_ingestion.config import config

    # Check if we have AWS credentials (they're optional for mocked tests)
    aws_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not aws_key_id or not aws_secret:
        print(
            "\n[INFO] AWS credentials not set in environment."
            "\n       - For local testing with mocks: This is OK (tests use MagicMock)"
            "\n       - For real S3 operations: Add AWS keys to .env file or set env vars"
        )
    else:
        print("\n[OK] AWS credentials detected in environment")

    # Log warnings if OpenAI/Pinecone keys are missing
    # Individual tests will use these and fail appropriately if needed
    if not config.OPENAI_API_KEY:
        print("\n[INFO] OPENAI_API_KEY not set - Stage 2+ tests may fail if not mocked")
    if not config.PINECONE_API_KEY:
        print(
            "\n[INFO] PINECONE_API_KEY not set - Stage 2+ tests may fail if not mocked"
        )

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
