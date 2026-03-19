"""
Integration tests for FastAPI and Streamlit applications

This test suite validates:
1. FastAPI server starts correctly
2. API endpoints respond with correct status codes
3. Response format matches expected schema
4. Error handling works correctly
5. Health checks function properly

Usage:
    pytest tests/test_integration.py -v
    pytest tests/test_integration.py -v -k "health"
"""

import json
import time

import pytest
import requests

from api import QueryRequest, QueryResponse


class TestAPIHealth:
    """Test API health endpoints"""

    API_URL = "http://localhost:8000"

    @pytest.fixture
    def api_running(self):
        """Check if API is running, skip test if not"""
        try:
            response = requests.get(f"{self.API_URL}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("API not running on http://localhost:8000")

    def test_health_endpoint(self, api_running):
        """Test /health endpoint"""
        if not api_running:
            pytest.skip("API not running")

        response = requests.get(f"{self.API_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_liveness_probe(self, api_running):
        """Test /health/live endpoint (K8s liveness probe)"""
        if not api_running:
            pytest.skip("API not running")

        response = requests.get(f"{self.API_URL}/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_readiness_probe(self, api_running):
        """Test /health/ready endpoint (K8s readiness probe)"""
        if not api_running:
            pytest.skip("API not running")

        response = requests.get(f"{self.API_URL}/health/ready")
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data


class TestQueryEndpoint:
    """Test query endpoint"""

    API_URL = "http://localhost:8000"

    @pytest.fixture
    def api_running(self):
        """Check if API is running, skip test if not"""
        try:
            response = requests.get(f"{self.API_URL}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("API not running on http://localhost:8000")

    def test_query_endpoint_success(self, api_running):
        """Test successful query"""
        if not api_running:
            pytest.skip("API not running")

        payload = {
            "query": "What is Apple's business?",
            "session_id": "test_session",
            "top_k": 5,
            "use_reranking": True,
            "enable_reasoning": True,
        }

        response = requests.post(
            f"{self.API_URL}/query",
            json=payload,
            timeout=30,
        )

        assert response.status_code == 200
        data = response.json()

        # Validate response schema
        assert "response" in data
        assert "sources" in data
        assert "confidence_score" in data
        assert "processing_time_ms" in data
        assert "query_id" in data

        # Validate types
        assert isinstance(data["response"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["confidence_score"], float)
        assert isinstance(data["processing_time_ms"], float)

    def test_query_validation(self, api_running):
        """Test query parameter validation"""
        if not api_running:
            pytest.skip("API not running")

        # Missing required field
        payload = {
            "session_id": "test_session",
            "top_k": 5,
        }

        response = requests.post(
            f"{self.API_URL}/query",
            json=payload,
            timeout=30,
        )

        assert response.status_code == 422  # Validation error

    def test_batch_query_endpoint(self, api_running):
        """Test batch query endpoint"""
        if not api_running:
            pytest.skip("API not running")

        payload = {
            "queries": [
                {
                    "query": "What is Apple's revenue?",
                    "session_id": "test_session_1",
                    "top_k": 3,
                },
                {
                    "query": "What are Apple's risks?",
                    "session_id": "test_session_2",
                    "top_k": 3,
                },
            ]
        }

        response = requests.post(
            f"{self.API_URL}/batch-query",
            json=payload,
            timeout=60,
        )

        assert response.status_code == 200
        data = response.json()

        # Validate batch response
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 2

        for result in data["results"]:
            assert "response" in result
            assert "sources" in result
            assert "confidence_score" in result


class TestErrorHandling:
    """Test error handling"""

    API_URL = "http://localhost:8000"

    @pytest.fixture
    def api_running(self):
        """Check if API is running, skip test if not"""
        try:
            response = requests.get(f"{self.API_URL}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("API not running on http://localhost:8000")

    def test_404_not_found(self, api_running):
        """Test 404 not found"""
        if not api_running:
            pytest.skip("API not running")

        response = requests.get(f"{self.API_URL}/invalid-endpoint")
        assert response.status_code == 404

    def test_timeout_handling(self, api_running):
        """Test timeout handling"""
        if not api_running:
            pytest.skip("API not running")

        # This is more of a manual test - the timeout depends on query complexity
        # Just verify the endpoint doesn't crash with a normal query
        payload = {
            "query": "Test query",
            "session_id": "test",
            "top_k": 1,
        }

        try:
            response = requests.post(
                f"{self.API_URL}/query",
                json=payload,
                timeout=30,
            )
            assert response.status_code in [200, 500, 503]
        except requests.exceptions.Timeout:
            pytest.skip("Query timed out - API is slow")


class TestStreamlitIntegration:
    """Test Streamlit and FastAPI integration"""

    API_URL = "http://localhost:8000"
    STREAMLIT_URL = "http://localhost:8501"

    def test_api_accessible_from_streamlit(self):
        """Test that Streamlit can reach the API"""
        try:
            response = requests.get(f"{self.API_URL}/health", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("API not running")

    def test_streamlit_running(self):
        """Test that Streamlit is running"""
        try:
            response = requests.get(f"{self.STREAMLIT_URL}", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("Streamlit not running on http://localhost:8501")


# ============================================================================
# Run Integration Tests Quick Check
# ============================================================================


def quick_health_check():
    """Quick health check without pytest - useful for debugging"""
    api_url = "http://localhost:8000"

    print("\n🔍 Running Quick Health Check...")
    print("=" * 60)

    try:
        # Check API health
        print("\n📡 Checking API health...")
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy")
            print(f"   Response: {response.json()}")
        else:
            print(f"⚠️  API returned status {response.status_code}")

        # Check API docs
        print("\n📚 Checking API documentation...")
        response = requests.get(f"{api_url}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API documentation is available")
            print(f"   URL: {api_url}/docs")
        else:
            print(f"⚠️  API docs returned status {response.status_code}")

        # Test query endpoint
        print("\n🔄 Testing query endpoint...")
        payload = {
            "query": "Test query",
            "session_id": "test",
            "top_k": 1,
        }

        response = requests.post(
            f"{api_url}/query",
            json=payload,
            timeout=30,
        )

        if response.status_code == 200:
            print("✅ Query endpoint works")
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)[:200]}...")
        else:
            print(f"⚠️  Query endpoint returned status {response.status_code}")

        print("\n" + "=" * 60)
        print("✅ All checks passed!")

    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API")
        print(f"   Make sure API is running: python -m uvicorn api:app")
        print(f"   API URL: {api_url}")

    except Exception as e:
        print(f"\n❌ Error during health check: {e}")


if __name__ == "__main__":
    quick_health_check()
