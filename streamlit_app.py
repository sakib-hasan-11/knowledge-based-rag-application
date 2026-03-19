"""
Streamlit UI for Knowledge-Based RAG Application
- Uses HOST_API environment variable for backend connectivity
- Minimal UI with query interface and error tracking
"""

import os
import time
from datetime import datetime
from typing import Dict

import requests
import streamlit as st

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="RAG Query UI",
    page_icon="🤖",
    layout="wide",
)

# ============================================================================
# Configuration
# ============================================================================

HOST_API = os.getenv("HOST_API", "http://localhost:8000")
API_QUERY_ENDPOINT = f"{HOST_API}/query"
API_HEALTH_ENDPOINT = f"{HOST_API}/health"

# ============================================================================
# Session State
# ============================================================================

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "errors_list" not in st.session_state:
    st.session_state.errors_list = []


# ============================================================================
# Helper Functions
# ============================================================================


def check_api_health() -> bool:
    """Check if API is healthy"""
    try:
        response = requests.get(API_HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except:
        return False


def query_rag_api(query: str, top_k: int = 5, use_reranking: bool = True) -> Dict:
    """Send query to RAG API"""
    try:
        payload = {
            "query": query,
            "top_k": top_k,
            "use_reranking": use_reranking,
        }

        response = requests.post(API_QUERY_ENDPOINT, json=payload, timeout=30)

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False,
                "error": f"API Error ({response.status_code})",
            }

    except requests.exceptions.Timeout:
        return {"success": False, "error": "API timeout (30s)"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": f"Cannot connect to {HOST_API}"}
    except Exception as e:
        return {"success": False, "error": f"Error: {str(e)}"}


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("## System Status")

    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Online", icon="✓")
    else:
        st.error("❌ API Offline", icon="✗")

    st.markdown(f"**Host:** {HOST_API}")

    st.divider()

    if st.session_state.errors_list:
        st.markdown(f"## Errors ({len(st.session_state.errors_list)})")
        for err in st.session_state.errors_list[:5]:
            st.caption(f"🔴 {err['error']}")
        if st.button("Clear Errors"):
            st.session_state.errors_list = []
            st.rerun()
    else:
        st.markdown("## Errors")
        st.caption("✅ No errors")


# ============================================================================
# Main Page
# ============================================================================

st.title("🤖 RAG Query Interface")

# Query input
query = st.text_input(
    "Enter your question:", placeholder="e.g., What is Apple's business?"
)

if query:
    if st.button("Send Query"):
        with st.spinner("Processing..."):
            result = query_rag_api(query)

            if result["success"]:
                data = result["data"]
                st.success("✅ Query processed")

                # Display response
                st.markdown("### Response")
                st.write(data["response"])

                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{data['confidence_score']:.0%}")
                with col2:
                    st.metric("Time", f"{data['processing_time_ms']:.0f}ms")
                with col3:
                    st.metric("Sources", len(data["sources"]))

                # Sources
                if data.get("sources"):
                    st.markdown("### Sources")
                    for src in data["sources"]:
                        st.markdown(
                            f"- **{src.get('source', 'Unknown')}** ({src.get('section', 'N/A')}) "
                            f"- Relevance: {src.get('relevance_score', 0):.0%}"
                        )

                # Store in history
                st.session_state.conversation_history.insert(
                    0, {"query": query, "response": data, "time": datetime.now()}
                )

            else:
                error = result["error"]
                st.error(f"❌ {error}")
                st.session_state.errors_list.insert(0, {"error": error})

# Recent queries
if st.session_state.conversation_history:
    st.divider()
    st.markdown("### Recent Queries")
    for idx, item in enumerate(st.session_state.conversation_history[:3]):
        st.caption(f"{idx + 1}. {item['query'][:60]}...")
