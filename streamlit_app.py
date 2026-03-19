"""
Streamlit UI for Knowledge-Based RAG Application
- Non-blocking startup
- Graceful API connection handling
"""

import os
from datetime import datetime
from typing import Dict

import requests
import streamlit as st

# ============================================================================
# CONFIGURATION (No blocking calls here)
# ============================================================================

st.set_page_config(
    page_title="RAG Query UI",
    page_icon="",
    layout="wide",
)

HOST_API = os.getenv("HOST_API", "http://localhost:8000")
API_QUERY_ENDPOINT = f"{HOST_API}/query"
API_HEALTH_ENDPOINT = f"{HOST_API}/health"

# ============================================================================
# SESSION STATE (Initialize early)
# ============================================================================

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "errors_list" not in st.session_state:
    st.session_state.errors_list = []


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def check_api_health() -> bool:
    """Non-blocking API health check"""
    try:
        response = requests.get(API_HEALTH_ENDPOINT, timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def query_rag_api(query: str, top_k: int = 5, use_reranking: bool = True) -> Dict:
    """Send query to RAG API"""
    try:
        payload = {
            "query": query,
            "top_k": top_k,
            "use_reranking": use_reranking,
        }

        response = requests.post(API_QUERY_ENDPOINT, json=payload, timeout=60)

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False,
                "error": f"API Error ({response.status_code})",
            }

    except requests.exceptions.Timeout:
        return {"success": False, "error": "API timeout (60s)"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": f"Cannot connect to {HOST_API}"}
    except Exception as e:
        return {"success": False, "error": f"Error: {str(e)}"}


# ============================================================================
# MAIN UI (No blocking operations)
# ============================================================================

st.title("RAG Query Interface")

st.markdown("Ask questions about Apple's business based on their 10-K filing.")

st.divider()

# Query Input
query = st.text_input(
    "Enter your question:",
    placeholder="e.g., What is Apple's main business?",
)

if query:
    if st.button("Send Query", type="primary"):
        with st.spinner("Processing your query..."):
            result = query_rag_api(query)

            if result["success"]:
                data = result["data"]

                # Display response
                st.success("Query processed successfully!")

                st.markdown("### Response")
                response_text = data.get("response", "")
                if response_text:
                    st.info(response_text)
                else:
                    st.warning("No response generated")

                # Debug info
                with st.expander("View Raw Response & Debug"):
                    st.json(data)

                # Metrics
                st.divider()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Confidence",
                        f"{data.get('confidence_score', 0):.0%}",
                    )
                with col2:
                    st.metric(
                        "Processing Time",
                        f"{data.get('processing_time_ms', 0):.0f}ms",
                    )
                with col3:
                    st.metric("Sources Found", len(data.get("sources", [])))

                # Sources
                if data.get("sources"):
                    st.markdown("### Referenced Sources")
                    for i, src in enumerate(data["sources"][:3], 1):
                        section = src.get("metadata", {}).get("section", "Unknown")
                        st.markdown(
                            f"**{i}. {section}** - Relevance: {src.get('score', 0):.2%}"
                        )

                # Store in history
                st.session_state.conversation_history.insert(
                    0,
                    {
                        "query": query,
                        "response": data,
                        "time": datetime.now().strftime("%I:%M %p"),
                    },
                )

            else:
                error = result["error"]
                st.error(f"Query Failed: {error}")
                st.session_state.errors_list.insert(0, {"error": error})

# Sidebar
with st.sidebar:
    st.markdown("## About")
    st.write("Knowledge-Based RAG Application")
    st.write("Retrieves information from Apple's 10-K filing and generates AI answers.")

    st.divider()

    st.markdown("## Configuration")
    st.write(f"**API Host:** {HOST_API}")

    # API Status check button
    if st.button("Check API Status"):
        if check_api_health():
            st.success("API Online")
        else:
            st.error(f"API Offline - {HOST_API}")

    st.divider()

    st.markdown("## History")
    if st.session_state.conversation_history:
        st.write(f"Queries: {len(st.session_state.conversation_history)}")
        for i, item in enumerate(st.session_state.conversation_history[:5], 1):
            st.caption(f"{i}. {item['query'][:50]}... ({item['time']})")
    else:
        st.write("No queries yet")

    if st.button("Clear History"):
        st.session_state.conversation_history = []
        st.rerun()
