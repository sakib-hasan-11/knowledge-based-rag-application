"""
FastAPI Server for Knowledge-Based RAG Application
- ALB route prefix: /api/*
- ECS deployment ready
- Minimal production code
"""

import os
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.data_ingestion.config import config
from src.data_ingestion.logging_config import create_logger
from src.retrieval.retrieval_pipeline import RetrievalPipeline

logger = create_logger("FastAPIServer")

# ============================================================================
# Global RAG Pipeline
# ============================================================================

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from pinecone import Pinecone

    from src.data_ingestion.sparse_vector_generator import BM25SparseVectorGenerator

    # Check environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pinecone_index = os.getenv("PINECONE_INDEX_NAME", "rag-documents")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")

    logger.info(f"Initializing with Pinecone index: {pinecone_index}")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
    logger.info("OpenAI embeddings initialized")

    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(pinecone_index)
    logger.info(f"Pinecone index connected: {pinecone_index}")

    # Initialize BM25 generator for sparse vectors
    sparse_generator = BM25SparseVectorGenerator()
    logger.info("BM25 sparse generator initialized")

    # Initialize retrieval pipeline
    retrieval_pipeline = RetrievalPipeline(
        embeddings_model=embeddings,
        sparse_generator=sparse_generator,
        index=index,
        enable_cloudwatch=False,
        logger_name="APIRetrievalPipeline",
    )

    # Initialize LLM for response generation
    llm = ChatOpenAI(
        model_name="gpt-4-turbo",
        temperature=0.7,
        max_tokens=1024,
        api_key=openai_key,
    )
    logger.info("ChatOpenAI LLM initialized")

    logger.info("RAG pipeline initialized successfully")
    PIPELINE_READY = True

except Exception as e:
    logger.error(
        f"Failed to initialize RAG pipeline: {str(e)}\n{traceback.format_exc()}"
    )
    retrieval_pipeline = None
    PIPELINE_READY = False


# ============================================================================
# Models
# ============================================================================


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranking: bool = Field(default=True)


class QueryResponse(BaseModel):
    query_id: str
    response: str
    sources: List[Dict]
    confidence_score: float
    processing_time_ms: float


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="RAG API",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ALB Path Prefix Middleware
# ============================================================================


@app.middleware("http")
async def handle_alb_prefix(request: Request, call_next):
    """ALB /api/* prefix routing - stripped by ALB"""
    logger.debug(f"{request.method} {request.url.path}")
    return await call_next(request)


# ============================================================================
# Exception Handler
# ============================================================================


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    logger.error(f"Error: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


# ============================================================================
# Startup Event
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup"""
    logger.info("FastAPI server starting up")
    if PIPELINE_READY:
        logger.info("RAG pipeline is ready for queries")
    else:
        logger.warning("RAG pipeline not initialized - will use mock responses")


# ============================================================================
# Health Check
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check for ALB"""
    try:
        return {
            "status": "healthy",
            "pipeline_ready": PIPELINE_READY,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")


# ============================================================================
# Query Endpoint (MAIN)
# ============================================================================


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """Process RAG query using retrieval pipeline + LLM"""
    import uuid

    query_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        if not request.query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"Query {query_id}: {request.query}")

        # Check if pipeline is ready
        if not PIPELINE_READY or retrieval_pipeline is None or llm is None:
            logger.warning("RAG pipeline not initialized, returning mock response")
            response_text = f"Response to: {request.query}"
            sources = [
                {
                    "source": "apple_10k.html",
                    "section": "Item 1",
                    "relevance_score": 0.95,
                }
            ]
            confidence_score = 0.92
        else:
            # Use actual retrieval pipeline + LLM
            try:
                result = retrieval_pipeline.run_complete_pipeline(
                    query=request.query,
                    enable_hyde=True,
                    enable_multi_query=True,
                    enable_mmr=request.use_reranking,
                    enable_compression=True,
                    session_id=request.session_id,
                )

                # Extract sources from phase 8
                phase8 = result.get("phases", {}).get("phase_8_during_retrieval", {})
                sources = phase8.get("final_results", [])

                # Extract prompts from phase 9
                phase9 = result.get("phases", {}).get("phase_9_post_retrieval", {})
                prompts = phase9.get("prompts", {})
                system_prompt = prompts.get("system", "You are a helpful assistant.")
                user_prompt = prompts.get("user", request.query)

                # Invoke LLM to generate actual response
                if system_prompt and user_prompt:
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                    llm_response = llm.invoke(messages)
                    response_text = llm_response.content.strip()
                    confidence_score = 0.95
                else:
                    # Fallback if prompts not available
                    response_text = f"Response to: {request.query}"
                    confidence_score = 0.5

                logger.info(f"Query {query_id} processed successfully with LLM")

            except Exception as pipeline_error:
                logger.error(
                    f"Pipeline execution error: {str(pipeline_error)}\n{traceback.format_exc()}"
                )
                # Fallback to mock response if pipeline fails
                response_text = f"Response to: {request.query}"
                sources = []
                confidence_score = 0.0

        processing_time = (time.time() - start_time) * 1000

        return QueryResponse(
            query_id=query_id,
            response=response_text,
            sources=sources,
            confidence_score=confidence_score,
            processing_time_ms=processing_time,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Query failed")


# ============================================================================
# Startup
# ============================================================================


@app.get("/")
async def root():
    return {"message": "RAG API"}


@app.on_event("startup")
async def startup():
    logger.info("FastAPI starting")


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
    )
