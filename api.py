"""
FastAPI Server for Knowledge-Based RAG Application
- ALB route prefix: /api/*
- ECS deployment ready
- Minimal production code
"""

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

logger = create_logger("FastAPIServer")


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
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
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
# Health Check
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check for ALB"""
    try:
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")


# ============================================================================
# Query Endpoint (MAIN)
# ============================================================================


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """Process RAG query"""
    import uuid

    query_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        if not request.query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"Query {query_id}")

        # Mock response - replace with actual RAG pipeline
        response_text = f"Response to: {request.query}"
        sources = [
            {
                "source": "apple_10k.html",
                "section": "Item 1",
                "relevance_score": 0.95,
            }
        ]

        processing_time = (time.time() - start_time) * 1000

        return QueryResponse(
            query_id=query_id,
            response=response_text,
            sources=sources,
            confidence_score=0.92,
            processing_time_ms=processing_time,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
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
