# RAG Application CI/CD Testing Strategy

## Executive Summary

This document outlines the comprehensive 3-stage CI/CD testing pipeline for the Knowledge-Based RAG (Retrieval-Augmented Generation) application. The pipeline ensures production-ready code quality through automated testing of all critical components with **specific focus on Legal Proceedings (Item 1C) and Cybersecurity Incidents (Item 3)** from 10-K financial disclosures.

**Test Focus:**
- **Legal Proceedings (Item 1C)**: Litigation, regulatory proceedings, financial exposure, accrued liabilities
- **Cybersecurity Incidents (Item 3)**: Incident response, security architecture, threat detection, board oversight

**Total Testing Coverage:**
- 250+ test cases across 4 modules
- Edge case coverage for production scenarios
- RAGAS metrics for quality evaluation
- LangSmith integration for tracing
- Automated regression detection
- Specialized tests for Legal and Cybersecurity sections

---

## Table of Contents

1. [Testing Strategy Overview](#testing-strategy-overview)
2. [Stage 1: Data Ingestion Module Tests](#stage-1-data-ingestion-module-tests)
3. [Stage 2: Retrieval Module Tests](#stage-2-retrieval-module-tests)
4. [Stage 3: Argumentation & Evaluation Tests](#stage-3-argumentation--evaluation-tests)
5. [GitHub Actions Workflow](#github-actions-workflow)
6. [GitHub Secrets Configuration](#github-secrets-configuration)
7. [Environment Variables](#environment-variables)
8. [Running Tests Locally](#running-tests-locally)
9. [Interpreting Test Results](#interpreting-test-results)
10. [Troubleshooting](#troubleshooting)

---

## Testing Strategy Overview

### Test Focus: Legal Proceedings (Item 1C) & Cybersecurity Incidents (Item 3)

This CI/CD pipeline **specifically focuses on testing two critical SEC disclosure sections**:

- **Item 1C: Legal Proceedings** - Litigation, regulatory proceedings, financial exposure, accrued liabilities
- **Item 3: Cybersecurity Incidents** - Incident response, security architecture, threat detection, board oversight

Rather than testing generic production cases, all test suites are tailored to validate extraction, retrieval, generation, and evaluation of these two important document sections.

### Three-Stage Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│          GitHub Actions CI/CD Pipeline (ci.yml)            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐   ┌──────────────────┐   ┌─────────┐ │
│  │  STAGE 1         │   │  STAGE 2         │   │ STAGE 3 │ │
│  │ Data Ingestion   │──▶│ Retrieval        │──▶│ Argum. &│ │
│  │ Legal & Cyber    │   │ Legal & Cyber    │   │ Eval.   │ │
│  └──────────────────┘   └──────────────────┘   └─────────┘ │
│       ▼                   ▼                        ▼        │
│   [Item 1C, 3]      [Query Routing]   [RAGAS, LangSmith]  │
│   [S3, Pinecone]    [Hybrid Search]   [Regression Test]   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Code Quality & Coverage Analysis (All Modules)      │   │
│  │ - pytest coverage reports                           │   │
│  │ - pylint code analysis                              │   │
│  │ - flake8 style checking                             │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ▼                                  │
│              ┌──────────────────────────┐                   │
│              │ Final Report & Notify    │                   │
│              │ Generate CI Summary      │                   │
│              └──────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### Test Dependencies

- **Stage 1** runs first (no dependencies)
- **Stage 2** requires Stage 1 to pass (uses validated data ingestion)
- **Stage 3** requires Stages 1 & 2 to pass (end-to-end tests)
- **Code Quality** runs in parallel after all modules complete

### Key Library Versions

All dependencies are pinned to specific versions in `requirements.txt` for Python 3.11.14:

| Library | Version | Purpose |
|---------|---------|---------|
| langchain | 0.3.27 | LLM orchestration |
| langchain-core | 0.3.83 | Core abstractions |
| langchain-community | 0.3.31 | Community integrations |
| langchain-openai | 0.3.35 | OpenAI adapter |
| openai | 2.28.0 | OpenAI API client |
| pinecone | 8.1.0 | Vector database |
| beautifulsoup4 | 4.14.3 | HTML parsing |
| rank-bm25 | 0.2.2 | Sparse vector generation |
| pytest | 9.0.2 | Test framework |
| python-json-logger | 4.0.0 | JSON logging |
| ragas | Latest | Evaluation metrics |

---

## Stage 1: Data Ingestion Module Tests

### Purpose
Validate that documents can be loaded from S3, parsed correctly for **Legal Proceedings (Item 1C)** and **Cybersecurity Incidents (Item 3)** sections, chunked semantically, converted to embeddings, and uploaded to Pinecone vector database with proper error handling. This stage focuses on extracting and processing these critical disclosure sections from 10-K filings.

### Test Duration: ~30 minutes

### Components Tested

#### 1.1 S3DocumentLoader Tests
**File**: `tests/test_data_ingestion.py::TestS3DocumentLoader`

Tests the AWS S3 integration for loading HTML documents:

| Test Case | Purpose | Edge Cases Covered |
|-----------|---------|-------------------|
| `test_s3_client_initialization_success` | Verify S3 connection | Valid AWS credentials, bucket exists |
| `test_s3_client_initialization_failure` | Handle connection failures | Invalid credentials, bucket not found |
| `test_list_documents_with_filtering` | Filter documents by extension | Mixed file types in S3 |
| `test_list_documents_empty_bucket` | Handle empty bucket | No documents match criteria |
| `test_load_document_from_s3` | Load specific document | Document encoding issues |
| `test_load_document_with_retry_logic` | Implement retry mechanism | Transient network failures |
| `test_load_document_encoding_issues` | Handle UTF-8 encoding | Special characters, emojis |

**Production Edge Cases:**
- S3 connectivity timeouts
- Rate limiting on ListObjects API
- Large HTML files (>100MB)
- Corrupted/incomplete file downloads

---

#### 1.2 HTMLDocumentParser Tests
**File**: `tests/test_data_ingestion.py::TestHTMLDocumentParser`

Tests extraction of **Legal Proceedings (Item 1C)** and **Cybersecurity Incidents (Item 3)** from financial documents (10-K filings). This test suite focuses specifically on these two critical disclosure sections.

| Test Case | Purpose | Section & Details |
|-----------|---------|-------------------|
| `test_extract_legal_proceedings_item_1c` | Extract Legal Proceedings (Item 1C) | Patent litigation, regulatory proceedings, contract disputes |
| `test_extract_cybersecurity_incidents_item_3` | Extract Cybersecurity Incidents (Item 3) | Incident response, zero-trust architecture, material incidents |
| `test_legal_proceedings_extraction_edge_case_missing_details` | Handle minimal legal information | No material proceedings scenarios |
| `test_cybersecurity_extraction_edge_case_multiple_incidents` | Extract multiple cyber incidents | Phishing, DDoS, malware, supply chain incidents |
| `test_legal_and_cybersecurity_metadata_extraction` | Extract section metadata | Item 1C & Item 3 specific metadata |
| `test_legal_proceedings_financial_impact_extraction` | Extract litigation financial data | Estimated exposure, accrued reserves, insurance coverage |
| `test_cybersecurity_zerorust_architecture_extraction` | Extract security architecture details | Zero-trust, MFA, encryption, monitoring |

**Test Focus - Legal Proceedings (Item 1C):**
- Patent litigation and IP disputes
- Regulatory proceedings (SEC investigations)
- Contract disputes with financial exposure
- Accrued liabilities and reserves
- Insurance coverage details
- Material contingent liabilities

**Test Focus - Cybersecurity Incidents (Item 3):**
- Current security incidents (year/quarter specific)
- Incident response procedures and board oversight
- Zero-trust architecture implementation
- Multi-factor authentication enforcement
- Encryption standards (AES-256, TLS 1.3)
- Continuous monitoring and threat detection
- Supply chain security risks
- Customer notification procedures

**Edge Cases Covered:**
- Missing or minimal disclosure information
- Multiple incidents in same quarter
- Financial impact quantification (loss estimates, prevention value)
- Regulatory notification requirements
- Redacted information handling
- Complex security architecture descriptions

---

#### 1.3 SemanticTextChunker Tests
**File**: `tests/test_data_ingestion.py::TestSemanticTextChunker`

Tests semantic document chunking for RAG context:

| Test Case | Purpose | Validation |
|-----------|---------|-----------|
| `test_chunker_initialization` | Initialize splitter | Default chunk_size=500, overlap=100 |
| `test_chunk_empty_documents` | Handle empty input | Return empty list |
| `test_chunk_single_document` | Chunk basic document | Verify split count |
| `test_chunk_preservation_of_metadata` | Preserve metadata | Original fields retained |
| `test_chunk_overlap_functionality` | Verify overlap works | Content appears in adjacent chunks |
| `test_chunk_size_handling` | Test different sizes | 100, 500, 1000 bytes |
| `test_chunk_very_long_document` | Handle large documents | 100K+ character documents |
| `test_chunk_special_characters` | Handle special chars | UTF-8, emojis, symbols |

**Production Edge Cases:**
- Documents > 1MB
- Boundary conditions at chunk limits
- Nested formatting (bullet points, tables)
- Code snippets or formatted data

---

#### 1.4 EmbeddingsGenerator Tests
**File**: `tests/test_data_ingestion.py::TestEmbeddingsGenerator`

Tests OpenAI embedding generation with batching and error handling:

| Test Case | Purpose | API Coverage |
|-----------|---------|--------------|
| `test_embeddings_client_initialization` | Initialize OpenAI client | Model: text-embedding-3-small |
| `test_generate_embeddings_for_documents` | Generate embeddings | Batch of 3 documents |
| `test_embeddings_batching` | Batch processing | 100 documents with batch_size=32 |
| `test_embeddings_api_rate_limiting` | Handle rate limits | Retry with exponential backoff |
| `test_embeddings_dimension_validation` | Validate dimensions | 1536 dimensions for small model |
| `test_empty_document_handling` | Handle empty docs | Skip or default vector |

**Production Edge Cases:**
- OpenAI API rate limiting (RateLimit 429)
- Token limit exceeded during embedding
- Network timeouts during API calls
- Cost optimization for large batches

---

#### 1.5 BM25SparseVectorGenerator Tests
**File**: `tests/test_data_ingestion.py::TestBM25SparseVectorGenerator`

Tests BM25 sparse vector generation for hybrid search:

| Test Case | Purpose | Validation |
|-----------|---------|-----------|
| `test_bm25_generator_initialization` | Initialize BM25 model | Stopwords loaded |
| `test_generate_sparse_vectors` | Generate sparse vectors | 3 test documents |
| `test_stopwords_filtering` | Verify stopwords removed | Common English words filtered |
| `test_empty_document_handling_sparse` | Handle empty docs | Zero vector or skip |
| `test_special_character_handling` | Process special chars | Symbols, numbers, mixed case |

**Production Edge Cases:**
- Non-English documents
- Numerical content (years, numbers)
- Custom domain-specific stopwords
- Very short documents

---

#### 1.6 PineconeUploader Tests
**File**: `tests/test_data_ingestion.py::TestPineconeUploader`

Tests Pinecone vector database integration with demo index:

| Test Case | Purpose | Pinecone Features |
|-----------|---------|------------------|
| `test_pinecone_client_initialization` | Initialize client | Index: rag-ci-test |
| `test_create_index_if_needed` | Create or verify index | Dimensions: 1536 |
| `test_upsert_hybrid_vectors` | Upload dense + sparse vectors | Metadata included |
| `test_batch_upsert_with_demo_index` | Batch uploading | Demo index persistence |
| `test_index_connection_failure` | Handle connection errors | Graceful degradation |
| `test_large_batch_handling` | Process large batches | Pagination support |

**Demo Index Configuration:**
- **Index Name**: `rag-ci-test`
- **Environment**: `us-east-1` (Serverless)
- **Metric**: `cosine` similarity
- **Dimensions**: 1536 (OpenAI embedding)
- **Hybrid Enabled**: Yes (dense + sparse support)

**Production Edge Cases:**
- Index quota exceeded
- Duplicate document IDs
- Invalid metadata format
- Network interruption during upsert

---

#### 1.7 Integration Tests
**File**: `tests/test_data_ingestion.py::TestDataIngestionPipeline`

End-to-end pipeline tests:

| Test Case | Focus |
|-----------|-------|
| `test_end_to_end_ingestion` | Full S3→Parse→Chunk→Embed→Upload flow |
| `test_pipeline_with_cybersecurity_content` | Cybersecurity section handling |
| `test_pipeline_with_legal_proceedings_content` | Legal section handling |
| `test_pipeline_error_recovery` | Error handling throughout |
| `test_pipeline_duplicate_handling` | Deduplicate documents |
| `test_pipeline_metadata_propagation` | Metadata through all stages |

---

## Stage 2: Retrieval Module Tests

### Purpose
Validate that queries are optimized, hybrid search (dense + sparse) returns relevant documents, reranking produces quality results, and memory is properly managed.

### Test Duration: ~45 minutes

### Components Tested

#### 2.1 Pre-Retrieval: Query Optimization
**File**: `tests/test_retrieval.py::TestQueryRewriter`

Tests LLM-based query rewriting for clarity:

| Test Case | Purpose |
|-----------|---------|
| `test_query_rewriter_initialization` | Initialize with gpt-4-turbo |
| `test_rewrite_simple_query` | Enhance simple queries |
| `test_rewrite_ambiguous_query` | Clarify vague queries |
| `test_rewrite_empty_query` | Handle empty input |
| `test_rewrite_special_characters` | Process special chars |
| `test_rewrite_very_long_query` | Handle long queries |

**Edge Cases:**
- Query with typos
- Multi-language queries
- Extremely short queries (1-2 words)

---

#### 2.2 Multi-Query Generation & HyDE
**File**: `tests/test_retrieval.py::TestMultiQueryGenerator`, `TestHyDEGenerator`

Tests generation of alternative query perspectives:

| Component | Purpose | Edge Cases |
|-----------|---------|-----------|
| **MultiQueryGenerator** | Create query variations | Semantic diversity |
| **HyDEGenerator** | Generate hypothetical documents | Relevance vs specificity |

---

#### 2.3 Domain Router
**File**: `tests/test_retrieval.py::TestDomainRouter`

Tests classification and routing to **Legal Proceedings (Item 1C)** and **Cybersecurity Incidents (Item 3)** sections. Focuses on accurate query-to-section mapping for these critical disclosure items.

| Test Case | Focus | Query Example |
|-----------|-------|-------|
| `test_route_cybersecurity_incidents_query` | Route to Item 3 cybersecurity | "What were the cybersecurity incidents in 2023?" |
| `test_route_cybersecurity_threat_detection_query` | Route cyber threat queries | "What threat detection systems do we have?" |
| `test_route_legal_proceedings_query` | Route to Item 1C legal | "What ongoing litigation cases does the company have?" |
| `test_route_legal_regulatory_query` | Route regulatory queries | "Are there any SEC investigations?" |
| `test_route_mixed_legal_cybersecurity_query` | Handle cross-section queries | "What legal implications do our cybersecurity incidents have?" |
| `test_route_cybersecurity_zero_trust_query` | Route security architecture | "Describe our zero-trust architecture" |
| `test_route_litigation_financial_impact_query` | Route financial litigation impact | "What are the estimated costs of ongoing litigation?" |

**Legal Proceedings (Item 1C) Routing:**
- Patent litigation and IP disputes
- Regulatory proceedings and investigations
- Contract disputes
- Litigation estimates and accrued reserves
- SEC/regulatory investigations
- Material contingent liabilities

**Cybersecurity Incidents (Item 3) Routing:**
- Cyber incidents and incidents response
- Zero-trust architecture and security measures
- Threat detection and monitoring
- Multi-factor authentication
- Encryption implementations
- Supply chain security
- Board cybersecurity oversight

---

#### 2.4 During-Retrieval: Hybrid Search
**File**: `tests/test_retrieval.py::TestHybridRetriever`

Tests hybrid vector retrieval combining dense and sparse:

| Test Case | Purpose |
|-----------|---------|
| `test_hybrid_retriever_initialization` | Init with alpha=0.5 |
| `test_hybrid_retrieve_basic` | Dense + sparse merge |
| `test_normalize_scores` | Scale scores to [0,1] |
| `test_normalize_empty_scores` | Handle empty input |
| `test_normalize_identical_scores` | Equal scores handling |
| `test_zero_results_handling` | No matches scenario |
| `test_alpha_weighting` | Test different alpha values |

**Hybrid Formula:**
$$\text{score}_{\text{hybrid}} = \alpha \times \text{score}_{\text{dense}} + (1-\alpha) \times \text{score}_{\text{sparse}}$$

**Alpha Values Tested:**
- 0.0: Pure sparse (BM25)
- 0.25: 25% dense
- 0.5: Balanced (DEFAULT)
- 0.75: 75% dense
- 1.0: Pure dense

---

#### 2.5 Reranking Strategies
**File**: `tests/test_retrieval.py::TestMMRReranker`, `TestCrossEncoderReranker`

Tests multiple reranking methods:

| Reranker | Purpose | Edge Cases |
|----------|---------|-----------|
| **MMRReranker** | Remove redundancy, ensure diversity | Single result, identical documents |
| **CrossEncoderReranker** | Fine-tune ranking with BERT-style models | Semantic similarity edge cases |

---

#### 2.6 Post-Retrieval Processing
**File**: `tests/test_retrieval.py::TestTokenCounter`, `TestContextualCompressor`, `TestPromptTemplateBuilder`

Tests context preparation and prompt building:

| Component | Purpose |
|-----------|---------|
| **TokenCounter** | Enforce token budgets |
| **ContextualCompressor** | Extract query-relevant segments |
| **PromptTemplateBuilder** | Build anti-hallucination prompts |

**Production Constraints:**
- Max tokens: 4000
- Max context: 2000 chars
- Include uncertainty guardrails

---

#### 2.7 Memory Management
**File**: `tests/test_retrieval.py::TestConversationMemoryManager`

Tests conversation history persistence in S3:

| Test Case | Purpose |
|-----------|---------|
| `test_memory_manager_initialization` | Initialize with S3 bucket |
| `test_save_conversation_to_s3` | Persist conversation |
| `test_load_conversation_from_s3` | Retrieve conversation |
| `test_load_conversation_not_found` | Handle missing conversation |
| `test_append_to_conversation` | Add messages to history |
| `test_memory_size_limit` | Enforce max messages |
| `test_conversation_context_window` | Respect LLM context window |

---

#### 2.8 Reasoning Enhancement
**File**: `tests/test_retrieval.py::TestChainOfThoughtReasoner`

Tests step-by-step reasoning for complex queries:

| Test Case | Purpose |
|-----------|---------|
| `test_generate_reasoning_steps` | Break down questions |
| `test_reasoning_step_validation` | Validate logic |

---

#### 2.9 Integration Tests
**File**: `tests/test_retrieval.py::TestRetrievalPipelineIntegration`

End-to-end retrieval pipeline:

| Test Case | Focus |
|-----------|-------|
| `test_complete_retrieval_flow` | All phases together |
| `test_pre_retrieval_phase` | Query optimization only |
| `test_during_retrieval_phase` | Search only |
| `test_post_retrieval_phase` | Processing only |
| `test_retrieval_with_zero_results` | No matches handling |
| `test_retrieval_token_limit` | Token budget enforcement |
| `test_concurrent_queries` | Thread safety |
| `test_retrieval_quality_metrics` | Relevance tracking |

---

## Stage 3: Argumentation & Evaluation Tests

### Purpose
Validate response generation quality, LangSmith tracing integration, RAGAS metrics computation, and regression detection.

### Test Duration: ~60 minutes

### Components Tested

#### 3.1 Prompt Engineering
**File**: `tests/test_argumentation.py::TestPromptTemplateBuilder`

Tests AI prompt construction:

| Test Case | Purpose |
|-----------|---------|
| `test_builder_initialization` | Initialize prompt builder |
| `test_build_basic_prompt` | Create basic prompt |
| `test_build_prompt_with_empty_context` | Empty context handling |
| `test_build_prompt_with_conversation_history` | Integrate conversation |
| `test_anti_hallucination_guardrails` | Include safety instructions |
| `test_prompt_length_validation` | Length constraint checking |
| `test_special_character_handling_in_prompt` | Quote/escape handling |

**Prompt Template Components:**
- System role definition
- Retrieved context
- Conversation history (if any)
- Anti-hallucination instructions
- Output format specification

---

#### 3.2 Conversation Memory
**File**: `tests/test_argumentation.py::TestConversationMemoryManager`

Tests S3-based conversation persistence:

| Test Case | Purpose |
|-----------|---------|
| `test_memory_manager_init` | S3 connection setup |
| `test_save_conversation_to_s3` | Persist with metadata |
| `test_load_conversation_from_s3` | Retrieve conversation |
| `test_load_conversation_not_found` | Graceful error handling |
| `test_append_to_conversation` | Add new messages |
| `test_memory_size_limit` | Enforce 10-message limit |
| `test_conversation_context_window` | 4K token limit |

---

#### 3.3 Chain-of-Thought Reasoning
**File**: `tests/test_argumentation.py::TestChainOfThoughtReasoner`

Tests step-by-step reasoning:

| Test Case | Purpose |
|-----------|---------|
| `test_reasoner_initialization` | Initialize reasoner |
| `test_generate_reasoning_steps` | Generate steps |
| `test_reasoning_validation` | Validate logic |
| `test_complex_query_reasoning` | Multi-step queries |

---

#### 3.4 Complete Generation Pipeline
**File**: `tests/test_argumentation.py::TestArgumentationPipeline`

End-to-end response generation:

| Test Case | Purpose |
|-----------|---------|
| `test_pipeline_initialization` | Initialize all components |
| `test_generate_response_basic` | Simple response generation |
| `test_generate_with_conversation_history` | Follow-up responses |
| `test_generate_with_chain_of_thought` | Reasoning-enabled responses |
| `test_pipeline_token_limit` | 500 token max response |
| `test_empty_query_handling` | Empty input handling |
| `test_llm_api_error_handling` | Retry on API failure |
| `test_memory_save_failure` | S3 error recovery |
| `test_hallucination_detection` | Detect ungrounded claims |
| `test_response_quality_metric` | Track quality scores |
| `test_concurrent_generation_requests` | Concurrent safety |

**Production Features:**
- Enable/disable chain-of-thought
- Enable/disable hallucination check
- Token budget enforcement
- S3 conversation persistence
- LLM error retry logic

---

#### 3.5 RAG Quality Evaluation
**File**: `tests/test_evaluation.py::TestRAGEvaluator`

Tests answer quality metrics:

| Metric | Purpose | Range |
|--------|---------|-------|
| **Answer Relevance** | Query-answer similarity | 0.0-1.0 |
| **Faithfulness** | Answer grounded in context | 0.0-1.0 |
| **Context Precision** | % of retrieved docs relevant | 0.0-1.0 |
| **Context Recall** | % of required info in context | 0.0-1.0 |

---

#### 3.6 RAGAS Metrics
**File**: `tests/test_evaluation.py::TestRAGASMetricsEvaluator`

Tests automated evaluation metrics:

| Metric | Computation | Thresholds |
|--------|------------|-----------|
| **faithfulness** | LLM-based fact check | >0.85 = Pass |
| **answer_relevance** | Query-answer similarity | >0.80 = Pass |
| **context_precision** | Relevant doc ratio | >0.75 = Pass |
| **context_recall** | Required info coverage | >0.70 = Pass |

**Test Cases:**
- `test_faithfulness_metric` - Fact verification
- `test_answer_relevance_metric` - Query fit
- `test_context_precision_metric` - Doc quality
- `test_context_recall_metric` - Info completeness
- `test_compute_all_metrics` - Batch computation
- `test_ragas_with_empty_context` - Edge case
- `test_ragas_with_hallucinated_answer` - Detection
- `test_ragas_metric_aggregation` - Batch averages

---

#### 3.7 Regression Testing
**File**: `tests/test_evaluation.py::TestRegressionTester`

Tests baseline comparison:

| Test Case | Purpose |
|-----------|---------|
| `test_regression_tester_initialization` | Initialize with baseline |
| `test_save_baseline_metrics` | Store to S3 |
| `test_load_baseline_metrics` | Retrieve from S3 |
| `test_compare_metrics` | Current vs baseline |
| `test_detect_regression` | 5% threshold detection |
| `test_baseline_not_found` | Graceful first-run |

**Baseline Strategy:**
- Version: `v1.0` (stored in S3)
- Metrics tracked:
  - Average faithfulness
  - Average answer relevance
  - Average context precision
  - Average context recall
- Regression threshold: 5% decline

---

#### 3.8 LangSmith Integration
**File**: `tests/test_evaluation.py::TestLangSmithIntegration`

Tests tracing and monitoring:

| Test Case | Purpose |
|-----------|---------|
| `test_langsmith_client_initialization` | Connect to LangSmith |
| `test_create_langsmith_project` | Create `rag-ci-test` project |
| `test_log_run_to_langsmith` | Log evaluation runs |
| `test_trace_query_execution` | Full execution trace |
| `test_langsmith_connectivity_failure` | Graceful offline mode |

**LangSmith Configuration:**
- **Project**: `rag-ci-test`
- **Endpoint**: `https://api.smith.langchain.com`
- **Tracing**: Full request/response capture

---

#### 3.9 Complete Evaluation Pipeline
**File**: `tests/test_evaluation.py::TestEvaluationPipeline`

End-to-end evaluation:

| Test Case | Purpose |
|-----------|---------|
| `test_pipeline_initialization` | Initialize all evaluators |
| `test_evaluate_single_sample` | Evaluate one Q&A pair |
| `test_evaluate_batch` | Batch evaluation |
| `test_generate_evaluation_report` | Create summary report |
| `test_regression_detection_in_pipeline` | Auto-detect regression |
| `test_pipeline_with_ground_truth` | Compare to expected answer |
| `test_pipeline_without_ground_truth` | Reference-free evaluation |
| `test_pipeline_timeout_handling` | 5-minute timeout |
| `test_pipeline_error_recovery` | Partial failure recovery |

**Evaluation Report Contents:**
- Timestamp and build info
- RAGAS metrics averages
- Regression status
- Failed assertions
- LangSmith trace links

---

#### 3.10 Reporting
**File**: `tests/test_evaluation.py::TestEvaluationReporter`

Tests report generation:

| Component | Purpose |
|-----------|---------|
| **Summary Report** | High-level metrics |
| **Detailed Report** | Per-sample results |
| **Regression Report** | Baseline comparison |
| **S3 Persistence** | Upload to bucket |

---

## GitHub Actions Workflow

### File Location
`.github/workflows/ci.yml`

### Workflow Triggers

```yaml
on:
  push:
    branches: [main, develop, feature/**]
  pull_request:
    branches: [main, develop]
  schedule:
    - cron: '0 2 * * *'  # Daily 2 AM UTC
```

### Job Orchestration

```yaml
test-data-ingestion
      ↓
test-retrieval (waits for Stage 1)
      ↓
test-argumentation-evaluation (waits for Stages 1 & 2)
      ↓
code-quality (parallel with Stage 3)
      ↓
final-report (waits for all jobs)
```

### Timeouts

| Stage | Timeout |
|-------|---------|
| Data Ingestion | 30 min |
| Retrieval | 45 min |
| Argumentation & Evaluation | 60 min |
| Code Quality | 30 min |
| **Total** | ~120 min |

---

## GitHub Secrets Configuration

### Required Secrets to Store

Create these secrets in GitHub repository settings:
`Settings → Secrets and variables → Actions → New repository secret`

| Secret Name | Value | Purpose | Retrieved From |
|------------|-------|---------|-----------------|
| `AWS_ACCESS_KEY_ID` | Your AWS access key | S3 document loading | AWS IAM Console |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key | S3 authentication | AWS IAM Console |
| `S3_TEST_BUCKET_NAME` | `rag-ci-test-docs` | Test document bucket | AWS S3 |
| `OPENAI_API_KEY` | Your OpenAI API key | GPT & embeddings | OpenAI Dashboard |
| `PINECONE_API_KEY` | Your Pinecone API key | Vector database auth | Pinecone Console |
| `PINECONE_HOST` | Your Pinecone index host | Vector database endpoint | Pinecone Console (Index details) |
| `LANGSMITH_API_KEY` | Your LangSmith API key | Tracing & monitoring | LangSmith Dashboard |

### How to Create Secrets

#### Step 1: Generate AWS Credentials
```bash
# In AWS Console:
# 1. IAM → Users → Create User
# 2. Attach Policy: S3 full access, EC2 permissions
# 3. Generate Access Key
# 4. Copy Access Key ID and Secret Access Key
```

#### Step 2: Create S3 Test Bucket
```bash
aws s3 mb s3://rag-ci-test-docs --region us-east-1
# Upload test documents:
aws s3 cp test-documents/ s3://rag-ci-test-docs/ --recursive
```

#### Step 3: Create OpenAI API Key
```
Visit: https://platform.openai.com/api-keys
- Click "Create new secret key"
- Copy the key
- Store in GitHub secret: OPENAI_API_KEY
```

#### Step 4: Create Pinecone API Key & Host
```
Visit: https://app.pinecone.io

1. Create API Key:
   - Go to "API Keys" (left sidebar)
   - Click "Create API Key"
   - Copy the key
   - Store in GitHub secret: PINECONE_API_KEY

2. Get Index Host:
   - Go to "Indexes" (left sidebar)
   - Create new serverless index:
     * Name: rag-ci-test
     * Region: us-east-1
     * Dimension: 1536
     * Metric: cosine
   - Click on the index to view details
   - Copy the Host URL (format: rag-ci-test-a1b2c3d4.svc.aind.pinecone.io)
   - Store in GitHub secret: PINECONE_HOST

⚠️ CRITICAL: Without PINECONE_HOST, data injection into the index will fail.
The host is the direct endpoint URL required for upserting vectors.
```

#### Step 5: Create LangSmith API Key
```
Visit: https://smith.langchain.com
- Go to "API Keys"
- Click "New API Key"
- Copy the key
- Store in GitHub secret: LANGSMITH_API_KEY
- Project will auto-create: rag-ci-test
```

### Validating Secrets

Add this workflow step to verify secrets exist:

```yaml
- name: Validate GitHub Secrets
  run: |
    if [ -z "${{ secrets.AWS_ACCESS_KEY_ID }}" ]; then
      echo "❌ AWS_ACCESS_KEY_ID not configured"
      exit 1
    fi
    if [ -z "${{ secrets.OPENAI_API_KEY }}" ]; then
      echo "❌ OPENAI_API_KEY not configured"
      exit 1
    fi
    if [ -z "${{ secrets.PINECONE_API_KEY }}" ]; then
      echo "❌ PINECONE_API_KEY not configured"
      exit 1
    fi
    if [ -z "${{ secrets.PINECONE_HOST }}" ]; then
      echo "❌ PINECONE_HOST not configured - required for data injection"
      exit 1
    fi
    echo "✅ All secrets configured"
```

---

## Environment Variables

### Configuration Files

#### `.env.test` (Committed to repo - Test Defaults)
Contains test environment variable templates:
```env
AWS_REGION=us-east-1
EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=500
PINECONE_INDEX_NAME=rag-ci-test
LANGSMITH_PROJECT_NAME=rag-ci-test
# ... actual values injected from secrets
```

#### Environment Variables Injected from Secrets
During CI/CD, these are resolved:
```bash
AWS_ACCESS_KEY_ID=${{ secrets.AWS_ACCESS_KEY_ID }}
AWS_SECRET_ACCESS_KEY=${{ secrets.AWS_SECRET_ACCESS_KEY }}
OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }}
PINECONE_API_KEY=${{ secrets.PINECONE_API_KEY }}
LANGSMITH_API_KEY=${{ secrets.LANGSMITH_API_KEY }}
```

### Critical Environment Variables

| Variable | Value | Impact |
|----------|-------|--------|
| `PINECONE_INDEX_NAME` | `rag-ci-test` | Must match created index |
| `LANGSMITH_PROJECT_NAME` | `rag-ci-test` | Auto-created on first run |
| `REGRESSION_THRESHOLD` | `0.05` | Fail if >5% metric drop |
| `MAX_RETRIES` | `3` | API retry count |

---

## Running Tests Locally

### Prerequisites

```bash
# Python 3.11.14
python --version

# Install dependencies (all pinned versions from requirements.txt)
pip install -r requirements.txt
```

### Setup Local Environment

```bash
# Copy test environment
cp .env.test .env.local

# Add your credentials
nano .env.local
# Edit: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_HOST, LANGSMITH_API_KEY

# Load environment
export $(cat .env.local | xargs)
```

### Run Individual Tests

```bash
# Stage 1: Data Ingestion
pytest tests/test_data_ingestion.py -v

# Stage 2: Retrieval
pytest tests/test_retrieval.py -v

# Stage 3: Argumentation
pytest tests/test_argumentation.py -v

# Evaluation
pytest tests/test_evaluation.py -v

# Specific test
pytest tests/test_data_ingestion.py::TestS3DocumentLoader::test_s3_client_initialization_success -v
```

### Run with Coverage

```bash
pytest tests/ \
  --cov=src \
  --cov-report=html \
  --cov-report=term
# Open: htmlcov/index.html
```

### Run with Parallel Execution

```bash
pytest tests/ -n auto
# Runs tests in parallel using available CPU cores
```

---

## Interpreting Test Results

### GitHub Actions Output

#### Stage Success Output
```
✅ Stage 1: Data Ingestion Module Tests - PASSED
  - S3DocumentLoader: 7/7 tests passed
  - HTMLDocumentParser: 8/8 tests passed
  - SemanticTextChunker: 7/7 tests passed
  - EmbeddingsGenerator: 4/4 tests passed
  - BM25SparseVectorGenerator: 4/4 tests passed
  - PineconeUploader: 6/6 tests passed
  - Integration tests: 5/5 tests passed
  Total: 41/41 passed in 15m 23s
```

#### Metric Report
```json
{
  "stage": "Evaluation",
  "metrics": {
    "faithfulness": {
      "value": 0.92,
      "threshold": 0.85,
      "status": "PASS"
    },
    "answer_relevance": {
      "value": 0.89,
      "threshold": 0.80,
      "status": "PASS"
    },
    "context_precision": {
      "value": 0.87,
      "threshold": 0.75,
      "status": "PASS"
    },
    "context_recall": {
      "value": 0.81,
      "threshold": 0.70,
      "status": "PASS"
    }
  },
  "regression": {
    "baseline": "v1.0",
    "status": "NO REGRESSION"
  }
}
```

### Common Test Failures

#### ❌ S3 Connection Failure
```
ERROR: Failed to connect to S3 bucket: rag-ci-test-docs
  Error Code: NoSuchBucket
```
**Fix**: Create S3 bucket and update `S3_TEST_BUCKET_NAME` secret

#### ❌ Pinecone Index Error
```
ERROR: Index rag-ci-test not found
```
**Fix**: Create Pinecone index:
```bash
# In code:
pc = Pinecone(api_key="...")
pc.create_index(
    name="rag-ci-test",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
```

#### ❌ OpenAI API Error
```
RateLimitError: Rate limit exceeded
```
**Fix**: Increase rate limit or add backoff:
```python
max_tokens = 100  # Reduce batch size
time.sleep(5)  # Add delay between requests
```

#### ❌ RAGAS Metric Computation Timeout
```
Timeout error evaluating faithfulness metric
```
**Fix**: Increase timeout or simplify batch:
```python
RAGAS_TIMEOUT=60  # Increase from 30
RAGAS_BATCH_SIZE=5  # Reduce from 10
```

---

## Troubleshooting

### Debugging Tips

#### 1. Enable Verbose Logging
```bash
export LOG_LEVEL=DEBUG
pytest tests/ -v --log-cli-level=DEBUG
```

#### 2. Run Single Test with Pdb
```bash
pytest tests/test_data_ingestion.py::TestS3DocumentLoader::test_s3_client_initialization_success -v -s --pdb
```

#### 3. Check Environment Variables
```bash
python -c "import os; print({k:v for k,v in os.environ.items() if 'AWS' in k or 'OPENAI' in k})"
```

#### 4. Validate API Keys
```bash
# Test AWS
aws s3 ls --region us-east-1

# Test OpenAI
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Test Pinecone
curl https://api.pinecone.io/indexes \
  -H "Api-Key: $PINECONE_API_KEY"

# Test LangSmith
curl https://api.smith.langchain.com/sessions \
  -H "X-API-Key: $LANGSMITH_API_KEY"
```

#### 5. Check Test Artifacts
GitHub Actions produces artifacts from test runs:
- `data-ingestion-test-results/` - Stage 1 results
- `retrieval-test-results/` - Stage 2 results
- `evaluation-test-results/` - Stage 3 results
- `code-quality-reports/` - Coverage & lint reports
- `final-ci-report/` - Summary report

**Download from**: Actions tab → Specific workflow run → Artifacts

---

### Expected Test Duration

| Stage | Duration | Key Operations |
|-------|----------|-----------------|
| Data Ingestion | 25-35 min | S3 I/O, OpenAI API calls |
| Retrieval | 40-50 min | LLM calls, hybrid search |
| Evaluation | 50-70 min | RAGAS metrics, LangSmith upload |
| Code Quality | 10-15 min | Coverage analysis |
| **Total** | 100-140 min | Full suite |

### Optimization Tips

1. **Cache Python Dependencies**: GitHub Actions caches pip by default
2. **Parallel Test Execution**: Use `pytest -n auto` (xdist)
3. **Skip Slow Tests**: Mark with `@pytest.mark.slow` and skip in CI
4. **Use Mocking**: Mock external APIs in unit tests

---

## Maintenance & Updates

###  Updating Test Suites

When code changes, update tests:

```bash
# 1. Modify test file
vim tests/test_retrieval.py

# 2. Run locally
pytest tests/test_retrieval.py -v

# 3. Commit and push
git add tests/
git commit -m "Update retrieval tests for new feature"
git push origin feature/xyz

# 4. CI validates automatically
```

### Updating Baselines

When intentional improvements occur:

```bash
# 1. Trigger evaluation
pytest tests/test_evaluation.py::TestEvaluationPipeline -v

# 2. If metrics improve, save new baseline
aws s3 cp evaluation-results/metrics.json \
  s3://rag-ci-test-docs/baselines/v1.1.json

# 3. Update in code
export BASELINE_VERSION=v1.1

# 4. Reset regression threshold if needed
export REGRESSION_THRESHOLD=0.03
```

---

## Summary & Checklist

### Implementation Checklist

- [ ] Create `.env.test` file
- [ ] Create `.github/workflows/ci.yml` workflow
- [ ] Create test files:
  - [ ] `tests/test_data_ingestion.py`
  - [ ] `tests/test_retrieval.py`
  - [ ] `tests/test_argumentation.py`
  - [ ] `tests/test_evaluation.py`
- [ ] Setup GitHub Secrets:
  - [ ] `AWS_ACCESS_KEY_ID`
  - [ ] `AWS_SECRET_ACCESS_KEY`
  - [ ] `S3_TEST_BUCKET_NAME`
  - [ ] `OPENAI_API_KEY`
  - [ ] `PINECONE_API_KEY`
  - [ ] `LANGSMITH_API_KEY`
- [ ] Create test S3 bucket
- [ ] Create Pinecone demo index (`rag-ci-test`)
- [ ] Create LangSmith project (`rag-ci-test`)
- [ ] Run local tests successfully
- [ ] Push to repository
- [ ] Verify GitHub Actions workflow runs

---

## Contact & Support

For issues or questions:
1. Check GitHub Actions logs: Actions tab → Workflow → Run → Logs
2. Review test artifacts for detailed error messages
3. Consult Troubleshooting section above

---

**Last Updated**: March 18, 2026
**Version**: 1.0
