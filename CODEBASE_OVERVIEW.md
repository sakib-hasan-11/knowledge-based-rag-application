# Knowledge-Based RAG Application - Complete Codebase Overview

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Data Flow Architecture](#data-flow-architecture)
3. [Module 1: Data Ingestion Pipeline](#module-1-data-ingestion-pipeline)
4. [Module 2: Retrieval Pipeline](#module-2-retrieval-pipeline)
5. [Module 3: Argumentation & Generation Pipeline](#module-3-argumentation--generation-pipeline)
6. [Module 4: Evaluation & Tracing Pipeline](#module-4-evaluation--tracing-pipeline)
7. [Integration Points](#integration-points)

---

## Architecture Overview

The Knowledge-Based RAG Application consists of 4 integrated modules that form a complete end-to-end pipeline:

```
Input Documents → Data Ingestion → Vector DB → Retrieval → Argumentation → Output Response
                                                                ↓
                                                          Evaluation & Tracing
```

Each module is production-grade with comprehensive error handling, CloudWatch logging, and modular components.

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE-BASED RAG APPLICATION FLOW                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: DATA INGESTION (Load, Parse, Chunk, Embed, Store)                 │
│  ───────────────────────────────────────────────────────────────────────    │
│  Input: HTML Documents (S3 or Local)                                        │
│         ↓                                                                   │
│  Processing:                                                                │
│    • Document Loading (S3DocumentLoader)                                    │
│    • HTML Parsing (HTMLDocumentParser)                                      │
│    • Semantic Chunking (SemanticTextChunker)                                │
│    • Dense Embeddings (EmbeddingsGenerator - OpenAI)                        │
│    • Sparse Vectors (BM25SparseVectorGenerator)                             │
│    • Pinecone Upload (PineconeUploader)                                     │
│         ↓                                                                   │
│  Output: Hybrid Vector Index (Pinecone)                                     │
│                                                                             │
│  PHASE 2: RETRIEVAL (Query Optimization → Search → Rerank)                  │
│  ──────────────────────────────────────────────────────────                 │
│  Input: User Query + Session Context                                        │
│         ↓                                                                   │
│  Processing:                                                                │
│    • Query Rewriting (QueryRewriter)                                        │
│    • Multi-Query Generation (MultiQueryGenerator)                           │
│    • HyDE Document Generation (HyDEGenerator)                               │
│    • Domain Routing (DomainRouter)                                          │
│    • Hybrid Retrieval (HybridRetriever - Dense + Sparse)                    │
│    • MMR Reranking (MMRReranker)                                            │
│    • Cross-Encoder Reranking (CrossEncoderReranker)                         │
│         ↓                                                                   │
│  Output: Ranked Relevant Documents                                          │
│                                                                             │
│  PHASE 3: ARGUMENTATION & GENERATION (Response Building)                    │
│  ──────────────────────────────────────────────────────                     │
│  Input: Query + Retrieved Documents + Session Memory                        │
│         ↓                                                                   │
│  Processing:                                                                │
│    • Token Counting (TokenCounter)                                          │
│    • Context Compression (ContextualCompressor)                             │
│    • Prompt Building (PromptTemplateBuilder)                                │
│    • Conversation Memory (ConversationMemoryManager)                        │
│    • Complexity Analysis (ChainOfThoughtReasoner)                           │
│    • LLM Generation (ArgumentationPipeline)                                 │
│    • Citation Extraction (Built-in to ArgumentationPipeline)                │
│         ↓                                                                   │
│  Output: Response with Citations + Session Metadata                         │
│                                                                             │
│  PHASE 4: EVALUATION & TRACING (Quality Assurance)                          │
│  ──────────────────────────────────────────────────                         │
│  Input: Responses + Reference Data + Baseline Metrics                       │
│         ↓                                                                   │
│  Processing:                                                                │
│    • RAG Quality Evaluation (RAGEvaluator)                                  │
│    • RAGAS Metrics (RAGASMetricsEvaluator)                                  │
│    • Regression Testing (RegressionTester)                                  │
│    • Report Generation (EvaluationReporter)                                 │
│    • LangSmith Tracing (Optional)                                           │
│         ↓                                                                   │
│  Output: Metrics Report + Regressions + Recommendations                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 1: Data Ingestion Pipeline

**Location:** `src/data_ingestion/`

### 1.1 Configuration Management

#### Class: `Config` (config.py)
**Purpose:** Centralized configuration management from environment variables

**Attributes:**
- AWS credentials & region configuration
- S3 bucket and document prefix settings
- OpenAI API key and embedding model (text-embedding-3-small, 1536 dimensions)
- Pinecone API key, index name, and environment
- Pipeline parameters (chunk_size=500, chunk_overlap=100, batch_size=100)
- Logging configuration (CloudWatch support)

**Methods:**
- `__init__(env_file)` - Load env variables from .env file
- `_load_configuration()` - Parse all config values
- `validate()` - Validate required configurations

**Input:** `.env` file  
**Output:** Configuration object with all pipeline parameters

---

### 1.2 Logging System

#### Class: `PipelineLogger` (logging_config.py)
**Purpose:** CloudWatch-compatible logging for AWS Lambda deployment

**Attributes:**
- Logger instance with console handler
- CloudWatch handler (optional)
- JSON and plain text format support

**Methods:**
- `__init__(name, level, log_format, enable_cloudwatch)` - Initialize logger
- `_add_cloudwatch_handler(log_group, log_stream)` - Add CloudWatch integration
- `create_logger(name, **kwargs)` - Factory method for creating loggers

**Class: `CloudWatchFormatter` (logging_config.py)
**Purpose:** Formats logs as JSON for CloudWatch compatibility

**Input:** Logging records  
**Output:** JSON-formatted log strings with timestamp, level, exception info

---

### 1.3 Document Loading (Phase 2.5)

#### Class: `S3DocumentLoader` (document_loader.py)
**Purpose:** Handles document loading from AWS S3 or local filesystem

**Attributes:**
- s3_client: boto3 S3 client
- bucket_name: S3 bucket for documents
- prefix: S3 prefix for filtering

**Methods:**
- `__init__(s3_client, bucket_name, prefix)` - Initialize S3 connection
- `list_documents(file_extension)` - List all documents in S3
  - **Input:** File extension filter (e.g., '.html')
  - **Output:** List of S3 object keys
- `load_document(s3_key)` - Load single document from S3
  - **Input:** S3 object key
  - **Output:** Document content as string
- `load_local_html(file_path)` - Load local HTML file
  - **Input:** Local file path
  - **Output:** Document content as string
- `get_file_metadata(s3_key)` - Extract metadata from S3 path
  - **Input:** S3 object key (format: company/fiscal_year/filing_type/file.html)
  - **Output:** Dict with company, fiscal_year, filing_type, source_file

**Error Handling:** Try-catch blocks for S3 errors, file not found, decode errors

**Logging:** CloudWatch-compatible JSON logs for all operations

---

### 1.4 Document Parsing (Phase 2.4, 2.6)

#### Class: `HTMLDocumentParser` (document_parser.py)
**Purpose:** Parse HTML and extract financial document sections

**Attributes:**
- target_sections: Dict of sections to extract (Item 1, 1A, 7, 8)
- logger: Pipeline logger instance

**Methods:**
- `__init__(target_sections)` - Initialize parser with target sections
- `parse_html_content(html_content)` - Parse HTML using BeautifulSoup
  - **Input:** Raw HTML string
  - **Output:** BeautifulSoup parsed object or None
- `extract_section(soup, section_key)` - Extract specific section
  - **Input:** BeautifulSoup object, section key (e.g., 'Item 1')
  - **Output:** Dict with {found, text, start_idx, end_idx, length}
- `extract_all_sections(html_content)` - Extract all target sections
  - **Input:** Raw HTML string
  - **Output:** Dict mapping section keys to extracted content
- `clean_text(text)` - Remove extra whitespace and artifacts
  - **Input:** Raw text
  - **Output:** Cleaned text string

#### Class: `DocumentMetadata` (document_parser.py)
**Purpose:** Dataclass for document section metadata

**Attributes:**
- section: Section identifier (e.g., "Item 1")
- section_full_name: Full section name (e.g., "Business")
- company: Company name
- fiscal_year: Fiscal year of document
- filing_type: Type of filing (10-K, 10-Q)
- source_file: S3 key or local file path
- extraction_date: ISO format timestamp

#### Class: `DocumentIngestionPipeline` (document_parser.py)
**Purpose:** End-to-end document loading, parsing, and LangChain Document creation

**Attributes:**
- s3_loader: S3DocumentLoader instance
- html_parser: HTMLDocumentParser instance
- extracted_documents: List of LangChain Documents

**Methods:**
- `__init__(s3_loader, html_parser)` - Initialize pipeline components
- `process_document(document_source, from_s3)` - Process single document
  - **Input:** S3 key or file path, source type flag
  - **Output:** List of LangChain Document objects with metadata
- Returns Document objects ready for chunking

---

### 1.5 Semantic Chunking (Phase 3)

#### Class: `SemanticTextChunker` (text_chunker.py)
**Purpose:** Split documents into semantic chunks with metadata preservation

**Attributes:**
- chunk_size: Size of chunks in characters (default: 500)
- chunk_overlap: Overlap between chunks (default: 100)
- text_splitter: RecursiveCharacterTextSplitter from LangChain

**Methods:**
- `__init__(chunk_size, chunk_overlap)` - Initialize with RecursiveCharacterTextSplitter
- `chunk_documents(documents)` - Split documents into chunks
  - **Input:** List of LangChain Document objects
  - **Output:** List of chunked Document objects with {chunk_id, chunk_count, original_length}
- `get_chunk_statistics(chunked_documents)` - Calculate chunk statistics
  - **Input:** List of chunked documents
  - **Output:** Dict with {total_chunks, avg_chunk_size, min_chunk_size, max_chunk_size, median_chunk_size}

**Separators Used:** `["\n\n", "\n", ". ", " ", ""]` (hierarchical splitting for semantic preservation)

---

### 1.6 Dense Embeddings (Phase 4)

#### Class: `EmbeddingsGenerator` (embeddings_generator.py)
**Purpose:** Generate dense vector embeddings using OpenAI API

**Attributes:**
- embedding_model: OpenAI model name (text-embedding-3-small)
- dimensions: Vector dimension (1536)
- batch_size: Batch size for processing (100)
- embeddings_client: OpenAIEmbeddings client

**Methods:**
- `__init__(embedding_model, dimensions, batch_size)` - Initialize OpenAI client
- `_initialize_embeddings()` - Create OpenAI embeddings client
- `generate_embeddings(documents)` - Generate embeddings for batch
  - **Input:** List of LangChain Document objects
  - **Output:** List of dicts: {id, values (1536D vector), metadata, text_preview}
  - **Error Handling:** Per-batch error recovery
- `generate_query_embedding(query_text)` - Embed single query
  - **Input:** Query string
  - **Output:** 1536-dimensional embedding vector
- `get_embedding_statistics(embeddings_data)` - Generate embedding metrics
  - **Input:** List of embedding dicts
  - **Output:** Dict with {total_embeddings, vector_dimension, successful, failed}

**Batching:** Processes documents in configurable batches to manage API rate limits and memory

---

### 1.7 Sparse Vectors (Phase 5)

#### Class: `BM25SparseVectorGenerator` (sparse_vector_generator.py)
**Purpose:** Generate sparse vectors using BM25 algorithm for hybrid search

**Attributes:**
- bm25_model: BM25Okapi model instance
- corpus_tokens: List of tokenized documents
- vocabulary: Dict of token frequencies
- token_to_idx: Dict mapping tokens to indices
- stopwords: Set of English stopwords

**Methods:**
- `__init__()` - Initialize BM25 generator with stopwords
- `_tokenize(text)` - Tokenize text with stopword removal
  - **Input:** Raw text string
  - **Output:** List of tokens (>2 chars, excluding stopwords)
- `build_corpus(documents)` - Build BM25 corpus
  - **Input:** List of LangChain Document objects
  - **Output:** Boolean (success/failure)
  - Creates BM25Okapi model from corpus
- `get_sparse_vector(text)` - Generate sparse vector for query/document
  - **Input:** Text string
  - **Output:** Dict {token_idx: normalized_bm25_score}
- `generate_all_sparse_vectors(documents)` - Batch sparse vector generation
  - **Input:** List of documents
  - **Output:** List of dicts {id, sparse_values (dict), bm25_scores}

**Sparse Format:** Dictionary with token indices as keys and BM25 scores as values

---

### 1.8 Pinecone Upload (Phase 6)

#### Class: `PineconeUploader` (pinecone_uploader.py)
**Purpose:** Manage Pinecone vector database and hybrid vector upload

**Attributes:**
- api_key: Pinecone API key
- index_name: Name of Pinecone index
- environment: Pinecone environment (e.g., us-east-1-aws)
- metric: Distance metric (cosine)
- pc: Pinecone client
- index: Pinecone index reference

**Methods:**
- `__init__(api_key, index_name, environment)` - Initialize Pinecone client
- `_initialize_pinecone()` - Create Pinecone client connection
- `create_index_if_needed(dimension, timeout)` - Create serverless index
  - **Input:** Vector dimension (1536), timeout in seconds
  - **Output:** Boolean (index ready or exists)
  - Creates: ServerlessSpec with AWS cloud provider
  - Polls: Up to 10 attempts to verify readiness
- `get_index_reference()` - Get reference to existing index
  - **Output:** Boolean (success/failure)
- `upsert_hybrid_vectors(vectors, batch_size)` - Upload hybrid vectors
  - **Input:** List of {id, values (dense), sparse_values, metadata}
  - **Output:** Dict with {upserted_count, failed_count, total_time}
  - **Vector Format:** Hybrid vectors with dense (1536D) + sparse components

---

### 1.9 Data Ingestion Pipeline Orchestrator

#### Class: `DataIngestionPipeline` (pipeline.py)
**Purpose:** Orchestrate all 6 phases of data ingestion (Phases 2-6)

**Initialization Components:**
- S3DocumentLoader
- HTMLDocumentParser
- DocumentIngestionPipeline
- SemanticTextChunker
- EmbeddingsGenerator
- BM25SparseVectorGenerator
- PineconeUploader

**Methods:**
- `__init__(enable_cloudwatch)` - Initialize all ingestion components
- `validate_configuration()` - Validate all required configs
- `run_phase_2_document_ingestion(document_sources, from_s3)` - Load & parse documents
  - **Input:** List of S3 keys/file paths, source type
  - **Output:** Tuple(success, extracted_documents)
- `run_phase_3_chunking(documents)` - Chunk documents
  - **Input:** List of documents
  - **Output:** Tuple(success, chunked_documents)
- `run_phase_4_embeddings(documents)` - Generate dense embeddings
  - **Input:** List of documents
  - **Output:** Tuple(success, embeddings_data)
- `run_phase_5_sparse_vectors(documents)` - Generate sparse vectors
  - **Input:** List of documents
  - **Output:** Tuple(success, sparse_vectors)
- `run_phase_6_pinecone_upload(dense_vectors, sparse_vectors)` - Upload hybrid vectors
  - **Input:** Dense and sparse vector lists
  - **Output:** Tuple(success, upload_statistics)
- `run_complete_pipeline(document_sources, from_s3)` - Execute all phases
  - **Input:** Document sources and source type
  - **Output:** Comprehensive pipeline results dict with all phase outputs

**Error Handling:** Phase-by-phase try-catch with partial result recovery

---

## Module 2: Retrieval Pipeline

**Location:** `src/retrieval/`

### 2.1 Pre-Retrieval Optimization (Phase 7)

#### Class: `QueryRewriter` (pre_retrieval.py)
**Purpose:** Improve query clarity using LLM

**Attributes:**
- llm: ChatOpenAI model (gpt-4-turbo)
- rewrite_prompt: PromptTemplate for rewriting

**Methods:**
- `__init__(model_name, temperature)` - Initialize with LLM
- `rewrite(query)` - Rewrite single query
  - **Input:** Original query string
  - **Output:** Rewritten query string
  - **Fallback:** Returns original if rewrite fails
- `batch_rewrite(queries)` - Rewrite multiple queries
  - **Input:** List of query strings
  - **Output:** List of rewritten queries

---

#### Class: `MultiQueryGenerator` (pre_retrieval.py)
**Purpose:** Generate query variations for multi-perspective search

**Attributes:**
- llm: ChatOpenAI model
- num_queries: Number of variations (default: 4)
- generation_prompt: PromptTemplate

**Methods:**
- `__init__(model_name, num_queries, temperature)` - Initialize
- `generate_queries(query)` - Generate query variations
  - **Input:** Original query string
  - **Output:** List of query variations
- `merge_results(variations_results)` - Merge results from multiple queries
  - **Input:** List of result sets from different queries
  - **Output:** Merged and deduplicated results

---

#### Class: `HyDEGenerator` (pre_retrieval.py)
**Purpose:** Generate hypothetical documents for semantic enhancement

**Attributes:**
- llm: ChatOpenAI model
- hyde_prompt: PromptTemplate for hypothetical doc generation

**Methods:**
- `__init__()` - Initialize generator
- `generate_hyde_docs(query)` - Generate hypothetical documents
  - **Input:** Query string
  - **Output:** List of hypothetical document strings
- `get_hyde_embeddings(hyde_docs, embeddings_model)` - Embed hypothetical docs
  - **Input:** List of hypothetical documents, embeddings model
  - **Output:** List of embeddings

---

#### Class: `DomainRouter` (pre_retrieval.py)
**Purpose:** Route queries to appropriate document sections

**Attributes:**
- domain_classifier: LLM or keyword-based classifier
- section_mappings: Dict mapping domains to sections

**Methods:**
- `__init__()` - Initialize router with domain mappings
- `classify_domain(query)` - Classify query domain
  - **Input:** Query string
  - **Output:** Dict {domain, confidence, section_recommendations}
- `get_routing_config(domain)` - Get retrieval config for domain
  - **Input:** Domain name
  - **Output:** Dict with section filters and reranker configs

---

### 2.2 During-Retrieval Search & Reranking (Phase 8)

#### Class: `HybridRetriever` (during_retrieval.py)
**Purpose:** Combine dense and sparse retrieval with scoring

**Attributes:**
- embeddings_model: OpenAIEmbeddings instance
- sparse_generator: BM25SparseVectorGenerator instance
- index: Pinecone index reference
- alpha: Weight for dense vs sparse (default: 0.5)
- top_k: Number of results (default: 10)

**Methods:**
- `__init__(embeddings_model, sparse_generator, index, alpha, top_k)` - Initialize
- `retrieve(query)` - Perform hybrid retrieval
  - **Input:** Query string
  - **Output:** List of merged documents with hybrid scores
- `_dense_retrieve(query)` - Dense vector retrieval
  - **Input:** Query string
  - **Output:** List of dense results {id, score, metadata, source}
- `_sparse_retrieve(query)` - Sparse vector retrieval
  - **Input:** Query string
  - **Output:** List of sparse results {sparse_scores, source}
- `_merge_results(dense_results, sparse_results)` - Merge & score
  - **Input:** Dense and sparse result lists
  - **Output:** Merged results with hybrid_score = alpha * dense + (1-alpha) * sparse
  - **Normalization:** Min-max normalization to [0,1]

**Hybrid Scoring:** `hybrid_score = alpha * normalized_dense_score + (1-alpha) * normalized_sparse_score`

---

#### Class: `MMRReranker` (during_retrieval.py)
**Purpose:** Maximal Marginal Relevance for diversity

**Attributes:**
- lambda_param: Diversity-relevance tradeoff (default: 0.5)
- top_k: Number of final results

**Methods:**
- `__init__(lambda_param, top_k)` - Initialize MMR reranker
- `rerank(documents)` - Rerank using MMR
  - **Input:** List of documents with relevance scores
  - **Output:** Reranked list with MMR scores (diverse subset)
  - **Formula:** MMR = λ * relevance - (1-λ) * max_similarity_to_selected

---

#### Class: `CrossEncoderReranker` (during_retrieval.py)
**Purpose:** Fine-grained reranking using cross-encoder

**Attributes:**
- model_name: Cross-encoder model (e.g., BAAI/bge-reranker-large)
- cross_encoder: HuggingFaceCrossEncoder instance

**Methods:**
- `__init__(model_name)` - Initialize cross-encoder
- `rerank(query, documents)` - Rerank documents
  - **Input:** Query string, list of documents
  - **Output:** Documents re-ranked by cross-encoder scores
  - Uses: Pre-computed cross-encoder scores

---

### 2.3 Retrieval Pipeline Orchestrator

#### Class: `RetrievalPipeline` (retrieval_pipeline.py)
**Purpose:** Orchestrate all 3 phases of retrieval (Phases 7-9 integration)

**Attributes:**
- pre_retrieval_components: QueryRewriter, MultiQueryGenerator, HyDEGenerator, DomainRouter
- retriever: HybridRetriever
- mmr_reranker: MMRReranker
- cross_encoder_reranker: CrossEncoderReranker

**Methods:**
- `__init__(embeddings_model, sparse_generator, index)` - Initialize all components
- `run_phase_7_pre_retrieval(query, verbose)` - Pre-retrieval optimization
  - **Input:** User query, verbose flag
  - **Output:** Optimized queries {rewritten, variations, hyde_docs, domain}
- `run_phase_8_during_retrieval(queries, verbose)` - Search and rerank
  - **Input:** List of queries to search
  - **Output:** Hybrid-retrieved documents with reranking scores
- `run_complete_pipeline(query, verbose)` - End-to-end retrieval
  - **Input:** User query, verbose flag
  - **Output:** Final ranked documents ready for generation

---

## Module 3: Argumentation & Generation Pipeline

**Location:** `src/argumentation/`

### 3.1 Response Generation Components (Phase 10)

#### Class: `PromptTemplateBuilder` (generation_components.py)
**Purpose:** Construct system and user prompts with anti-hallucination constraints

**Attributes:**
- SYSTEM_PROMPT: Hard prompt with retrieval constraints
- FEW_SHOT_EXAMPLES: List of example Q&A pairs

**Methods:**
- `__init__(custom_system_prompt)` - Initialize builder
- `build_system_prompt()` - Get system prompt
  - **Output:** System prompt string with anti-hallucination rules
- `build_user_prompt(query, context, memory_summary)` - Build complete user prompt
  - **Input:** Query, retrieved context, conversation history
  - **Output:** Formatted user prompt with context injection
- `get_few_shot_examples()` - Get example pairs
  - **Output:** List of {query, reasoning, answer} dicts
- `add_few_shot_to_prompt(user_prompt, indicators)` - Add relevant examples
  - **Input:** Base prompt, query indicators {comparison, calculation, etc}
  - **Output:** User prompt with few-shot examples appended

---

#### Class: `ConversationMemoryManager` (generation_components.py)
**Purpose:** Manage conversation history with S3 persistence

**Attributes:**
- session_id: UUID for conversation session
- conversation_buffer: List of recent interactions
- summary: High-level conversation summary
- s3_client: boto3 S3 client
- bucket_name: S3 bucket for chat history

**Methods:**
- `__init__(s3_client, bucket_name, max_window)` - Initialize manager
  - max_window: Number of recent interactions to keep (default: 5)
- `add_interaction(query, response, sources, reasoning)` - Add Q&A to buffer
  - **Input:** Query, response, source docs, reasoning steps
- `update_summary(summary_text)` - Update conversation summary
- `get_memory_string()` - Format memory for prompt injection
  - **Output:** Formatted string with recent questions and summary
- `save_to_s3()` - Persist session to S3
  - **Output:** Boolean (success/failure)
  - Format: S3 path: `chat-history/{session_id}/session.json`
- `load_session_from_s3(session_id)` - Load previous session
  - **Input:** Session UUID
  - **Output:** Boolean (success/failure)

---

#### Class: `ChainOfThoughtReasoner` (generation_components.py)
**Purpose:** Analyze query complexity and generate reasoning prompts

**Attributes:**
- Complexity keywords for different query types
- Reasoning prompt templates

**Methods:**
- `analyze_query_complexity(query)` - Determine CoT necessity
  - **Input:** Query string
  - **Output:** Dict {requires_cot, indicators, complexity_score (0-5)}
  - **Indicators:** {multi_part, comparison, calculation, explanation, historical}
- `build_cot_prompt(query, context)` - Build step-by-step reasoning prompt
  - **Input:** Query, retrieved context
  - **Output:** Formatted CoT prompt template
- `extract_reasoning_steps(llm_response)` - Parse reasoning from response
  - **Input:** LLM response string
  - **Output:** Dict {reasoning_steps (list), step_count (int)}

---

### 3.2 Argumentation Pipeline Orchestrator

#### Class: `ArgumentationPipeline` (generation_pipeline.py)
**Purpose:** Orchestrate end-to-end response generation

**Attributes:**
- llm_client: OpenAI API client
- prompt_builder: PromptTemplateBuilder
- memory_manager: ConversationMemoryManager
- reasoner: ChainOfThoughtReasoner

**Methods:**
- `__init__(llm_client, s3_client, bucket_name, model_name, temperature, max_tokens)` - Initialize all components
- `generate_response(query, retrieved_documents, session_id, use_cot, verbose)` - Generate complete response
  - **Input:** Query, retrieved docs, session ID, CoT flag, verbose flag
  - **Output:** Dict with {response, citations, session_id, used_cot, complexity_score}
  - **12-Step Process:**
    1. Load session if provided
    2. Format context from documents
    3. Get conversation memory
    4. Analyze query complexity
    5. Build system prompt
    6. Build user prompt with context & memory
    7. Add CoT reasoning if needed
    8. Add few-shot examples
    9. Call LLM
    10. Extract citations
    11. Extract reasoning steps
    12. Save to memory & S3
- `_format_context(documents, verbose)` - Format retrieved docs
  - **Input:** List of document dicts
  - **Output:** Formatted context string with [Doc X: Title] format
- `_call_llm(system_prompt, user_prompt, verbose)` - Invoke LLM
  - **Input:** System and user prompts
  - **Output:** LLM response string or error message
  - **Params:** temperature=0.3, max_tokens=500
- `_extract_citations(response, documents)` - Extract document citations
  - **Input:** Response text, source documents
  - **Output:** List of {title, source, relevance_score}
- `update_session_summary(summary_text)` - Update and save summary
- `get_session_memory()` - Retrieve current session memory
- `clear_session()` - Clear session and create new one

---

#### Class: `GenerationStatistics` (generation_pipeline.py)
**Purpose:** Collect statistics across generation runs

**Methods:**
- `record_generation(result)` - Record single generation result
- `get_summary()` - Get aggregated statistics
  - **Output:** Dict {total_queries, successful, failed, avg_time, tokens_generated, cot_used_count}

---

## Module 4: Evaluation & Tracing Pipeline

**Location:** `src/evaluation/`

### 4.1 Evaluation Components (Phase 11)

#### Class: `RAGEvaluator` (evaluation_components.py)
**Purpose:** Evaluate RAG system quality on test queries

**Attributes:**
- llm_client: LLM for evaluation
- retrieval_pipeline: Retrieval pipeline instance
- generation_pipeline: Generation pipeline instance
- timeout_seconds: Query timeout (default: 30)

**Methods:**
- `__init__(llm_client, retrieval_pipeline, generation_pipeline)` - Initialize
- `evaluate_single_query(query, verbose)` - Evaluate single query
  - **Input:** Query string, verbose flag
  - **Output:** Dict with {query, latencies, retrieved_docs, response, error}
  - **Latencies:** {retrieval_ms, generation_ms, total_ms}
- `batch_evaluate(queries, verbose)` - Evaluate multiple queries
  - **Input:** List of queries
  - **Output:** List of evaluation results dicts

---

#### Class: `RAGASMetricsEvaluator` (evaluation_components.py)
**Purpose:** Compute RAGAS metrics (ground-truth-free evaluation)

**Attributes:**
- metrics: List of RAGAS metric objects
  - faithfulness: Is answer grounded in context?
  - answer_relevancy: Is answer relevant to query?
  - context_precision: Are retrieved docs relevant?
  - context_recall: Did we retrieve all relevant docs?

**Methods:**
- `__init__()` - Initialize with RAGAS metrics
- `prepare_evaluation_dataset(evaluation_results)` - Convert to RAGAS format
  - **Input:** List of evaluation results
  - **Output:** RAGAS Dataset with {question, answer, contexts}
- `evaluate(evaluation_results)` - Run RAGAS evaluation
  - **Input:** Evaluation results
  - **Output:** Dict with {overall_scores, individual_scores, sample_count}
- `print_metric_report(metrics_results)` - Print metrics summary

---

#### Class: `RegressionTester` (evaluation_components.py)
**Purpose:** Detect performance regressions vs baseline

**Attributes:**
- s3_client: boto3 S3 client
- bucket_name: Bucket for baseline storage
- baseline_metrics: Loaded baseline dict

**Methods:**
- `__init__(s3_client, bucket_name)` - Initialize tester
- `load_baseline()` - Load saved baseline from S3
  - **Output:** Boolean (success/failure)
  - **S3 key:** `evaluation/baseline_metrics.json`
- `save_baseline(metrics)` - Save current metrics as baseline
  - **Input:** Metrics dict
  - **Output:** Boolean (success/failure)
- `detect_regressions(current_metrics, threshold_percent)` - Compare with baseline
  - **Input:** Current metrics, threshold % (default: 10%)
  - **Output:** Dict with {baseline_available, regressions, improvements}
  - **Each regression:** {metric, baseline, current, change_percent}

---

### 4.2 Evaluation Reporter

#### Class: `EvaluationReporter` (evaluation_reporter.py)
**Purpose:** Generate comprehensive evaluation reports

**Attributes:**
- s3_client: boto3 S3 client
- bucket_name: Bucket for report storage

**Methods:**
- `__init__(s3_client, bucket_name)` - Initialize reporter
- `generate_report(rag_results, ragas_metrics, regression_analysis, report_name)` - Create comprehensive report
  - **Input:** Evaluation results, metrics, regression analysis, report name
  - **Output:** Dict with:
    - timestamp: ISO timestamp
    - summary: {total_queries, success_rate, avg_latency}
    - metrics: RAGAS scores
    - latency_analysis: {retrieval, generation, total} stats
    - regression_analysis: Detected regressions/improvements
    - recommendations: Actionable suggestions
- `_generate_summary(results, metrics)` - Create summary stats
- `_analyze_latencies(results)` - Analyze latency distribution
  - **Output:** Dict with mean, min, max, median, p95, p99 for each phase
- `_generate_recommendations(metrics)` - Generate improvement suggestions
  - **Output:** List of recommendation strings
- `save_report(report, format)` - Save to S3
  - **Input:** Report dict, format (json/csv)
  - **Output:** Boolean (success/failure)
  - **S3 path:** `rag-reports/{report_name}_{timestamp}.{format}`
- `print_report(report)` - Print formatted report to logger

---

### 4.3 Evaluation Pipeline Orchestrator

#### Class: `EvaluationPipeline` (evaluation_pipeline.py)
**Purpose:** Orchestrate complete evaluation workflow

**Attributes:**
- rag_evaluator: RAGEvaluator instance
- ragas_evaluator: RAGASMetricsEvaluator instance
- regression_tester: RegressionTester instance
- reporter: EvaluationReporter instance
- langsmith_tracer: Optional LangSmith tracing

**Methods:**
- `__init__(llm_client, s3_client, bucket_name, retrieval_pipeline, generation_pipeline, enable_tracing)` - Initialize
- `run_complete_evaluation(queries, report_name, save_report, check_regressions, update_baseline, verbose)` - Full evaluation
  - **Input:** Evaluation queries, config flags
  - **Output:** Dict with {status, rag_results, ragas_metrics, regression_analysis, report, execution_time}
  - **6-Step Process:**
    1. Batch RAG evaluation (latency & quality)
    2. RAGAS metrics computation
    3. Regression detection (if enabled)
    4. Report generation
    5. Baseline update (if requested)
    6. S3 save (if requested)
- `run_quick_evaluation(queries, verbose)` - Fast evaluation (RAG only)
  - **Input:** Queries, verbose flag
  - **Output:** Quick evaluation summary {total_queries, successful, failed, avg_latency, results}

---

#### Class: `EvaluationStatistics` (evaluation_pipeline.py)
**Purpose:** Aggregate statistics across multiple evaluation runs

**Methods:**
- `record_evaluation_run(result)` - Record run results
- `get_summary()` - Get aggregate stats
  - **Output:** Dict {total_runs, total_queries, success_rate, avg_time, regressions_detected}

---

## Integration Points

### Data Flow Between Modules

```
┌─────────────────────────────────────────────────────────────────┐
│ MODULE 1: DATA INGESTION                                        │
│ ─────────────────────────────────────────────────────────────── │
│ Input:  HTML Documents (S3/Local)                               │
│ Output: Pinecone Index (Hybrid Vectors)                         │
│         - Dense embeddings: 1536D OpenAI vectors                │
│         - Sparse vectors: BM25 token indices & scores           │
│         - Metadata: section, company, fiscal_year, filing_type  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ MODULE 2: RETRIEVAL                                              │
│ ─────────────────────────────────────────────────────────────── │
│ Input:  User Query + Retrieved Index                            │
│ Processing:                                                      │
│   1. Query Rewriting (LLM)                                       │
│   2. Multi-Query Generation                                      │
│   3. HyDE Document Generation                                    │
│   4. Domain Routing                                              │
│   5. Hybrid Retrieval (Dense + Sparse)                          │
│   6. MMR Reranking (Diversity)                                   │
│   7. Cross-Encoder Reranking (Fine-tuning)                      │
│ Output: Ranked Relevant Documents                               │
│         - Top-5 documents with relevance scores                 │
│         - Metadata and content preserved                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ MODULE 3: ARGUMENTATION & GENERATION                             │
│ ─────────────────────────────────────────────────────────────── │
│ Input:  Query + Retrieved Docs + Session Context                │
│ Processing:                                                      │
│   1. Token Counting (Budget Management)                          │
│   2. Context Compression (Query-relevant extraction)            │
│   3. Prompt Building (Anti-hallucination)                        │
│   4. Memory Management (Conversation history)                    │
│   5. Complexity Analysis (CoT decision)                          │
│   6. LLM Invocation (Response generation)                        │
│   7. Citation Extraction                                         │
│ Output: Response with Citations                                  │
│         - Full response text                                     │
│         - Extracted citations                                    │
│         - Session metadata                                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ MODULE 4: EVALUATION & TRACING                                   │
│ ─────────────────────────────────────────────────────────────── │
│ Input:  Responses + Reference Data + Baseline                   │
│ Processing:                                                      │
│   1. RAG Evaluation (Latency)                                     │
│   2. RAGAS Metrics (Faithfulness, Relevancy, etc)               │
│   3. Regression Detection                                        │
│   4. Report Generation                                           │
│   5. Recommendations                                             │
│ Output: Evaluation Report                                        │
│         - Metrics scores                                         │
│         - Latency analysis                                       │
│         - Regressions/Improvements                               │
│         - Actionable recommendations                             │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration & Logging

**Centralized Configuration:** `Config` class (data_ingestion/config.py)
- Used by all modules
- Loads from .env file
- Provides all API keys, model names, and pipeline parameters

**Centralized Logging:** `PipelineLogger` class (data_ingestion/logging_config.py)
- CloudWatch-compatible JSON format
- Used by all modules
- Supports plain text and JSON formats

### Shared Interfaces

1. **LangChain Document:** Standard document format across all modules
   - `page_content`: Text content
   - `metadata`: Dict with section, company, fiscal_year, source, etc

2. **Embedding Model:** Shared OpenAI embeddings across modules
   - Model: text-embedding-3-small
   - Dimension: 1536

3. **Sparse Vector Format:** Dict-based for BM25 scores
   - Key: Token index
   - Value: BM25 score (0-1 normalized)

4. **Session Management:** UUID-based sessions persisted to S3
   - Format: `chat-history/{session_id}/session.json`
   - Includes: conversation history, summary, metadata

---

## Module Dependency Graph

```
Config ────────────┬──────────────────────────┬──────────────────┐
                   │                          │                  │
PipelineLogger ────┴──────────────────────────┴──────────────────┤
                                              │                  │
                   ┌──────────────────────────┴──────────────────┤
                   │                                              │
      Data Ingestion Module                              Retrieval Module
      ────────────────────                              ─────────────────
      • S3DocumentLoader                               • QueryRewriter
      • HTMLDocumentParser                             • MultiQueryGenerator
      • DocumentIngestionPipeline                      • HyDEGenerator
      • SemanticTextChunker                            • DomainRouter
      • EmbeddingsGenerator                            • HybridRetriever
      • BM25SparseVectorGenerator                      • MMRReranker
      • PineconeUploader                               • CrossEncoderReranker
      • DataIngestionPipeline                          • RetrievalPipeline
                   │                                           │
                   └────────────┬────────────────────────────┬──┘
                                │                            │
                                ▼                            ▼
                    Argumentation Module            Evaluation Module
                    ───────────────────             ──────────────────
                    • PromptTemplateBuilder         • RAGEvaluator
                    • ConversationMemoryManager     • RAGASMetricsEvaluator
                    • ChainOfThoughtReasoner        • RegressionTester
                    • ArgumentationPipeline         • EvaluationReporter
                                                    • EvaluationPipeline
                                                    • EvaluationStatistics
```

---

## Summary Table

| Module | Location | Purpose | Key Classes | Input | Output |
|--------|----------|---------|-------------|-------|--------|
| **Data Ingestion** | `src/data_ingestion/` | Load, parse, chunk, embed, store documents | 10 classes including `DataIngestionPipeline` | HTML documents (S3/local) | Hybrid vectors in Pinecone |
| **Retrieval** | `src/retrieval/` | Optimize query, search, rerank documents | 7 classes including `RetrievalPipeline` | User query + Pinecone index | Ranked relevant documents |
| **Argumentation** | `src/argumentation/` | Build prompts, manage memory, generate response | 4 classes including `ArgumentationPipeline` | Query + docs + memory | Response with citations |
| **Evaluation** | `src/evaluation/` | Evaluate quality, detect regressions, report | 5 classes including `EvaluationPipeline` | Responses + reference data | Metrics report + recommendations |

---

## End of Codebase Overview

This documentation captures the complete architecture, class structure, inputs/outputs, and integration points for the Knowledge-Based RAG Application codebase in the `src/` folder.
