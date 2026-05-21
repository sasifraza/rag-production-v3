# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production RAG (Retrieval-Augmented Generation) API built with FastAPI and LangGraph. Supports document ingestion (PDF), vector search with Chroma, cross-encoder reranking, and multi-session conversation memory.

## Setup & Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (requires .env with keys set)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Docker
docker build -t rag-production-v2 .
docker run -p 8000:8000 --env-file .env rag-production-v2
```

Required environment variables (in `.env`):
- `OPENAI_API_KEY` — used for HuggingFace embeddings (ingest/retrieval pipeline)
- `ANTHROPIC_API_KEY` — used for generation (Sonnet) and grading/rewriting (Haiku)
- `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_PROJECT` — optional LLM observability

## Architecture

### Request Flow

```
POST /query → app/main.py → rag/graph.py run_query() → retrieve → grade_documents → generate → quality_check
                                                                              ↓ (no relevant docs)
                                                                           rewrite → retrieve (max 2x)
                                                                              ↓ (max rewrites hit)
                                                                            no_docs → END
POST /upload → app/main.py → rag/ingest.py → chunk → embed → Chroma (data/chroma/)
```

### Key Modules

- **`app/main.py`** — FastAPI app with four endpoints: `GET /health`, `POST /upload`, `POST /query`, `GET /metrics`. Holds the in-memory `conversation_store` dict (keyed by `session_id`).
- **`rag/graph.py`** — LangGraph `StateGraph` with 5 nodes: `retrieve`, `grade_documents`, `generate`, `rewrite`, `no_docs`. Uses Haiku (`claude-haiku-4-5-20251001`) for per-document relevance grading and query rewriting, Sonnet (`claude-sonnet-4-20250514`) for final generation. `quality_check` is a conditional edge (not a node) that retries retrieval if the answer is under 20 chars. `rewrite` is capped at `MAX_REWRITES = 2`; exceeding it routes to `no_docs`. Entry point for callers is `run_query(query, session_id, chat_history) → {answer, sources, chat_history}`. All 5 nodes are decorated with `@traceable` for LangSmith. `RAGState` carries: `query`, `context`, `answer`, `chat_history`, `rewrite_count`, `sources`.
- **`rag/retriever.py`** — Wraps Chroma with HuggingFace embeddings (`all-MiniLM-L6-v2`). Retrieves top-5 by vector similarity (`TOP_K_RETRIEVE = 5`), reranks to top-3 using cross-encoder (`TOP_K_FINAL = 3`, `RERANK_MODEL = "BAAI/bge-reranker-base"`). Chroma path set via `CHROMA_PERSIST_DIR = "data/chroma/"`. Both models are downloaded on first use. Exposes `retrieve_and_rerank(query) -> list[Document]` using a module-level lazy-initialized retriever instance (loaded once per process).
- **`rag/ingest.py`** — Loads PDFs with `PyPDFLoader`, splits with `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=50), stores embeddings in Chroma at `data/chroma/`.

### State & Memory

- Conversation history lives in `conversation_store` in `app/main.py` — it is **in-memory only** and lost on restart.
- The LangGraph state (`RAGState`) carries: `query`, `context`, `answer`, `chat_history`, `rewrite_count`, `sources`.
- `chat_history` uses `Annotated[List, operator.add]` — LangGraph appends the two messages returned by `generate` automatically; callers must not manually concatenate.
- Only the last 4 messages from history are passed to the generation prompt.

### Storage

- Chroma vector DB persists to `data/chroma/` (created by Docker; must exist locally before running).
- No other database is used.

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Returns status and version |
| POST | `/upload` | Accepts PDF file, ingests into Chroma |
| POST | `/query` | Accepts `{"query": "...", "session_id": "..."}`, returns answer |
| GET | `/metrics` | Returns active session count and feature list |
