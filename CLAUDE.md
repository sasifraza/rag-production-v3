# CLAUDE.md — RAG Production v3
# Last updated: May 2026

## Project Overview
Production RAG API — FastAPI + LangGraph + ChromaDB + Anthropic Claude.
Supports PDF ingestion, vector search, cross-encoder reranking, 
multi-session conversation memory, LangSmith observability.

## Stack
- LLM: Anthropic Claude — Haiku (grading/rewriting) + Sonnet (generation)
- Embeddings: HuggingFace all-MiniLM-L6-v2 via langchain_huggingface (free, downloads on first use)
- Vector DB: ChromaDB at data/chroma/ (must exist before running locally)
- Reranker: BAAI/bge-reranker-base, top-5 retrieve → top-3 rerank
- Framework: LangGraph + LangChain + FastAPI
- Observability: LangSmith (@traceable on every node)

## Architecture

### Request Flow
POST /query → app/main.py → rag/graph.py (LangGraph) → retrieve → 
grade_documents → generate (or rewrite → retrieve again) → quality_check
POST /upload → app/main.py → rag/ingest.py → chunk → embed → Chroma

### LangGraph Nodes (graph.py)
- retrieve: calls retriever.retrieve_and_rerank(query)
- grade_documents: Haiku grades each doc yes/no, filters irrelevant
- generate: Sonnet answers using filtered docs + last 4 messages history
- rewrite: Haiku rewrites query if no relevant docs (max 2 retries)
- no_docs: fallback message if max rewrites hit
- quality_check: retry if answer under 20 chars

### RAGState TypedDict
query, context, answer, chat_history, documents, relevance, rewrite_count

## Key Modules
- app/main.py — FastAPI, 4 endpoints, in-memory conversation_store 
  (keyed by session_id, LOST ON RESTART)
- rag/graph.py — LangGraph StateGraph, Anthropic Claude calls
- rag/retriever.py — ChromaDB + HuggingFace + cross-encoder reranking
- rag/ingest.py — PyPDFLoader, chunk_size=500, overlap=50, saves to data/chroma/

## Retriever Config (retriever.py)
CHROMA_PERSIST_DIR = "data/chroma/"
COLLECTION_NAME    = "rag_v3_docs"
EMBED_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL       = "BAAI/bge-reranker-base"
TOP_K_RETRIEVE     = 5
TOP_K_FINAL        = 3

## API Endpoints
| Method | Path     | Purpose                                              |
|--------|----------|------------------------------------------------------|
| GET    | /health  | Returns status and version                           |
| POST   | /upload  | Accepts PDF, ingests into Chroma                     |
| POST   | /query   | {"query": "...", "session_id": "..."} → answer       |
| GET    | /metrics | Active session count and feature list                |

## Environment Variables (.env)
ANTHROPIC_API_KEY   — Claude Haiku + Sonnet
LANGSMITH_API_KEY   — LangSmith observability
LANGSMITH_TRACING   — true
LANGSMITH_PROJECT   — rag-v3-langgraph

## Local Development
pip install -r requirements.txt
mkdir -p data/chroma
uvicorn app.main:app --host 0.0.0.0 --port 8000

## Deployment Architecture
- Mac is Apple Silicon (ARM) — MUST build with --platform linux/amd64
- Azure Container Apps runs linux/amd64 — ARM image fails silently
- Docker built LOCALLY then pushed to ACR
- GitHub Actions (.github/workflows/) ONLY runs az containerapp update (no Docker build in CI)

### Local Build & Push
bash scripts/build_and_push.sh

### build_and_push.sh does:
1. docker buildx build --platform linux/amd64 -t ragv3acr.azurecr.io/rag-production-v3:latest .
2. az acr update --admin-enabled true --name ragv3acr
3. docker login ragv3acr.azurecr.io (with ACR admin creds)
4. docker push ragv3acr.azurecr.io/rag-production-v3:latest

### Azure Resources
- Subscription: 903e7f42-b475-4585-a0b5-6d4453deb671
- Resource Group: rag-production-v3-rg
- ACR: ragv3acr (ragv3acr.azurecr.io)
- Container App: rag-v3-app
- Region: eastus

### Azure Deployment Rules (NEVER SKIP)
- Always: az acr update --admin-enabled true
- Always: --min-replicas 1
- Always: --revision-weight latest=100
- Always: docker login NOT az acr login
- Always: --platform linux/amd64 on Mac

### GitHub Secrets Required
- AZURE_CREDENTIALS — full JSON from az ad sp create-for-rbac --sdk-auth
- ACR_USERNAME      — ragv3acr
- ACR_PASSWORD      — from az acr credential show --name ragv3acr
- ANTHROPIC_API_KEY
- LANGSMITH_API_KEY

## Known Errors & Fixes
1. click==8.1.8 conflict in requirements.txt
   FIX: Remove all version pins, keep only direct dependencies unpinned

2. azure/login@v2 fails — "client-id and tenant-id not supplied"
   FIX: Always use azure/login@v1 with creds: ${{ secrets.AZURE_CREDENTIALS }}

3. Docker build on GitHub Actions too expensive
   FIX: Build locally, push to ACR, CI only runs az containerapp update

4. ARM/AMD64 mismatch — Mac builds arm64, Azure needs amd64
   FIX: Always docker buildx build --platform linux/amd64

5. Host key verification failed for GitHub SSH
   FIX: ssh-keyscan github.com >> ~/.ssh/known_hosts

6. Chroma data/chroma/ directory missing locally
   FIX: mkdir -p data/chroma before first run
