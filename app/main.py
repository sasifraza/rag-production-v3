from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rag.graph import rag_graph
from rag.ingest import ingest_document
import shutil
import os

app = FastAPI(title="RAG Production v2 API")

# Conversation memory store
conversation_store = {}

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"

class QueryResponse(BaseModel):
    answer: str
    session_id: str

# Health endpoint
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0"}

# Upload document endpoint
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    chunks = ingest_document(file_path)
    return {
        "message": f"Document ingested successfully",
        "filename": file.filename,
        "chunks": chunks
    }

# Query endpoint with conversation memory
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    # Get conversation history
    chat_history = conversation_store.get(request.session_id, [])
    
    # Run RAG graph
    result = rag_graph.invoke({
        "query": request.query,
        "context": [],
        "answer": "",
        "chat_history": chat_history
    })
    
    # Update conversation store
    conversation_store[request.session_id] = result["chat_history"]
    
    return QueryResponse(
        answer=result["answer"],
        session_id=request.session_id
    )

# Metrics endpoint
@app.get("/metrics")
def metrics():
    return {
        "active_sessions": len(conversation_store),
        "version": "2.0",
        "features": [
            "LangGraph agentic flow",
            "Multi-document support",
            "Conversation memory",
            "Cross-encoder reranking"
        ]
    }