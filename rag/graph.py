from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from langsmith import traceable
from rag import retriever as retriever_module
from typing import TypedDict, List, Annotated
import operator

HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-20250514"
MAX_REWRITES = 2


class RAGState(TypedDict):
    query: str
    context: List[str]
    answer: str
    chat_history: Annotated[List, operator.add]
    rewrite_count: int
    sources: List[str]


haiku = ChatAnthropic(model=HAIKU_MODEL, temperature=0)
sonnet = ChatAnthropic(model=SONNET_MODEL, temperature=0)


@traceable
def retrieve(state: RAGState) -> dict:
    docs = retriever_module.retrieve_and_rerank(state["query"])
    return {
        "context": [doc.page_content for doc in docs],
        "sources": [doc.metadata.get("source", "") for doc in docs],
    }


@traceable
def grade_documents(state: RAGState) -> dict:
    relevant = []
    for doc in state["context"]:
        prompt = (
            f"Is the following document relevant to answering the query?\n\n"
            f"Query: {state['query']}\n\n"
            f"Document: {doc}\n\n"
            f"Reply with only 'yes' or 'no'."
        )
        response = haiku.invoke([HumanMessage(content=prompt)])
        if "yes" in response.content.strip().lower():
            relevant.append(doc)
    return {"context": relevant}


@traceable
def generate(state: RAGState) -> dict:
    context = "\n\n".join(state["context"])
    chat_history = state.get("chat_history", [])
    messages = [
        SystemMessage(content=(
            "You are a helpful assistant. Answer based ONLY on the provided context. "
            "If you cannot find the answer in the context, say so.\n\n"
            f"Context:\n{context}"
        ))
    ]
    for msg in chat_history[-4:]:
        messages.append(msg)
    messages.append(HumanMessage(content=state["query"]))
    response = sonnet.invoke(messages)
    return {
        "answer": response.content,
        "chat_history": [HumanMessage(content=state["query"]), response],
    }


@traceable
def rewrite(state: RAGState) -> dict:
    count = state.get("rewrite_count", 0) + 1
    prompt = (
        f"Rewrite the following search query to improve document retrieval. "
        f"Return only the rewritten query, nothing else.\n\nQuery: {state['query']}"
    )
    response = haiku.invoke([HumanMessage(content=prompt)])
    return {"query": response.content.strip(), "rewrite_count": count}


@traceable
def no_docs(state: RAGState) -> dict:
    return {
        "answer": (
            "I could not find relevant documents to answer your question. "
            "Please try rephrasing or uploading related documents."
        )
    }


def route_after_grading(state: RAGState) -> str:
    if state["context"]:
        return "generate"
    if state.get("rewrite_count", 0) >= MAX_REWRITES:
        return "no_docs"
    return "rewrite"


def quality_check(state: RAGState) -> str:
    if len(state.get("answer", "")) < 20:
        return "retrieve"
    return END


def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("rewrite", rewrite)
    graph.add_node("no_docs", no_docs)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate": "generate", "rewrite": "rewrite", "no_docs": "no_docs"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("no_docs", END)
    graph.add_conditional_edges(
        "generate",
        quality_check,
        {"retrieve": "retrieve", END: END},
    )

    return graph.compile()


rag_graph = build_rag_graph()


def run_query(query: str, session_id: str, chat_history: list) -> dict:
    initial_state = {
        "query": query,
        "context": [],
        "answer": "",
        "chat_history": chat_history,
        "rewrite_count": 0,
        "sources": [],
    }
    result = rag_graph.invoke(initial_state)
    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "chat_history": result["chat_history"],
    }
