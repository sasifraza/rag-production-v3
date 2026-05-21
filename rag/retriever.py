from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from rag.ingest import get_vectorstore

RERANK_MODEL = "BAAI/bge-reranker-base"
CHROMA_PERSIST_DIR = "data/chroma/"
TOP_K_RETRIEVE = 5
TOP_K_FINAL = 3

def get_retriever():
    vectorstore = get_vectorstore()

    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K_RETRIEVE}
    )

    model = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)
    compressor = CrossEncoderReranker(model=model, top_n=TOP_K_FINAL)

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    return retriever


_retriever_instance = None


def retrieve_and_rerank(query: str) -> list:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = get_retriever()
    return _retriever_instance.invoke(query)
