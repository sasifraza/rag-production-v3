import os 
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os 

CHROMA_PATH = "data/chroma"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


def ingest_document(file_path: str):
    # load document 
    loader = PyPDFLoader(file_path)
    
    documents = loader.load()

    # split into chunks
    
    splitter = RecursiveCharacterTextSplitter (
        chunk_size = 500,
        chunk_overlap = 50
    )  

    chunks=  splitter.split_documents(documents)

    # Create embeddings and store 

    embeddings = OpenAIEmbeddings()
    
    vectorstore= Chroma.from_documents(
      documents= chunks,
      embedding=embeddings,
      persist_directory= CHROMA_PATH
    )

def get_vectorstore():
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory= CHROMA_PATH,
        embedding_function= embeddings
    )




