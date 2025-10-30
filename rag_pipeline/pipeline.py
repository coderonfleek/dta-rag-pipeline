from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

try:
	from langchain_core.documents import Document
except Exception:  # fallback for older langchain
	from langchain.schema import Document  # type: ignore

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

DEFAULT_COLLECTION_NAME = "rag-docs"


def chunk_documents(
	documents: List[Document],
	chunk_size: int = 1000,
	chunk_overlap: int = 200,
) -> List[Document]:
	"""Split documents into overlapping chunks."""
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		length_function=len,
	)
	return splitter.split_documents(documents)


def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
	"""Create a HuggingFace embeddings instance."""
	return HuggingFaceEmbeddings(model_name=model_name)


def build_vectorstore(
    chunked_documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_directory: Optional[str] = None,
) -> Chroma:
    """Build a Chroma vectorstore from chunked documents.

    With langchain-chroma, persistence is handled via a Chroma `PersistentClient`.
    We create a client at `persist_directory` and bind a stable `collection_name`.
    """
    client = None
    if persist_directory:
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_directory)

    return Chroma.from_documents(
        chunked_documents,
        embedding=embeddings,
        client=client,
        collection_name=DEFAULT_COLLECTION_NAME,
    )


def save_vectorstore(vectorstore: Chroma, path: str) -> None:
    """No-op for langchain-chroma: persistence is handled by the client."""
    Path(path).mkdir(parents=True, exist_ok=True)
    # No explicit persist() in langchain-chroma. Data is written via the client.


def load_vectorstore(path: str, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Load a Chroma vectorstore from a local directory using PersistentClient."""
    client = chromadb.PersistentClient(path=path)
    return Chroma(
        client=client,
        collection_name=DEFAULT_COLLECTION_NAME,
        embedding_function=embeddings,
    )


def get_retriever(vectorstore: Chroma, k: int = 4):
	"""Create a retriever from the vectorstore."""
	return vectorstore.as_retriever(search_kwargs={"k": k})
