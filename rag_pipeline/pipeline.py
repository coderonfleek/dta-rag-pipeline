from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

try:
	from langchain_core.documents import Document
except Exception:  # fallback for older langchain
	from langchain.schema import Document  # type: ignore

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


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
	chunked_documents: List[Document], embeddings: HuggingFaceEmbeddings
) -> Chroma:
	"""Build a Chroma vectorstore from chunked documents (in-memory)."""
	return Chroma.from_documents(chunked_documents, embedding=embeddings)


def save_vectorstore(vectorstore: Chroma, path: str) -> None:
	"""Persist Chroma vectorstore to a local directory."""
	# Chroma uses a persist directory; set it if not set and persist
	vectorstore._persist_directory = path  # set target path before persisting
	Path(path).mkdir(parents=True, exist_ok=True)
	vectorstore.persist()


def load_vectorstore(path: str, embeddings: HuggingFaceEmbeddings) -> Chroma:
	"""Load a Chroma vectorstore from a local directory."""
	return Chroma(persist_directory=path, embedding_function=embeddings)


def get_retriever(vectorstore: Chroma, k: int = 4):
	"""Create a retriever from the vectorstore."""
	return vectorstore.as_retriever(search_kwargs={"k": k})
