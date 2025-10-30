from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

try:
	from langchain_core.documents import Document
except Exception:  # fallback for older langchain
	from langchain.schema import Document  # type: ignore

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
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
) -> FAISS:
	"""Build a FAISS vectorstore from chunked documents."""
	return FAISS.from_documents(chunked_documents, embeddings)


def save_vectorstore(vectorstore: FAISS, path: str) -> None:
	"""Persist FAISS vectorstore to local folder."""
	Path(path).mkdir(parents=True, exist_ok=True)
	vectorstore.save_local(path)


def load_vectorstore(path: str, embeddings: HuggingFaceEmbeddings) -> FAISS:
	"""Load a FAISS vectorstore from local folder."""
	return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def get_retriever(vectorstore: FAISS, k: int = 4):
	"""Create a retriever from the vectorstore."""
	return vectorstore.as_retriever(search_kwargs={"k": k})
