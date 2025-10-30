from __future__ import annotations

import argparse
from typing import Optional

from .db import fetch_all_documents
from .ingest import ingest_files_to_db
from .pipeline import (
	build_vectorstore,
	chunk_documents,
	get_embeddings,
	get_retriever,
	load_vectorstore,
	save_vectorstore,
)


def run_build(
	source_dir: str,
	db_path: str,
	vectorstore_path: str,
	chunk_size: int,
	chunk_overlap: int,
	model_name: str,
) -> None:
	# Step 1: Pull source documents into a database
	count = ingest_files_to_db(source_dir=source_dir, db_path=db_path)
	print(f"Ingested {count} files into SQLite at {db_path}")

	# Step 2: Pull data into RAG pipeline by first chunking them
	documents = fetch_all_documents(db_path)
	print(f"Fetched {len(documents)} raw documents from DB")
	chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
	print(f"Created {len(chunks)} chunks")

	# Step 3: Embed and save your documents in a vector database
	embeddings_model = get_embeddings(model_name=model_name)

	vs = build_vectorstore(chunks, embeddings_model)
	save_vectorstore(vs, vectorstore_path)
	print(f"Saved FAISS vectorstore to {vectorstore_path}")


def run_query(vectorstore_path: str, query: str, model_name: str, k: int = 4) -> None:
	embeddings_model = get_embeddings(model_name=model_name)
	vs = load_vectorstore(vectorstore_path, embeddings_model)
	retriever = get_retriever(vs, k=k)
	docs = retriever.invoke(query)
	print("Top Documents:")
	for i, d in enumerate(docs, 1):
		meta = d.metadata or {}
		print(f"[{i}] {meta.get('source', 'unknown')} :: {meta.get('filename', '')}")
		print(d.page_content[:500])
		print("-" * 80)


def main() -> None:
	parser = argparse.ArgumentParser(description="RAG Pipeline Demo")
	parser.add_argument("--source_dir", type=str, default="./data", help="Folder of input files")
	parser.add_argument("--db_path", type=str, default="./rag_raw_docs.sqlite", help="SQLite DB path")
	parser.add_argument("--vectorstore_path", type=str, default="./faiss_store", help="Path to save/load FAISS")
	parser.add_argument("--chunk_size", type=int, default=1000)
	parser.add_argument("--chunk_overlap", type=int, default=200)
	parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
	parser.add_argument("--query", type=str, default=None, help="Run a query against the saved vectorstore")
	parser.add_argument("--k", type=int, default=4, help="Top-K documents to retrieve")

	args = parser.parse_args()

	if args.query:
		run_query(args.vectorstore_path, args.query, args.model_name, args.k)
	else:
		run_build(
			source_dir=args.source_dir,
			db_path=args.db_path,
			vectorstore_path=args.vectorstore_path,
			chunk_size=args.chunk_size,
			chunk_overlap=args.chunk_overlap,
			model_name=args.model_name,
		)


if __name__ == "__main__":
	main()
