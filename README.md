## RAG Demo (LangChain + SQLite + FAISS)

This project demonstrates a simple RAG pipeline with self-contained functions:

- Pull source documents into a SQLite database
- Chunk documents
- Embed documents with HuggingFace
- Save and load vectors using FAISS (local vector database)

### Requirements

- Python 3.10+
- Install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Project Layout

- `rag_pipeline/db.py`: SQLite initialization and raw document storage
- `rag_pipeline/ingest.py`: Ingest files from a directory into SQLite
- `rag_pipeline/pipeline.py`: Chunking, embedding, and FAISS vector store utilities
- `rag_pipeline/main.py`: Example script wiring all steps together

### Usage

1) Prepare a folder with your documents (e.g., `.txt`, `.md`, `.pdf`).

2) Run the pipeline example:

```bash
python -m rag_pipeline.main \
  --source_dir ./data \
  --db_path ./rag_raw_docs.sqlite \
  --vectorstore_path ./faiss_store \
  --chunk_size 1000 \
  --chunk_overlap 200
```

3) Example query against the saved vectorstore:

```bash
python -m rag_pipeline.main --query "What does this corpus talk about?" --vectorstore_path ./faiss_store
```

Notes:
- Embeddings use `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace, no API key required.
- FAISS is saved locally to `--vectorstore_path` so you can reuse it without re-ingesting.

### Extending

- Swap FAISS for another LangChain-supported vector DB.
- Replace the embeddings model with your preferred model.
- Wire the retriever into your favorite LLM for a full RAG QA chain.
