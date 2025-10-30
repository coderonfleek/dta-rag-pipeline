## RAG Demo (LangChain + SQLite + Chroma)

This project demonstrates a simple RAG pipeline with self-contained functions:

- Pull source documents into a SQLite database
- Chunk documents
- Embed documents with HuggingFace (Only)
- Save and load vectors using Chroma (local vector database)

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
- `rag_pipeline/pipeline.py`: Chunking, embedding, and Chroma vector store utilities
- `rag_pipeline/main.py`: Example script wiring all steps together

### Usage

1) Prepare a folder with your documents (e.g., `.txt`, `.md`, `.pdf`).

2) Run the pipeline example (build and persist Chroma to `--vectorstore_path`):

```bash
python -m rag_pipeline.main \
  --source_dir ./raw_docs \
  --db_path ./rag_raw_docs.sqlite \
  --vectorstore_path ./chroma_store \
  --chunk_size 1000 \
  --chunk_overlap 200 \
  --model_name sentence-transformers/all-MiniLM-L6-v2
```

3) Example query against the saved vectorstore:

```bash
python -m rag_pipeline.main \
  --vectorstore_path ./chroma_store \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --query "What does this corpus talk about?"
```

Notes:
- Embeddings use `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace, no API key required.
- Chroma persistence is handled via a `chromadb.PersistentClient` at `--vectorstore_path`. Use the same path for build and query.
- Ensure the same `--model_name` is used for build and query so embedding spaces match.

### Extending

- Swap Chroma for another LangChain-supported vector DB.
- Replace the embeddings model with your preferred model.
- Wire the retriever into your favorite LLM for a full RAG QA chain.
