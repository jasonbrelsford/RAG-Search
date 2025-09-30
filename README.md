# RAG Search Demo

A minimal **Retrieval-Augmented Generation** service built with Python, FastAPI, and FAISS.

It:
- Ingests text files, turns them into embeddings
- Stores them in a vector index (FAISS)
- Exposes a `/search` API that returns the most relevant docs to your query

## Quick Start

```bash
# 1. Clone (after unzipping)
cd rag-search

# 2. Create FAISS index (embeds sample docs)
python src/ingest.py

# 3. Run API (Docker recommended)
docker compose up --build

# 4. Query
curl -X POST http://localhost:8000/search          -H "Content-Type: application/json"          -d '{"question":"What is a vector database?"}'
```

## Project Layout
```
src/
  ingest.py
  search.py
  app.py
data/sample_docs/
Dockerfile
docker-compose.yml
requirements.txt
tests/test_search.py
.github/workflows/ci.yml
```

## Notes
- This starter uses `faiss-cpu` and `sentence-transformers` (model: `all-MiniLM-L6-v2`).
- Distances returned by FAISS (`IndexFlatL2`) are **squared Euclidean** distances (non‑negative floats). Smaller is more similar.
- You can later swap FAISS for `pgvector`/Pinecone/Weaviate if you want a managed vector DB.
