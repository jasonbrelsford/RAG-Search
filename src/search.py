import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_FILE = "faiss.index"
DOCS_FILE = "docs.json"

# Load model, index, and docs once at import for simplicity
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_FILE)
with open(DOCS_FILE, "r", encoding="utf-8") as f:
    docs = json.load(f)

def search(query: str, k: int = 3):
    """Return top-k nearest docs by L2 distance (smaller = closer)."""
    q_emb = model.encode([query]).astype(np.float32)
    distances, indices = index.search(q_emb, k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        # dist is a non-negative float (squared Euclidean). Lower = more similar.
        results.append({
            "id": docs[idx]["id"],
            "distance": float(dist),
            "text": docs[idx]["text"]
        })
    return results
