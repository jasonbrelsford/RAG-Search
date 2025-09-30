import faiss
import os
import json
from sentence_transformers import SentenceTransformer

DATA_DIR = "data/sample_docs"
INDEX_FILE = "faiss.index"

def build_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = []
    for fname in sorted(os.listdir(DATA_DIR)):
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            docs.append({"id": fname, "text": f.read()})
    if not docs:
        raise RuntimeError(f"No documents found in {DATA_DIR}")

    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts)
    dim = embeddings.shape[1]

    # Flat L2 index: simple and fine for demos/small corpora
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open("docs.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    build_index()
    print(f"Index built and saved to {INDEX_FILE}. Docs metadata saved to docs.json.")
