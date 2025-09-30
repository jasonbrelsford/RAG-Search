from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Dict
from src.search import search as vector_search

app = FastAPI(title="RAG Search Demo")

class Query(BaseModel):
    question: str
    k: int = 3

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}

@app.post("/search")
def do_search(q: Query):
    try:
        results = vector_search(q.question, k=q.k)
        return {"results": results}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Missing index or docs: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
