from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import RAG

app = FastAPI(
    title="RAG API",
    version="1.0.0",
)

# --- Инициализация RAG один раз при старте ---
rag = RAG(
    embed_model_name="deepvk/USER-bge-m3",
    embed_index_name="recursive_USER-bge-m3",
)

# --- Request / Response схемы ---

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    reason: str


# --- Endpoint ---

@app.post("/rag", response_model=QueryResponse)
def rag_query(request: QueryRequest):
    try:
        result = rag.invoke(request.query)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# --- Healthcheck ---

@app.get("/health")
def health():
    return {"status": "ok"}


# --- Entry point ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
