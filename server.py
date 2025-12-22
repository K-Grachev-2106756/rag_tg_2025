from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src import RAG
from src.db_utils.history_utils import (
    init_history_table,
    log_query,
    get_all_history,
    get_history_by_dialogue,
    search_history,
    get_history_stats,
    delete_history,
    get_recent_dialogues
)


# --- Lifespan для инициализации при старте ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: инициализация таблицы истории
    try:
        init_history_table()
    except Exception as e:
        print(f"⚠️ Не удалось инициализировать таблицу истории: {e}")
    yield
    # Shutdown: ничего не делаем


app = FastAPI(
    title="RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Инициализация RAG один раз при старте ---
rag = RAG(
    embed_model_name="deepvk/USER-bge-m3",
    embed_index_name="recursive_USER-bge-m3",
)


# --- Request / Response схемы ---

class QueryRequest(BaseModel):
    query: str
    dialogue_id: Optional[str] = None  # Для будущей поддержки диалогов


class QueryResponse(BaseModel):
    answer: str
    reason: str
    query_id: Optional[int] = None  # ID записи в истории


class HistoryEntry(BaseModel):
    id: int
    timestamp: str
    dialogue_id: str
    query: str
    answer: str
    reason: Optional[str] = None
    search_period: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class HistoryStats(BaseModel):
    total_queries: int
    unique_dialogues: int
    last_query_time: Optional[str] = None
    first_query_time: Optional[str] = None


class DialogueInfo(BaseModel):
    dialogue_id: str
    message_count: int
    started_at: Optional[str] = None
    last_message_at: Optional[str] = None


# --- RAG Endpoint ---

@app.post("/rag", response_model=QueryResponse)
def rag_query(request: QueryRequest):
    """Основной endpoint для запросов к RAG. Логирует запрос после получения ответа."""

    # Получаем ответ от RAG
    result = rag.invoke(request.query)
    
    # Логируем в историю
    query_id = log_query(
        query=request.query,
        answer=result.get("answer", ""),
        reason=result.get("reason", ""),
        dialogue_id=request.dialogue_id
    )
    
    return QueryResponse(
        answer=result.get("answer", ""),
        reason=result.get("reason", ""),
        query_id=query_id
    )
    # except Exception as e:
    #     raise HTTPException(
    #         status_code=500,
    #         detail=str(e)
    #     )


# --- History Endpoints ---

@app.get("/history", response_model=List[HistoryEntry])
def get_history(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    """Получить историю запросов"""
    return get_all_history(limit=limit, offset=offset)


@app.get("/history/stats", response_model=HistoryStats)
def get_stats():
    """Получить статистику по истории"""
    stats = get_history_stats()
    return HistoryStats(
        total_queries=stats.get("total_queries", 0),
        unique_dialogues=stats.get("unique_dialogues", 0),
        last_query_time=stats.get("last_query_time"),
        first_query_time=stats.get("first_query_time")
    )


@app.get("/history/search", response_model=List[HistoryEntry])
def search_in_history(
    q: str = Query(..., min_length=1, description="Текст для поиска"),
    limit: int = Query(default=50, ge=1, le=500)
):
    """Поиск по истории запросов"""
    return search_history(search_text=q, limit=limit)


@app.get("/history/dialogues", response_model=List[DialogueInfo])
def get_dialogues(
    limit: int = Query(default=10, ge=1, le=100)
):
    """Получить список последних диалогов"""
    return get_recent_dialogues(limit=limit)


@app.get("/history/dialogue/{dialogue_id}", response_model=List[HistoryEntry])
def get_dialogue(dialogue_id: str):
    """Получить историю конкретного диалога"""
    return get_history_by_dialogue(dialogue_id)


@app.delete("/history")
def clear_history(dialogue_id: Optional[str] = None):
    """Удалить историю (всю или конкретного диалога)"""
    try:
        delete_history(dialogue_id=dialogue_id)
        if dialogue_id:
            return {"message": f"История диалога {dialogue_id} удалена"}
        return {"message": "Вся история удалена"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
