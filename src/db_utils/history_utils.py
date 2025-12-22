"""
Утилиты для работы с историей запросов в PostgreSQL
Используется на бэкенде для логирования запросов к RAG
"""
from datetime import datetime
from typing import List, Dict, Optional
import json

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.config import sql_client


def init_history_table():
    """
    Инициализация таблицы истории запросов
    Создает таблицу, если она не существует
    """
    try:
        with sql_client.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS query_history (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                    dialogue_id VARCHAR(255) NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    reason TEXT,
                    search_period JSONB,
                    metadata JSONB
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_query_history_dialogue_id 
                ON query_history(dialogue_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_query_history_timestamp 
                ON query_history(timestamp DESC)
            """))
        print("✅ Таблица query_history инициализирована")
    except SQLAlchemyError as e:
        print(f"❌ Ошибка при инициализации таблицы: {e}")
        raise


def log_query(
    query: str,
    answer: str,
    reason: str,
    dialogue_id: Optional[str] = None,
    search_period: Optional[Dict] = None,
    metadata_: Optional[Dict] = None
) -> Optional[int]:
    """
    Логировать запрос в историю (вызывается бэкендом после получения ответа от LLM)
    
    Args:
        query: Текст вопроса пользователя
        answer: Ответ системы
        reason: Обоснование ответа
        dialogue_id: ID диалога (опционально)
        search_period: Период поиска
        metadata_: Дополнительные метаданные
        
    Returns:
        ID созданной записи или None при ошибке
    """
    # Генерируем dialogue_id если не передан
    if not dialogue_id:
        dialogue_id = f"single_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    try:
        with sql_client.begin() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO query_history 
                    (timestamp, dialogue_id, query, answer, reason, search_period, metadata)
                    VALUES (:timestamp, :dialogue_id, :query, :answer, :reason, 
                            CAST(:search_period AS JSONB), CAST(:metadata AS JSONB))
                    RETURNING id
                """),
                {
                    "timestamp": datetime.now(),
                    "dialogue_id": dialogue_id,
                    "query": query,
                    "answer": answer,
                    "reason": reason,
                    "search_period": json.dumps(search_period or {}),
                    "metadata": json.dumps(metadata_ or {})
                }
            )
            query_id = result.scalar()
            return query_id
    except SQLAlchemyError as e:
        print(f"❌ Ошибка при логировании запроса: {e}")
        return None


def get_all_history(limit: int = 100, offset: int = 0) -> List[Dict]:
    """Получить всю историю запросов"""
    try:
        with sql_client.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT id, timestamp, dialogue_id, query, answer, reason, 
                           search_period, metadata
                    FROM query_history
                    ORDER BY timestamp DESC
                    LIMIT :limit OFFSET :offset
                """),
                {"limit": limit, "offset": offset}
            )
            
            rows = result.mappings().all()
            # Конвертируем datetime в ISO строку для JSON сериализации
            return [
                {
                    **dict(row),
                    "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None
                }
                for row in rows
            ]
    except SQLAlchemyError as e:
        print(f"❌ Ошибка при получении истории: {e}")
        return []


def get_history_by_dialogue(dialogue_id: str) -> List[Dict]:
    """Получить историю конкретного диалога"""
    try:
        with sql_client.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT id, timestamp, dialogue_id, query, answer, reason, 
                           search_period, metadata
                    FROM query_history
                    WHERE dialogue_id = :dialogue_id
                    ORDER BY timestamp ASC
                """),
                {"dialogue_id": dialogue_id}
            )
            
            rows = result.mappings().all()
            return [
                {
                    **dict(row),
                    "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None
                }
                for row in rows
            ]
    except SQLAlchemyError as e:
        print(f"❌ Ошибка при получении диалога: {e}")
        return []


def search_history(search_text: str, limit: int = 50) -> List[Dict]:
    """Поиск по истории запросов"""
    try:
        with sql_client.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT id, timestamp, dialogue_id, query, answer, reason, 
                           search_period, metadata
                    FROM query_history
                    WHERE query ILIKE :search_pattern 
                       OR answer ILIKE :search_pattern
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """),
                {
                    "search_pattern": f"%{search_text}%",
                    "limit": limit
                }
            )
            
            rows = result.mappings().all()
            return [
                {
                    **dict(row),
                    "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None
                }
                for row in rows
            ]
    except SQLAlchemyError as e:
        print(f"❌ Ошибка при поиске в истории: {e}")
        return []


def get_history_stats() -> Dict:
    """Получить статистику по истории запросов"""
    try:
        with sql_client.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT 
                        COUNT(*) as total_queries,
                        COUNT(DISTINCT dialogue_id) as unique_dialogues,
                        MAX(timestamp) as last_query_time,
                        MIN(timestamp) as first_query_time
                    FROM query_history
                """)
            )
            
            row = result.mappings().first()
            if row:
                return {
                    "total_queries": row["total_queries"],
                    "unique_dialogues": row["unique_dialogues"],
                    "last_query_time": row["last_query_time"].isoformat() if row["last_query_time"] else None,
                    "first_query_time": row["first_query_time"].isoformat() if row["first_query_time"] else None
                }
            return {}
    except SQLAlchemyError as e:
        print(f"❌ Ошибка при получении статистики: {e}")
        return {}


def delete_history(dialogue_id: Optional[str] = None):
    """Удалить историю"""
    try:
        with sql_client.begin() as conn:
            if dialogue_id:
                conn.execute(
                    text("DELETE FROM query_history WHERE dialogue_id = :dialogue_id"),
                    {"dialogue_id": dialogue_id}
                )
                print(f"✅ История диалога {dialogue_id} удалена")
            else:
                conn.execute(text("DELETE FROM query_history"))
                print("✅ Вся история удалена")
    except SQLAlchemyError as e:
        print(f"❌ Ошибка при удалении истории: {e}")
        raise


def get_recent_dialogues(limit: int = 10) -> List[Dict]:
    """Получить список последних диалогов"""
    try:
        with sql_client.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT 
                        dialogue_id,
                        COUNT(*) as message_count,
                        MIN(timestamp) as started_at,
                        MAX(timestamp) as last_message_at
                    FROM query_history
                    GROUP BY dialogue_id
                    ORDER BY MAX(timestamp) DESC
                    LIMIT :limit
                """),
                {"limit": limit}
            )
            
            rows = result.mappings().all()
            return [
                {
                    "dialogue_id": row["dialogue_id"],
                    "message_count": row["message_count"],
                    "started_at": row["started_at"].isoformat() if row["started_at"] else None,
                    "last_message_at": row["last_message_at"].isoformat() if row["last_message_at"] else None
                }
                for row in rows
            ]
    except SQLAlchemyError as e:
        print(f"❌ Ошибка при получении списка диалогов: {e}")
        return []
