import uuid
from typing import Literal, Any

import pandas as pd
from qdrant_client import models

from src.db_utils.db_clients import qdrant_client


def qdrant_create_index(
        index_name: str, 
        dim: int, 
        distance: Literal["cosine", "euclid", "manhattan"],
    ):
    distance_mode = None
    match distance:
        case "cosine":
            distance_mode = models.Distance.COSINE
        case "euclid":
            distance_mode = models.Distance.EUCLID
        case "manhattan":
            distance_mode = models.Distance.MANHATTAN
        case _:
            return ValueError(distance)
        
    return qdrant_client.create_collection(
        collection_name=index_name,
        vectors_config=models.VectorParams(
            size=dim, 
            distance=distance_mode,
        )
    )


def qdrant_insert(df: pd.DataFrame, index_name: str) -> Any:
    """
    df.columns == ["doc_id", "text", "vector"]
    """
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),  # уникальный id чанка
            vector=list(row.vector),  # вектор чанкa
            payload={
                "doc_id": row.doc_id,  # <--- связь с PostgreSQL
                "text": row.text,
            },
        ) for row in df.itertuples(index=False)
    ]

    return qdrant_client.upsert(collection_name=index_name, points=points)


def qdrant_search(index_name: str, vector: list, limit: int = 5) -> list:
    return qdrant_client.query_points(
        collection_name=index_name,
        query=vector,
        limit=limit,
    )
