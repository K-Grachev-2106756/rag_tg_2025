from typing import Hashable, Optional

import pandas as pd
from sqlalchemy import text

from src.db_utils.db_clients import sql_client


def sql_drop(table: str):
    try:
        with sql_client.begin() as conn:
            conn.execute(text(f"drop table if exists {table};"))
    except Exception as e:
        print("Ошибка:", e)


def sql_dump_df(df: pd.DataFrame, table: str) -> Optional[int]:
    return df.to_sql(table, sql_client, if_exists="replace", index=False)


def sql_get_table(table: str) -> pd.DataFrame:
    with sql_client.connect() as conn:
        df = pd.read_sql(f"""select * from {table}""", conn)

    return df


def sql_get_by_id(id_: Hashable) -> dict:
    with sql_client.connect() as conn:
        row = (
            conn.execute(
                text("SELECT * FROM posts WHERE ctid = :id"),
                {"id": id_},
            )
            .mappings()
            .first()
        )

    return row


def sql_get_by_ids(ids_: Hashable) -> list[dict]:
    with sql_client.connect() as conn:
        rows = (
            conn.execute(
                text("SELECT * FROM posts WHERE ctid = ANY(:ids)"),
                {"ids": ids_},
            )
            .mappings()
            .all()
        )

    return rows


def sql_fetch_batch(batch_size: int = 16, offset: int = 0):
    query = text("""
        SELECT ctid, content
        FROM posts
        ORDER BY ctid
        LIMIT :limit
        OFFSET :offset
    """)

    with sql_client.connect() as conn:
        rows = conn.execute(query, {"limit": batch_size, "offset": offset}).mappings().all()

    return rows
