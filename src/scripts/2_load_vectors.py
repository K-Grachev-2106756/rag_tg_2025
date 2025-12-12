import time

import pandas as pd
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from src.db_utils.sql_utils import sql_fetch_batch
from src.db_utils.qdrant_utils import qdrant_insert, qdrant_create_index
from src.data.splitter import Splitter




if __name__ == "__main__":
    splitter_mode = "recursive"
    model_name = "deepvk/USER-bge-m3"
    vector_index_name = f"{splitter_mode}_{model_name.split('/')[1]}"

    # Инициализация объектов
    splitter = Splitter(splitter_mode, chunk_size=256, chunk_overlap=64)
    emb = HuggingFaceEmbeddings(
        model_name=model_name, 
        encode_kwargs={"normalize_embeddings": True},
    )

    # Создание индекса
    qdrant_create_index(
        index_name=vector_index_name,
        dim=len(emb.embed_documents(["None"])[0]),
        distance="cosine",
    )

    # Загрузка документов батчами
    batch_size = 16
    offset = 0
    while True:
        rows = sql_fetch_batch(batch_size=batch_size, offset=offset)
        if not rows:
            break  # дошли до конца
        
        dfs = []
        for r in rows:
            chunks = splitter.split_text(r["content"])
            vectors = emb.embed_documents(chunks)

            dfs.append(pd.DataFrame({"doc_id": r["ctid"], "text": chunks, "vector": vectors}))
        
        print(f"{offset} - {offset + batch_size}:", qdrant_insert(pd.concat(dfs), vector_index_name))
        
        offset += batch_size

        time.sleep(0.3)
