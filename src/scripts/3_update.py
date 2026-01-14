import os
import sys
sys.path.append(os.getcwd())
import time
import datetime

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import pandas as pd

from src.config import pyro_source, CHANNEL_ID
from src.data.clean import clean_df
from src.db_utils.sql_utils import sql_dump_df, sql_get_by_date
from src.db_utils.qdrant_utils import qdrant_insert
from src.data.splitter import Splitter


today = datetime.datetime.today()

# Загрузка в sql
posts = pyro_source.load_days(
    channel_id=CHANNEL_ID,
    from_date=datetime.datetime.today(),
)

df = pd.DataFrame(posts)
df = clean_df(df)

sql_dump_df(df, "posts", if_exists="append")

# Загрузка в qdrant
splitter_mode = "recursive"
model_name = "deepvk/USER-bge-m3"
vector_index_name = f"{splitter_mode}_{model_name.split('/')[1]}"

splitter = Splitter(splitter_mode, chunk_size=256, chunk_overlap=64)
emb = HuggingFaceEmbeddings(
    model_name=model_name, 
    encode_kwargs={"normalize_embeddings": True},
)

batch_size = 16
offset = 0
rows = sql_get_by_date(today.date().isoformat())
for i in range(0, len(rows), batch_size):
    dfs = []
    for r in rows[i:i+batch_size]:
        chunks = splitter.split_text(r["content"])
        vectors = emb.embed_documents(chunks)

        dfs.append(pd.DataFrame({"doc_id": r["ctid"], "text": chunks, "vector": vectors}))

    print(f"{offset} - {offset + batch_size}:", qdrant_insert(pd.concat(dfs), vector_index_name))
    
    offset += batch_size

    time.sleep(0.3)
