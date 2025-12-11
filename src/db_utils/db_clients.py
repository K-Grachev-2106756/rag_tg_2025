import os

from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from dotenv import load_dotenv


load_dotenv()

# Sql
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
PATH_TO_CERT = os.getenv("PATH_TO_CERT")

connection_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

sql_client = create_engine(
    connection_str,
    connect_args={
        "sslmode": "verify-full",
        "sslrootcert": PATH_TO_CERT,
        "target_session_attrs": "read-write"
    }
)

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")

qdrant_client = QdrantClient(url=QDRANT_URL)