import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sqlalchemy import create_engine

from src.parser.pyrosource import PyroSource


load_dotenv()

# Парсер
API_ID = os.environ.get("TELEGRAM_API_ID")
API_HASH = os.environ.get("TELEGRAM_API_HASH")

pyro_source = PyroSource(api_id=API_ID, api_hash=API_HASH)

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

# LLM
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
