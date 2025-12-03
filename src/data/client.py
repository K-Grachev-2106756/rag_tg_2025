from data.pyrosource import PyroSource


import os
from dotenv import load_dotenv


load_dotenv()
API_ID = os.environ.get("TELEGRAM_API_ID")
API_HASH = os.environ.get("TELEGRAM_API_HASH")


pyro_source = PyroSource(api_id=API_ID, api_hash=API_HASH)
