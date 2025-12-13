from langchain_core.runnables import RunnableLambda
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from src.db_utils.qdrant_utils import qdrant_search


class Retriever:

    def __init__(self, embed_model_name: str, embed_index_name: str):
        self.embed_model = HuggingFaceEmbeddings(
            model_name=embed_model_name, 
            encode_kwargs={"normalize_embeddings": True},
        )
        self.embed_index_name = embed_index_name

        self.chain = RunnableLambda(self._retrieve)

    
    def _retrieve(self, query: str) -> str:
        docs = qdrant_search(
            self.embed_index_name, 
            self.embed_model.embed_query(query),
        )
        return "\n".join(
            f"{i}) {doc.payload['text']}"
            for i, doc in enumerate(docs.points, 1)
        )
