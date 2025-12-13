from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.rag.retriever import Retriever
from src.rag.llm import get_model
from src.config import MISTRAL_API_KEY, MISTRAL_MODEL


class LLMResponse(BaseModel):
    answer: str = Field(..., min_length=1, description="Прямой точный ответ на вопрос")
    reason: str = Field(..., min_length=1, description="Объяснение, почему ответ именно такой")


class RAG:

    def __init__(self, embed_model_name: str, embed_index_name: str):
        self.retriever = Retriever(embed_model_name, embed_index_name)
        self.parser = JsonOutputParser(pydantic_object=LLMResponse)
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Ты полезный и точный ассистент. "
                "Ответь на вопрос, опираясь ТОЛЬКО на предложенный контекст. "
                "Если в контексте нет ответа, ответь \"Не знаю.\""
            ),
            HumanMessagePromptTemplate.from_template(
                "{format_instructions}\n\n"
                "Контекст:\n{context}\n\n"
                "Вопрос:{question}"
            ),
        ])
        self.llm = get_model(MISTRAL_API_KEY, MISTRAL_MODEL)

        self.chain = (
            {
                "context": self.retriever.chain,
                "question": RunnablePassthrough(),
                "format_instructions": lambda _: self.parser.get_format_instructions(),
            }
            | self.prompt
            | self.llm
            | self.parser
        )


    def invoke(self, query: str):
        return self.chain.invoke(query)
