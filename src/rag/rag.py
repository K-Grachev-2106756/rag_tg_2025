from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from src.rag.retriever import Retriever
from src.rag.llm import get_model
from src.rag.question_enricher import QuestionEnricher
from src.config import LLM_API_KEY, LLM, CHAT_HISTORY_LENGTH, ENABLE_QUESTION_ENRICHMENT


class LLMResponse(BaseModel):
    answer: str = Field(..., min_length=1, description="Прямой точный ответ на вопрос")
    reason: str = Field(..., min_length=1, description="Объяснение, почему ответ именно такой")


class RAG:

    def __init__(self, embed_model_name: str, embed_index_name: str):
        self.retriever = Retriever(embed_model_name, embed_index_name)
        self.parser = JsonOutputParser(pydantic_object=LLMResponse)
        self.llm = get_model(LLM_API_KEY, LLM)
        self.history_length = CHAT_HISTORY_LENGTH
        self.enable_enrichment = ENABLE_QUESTION_ENRICHMENT
        
        # Initialize question enricher if enabled
        if self.enable_enrichment:
            self.question_enricher = QuestionEnricher()
        else:
            self.question_enricher = None

    def invoke(self, query: str, history: Optional[List[Dict]] = None):
        """
        Invoke RAG with optional chat history
        
        Args:
            query: User question
            history: List of previous messages [{"query": "...", "answer": "..."}, ...]
        """
        try:
            # Enrich question with context from history if enabled
            enriched_query = query
            
            if self.enable_enrichment and self.question_enricher and history:
                # Use last N messages for enrichment
                recent_history = history[-self.history_length:] if len(history) > self.history_length else history
                enriched_query = self.question_enricher.enrich(query, recent_history)
            
            # Get context from retriever using enriched query
            context = self.retriever.chain.invoke(enriched_query)
            
            # Build prompt without history (enriched question already contains context)
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "Ты полезный и точный ассистент. "
                    "Ответь на вопрос, опираясь ТОЛЬКО на предложенный контекст. "
                    "Если в контексте нет ответа, ответь \"Не знаю.\""
                ),
                HumanMessagePromptTemplate.from_template(
                    "{format_instructions}\n\n"
                    "Контекст:\n{context}\n\n"
                    "Вопрос: {question}"
                ),
            ])
            
            # Build chain
            chain = (
                prompt
                | self.llm
                | self.parser
            )
            
            # Invoke with enriched question
            result = chain.invoke({
                "context": context,
                "question": enriched_query,  # Use enriched question
                "format_instructions": self.parser.get_format_instructions(),
            })
            
            return result
            
        except (OutputParserException, ValidationError) as e:
            return LLMResponse(
                answer="Не знаю.",
                reason="Модель не смогла вернуть ответ в корректном формате."
            )

