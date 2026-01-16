"""
Question Enricher Agent
Обогащает вопрос пользователя контекстом из истории диалога
Заменяет местоимения и ссылки на конкретные сущности
"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from src.rag.llm import get_model
from src.config import LLM_API_KEY, LLM


class EnrichedQuestion(BaseModel):
    enriched_question: str = Field(
        ..., 
        min_length=1, 
        description="Обогащенный вопрос с заменой местоимений и добавлением контекста"
    )


class QuestionEnricher:
    """
    Агент для обогащения вопросов контекстом из истории диалога.
    Заменяет местоимения (он, она, это, там) и неполные ссылки на конкретные сущности.
    """
    
    def __init__(self):
        self.llm = get_model(LLM_API_KEY, LLM)
        self.parser = JsonOutputParser(pydantic_object=EnrichedQuestion)
        
        self.prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(
                "Ты помощник, который обогащает вопросы пользователя контекстом из истории диалога.\n"
                "Твоя задача:\n"
                "1. Заменить местоимения (он, она, оно, они, это, то, там, тогда и т.д.) на конкретные сущности из истории общения с пользователем\n"
                "2. Дополнить неполные вопросы (например, 'А вчера?' -> 'Какой был курс доллара вчера?')\n"
                "3. Сделать вопрос самодостаточным и понятным без контекста истории\n"
                "4. Сохранить смысл и намерение пользователя\n\n"
                "Если вопрос уже полный и не требует обогащения, верни его без изменений.\n"
                "Если не получается понять, как правильно обогатить какую-то часть вопроса, то ее следует оставить неизмененной."
            ),
            HumanMessagePromptTemplate.from_template(
                "{format_instructions}\n\n"
                "История диалога:\n{history}\n\n"
                "Новый вопрос пользователя: {question}\n\n"
                "Обогати вопрос контекстом из истории."
            )
        ])
        
        self.chain = self.prompt | self.llm | self.parser
    
    def _format_history(self, history: List[Dict]) -> str:
        """Format chat history for the prompt"""
        if not history:
            return "История диалога пуста."
        
        history_text = ""
        for i, msg in enumerate(history, 1):
            history_text += f"[{i}] Пользователь: {msg.get('query', '')}\n"
            history_text += f"    Ответ: {msg.get('answer', '')}\n\n"
        
        return history_text.strip()
    
    def enrich(self, question: str, history: Optional[List[Dict]] = None) -> Dict[str, str]:
        """
        Enrich question with context from history
        
        Args:
            question: Original user question
            history: List of previous messages [{"query": "...", "answer": "..."}, ...]
            
        Returns:
            Dict with enriched_question and explanation
        """
        # If no history, return original question
        if not history or len(history) == 0:
            return question
        
        try:
            # Format history
            history_text = self._format_history(history)
            
            # Invoke chain
            result = self.chain.invoke({
                "history": history_text,
                "question": question,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            return result.get("enriched_question", question)
            
        except Exception as e:
            # On any other error, return original question
            return question
