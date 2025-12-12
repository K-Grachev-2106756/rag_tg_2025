from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import pandas as pd


class AnswerEvaluation(BaseModel):
    is_valid: bool = Field(
        description="Является ли ответ валидным и корректным относительно вопроса и оригинального текста"
    )
    relevance_score: float = Field(
        description="Оценка релевантности ответа вопросу от 0.0 до 1.0",
        ge=0.0,
        le=1.0
    )
    completeness_score: float = Field(
        description="Оценка полноты ответа от 0.0 до 1.0 (насколько ответ покрывает всю необходимую информацию)",
        ge=0.0,
        le=1.0
    )
    factual_accuracy_score: float = Field(
        description="Оценка фактической точности ответа от 0.0 до 1.0 (соответствие фактам из оригинального текста)",
        ge=0.0,
        le=1.0
    )


class QuestionBatchIterator:
    def __init__(self, questions, batch_size):
        self.questions = questions
        self.batch_size = batch_size
        self.current_idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_idx >= len(self.questions):
            raise StopIteration
        
        batch = self.questions[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        return batch
    
    def __len__(self):
        return (len(self.questions) + self.batch_size - 1) // self.batch_size
    
    def reset(self):
        self.current_idx = 0


class QAEvaluator:
    def __init__(
        self,
        df,
        text_column="original_text",
        model="qwen/qwen3-next-80b-a3b-instruct",
        temperature=0.0,
        api_key=None,
        api_base="https://api.proxyapi.ru/openrouter/v1"
    ):
        self.df = df.copy()
        self.original_text_column = text_column
        self.api_key = api_key
        
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=self.api_key,
            openai_api_base=api_base,
        )
        
        self._setup_evaluation_agent()
        self._current_question_column = None
        self._questions_data = None
    
    def _setup_evaluation_agent(self):
        self.parser = PydanticOutputParser(pydantic_object=AnswerEvaluation)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты - эксперт по оценке качества ответов на вопросы по новостным текстам.

Твоя задача - оценить, насколько ответ корректен и полон относительно заданного вопроса и оригинального текста.

## Критерии оценки:

### is_valid (валидность):
- True: ответ корректно отвечает на вопрос и соответствует фактам из текста
- False: ответ неверный, не по теме, или содержит фактические ошибки

### relevance_score (релевантность, 0.0-1.0):
- 1.0: ответ полностью по теме вопроса
- 0.5: ответ частично по теме
- 0.0: ответ не имеет отношения к вопросу

### completeness_score (полнота, 0.0-1.0):
- 1.0: ответ содержит всю необходимую информацию
- 0.5: ответ содержит часть информации
- 0.0: ответ пустой или не содержит нужной информации

### factual_accuracy_score (фактическая точность, 0.0-1.0):
- 1.0: все факты в ответе соответствуют оригинальному тексту
- 0.5: есть небольшие неточности
- 0.0: факты в ответе противоречат оригинальному тексту

{format_instructions}"""),
            ("human", """Оцени следующий ответ:

## Оригинальный текст поста:
{original_text}

## Вопрос:
{question}

## Ответ для оценки:
{answer}

Проанализируй и выдай оценку.""")
        ])
        
        self.evaluation_chain = self.prompt | self.llm | self.parser
    
    def get_questions(self, question_column, batch_size=10):
        if question_column not in self.df.columns:
            raise ValueError(f"Колонка '{question_column}' не найдена в DataFrame. "
                           f"Доступные колонки: {list(self.df.columns)}")
        
        self._current_question_column = question_column
        
        self._questions_data = []
        for idx, row in self.df.iterrows():
            self._questions_data.append({
                "index": idx,
                "question": row[question_column],
                "original_text": row[self.original_text_column]
            })
        
        questions = [item["question"] for item in self._questions_data]
        
        return QuestionBatchIterator(questions, batch_size)
    
    def evaluate_answers(self, answers, show_progress=True):
        if self._questions_data is None:
            raise ValueError("Сначала вызовите get_questions() для получения вопросов")
        
        if len(answers) != len(self._questions_data):
            raise ValueError(
                f"Количество ответов ({len(answers)}) не совпадает с количеством "
                f"вопросов ({len(self._questions_data)})"
            )
        
        total_questions = len(answers)
        valid_answers = 0
        invalid_answers = 0
        detailed_results = []
        
        relevance_scores = []
        completeness_scores = []
        factual_accuracy_scores = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(
                zip(self._questions_data, answers),
                total=len(answers),
                desc="Оценка ответов"
            )
        else:
            iterator = zip(self._questions_data, answers)
        
        for qa_data, answer in iterator:
            try:
                evaluation = self._evaluate_single_answer(
                    original_text=qa_data["original_text"],
                    question=qa_data["question"],
                    answer=answer
                )
                
                if evaluation.is_valid:
                    valid_answers += 1
                else:
                    invalid_answers += 1
                
                relevance_scores.append(evaluation.relevance_score)
                completeness_scores.append(evaluation.completeness_score)
                factual_accuracy_scores.append(evaluation.factual_accuracy_score)
                
                detailed_results.append({
                    "index": qa_data["index"],
                    "question": qa_data["question"],
                    "answer": answer,
                    "is_valid": evaluation.is_valid,
                    "relevance_score": evaluation.relevance_score,
                    "completeness_score": evaluation.completeness_score,
                    "factual_accuracy_score": evaluation.factual_accuracy_score,
                })
                
            except Exception as e:
                print(f"Ошибка при оценке ответа: {e}")
                invalid_answers += 1
                relevance_scores.append(0.0)
                completeness_scores.append(0.0)
                factual_accuracy_scores.append(0.0)
                
                detailed_results.append({
                    "index": qa_data["index"],
                    "question": qa_data["question"],
                    "answer": answer,
                    "is_valid": False,
                    "relevance_score": 0.0,
                    "completeness_score": 0.0,
                    "factual_accuracy_score": 0.0
                })
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
        avg_factual_accuracy = sum(factual_accuracy_scores) / len(factual_accuracy_scores) if factual_accuracy_scores else 0.0
        
        accuracy = valid_answers / total_questions if total_questions > 0 else 0.0
        combined_score = (avg_relevance + avg_completeness + avg_factual_accuracy) / 3
        
        return {
            "total_questions": total_questions,
            "valid_answers": valid_answers,
            "invalid_answers": invalid_answers,
            "accuracy": accuracy,
            "avg_relevance": avg_relevance,
            "avg_completeness": avg_completeness,
            "avg_factual_accuracy": avg_factual_accuracy,
            "combined_score": combined_score,
            "detailed_results": detailed_results,
        }
    
    def _evaluate_single_answer(self, original_text, question, answer):
        if answer is None or (isinstance(answer, str) and answer.strip() == ""):
            return AnswerEvaluation(
                is_valid=False,
                relevance_score=0.0,
                completeness_score=0.0,
                factual_accuracy_score=0.0,
            )
        
        result = self.evaluation_chain.invoke({
            "original_text": original_text,
            "question": question,
            "answer": answer,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return result
    
    def get_detailed_results_df(self, metrics):
        return pd.DataFrame(metrics["detailed_results"])
