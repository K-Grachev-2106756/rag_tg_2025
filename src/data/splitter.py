from typing import Literal

from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter, NLTKTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


class Splitter:
    """
    Класс описывает функционал разделения текста на чанки тремя способами на выбор:
        - рекурсивно разбивая чанки различными разделителями 
            в порядке возрастания "жесткости" их эффекта;
        
        - объединяя выделенные с помощью библиотеки NLTK предложения 
            в чанки определенного размера и с наложением;
        
        - разбивая текст на семантически связанные блоки 
            с помощью векторных представлений текстов;
    """

    def __init__(
            self, 
            mode: Literal["recursive", "nltk", "semantic"], 
            model_name: str = "deepvk/USER-bge-m3",
            chunk_size: int = 256,
            chunk_overlap: int = 64,
            **splitter_kwargs,
        ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        match mode:

            case "recursive":
                self.splitter = RecursiveCharacterTextSplitter(
                    separators=[
                        "\n### ", "\n## ", "\n# ", 
                        "\n\n", "\n", 
                        "!", "?", ". ", ";", ",", ")", " ", "",
                    ],
                    keep_separator="end",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=lambda x: len(self.tokenizer.encode(x, add_special_tokens=False)),
                    **splitter_kwargs,
                )
                self.split_fn = self._recursive_split
            
            case "nltk":
                self.splitter = NLTKTextSplitter(
                    language="russian", 
                    **splitter_kwargs,
                )
                self.split_fn = self._nltk_split
            
            case "semantic":
                self.splitter = SemanticChunker(
                    HuggingFaceEmbeddings(
                        model_name=model_name, 
                        encode_kwargs={"normalize_embeddings": True},
                    ), 
                    **splitter_kwargs,
                )
                self.split_fn = self._semantic_split


    def split_text(self, text: str) -> list[str]:
        """
        Доступная пользователю функция разделения текста на чанки
        """
        return self.split_fn(text)
    

    def _recursive_split(self, text: str) -> list[str]:
        """
        Функция разделения текста на чанки при self.splitter == RecursiveCharacterTextSplitter
        """
        return [
            chunk 
            for chunk in self.splitter.split_text(text)
            if any(ch.isalpha() for ch in set(chunk))
        ]
    
    
    def _nltk_split(self, text: str) -> list[str]:
        """
        Функция разделения текста на чанки при self.splitter == NLTKTextSplitter
        """
        sentences = self.splitter.split_text(text)[0].split("\n\n")
        sent_sizes = [
            len(self.tokenizer.encode(sent, add_special_tokens=False)) 
            for sent in sentences
        ]

        chunks = []
        i, n = 0, len(sentences)
        while i < n:
            cur_len, cur_texts = 0, []

            # --- Собираем строки в чанк ---
            j = i
            while (j < n) and (cur_len + sent_sizes[j] <= self.chunk_size):
                cur_texts.append(sentences[j])
                cur_len += sent_sizes[j]
                j += 1

            chunks.append(cur_texts)

            # --- Сдвигаем окно с overlap ---
            if j >= n:
                break

            # Держим overlap в токенах, но не превышая его
            overlap_len, k = 0, j - 1
            while (k >= i) and (overlap_len + sent_sizes[k] <= self.chunk_overlap):
                overlap_len += sent_sizes[k]
                k -= 1  # идём назад от конца чанка

            # Следующий старт = k+1
            i = k + 1

        return chunks
    

    def _semantic_split(self, text: str) -> list[str]:
        """
        Функция разделения текста на чанки при self.splitter == SemanticChunker
        """
        return self.splitter.split_text(text)
