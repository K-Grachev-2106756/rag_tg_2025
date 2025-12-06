from typing import Literal

import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter, NLTKTextSplitter
from langchain_experimental.text_splitter import SemanticChunker


class EmbeddingWrapper:
    """
    Класс описывает кастомизацию langchain_core.embeddings.Embeddings 
        для использования внутри SemanticChunker
    """
    
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(self.device).eval()
    

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> torch.Tensor:
        """
        Функция для получения векторных представлений текстов
        """
        all_embeds = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                emb = self.model.encode(
                    batch, 
                    convert_to_tensor=True, 
                    normalize_embeddings=True,
                )
                all_embeds.append(emb.cpu())
            torch.cuda.empty_cache()
        
        return torch.cat(all_embeds, dim=0)


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
                    EmbeddingWrapper(model_name), 
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
