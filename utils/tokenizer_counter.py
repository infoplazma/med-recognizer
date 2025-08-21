from typing import Optional, List
from transformers import AutoTokenizer, PreTrainedTokenizer
import threading
import tiktoken


class Llama3TokenizerCounter:
    """
    Класс для оценки количества токенов в тексте для моделей семейства Llama3 (например, Llama3-Med42-8B).
    Токенизатор загружается только один раз и используется повторно.

    Пример использования:
        counter = Llama3TokenizerCounter("m42-health/Llama3-Med42-8B")
        num_tokens = counter.count_tokens("Ваш медицинский текст здесь...")
        print(f"Количество токенов: {num_tokens}")
    """
    _tokenizer: Optional[PreTrainedTokenizer] = None
    _lock = threading.Lock()

    def __init__(self, model_id_or_path: str = "m42-health/Llama3-Med42-8B") -> None:
        """
        :param model_id_or_path: Имя модели на HuggingFace или путь к локальной папке токенизатора.
        """
        self.model_id_or_path = model_id_or_path

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Внутренний метод для загрузки токенизатора. Выполняется только один раз.
        """
        with self._lock:
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id_or_path)
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """
        Подсчитывает количество токенов в переданном тексте.

        :param text: Строка с текстом для токенизации.
        :return: Количество токенов.
        """
        tokenizer = self._load_tokenizer()
        tokens: List[str] = tokenizer.tokenize(text)
        return len(tokens)

    def tokenize(self, text: str) -> List[str]:
        """
        Токенизирует текст и возвращает список токенов.

        :param text: Исходный текст.
        :return: Список токенов (строк).
        """
        tokenizer = self._load_tokenizer()
        return tokenizer.tokenize(text)

    def __call__(self, text: str) -> int:
        return self.count_tokens(text)


class BioMedGPTTokenizerCounter:
    """
    Класс для оценки количества токенов в тексте для моделей, основанных на Llama2,
    таких как BioMedGPT-LM-7B. Токенизатор загружается один раз и используется повторно.

    Пример использования:
        counter = BioMedGPTTokenizerCounter("PharMolix/BioMedGPT-LM-7B")
        num_tokens = counter.count_tokens("Cardiopulmonary resuscitation (CPR)...")
        print(f"Количество токенов: {num_tokens}")
    """
    _tokenizer: Optional[PreTrainedTokenizer] = None
    _lock = threading.Lock()
    _max_context_length = 2048  # Максимальное контекстное окно для BioMedGPT-LM-7B

    def __init__(self, model_id_or_path: str = "meta-llama/Llama-2-7b-chat-hf") -> None:
        """
        :param model_id_or_path: Имя модели на HuggingFace или путь к локальной папке токенизатора.
        """
        self.model_id_or_path = model_id_or_path

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Внутренний метод для загрузки токенизатора. Выполняется только один раз.

        :raises ValueError: Если токенизатор не удалось загрузить.
        :return: Загруженный токенизатор.
        """
        with self._lock:
            if self._tokenizer is None:
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model_id_or_path,
                        use_auth_token=True  # Для моделей с ограниченным доступом
                    )
                except Exception as e:
                    raise ValueError(f"Не удалось загрузить токенизатор для {self.model_id_or_path}: {str(e)}")
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """
        Подсчитывает количество токенов в переданном тексте.

        :param text: Строка с текстом для токенизации.
        :return: Количество токенов.
        :raises ValueError: Если текст пустой или не является строкой.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Текст должен быть непустой строкой")

        tokenizer = self._load_tokenizer()
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        token_count = len(token_ids)

        if token_count > self._max_context_length:
            print(f"Предупреждение: Количество токенов ({token_count}) превышает "
                  f"максимальное контекстное окно ({self._max_context_length})")

        return token_count

    def tokenize(self, text: str) -> List[str]:
        """
        Токенизирует текст и возвращает список токенов.

        :param text: Исходный текст.
        :return: Список токенов (строк).
        :raises ValueError: Если текст пустой или не является строкой.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Текст должен быть непустой строкой")

        tokenizer = self._load_tokenizer()
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        return [tokenizer.decode([tid]) for tid in token_ids]

    def __call__(self, text: str) -> int:
        """
        Вызывает count_tokens для удобства использования класса как функции.
        """
        return self.count_tokens(text)


def openai_tokens_counter(text: str, model: str = "gpt-4o") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def approximate_tokens_counter(text: str, model: str = "cl100k_base") -> int:
    # Загрузка токенизатора
    tokenizer = tiktoken.get_encoding(model)
    # Подсчет токенов
    tokens = tokenizer.encode(text)
    return len(tokens)


# --- Пример использования ---

if __name__ == "__main__":
    counter = Llama3TokenizerCounter("m42-health/Llama3-Med42-8B")
    text = "Your medical text is here..."
    num_tokens = counter.count_tokens(text)
    print(f"Количество токенов: {num_tokens}")

    # Для получения самих токенов:
    tokens = counter.tokenize(text)
    print("Токены:", tokens)

    num_tokens = approximate_tokens_counter(text)
    print("Приблизительно Токены:", num_tokens)

    gpt_counter = BioMedGPTTokenizerCounter()
    num_tokens = gpt_counter.count_tokens(text)
    print(f"Количество токенов GPT: {num_tokens}")

