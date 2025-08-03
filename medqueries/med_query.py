"""
med_query.py

Интерактивный модуль для поиска медицинских фактов через индекс llama-index и локальную LLM (LangChain).
Пользователь может ввести вопрос, получить сгенерированный ответ и посмотреть цитируемые источники.
Для каждого запроса показывается время ответа (мин:сек.миллисек). Пустая строка — выход из программы.

Требует:
- индекса (llama-index)
- настроенной LLM (например, Llama3-Med42-8B через LM Studio)
- settings.py с PERSISTED_INDEX_DIR, llama3_med42_llm
- utils.retriever_adapter с классом LlamaRetrieverForLangChain
"""

from typing import Any
from time import perf_counter

from llama_index.core import StorageContext, load_index_from_storage
from langchain.chains import RetrievalQA

from settings import PERSISTED_INDEX_DIR, default_llm
from utils.retriever_adapter import LlamaRetrieverForLangChain
from utils.formatting import format_time, display_quotes


def main() -> None:
    """
    Главная функция. Загружает индекс, создает retriever и RetrievalQA-цепочку,
    запускает интерактивный цикл вопросов-ответов с выводом источников и времени работы.
    """
    # Загружаем индекс и создаём retriever
    storage_context: StorageContext = StorageContext.from_defaults(persist_dir=PERSISTED_INDEX_DIR)
    loaded_index = load_index_from_storage(storage_context)
    llama_retriever = loaded_index.as_retriever(similarity_top_k=7)
    retriever = LlamaRetrieverForLangChain(llama_retriever=llama_retriever)

    qa: RetrievalQA = RetrievalQA.from_chain_type(
        llm=default_llm,
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

    print("Введите медицинский вопрос (пустая строка — выход):")
    while True:
        question: str = input("\nВопрос: ").strip()
        if not question:
            print("Выход.")
            break

        t1 = perf_counter()
        result: Any = qa.invoke({"query": question})
        t2 = perf_counter()

        print("\nОтвет:\n", result["result"])
        print(f"\nВремя ответа: {format_time(t2 - t1)}")
        display_quotes(result)


if __name__ == "__main__":
    main()
