"""
med_query_chain.py

Модуль для интерактивного поиска и анализа медицинских признаков по диагнозу
с помощью индекса llama-index и локальной LLM через LangChain.

Цепочка включает два шага:
1. Поиск симптомов по диагнозу (через RetrievalQA).
2. Классификация каждого найденного признака (симптом/анализ/медицинский показатель).

Ввод диагнозов — в бесконечном цикле, выход по пустой строке.
Для каждого шага и итога выводится время выполнения (мин:сек.миллисек).

Требует:
- настроенного индекса (llama-index)
- OpenAI-совместимой LLM (например, Llama3-Med42-8B через LM Studio)
- settings.py с переменными PERSISTED_INDEX_DIR, llama3_med42_llm
- utils.retriever_adapter и utils.formatting
"""

from typing import Any
from time import perf_counter

from llama_index.core import StorageContext, load_index_from_storage
from langchain.chains import RetrievalQA
from settings import PERSISTED_INDEX_DIR, default_llm
from utils.retriever_adapter import LlamaRetrieverForLangChain
from utils.formatting import format_time, display_quotes


def get_symptoms_for_diagnosis(
    diagnosis: str,
    qa_chain: RetrievalQA,
    verbose: bool = False
) -> str:
    """
    Получает список симптомов для заданного диагноза через RetrievalQA-цепочку.

    Args:
        diagnosis (str): Название диагноза.
        qa_chain (RetrievalQA): RetrievalQA-цепочка для поиска.
        verbose (bool): Если True — выводит найденные источники.

    Returns:
        str: Список симптомов, найденный по диагнозу (обычно нумерованный).
    """
    question = f"What are the symptoms of {diagnosis}? Give out only numbered symptoms"
    result = qa_chain.invoke({"query": question})
    print(f"\nСимптомы для диагноза '{diagnosis}':\n{result['result']}\n")
    if verbose:
        display_quotes(result)
    return result['result']


def analyze_feature_types(
    diagnosis: str,
    featurelist: str,
    llm: Any
) -> str:
    """
    Классифицирует признаки как "symptom", "analysis" или "medical indicator" с помощью LLM.

    Args:
        diagnosis (str): Название диагноза.
        featurelist (str): Список признаков (обычно из get_symptoms_for_diagnosis).
        llm (Any): LLM с методом invoke(prompt: str).

    Returns:
        str: Список пар "признак - тип" (в виде текста).
    """
    prompt = (
        f"Determine the type of {diagnosis} features, select for each of the feature from featurelist, "
        f"is it a symptom, analysis or medical indicator. "
        f"Give the answer in the form of a pair feature - type\n\n{featurelist}"
    )
    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    print(f"\nКлассификация признаков для '{diagnosis}':\n{text}\n")
    return text


def main() -> None:
    """
    Главная функция модуля.
    Загружает индекс, инициализирует retriever и цепочку,
    запускает интерактивный цикл анализа диагноза:
    - Запрашивает диагноз у пользователя.
    - Поиск симптомов через RetrievalQA.
    - Классификация признаков через LLM.
    - Выводит время каждого этапа и общее время.
    - Завершает работу при вводе пустой строки.
    """
    # Загружаем индекс и создаём retriever
    storage_context: StorageContext = StorageContext.from_defaults(persist_dir=PERSISTED_INDEX_DIR)
    loaded_index = load_index_from_storage(storage_context)
    llama_retriever = loaded_index.as_retriever(similarity_top_k=3)
    retriever = LlamaRetrieverForLangChain(llama_retriever=llama_retriever)

    # Готовим RetrievalQA chain для поиска по медицинским источникам
    qa: RetrievalQA = RetrievalQA.from_chain_type(
        llm=default_llm,
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

    print("Введите диагноз (пустая строка — выход):")
    while True:
        diagnosis: str = input("\nДиагноз: ").strip()
        if not diagnosis:
            print("Выход.")
            break

        # 1. Находим список симптомов по диагнозу, замер времени
        t1 = perf_counter()
        symptom_list: str = get_symptoms_for_diagnosis(diagnosis, qa, verbose=True)
        t2 = perf_counter()
        print(f"Время поиска симптомов: {format_time(t2 - t1)}")

        # 2. Классифицируем признаки, замер времени
        t3 = perf_counter()
        analyze_feature_types(diagnosis, symptom_list, default_llm)
        t4 = perf_counter()
        print(f"Время классификации признаков: {format_time(t4 - t3)}")

        print(f"Общее время: {format_time(t4 - t1)}")


if __name__ == "__main__":
    main()
