"""
summary.py

Генерация краткого поискового summary (4–10 слов) для каждого medical chunk через LLM.
"""

from typing import List, Dict, Any
from langchain_core.language_models.base import BaseLanguageModel
from settings import default_llm


def extract_chunk_summary(
        chunk_text: str,
        llm: BaseLanguageModel = default_llm,
        max_words: int = 30,
) -> str:
    """
    Генерирует короткое search-focused summary (4-17 слов) для текста чанка.
    """
    prompt = (
        "Below is a fragment from a medical document.\n"
        f"Summarize its main topic in 4 to {max_words} words (maximum 1 sentence, for search and indexing).\n"
        "Be specific, avoid generic words, do not start with 'This chunk...' or 'The document...'.\n"
        f"\nFragment:\n{chunk_text[:1700]}\n"
        "\nSummary:"
    )
    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        summary = text.strip().replace("\n", " ")
        # Удаляем точки в конце, если есть
        if summary.endswith('.'):
            summary = summary[:-1].strip()
        # Обрезаем если вдруг LLM вернул слишком длинно
        words = summary.split()
        if len(words) > max_words:
            summary = " ".join(words[:max_words])
        return summary
    except Exception as e:
        print(f"[generate_chunk_summary] LLM error: {e}")
        return ""


# --- Тестовый запуск ---
if __name__ == "__main__":
    import os
    import glob
    from settings import default_tokens_counter, openai_llm, openai_tokens_counter, DEFAULT_MODEL, MODEL

    print("\nСравнительный тест по llm:")
    print("-" * 100, "\n")
    test_dir = "../../data/test_chunks"
    for file_path in sorted(glob.glob(os.path.join(test_dir, "*.txt"))):
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as file:
            example = file.read()

        print(f"Default llm ({DEFAULT_MODEL}):")
        num_tokens = default_tokens_counter(example)
        print(f"Количество токенов в '{file_name}': {num_tokens}")
        diseases = extract_chunk_summary(example, max_words=17)
        print("Diseases found:", diseases)

        print(f"\nOpenAI llm ({MODEL}):")
        num_tokens = openai_tokens_counter(example)
        print(f"Количество токенов в '{file_name}': {num_tokens}")
        diseases = extract_chunk_summary(example, llm=openai_llm, max_words=17)
        print("Diseases found:", diseases)
        print("-" * 100)

