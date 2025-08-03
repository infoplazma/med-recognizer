"""
linked_diseases.py

Определение связанных заболеваний между двумя соседними медицинскими чанками.
Использует LLM для оценки, продолжается ли обсуждение заболевания.
"""
import json
from typing import List, Dict
from langchain_core.language_models.base import BaseLanguageModel
from settings import default_llm
from utils.json_response import clean_json_response


def find_linked_diseases_between_chunks(
        prev_text: str,
        next_text: str,
        llm: BaseLanguageModel = default_llm,
        max_keys: int = 5
) -> List[str]:
    """
    Определяет, обсуждается ли заболевание из текущего чанка в следующем чанке (есть ли смысловая связь).

    Args:
        prev_text (str): Текст текущего чанка.
        next_text (str): Текст следующего чанка.
        llm (BaseLanguageModel): LLM для анализа контекста.
        max_keys (int): Максимальное количество диагнозов для анализа.

    Returns:
        List[str]: Список диагнозов, по которым обсуждение связано между чанками.
    """
    prompt = (
            "Given two adjacent fragments from a medical document:\n"
            "1. List all diseases/diagnoses discussed in Fragment 1.\n"
            "2. For each, check if Fragment 2 contains any additional, new, or clarifying information about the same disease/diagnosis.\n"
            "If so, return a JSON array with those names. If none, return []. Only return names that are clearly referenced in Fragment 2."
            "\n---\nFragment 1:\n" + prev_text[-300:] +
            "\n---\nFragment 2:\n" + next_text[:300]
    )

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        diseases_raw = content.strip()
        # print(f"{diseases_raw=}")
        diseases = json.loads(clean_json_response(diseases_raw))
        return diseases
    except Exception as e:
        print(f"[extract_diseases] LLM error: {e}")
        return []


# --- Тестовый запуск ---
if __name__ == "__main__":
    import os
    import glob
    from settings import default_tokens_counter, openai_llm, openai_tokens_counter, DEFAULT_MODEL, MODEL

    print("\nСравнительный тест по llm:")
    print("-" * 100, "\n")
    test_dir = "../../data/test_chunks"
    for i, file_path in enumerate(sorted(glob.glob(os.path.join(test_dir, "*.txt")))):
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as file:
            curr_example = file.read()

        if i == 0:
            prev_example = curr_example
            prev_file_name = file_name
            continue

        print(f"\n=== {prev_file_name} {file_name} ===")

        print(f"Default llm ({DEFAULT_MODEL}):")
        prev_num_tokens = default_tokens_counter(prev_example)
        curr_num_tokens = default_tokens_counter(curr_example)
        print(f"Количество пред. и тек. токенов в '{file_name}': {prev_num_tokens}  {curr_num_tokens}")
        diseases = find_linked_diseases_between_chunks(prev_example, curr_example)
        print("Diseases found:", diseases)

        print(f"\nOpenAI llm ({MODEL}):")
        prev_num_tokens = openai_tokens_counter(prev_example)
        curr_num_tokens = openai_tokens_counter(curr_example)
        print(f"Количество пред. и тек. токенов в '{file_name}': {prev_num_tokens}  {curr_num_tokens}")
        diseases = find_linked_diseases_between_chunks(prev_example, curr_example, llm=openai_llm)
        print("Diseases found:", diseases)
        print("-" * 100)

        prev_example = curr_example
        prev_file_name = file_name
