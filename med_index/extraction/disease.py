"""
disease.py

Извлечение заболеваний (diseases/diagnoses) из медицинского текста с помощью LLM.
"""
import json
from typing import List
from langchain_core.language_models.base import BaseLanguageModel

# Импортируй свой llm из настроек, если надо, или прокидывай в функцию аргументом
from settings import default_llm
from utils.json_response import clean_json_response


def extract_diseases(
        text: str,
        llm: BaseLanguageModel = default_llm,
        max_length: int = 900
) -> List[str]:
    """
    Извлекает список заболеваний/диагнозов, упоминаемых или обсуждаемых в тексте.

    Args:
        text (str): Текст чанка (лучше < 1000 токенов).
        llm (BaseLanguageModel): LLM-инстанс (по умолчанию — твой llama3_med42_llm).
        max_length (int): Максимальная длина текста для передачи LLM (если чанк большой).

    Returns:
        List[str]: Список заболеваний (названий diagnosis/disease). Пустой список если не найдено.
    """
    # Safety: не превышаем окно модели
    snippet = text[:max_length]

    prompt = (
        "Below is a fragment from a medical document.\n"
        "Extract all diseases or diagnoses mentioned or discussed in the text. "
        "Do NOT include symptoms, signs, or findings. "
        "Return a valid JSON array of disease/diagnosis names. If there are none, return an empty array [].\n\n"
        f"Fragment:\n{snippet}"
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


# --- Для локального теста модуля ---
if __name__ == "__main__":
    import os
    import glob
    from settings import default_tokens_counter, openai_llm, openai_tokens_counter, DEFAULT_MODEL, MODEL

    print("\nСравнительный тест по llm:")
    print("-" * 100)
    test_dir = "../../data/test_chunks"
    for file_path in sorted(glob.glob(os.path.join(test_dir, "*.txt"))):
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as file:
            example = file.read()
            print(f"=== {file_name} ===\n")

        print(f"Default llm ({DEFAULT_MODEL}):")
        num_tokens = default_tokens_counter(example)
        print(f"Количество токенов: {num_tokens}")
        diseases = extract_diseases(example, max_length=2000)
        print("Diseases found:", diseases)

        print(f"\nOpenAI llm ({MODEL}):")
        num_tokens = openai_tokens_counter(example)
        print(f"Количество токенов: {num_tokens}")
        diseases = extract_diseases(example, llm=openai_llm, max_length=9000)
        print("Diseases found:", diseases)
        print("-"*100)
