# deterministic_pipeline_modern.py

"""
Детерминированная цепочка для извлечения и верификации диагнозов из текста (современный стиль).
"""

import json
from typing import List
from langchain.prompts import ChatPromptTemplate
from settings import default_llm, openai_llm
from utils.json_response import clean_json_response

LLM = default_llm

# --- Промпт 1: Извлечение диагнозов ---
extract_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a medical analyst. Identify which diseases or diagnoses are being discussed."),
    ("user",
     "Below is a fragment from a medical document.\n"
     "Your task: List the names of diseases or diagnoses that are mentioned or discussed in this fragment. "
     "Your answer must be a JSON array of names, or [] if there are none.\n\nFragment:\n{snippet}\n---")
])

# --- Промпт 2: Верификация ---
verify_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a medical analyst. Determine which of the following names are actual diseases or medical diagnoses."),
    ("user",
     "Your task: Verify which of the names in the list below are in fact used in medical practice as diseases or diagnoses.\n"
     "Return a JSON array of valid names, or [] if none are appropriate.\n\nDisease/diagnosis list:\n{diseases_raw}\n snippet:\n{snippet}\n---")
])

# --- Чистая пайп-цепочка ---
extract_chain = extract_prompt | LLM | (
    lambda x: {"diseases_raw": clean_json_response(x.content if hasattr(x, "content") else str(x))})
verify_chain = verify_prompt | LLM | (
    lambda x: {"diseases_verified": clean_json_response(x.content if hasattr(x, "content") else str(x))})


def pipeline(snippet: str) -> List[str]:
    """
    Последовательный вызов цепочек (pipeline) для извлечения и верификации диагнозов.
    Args:
        snippet (str): Фрагмент медицинского текста.
    Returns:
        List[str]: Финальный JSON-массив верифицированных диагнозов.
    """
    step1 = extract_chain.invoke({"snippet": snippet})
    step2 = verify_chain.invoke({"snippet": snippet, "diseases_raw": step1["diseases_raw"]})
    try:
        diseases_ = json.loads(step2["diseases_verified"])
    except Exception:
        diseases_ = []
    return diseases_


# --- Тест ---
if __name__ == "__main__":
    import os
    import glob

    print("STARTED!")

    test_dir = "../data/test_chunks"
    for file_path in sorted(glob.glob(os.path.join(test_dir, "*.txt"))):
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            print(f"\n--- {file_name} ---")
            diseases_ = pipeline({"snippet": text})
            # Разные варианты вызова pipeline
            # pipeline(text)
            # result = pipeline.invoke({
            #     "snippet": text,
            #     "meta": "any_additional_context"
            # })
            print("Diseases found:", diseases_)
