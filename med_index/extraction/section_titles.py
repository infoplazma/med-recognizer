"""
section_titles.py

Модуль для извлечения и привязки актуальных section titles (в первую очередь — названия заболеваний или диагнозов)
к каждому чанку медицинского текста. Поддерживает два режима: regexp/эвристика и LLM.
Фокус на disease-centric структурировании данных для последующей индексации и explainability.

Функции:
    - extract_section_titles: Быстрое извлечение кандидатов на заголовки по тексту документа (regexp).
    - extract_section_titles_llm: Извлекает disease titles с помощью LLM (по всему тексту или среди кандидатов).
    - get_active_section_titles: Для заданного чанка возвращает disease titles, встречающиеся в его тексте.
"""

import re
from typing import Dict, List, Optional
from langchain_core.language_models.base import BaseLanguageModel


def extract_section_titles(text: str) -> Dict[str, int]:
    """
    Извлекает кандидаты на section titles (названия диагнозов/заболеваний) по тексту документа с помощью regexp.

    Заголовок: строка, начинается с заглавной буквы, не содержит спецсимволов (кроме - / ( ) '), запятых и точек,
    допускается только * в конце. Для каждого найденного названия сохраняется его индекс первого вхождения.

    Args:
        text (str): Весь текст документа или чанка.

    Returns:
        Dict[str, int]: Словарь {title: position_in_text, ...} (title уже очищен, без спецсимволов).
    """
    # Паттерн: только разрешённые символы и опционально * в конце
    pattern = r"^[A-ZА-Я][A-Za-zА-Яа-я0-9 \-()/']{2,79}\*?$"
    titles: Dict[str, int] = dict()
    for m in re.findall(pattern, text, flags=re.MULTILINE):
        title = m.strip()
        # Удаляем * только если в конце
        if title.endswith("*"):
            title = title[:-1].strip()
        if title and len(title) > 2 and not title.islower():
            pos = text.find(title)
            # Сохраняем только первое вхождение
            if pos != -1 and title not in titles:
                titles[title] = pos
    return titles


def filter_section_titles_llm(
    text: str,
    llm: BaseLanguageModel,
    candidate_titles: Dict[str, int],
    window: int = 100
) -> Dict[str, int]:
    """
    Фильтрует кандидатные disease/diagnosis section titles с помощью LLM.
    Для каждого кандидата строит отдельный контекст вокруг него и спрашивает модель.

    Args:
        text (str): Фрагмент документа.
        llm (BaseLanguageModel): LLM.
        candidate_titles (Dict[str, int]): {title: pos}.
        window (int): Размер окна вокруг найденного кандидата для контекста.

    Returns:
        Dict[str, int]: Только те кандидаты, что реально диагнозы.
    """
    result = {}
    for title, pos in candidate_titles.items():
        # Контекст вокруг кандидата (±window)
        start = max(0, pos - window)
        end = min(len(text), pos + len(title) + window)
        snippet = text[start:end]

        prompt = (
            f"Context: '''{snippet}'''\n\n"
            f"Is the phrase '{title}' in this context a DISEASE or DIAGNOSIS that is being discussed? "
            f"Answer only 'Yes' or 'No'."
        )
        try:
            response = llm.invoke(prompt)
            answer = response.content.lower() if hasattr(response, "content") else str(response).lower()
            if "yes" in answer:
                result[title] = pos
        except Exception as e:
            print(f"[filter_section_titles_llm] Error for '{title}': {e}")
    return result


def get_active_section_titles(
    chunk_text: str,
    all_titles: Dict[str, int]
) -> List[str]:
    """
    Для заданного чанка возвращает disease titles (section titles), реально встречающиеся в его тексте.

    Args:
        chunk_text (str): Текст чанка.
        all_titles (Dict[str, int]): Словарь {title: pos_in_doc, ...} — все найденные в документе disease titles.

    Returns:
        List[str]: Список disease titles, реально встречающихся в тексте чанка.
    """
    titles_in_chunk = [t for t in all_titles if t and t in chunk_text]
    return titles_in_chunk


# --- Тестовый блок ---
if __name__ == "__main__":
    import os
    import glob
    from settings import default_llm, openai_llm, DEFAULT_MODEL, MODEL

    print("\nСравнительный тест extract_section_titles / LLM:")
    print("-" * 100)

    test_dir = "../../data/test_chunks"
    for file_path in sorted(glob.glob(os.path.join(test_dir, "*.txt"))):
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            print(f"\n=== {file_name} ===")

            # Regexp/эвристика (теперь словарь!)
            candidates = extract_section_titles(text)
            print(f"Section titles (regexp): {candidates}")

            # Сравнение с LLM (2 режима: raw text, candidate_titles)
            chunk_example = text[:1600]
            print(f"Default llm ({DEFAULT_MODEL}):")
            llm_titles_with_candidates = filter_section_titles_llm(
                chunk_example, default_llm, candidate_titles=candidates)
            print(f"Section titles (default_llm, from candidates): {llm_titles_with_candidates}")

            if openai_llm:
                print(f"\nOpenAI llm ({MODEL}):")
                oa_titles_with_candidates = filter_section_titles_llm(
                    chunk_example, openai_llm, candidate_titles=candidates)
                print(f"Section titles (openai_llm, from candidates): {oa_titles_with_candidates}")

            print("-" * 100)
