import re
from typing import List, Dict


def get_heading_candidates(text: str) -> List[Dict[str, str]]:
    """
    Извлекает кандидатов на заголовки заболеваний из текста с использованием регулярных выражений.

    Args:
        text (str): Входной текст для анализа.

    Returns:
        List[Dict[str, str]]: Список словарей с ключами 'heading' (название заболевания).
    """
    candidates = []
    lines = text.split('\n')

    # Регулярное выражение для заголовков: короткие строки (1-3 слова), возможно с '*'
    heading_pattern = r'^(.*?)(?:\*)?\s*$'

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Проверка, является ли строка потенциальным заголовком
        heading_match = re.match(heading_pattern, line)
        if heading_match:
            potential_heading = heading_match.group(1).strip()
            # Ограничиваем длину заголовка (1-3 слова) и проверяем, не пустой ли
            if len(potential_heading.split()) <= 3 and potential_heading:
                candidates.append({"heading": potential_heading})

    return candidates
