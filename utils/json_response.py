import re


def clean_json_response(raw_response: str) -> str:
    """
    Преобразует markdown/json ответ LLM к "чистому" JSON для json.loads.
    Удаляет markdown-обёртки, невалидные trailing/leading символы, комментарии, python-style булевые, лишние запятые и т.д.
    """
    cleaned = raw_response.strip()

    # 1. Удаление markdown-блока (например, ```json ... ```)
    # Ловит блоки в любом месте
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    # 2. Если что-то осталось (например, начинается с "```json"), режем вручную
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:].lstrip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:].lstrip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].rstrip()

    # 3. Удаляем управляющие символы, оставляя только валидные
    cleaned = ''.join(c for c in cleaned if ord(c) >= 32 or c in '\n\t\r')

    # 4. Удаляем комментарии (#, //)
    cleaned = re.sub(r'#.*', '', cleaned)
    cleaned = re.sub(r'//.*', '', cleaned)

    # 5. Приведение python-style к json
    cleaned = cleaned.replace('True', 'true').replace('False', 'false').replace('None', 'null').replace('Null', 'null')

    # 6. Удаляем висячие запятые перед скобками
    cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)

    # 7. Иногда LLM может забыть запятые между полями (очень сложно фиксить general-case, не всегда безопасно)
    # Можно добавить доп. регулярки при необходимости.

    return cleaned.strip()
