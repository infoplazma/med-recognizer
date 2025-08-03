# formatting.py

def format_time(t: float) -> str:
    """
    Форматирует время в секундах как "мм:сс.миллисек".
    """
    minutes = int(t // 60)
    seconds = int(t % 60)
    millis = int((t - int(t)) * 1000)
    return f"{minutes:02}:{seconds:02}.{millis:03}"


def display_quotes(result) -> None:
    print("\n--- Источники ---")
    for i, doc in enumerate(result["source_documents"]):
        print(f"--- Документ {i + 1} ---")
        print(doc.page_content[:300], "...")
        print(doc.metadata)
