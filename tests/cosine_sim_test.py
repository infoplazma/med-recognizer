from __future__ import annotations
from typing import Optional

# pip install sentence-transformers
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "sentence-transformers is not installed. "
        "Install it with: pip install sentence-transformers"
    ) from e

_ST_MODEL: Optional[SentenceTransformer] = None
_MODEL_NAME: str = "all-MiniLM-L6-v2"  # быстрый и точный для EN; при желании замените на all-mpnet-base-v2


def _get_st_model() -> SentenceTransformer:
    """Ленивая инициализация модели, чтобы не грузить её при каждом вызове."""
    global _ST_MODEL
    if _ST_MODEL is None:
        print(f"\rLoading...", end="")
        _ST_MODEL = SentenceTransformer(_MODEL_NAME)
        print(f"\rLoaded")
    return _ST_MODEL


def _cosine_sim(llm_heading: str, spam_text: str, *, model: Optional[SentenceTransformer] = None) -> float:
    """
    Косинусное сходство между строкой-заголовком и текстом спана с помощью SentenceTransformer.

    :param llm_heading: строка заголовка, предложенная LLM
    :param spam_text: текст спана/строки из PDF (опечатка в имени аргумента сохранена по ТЗ)
    :param model: уже созданная модель SentenceTransformer; если None — будет использована/создана дефолтная
    :return: косинусное сходство (-1..1), при L2-нормализации обычно в диапазоне [0..1]
    """
    if not llm_heading.strip() or not spam_text.strip():
        return 0.0

    m = model or _get_st_model()
    # Нормализуем эмбеддинги: косинус становится скалярным произведением
    emb = m.encode([llm_heading, spam_text], normalize_embeddings=True, batch_size=2, show_progress_bar=False)
    h, s = emb[0], emb[1]
    return float((h * s).sum())


if __name__ == "__main__":
    result = _cosine_sim("campylobacter", "Campylobacter*")
    print(f"{result=}")

