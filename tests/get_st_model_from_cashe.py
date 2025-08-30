"""
Если не хочешь хранить состояние в settings, сделай кэш через functools.lru_cache
"""
from typing import Optional
from functools import lru_cache
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_st_model_cached(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    print(f"{model_name} loading...", end="", flush=True)
    model = SentenceTransformer(model_name)
    print(f"\rLoaded")
    return model


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

    m = model or get_st_model_cached()
    # Нормализуем эмбеддинги: косинус становится скалярным произведением
    emb = m.encode([llm_heading, spam_text], normalize_embeddings=True, batch_size=2, show_progress_bar=False)
    h, s = emb[0], emb[1]
    return float((h * s).sum())


if __name__ == "__main__":
    result = _cosine_sim("campylobacter", "Campylobacter*")
    print(f"{result=}")
    result = _cosine_sim("campylobacter", "Campylobacter***")
    print(f"{result=}")
