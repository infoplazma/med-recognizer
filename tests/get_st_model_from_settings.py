"""
Для хранения модели в отдельном settings.py, то удобнее делать ленивую и потокобезопасную инициализацию так:
менять переменную прямо в settings и использовать «double-checked locking».
"""
from __future__ import annotations
from typing import Optional
import threading
import settings  # твой модуль с MODEL_NAME и ST_MODEL

_lock = threading.Lock()


def get_st_model(*, model_name: Optional[str] = None, verbose: bool = False):
    """
    Ленивая и потокобезопасная инициализация SentenceTransformer в settings.ST_MODEL.

    :param model_name: имя модели; если None — берётся settings.MODEL_NAME
    :param verbose: печатать сообщения о загрузке
    :return: инстанс модели SentenceTransformer
    """
    # Быстрая ветка: уже загружено
    if settings.ST_MODEL is not None:
        return settings.ST_MODEL

    # Потокобезопасная инициализация (double-checked locking)
    with _lock:
        if settings.ST_MODEL is None:
            if verbose:
                print("Loading SentenceTransformer...", end="", flush=True)
            from sentence_transformers import SentenceTransformer  # локальный импорт = ленивая загрузка пакета
            name = model_name or getattr(settings, "ST_MODEL_NAME", "all-MiniLM-L6-v2")
            settings.ST_MODEL = SentenceTransformer(name)
            if verbose:
                print("\rLoaded SentenceTransformer.     ")
    return settings.ST_MODEL
