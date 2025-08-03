"""
settings.py

Глобальный конфиг для пайплайна медицинского индекса.
Содержит настройки LLM, эмбеддингов, пути данных, storage, etc.
"""

import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from utils.tokenizer_counter import Llama3TokenizerCounter, openai_tokens_counter as otc

# --- Убираем лишние ворнинги torch ---
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
load_dotenv()

# ================== Пути к данным ==================
BASE_DIR: str = os.path.abspath(os.path.dirname(__file__))
DATA_DIR: str = os.path.abspath(os.path.join(BASE_DIR, "../data"))
STORAGE_DIR: str = os.path.abspath(os.path.join(BASE_DIR, "../storage"))

MED_SOURCE_DIR: str = os.path.join(DATA_DIR, "med_sources")
PERSISTED_INDEX_DIR: str = os.path.join(STORAGE_DIR, "storage")
INDEXED_FILES_PATH: str = os.path.join(STORAGE_DIR, "indexed_files.txt")

# ================== Дефолтная LLM и эмбеддинги ==================
# Векторные эмбеддинги (Huggingface sentence-transformers)
DEFAULT_MODEL = "Llama3-Med42-8B"

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("default embedding from HuggingFaceEmbedding loaded")

# LLM (LM Studio) для LangChain (OpenAI API-совместимая)
default_llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="not-needed-for-lm-studio",
    model_name=DEFAULT_MODEL,
    temperature=0.0,
)
default_tokens_counter = Llama3TokenizerCounter(f"m42-health/{DEFAULT_MODEL}")

# ================== OpenAI модель (по желанию) ==================
MODEL: str = "gpt-3.5-turbo"  # "gpt-4o" или "gpt-3.5-turbo"

API_KEY: str = os.getenv("API_KEY_OPENAI", "")
if not API_KEY:
    print("[settings.py] Не найден API_KEY_OPENAI в .env или окружении. Пропускаем openai_llm.")

openai_llm = ChatOpenAI(
    openai_api_key=API_KEY,
    model_name=MODEL,
    temperature=0.0,
    max_tokens=900,
) if API_KEY else None

openai_tokens_counter = (lambda text: otc(text, model=MODEL)) if API_KEY else None
