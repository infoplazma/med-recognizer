"""
Агент классифицирует тип страницы "CONTENT TABLE", "GENERAL", "DISEASES DESCRIPTION", "OTHER"
"""
# **********************************************************************************************************************
#                                          Импорты
# **********************************************************************************************************************
from typing import Dict, Literal, TypedDict, Annotated

from langchain_core.output_parsers import JsonOutputParser  # StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableParallel
from langchain.output_parsers import RetryOutputParser
# from langgraph.graph import END, START, StateGraph

import settings
from utils.custom_print import custom_pretty_print, pretty_print_json
from utils.read_pdf_page import read_page_text_blocks
from utils.tokenizer_counter import NumTokensCounter
from utils.chain_logging import log_chain_step
# **********************************************************************************************************************
#                                          Импорт промптов для Агента
# **********************************************************************************************************************
from toolkit.prompt_templates.one_page_reader_prompts import TYPE_SYSTEM_PROMPT, TYPE_HUMAN_PROMPT
# **********************************************************************************************************************
#                                          Установки
# **********************************************************************************************************************
# Инициализация LLM
llm = settings.default_llm
print(f"Модель: {llm.model_name}")
# Инициализация счетчика токенов
count_num_tokens = NumTokensCounter(llm=llm)


# **********************************************************************************************************************
#                                          Определение вспомогательных классов и функций
# **********************************************************************************************************************
# Схема ожидаемого JSON-ответа агента
class TextTypeClassifier(TypedDict):
    """Determines which category the text belongs to"""
    category: Annotated[Literal[
                            "CONTENT TABLE", "GENERAL", "DISEASES DESCRIPTION", "OTHER"], ..., "Must be one of: CONTENT TABLE, GENERAL, DISEASES DESCRIPTION, OTHER"]
    evidence: Annotated[str, ..., "An explanation for the chosen category, must not be empty"]


def log_extractor(input_dict: dict, config: RunnableConfig): return log_chain_step(input_dict, config, "extractor")


# **********************************************************************************************************************
#                                          Определение Агента
# **********************************************************************************************************************
def page_type_agent(page_text: str, config: RunnableConfig = None) -> Dict[str, Literal["CONTENT TABLE",
                                                                                        "GENERAL",
                                                                                        "DISEASES DESCRIPTION",
                                                                                        "OTHER"]]:
    """Агент классифицирует страницу и объясняет классификацию"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", TYPE_SYSTEM_PROMPT),
        ("human", TYPE_HUMAN_PROMPT),
    ])

    # 1) Главная ветка: нативные Structured Outputs → сразу JSON-модель
    structured_extractor = prompt | RunnableLambda(log_extractor) | llm.with_structured_output(TextTypeClassifier, method="json_mode")
    # structured_extractor_with_config = structured_extractor.with_config(CONFIG)

    # 2) Резервная ветка: LLM → JSON + RetryOutputParser
    #    (если structured_extractor упадёт исключением, пойдём сюда)
    json_parser = JsonOutputParser()
    retry_parser = RetryOutputParser.from_llm(parser=json_parser, llm=llm)

    # Сгенерируем и prompt_value (для retry_parser), и сырой текст
    gen_for_retry = RunnableParallel(
        prompt_value=prompt,  # ChatPromptValue (нужен для parse_with_prompt)
        completion=prompt | llm | (lambda m: m.content),  # сырой текст модели
    )

    def _parse_with_retry(payload: dict, config: RunnableConfig) -> dict:
        text = payload["completion"]
        p_val = payload["prompt_value"]
        try:
            return json_parser.parse(text)
        except Exception:
            # Альтернатива: ловить конкретный OutputParserException
            # except OutputParserException:
            return retry_parser.parse_with_prompt(text, p_val)

    fallback_extractor = gen_for_retry | RunnableLambda(_parse_with_retry)

    # 6) Собираем итоговую цепочку: основной путь + фолбэк + сетевые ретраи при необходимости
    chain = structured_extractor.with_config(config).with_fallbacks([fallback_extractor])

    response = chain.invoke(page_text)
    return response


# **********************************************************************************************************************
#                                          Тестирование Агента
# **********************************************************************************************************************
if __name__ == "__main__":
    import os

    # Источник для теста
    PAGE_NUM = 79  # Номер страницы (начинается с 1)
    PDF_SOURCE_FILES = ("Guide-to-Common-Childhood-Infections-2023_Final-Approved.pdf",
                        "Easy Paediatrics.pdf")
    PDF_SOURCE_FILE = PDF_SOURCE_FILES[1]
    PAGE_NUM -= 1
    PDF_PATH = os.path.join("../data", "med_sources", PDF_SOURCE_FILE)
    EXAMPLE_TEXT_PATH = os.path.join("../tests/data/example.txt")

    # Основной конфиг для "pipeline"
    CONFIG = RunnableConfig(
        tags=["extractor", "retry_parser"],
        metadata={"pdf_file": PDF_SOURCE_FILE, "page": PAGE_NUM, "max_tokens": 1000},
        run_name="PageTyping")
    custom_pretty_print(f"Config:", CONFIG)

    # Загрузка и разбиение документа
    print("\rLoading...", end="")
    sample_text = read_page_text_blocks(PDF_PATH, PAGE_NUM)
    sample_text = sample_text.strip()
    with open(EXAMPLE_TEXT_PATH, "w", encoding="utf-8") as fp:
        fp.write(sample_text)
    print(f"\rОбразец текста сохранен: {os.path.abspath(EXAMPLE_TEXT_PATH)}")

    agent_response = page_type_agent(sample_text, config=CONFIG)
    custom_pretty_print("\rAgent response:", agent_response)
