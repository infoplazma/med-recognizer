"""
Модуль: simple_agent.py
-----------------------

Назначение
~~~~~~~~~~
Универсальный «агент»-экстрактор поверх LangChain LCEL, который:
1) Пытается получить **структурированный JSON-ответ** через
   `llm.with_structured_output(...)` (основная ветка).
2) Если основной путь падает (нет поддержки/валидация), выполняет
   **резервную ветку**: LLM → `JsonOutputParser` → `RetryOutputParser`,
   который «чинит» вывод под схему.

Особенности
~~~~~~~~~~~
- Внутренние шаги логируются по тегу (при наличии в RunnableConfig).
- Можно передать `RunnableConfig` (теги, метаданные, run_name).
- Ожидается, что в `hum_prompt` используется плейсхолдер `{text}`.

Пример использования


Зависимости
~~~~~~~~~~~
- langchain, langchain-openai
- pydantic (если передаёте Pydantic-модель в validator)
"""
from typing import Any, List, Dict, Optional, TypedDict, Union, Annotated

from langchain.output_parsers import RetryOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser # StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableParallel

from utils.chain_logging import log_chain_step
from utils.tokenizer_counter import NumTokensCounter


class SimpleAgent:
    """
    Агент для получения структурированного JSON через LCEL с фолбэком на RetryOutputParser.

    Основной путь:
        prompt | llm.with_structured_output(validator, method="json_mode")

    Резервный путь:
        prompt → llm → JsonOutputParser (если падает) → RetryOutputParser.from_llm(...)

    :param llm: экземпляр ChatOpenAI (или совместимый чат-LLM)
    :param sys_prompt: системное сообщение для ChatPromptTemplate
    :param hum_prompt: пользовательское сообщение с плейсхолдером `{text}`
    :param schema: схема результата:
        - JSON Schema (TypeDict)
    :param tag: опциональный тег для выборочного логирования шагов
    :param msg: опциональное сообщения для тега при логировании
    """

    def __init__(self, llm: ChatOpenAI, sys_prompt: str, hum_prompt: str, schema: TypedDict, tag: str = None, msg: str = None):
        self.llm = llm
        self.sys_prompt = sys_prompt
        self.hum_prompt = hum_prompt
        self.schema = schema
        self.tag = tag
        self.msg = msg if msg is not None else "Пробую через"

        # Инициализация счетчика токенов
        self.count_num_tokens = NumTokensCounter(llm=llm)

    def __call__(self, input_data: Union[str, dict], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Запускает пайплайн экстракции для заданного текста.

        :param input_data: входной текст; будет подставлен в {text} в `hum_prompt`
        :param config: RunnableConfig (теги, метаданные и т.д.) или None
        :return: словарь с результатом (даже если основной путь вернул Pydantic-модель,
                 он будет приведён к dict)
        """
        # 1) Промпт: ожидается, что в hum_prompt есть {text}
        prompt = ChatPromptTemplate.from_messages(
            [("system", self.sys_prompt), ("human", self.hum_prompt)]
        )

        # Вспомогательные логгеры для разных веток
        def log_main_branch(input_dict: Dict[str, Any], config: RunnableConfig = None):
            return log_chain_step(input_dict, config, self.tag, msg=f"{self.msg} 'Main Branch'")

        def log_json_branch(input_dict: Dict[str, Any], config: Optional[RunnableConfig] = None):
            return log_chain_step(input_dict, config, self.tag, msg=f"{self.msg} 'JSONParser'")

        def log_fallback_branch(_: Dict[str, Any], config: Optional[RunnableConfig] = None):
            return log_chain_step({}, config, self.tag, msg=f"{self.msg} 'RetryOutputParser'")

        # 1) Главная ветка: нативные Structured Outputs → сразу JSON-модель
        structured_extractor = prompt | RunnableLambda(log_main_branch) | self.llm.with_structured_output(
            self.schema, method="json_schema")

        # 2) Резервная ветка: LLM → JSON + RetryOutputParser
        #    (если structured_extractor упадёт исключением, пойдём сюда)
        json_parser = JsonOutputParser()
        retry_parser = RetryOutputParser.from_llm(parser=json_parser, llm=self.llm)

        # Сгенерируем и prompt_value (для retry_parser), и сырой текст
        gen_for_retry = RunnableParallel(
            prompt_value=prompt,  # ChatPromptValue (нужен для parse_with_prompt)
            completion=prompt | RunnableLambda(log_json_branch) | self.llm | StrOutputParser()  # сырой текст модели
        )

        def _parse_with_retry(payload: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
            """
            Парсит JSON; при ошибке — исправляет через RetryOutputParser.
            """
            text_out: str = payload["completion"]
            p_val = payload["prompt_value"]
            try:
                print(f"{text_out=}")
                return json_parser.parse(text_out)
            except Exception:
                log_fallback_branch({}, config=config)
                return retry_parser.parse_with_prompt(text_out, p_val)

        fallback_extractor = gen_for_retry | RunnableLambda(_parse_with_retry)

        # 6) Собираем итоговую цепочку: основной путь + фолбэк + сетевые ретраи при необходимости
        # Привязать config ко всей цепочке (если он есть)
        chain = (structured_extractor.with_fallbacks([fallback_extractor])
                 .with_config(config) if config else
                 structured_extractor.with_fallbacks([fallback_extractor]))

        response = chain.invoke(input_data)
        return response


# **********************************************************************************************************************
#                                          Тестирование Агента
# **********************************************************************************************************************
if __name__ == "__main__":
    import os

    import settings
    from utils.custom_print import custom_pretty_print
    from utils.read_pdf_page import read_page_text_blocks
    # ******************************************************************************************************************
    #                                          Импорт промптов для Агента
    # ******************************************************************************************************************
    from toolkit.prompt_templates.disease_name_agent_prompts import SYSTEM_PROMPT, HUMAN_PROMPT, DiseaseListSchema
    from toolkit.prompt_templates.one_page_reader_prompts import TYPE_SYSTEM_PROMPT, TYPE_HUMAN_PROMPT, TypeScheme

    # Источник для теста
    PAGE_NUM = 86  # Номер страницы (начинается с 1)
    PDF_SOURCE_FILES = ("Guide-to-Common-Childhood-Infections-2023_Final-Approved.pdf",
                        "Easy Paediatrics.pdf")
    PDF_SOURCE_FILE = PDF_SOURCE_FILES[1]
    PAGE_NUM -= 1
    PDF_PATH = os.path.join("../data", "med_sources", PDF_SOURCE_FILE)
    EXAMPLE_TEXT_PATH = os.path.join("../tests/data/example.txt")

    # Загрузка и разбиение документа
    print("\rLoading...", end="")
    sample_text = read_page_text_blocks(PDF_PATH, PAGE_NUM)
    sample_text = sample_text.strip()
    with open(EXAMPLE_TEXT_PATH, "w", encoding="utf-8") as fp:
        fp.write(sample_text)
    print(f"\rОбразец текста сохранен: {os.path.abspath(EXAMPLE_TEXT_PATH)}")

    # Основной конфиг для "pipeline"
    CONFIG = RunnableConfig(
        tags=["extractor:diseases", "extractor:type_page"],
        metadata={"pdf_file": PDF_SOURCE_FILE, "page": PAGE_NUM, "max_tokens": 1000},
        run_name="Diseases extracting")
    custom_pretty_print(f"Config:", CONFIG)

    diseases_agent = SimpleAgent(
        llm=settings.default_llm,
        sys_prompt=SYSTEM_PROMPT,
        hum_prompt=HUMAN_PROMPT,
        schema=DiseaseListSchema,
        tag="extractor:diseases",
        msg="Пытаю извлечь заболевания используя"
    )

    agent_response = diseases_agent(sample_text, config=CONFIG)
    custom_pretty_print("\rAgent response:", agent_response)

    type_agent = SimpleAgent(
        llm=settings.default_llm,
        sys_prompt=TYPE_SYSTEM_PROMPT,
        hum_prompt=TYPE_HUMAN_PROMPT,
        schema=TypeScheme,
        tag="extractor:type_page",
        msg="Пытаю определить тип текста используя"
    )

    agent_response = type_agent(sample_text, config=CONFIG)
    custom_pretty_print("\rAgent response:", agent_response)
