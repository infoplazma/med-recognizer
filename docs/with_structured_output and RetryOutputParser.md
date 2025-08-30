Да — можно собрать «чистый» LCEL-пайплайн, где **главный путь** использует `with_structured_output(...)`, а **резервный** — `RetryOutputParser` на базе Pydantic-парсера. Ниже — самодостаточный пример извлечения структурированного ответа (тот же кейс с болезнями).

> Идея: сначала пытаемся получить Pydantic-модель нативно (structured outputs / tool use). Если провайдер не поддерживает или валидация упала — автоматически скатываемся на ветку `LLM → RetryOutputParser`, который «чинит» ответ под схему.

```python
# pip install -U langchain langchain-openai pydantic
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import RetryOutputParser   # именно тут
# (опционально) from langchain_core.output_parsers import OutputParserException

# 1) Схема ожидаемого JSON-ответа
class DiseaseDecision(BaseModel):
    case: Literal["NONE","CONTINUATION","CONTINUATION_AND_NEW","NEW"]
    current_diseases: List[str] = Field(default_factory=list)
    continued_diseases: List[str] = Field(default_factory=list)
    new_diseases: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list, max_items=2)

# 2) Модель
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3) Инструкции (system + user)
SYSTEM = """You extract diseases from clinical text and relate CURRENT to PREVIOUS.
Return the fields required by the schema exactly (no extra fields)."""

USER = """CURRENT chunk:
---
{context}
---

PREVIOUS summary (may be empty):
---
{prev_summary}
---

PREVIOUS diseases (comma-separated; may be empty):
{prev_diseases}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", USER),
])

# 4) Главная ветка: нативные Structured Outputs → сразу Pydantic-модель
structured_extractor = prompt | llm.with_structured_output(DiseaseDecision)

# 5) Резервная ветка: LLM → Pydantic + RetryOutputParser
#    (если structured_extractor упадёт исключением, пойдём сюда)
pyd_parser = PydanticOutputParser(pydantic_object=DiseaseDecision)
retry_parser = RetryOutputParser.from_llm(parser=pyd_parser, llm=llm)

# Сгенерируем и prompt_value (для retry_parser), и сырой текст
gen_for_retry = RunnableParallel(
    prompt_value = prompt,                                  # ChatPromptValue (нужен для parse_with_prompt)
    completion   = prompt | llm | (lambda m: m.content),    # сырой текст модели
)

def _parse_with_retry(payload: dict) -> DiseaseDecision:
    text = payload["completion"]
    pval = payload["prompt_value"]
    try:
        return pyd_parser.parse(text)
    except Exception:
        # Альтернатива: ловить конкретный OutputParserException
        # except OutputParserException:
        return retry_parser.parse_with_prompt(text, pval)

fallback_extractor = gen_for_retry | RunnableLambda(_parse_with_retry)

# 6) Собираем итоговую цепочку: основной путь + фолбэк + сетевые ретраи при необходимости
chain = structured_extractor.with_fallbacks([fallback_extractor]).with_retry()

# 7) Пример запуска
inputs = {
    "context": "The patient with T2D started insulin; new-onset heart failure was diagnosed.",
    "prev_summary": "Prior visit discussed poor glycemic control in type 2 diabetes.",
    "prev_diseases": "Type 2 diabetes",
}

out: DiseaseDecision = chain.invoke(inputs)
print(out.model_dump())
```

### Что здесь происходит

* **`llm.with_structured_output(DiseaseDecision)`** — просит модель вернуть строго эту Pydantic-схему (через native structured outputs / tool use, если провайдер умеет).
* **`with_fallbacks([fallback_extractor])`** — если главный runnable бросает исключение (нет поддержки, валидация упала и т.п.), вызывается запасной пайп.
* **Запасной пайп** собирает одновременно `prompt_value` и «сырой» текст, после чего `RetryOutputParser` делает дополнительный LLM-вызов и чинит ответ под схему.
* **`with_retry()`** сверху — для сетевых/временных сбоев (429/5xx/таймауты).

> Если хотите жёстче контролировать пустые результаты — добавьте в Pydantic-схему ограничения (`min_length`/`Literal`) и ловите `ValidationError` в хэндлере, либо доверьте это `with_fallbacks` (упадёт главная ветка → заработает `RetryOutputParser`).
