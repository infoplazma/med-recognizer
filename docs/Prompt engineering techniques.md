Коротко: «правила» и приёмы хороших промптов не в одном месте — это свод рекомендаций из официальных гайдлайнов провайдеров моделей и инструментов. Вот куда смотреть (с кратким комментарием):

* **OpenAI – Prompt engineering**: базовые приёмы (ролевая инструкция, примеры, ограничения), плюс «Structured outputs» и tool/function calling — то, что мы использовали для жёсткой схемы ответа. ([OpenAI Платформа][1])
* **Anthropic – Prompt engineering**: системный разбор целей промпта, когда помогает few-shot, отдельная заметка про multi-shot примеры. Полезно для задач классификации/извлечения. ([Anthropic][2])
* **LangChain – Structured outputs / Output parsers**: как получать структурированные ответы (`with_structured_output`) и когда лучше использовать парсеры (`JsonOutputParser`, `PydanticOutputParser`) + примеры. Это база для «железных» экстракторов. ([LangChain][3])
* **DeepLearning.AI (Ng & Fulford) – ChatGPT Prompt Engineering for Developers**: короткий бесплатный курс с практикой и best practices (разделение ролей, итеративное уточнение, few-shot). ([deeplearning.ai][4], [DeepLearning.AI - Learning Platform][5])
* **Microsoft Azure OpenAI – Best practices** и **AWS Bedrock – Guidelines**: корпоративные чеклисты (быть конкретным, ограничивать пространство решений, дублировать ключевые инструкции, давать примеры). Хорошо для «продуктовых» требований. ([Microsoft Learn][6], [Документация AWS][7])
* **Cohere – Prompt engineering basics / guide**: компактные обзоры с примерами и анти-паттернами; полезно как дополнительная перспектива. ([Cohere][8])

## Быстрая шпаргалка (что мы применяли в вашем промпте)

1. **Явная цель + критерии** (что извлекаем, что не извлекаем). ([Anthropic][2])
2. **Структурированный формат** (JSON-схема или чёткая строка), лучше через нативный structured output. ([LangChain][9])
3. **Few-shot с контр-примером** (включая случай BOTH/`CONTINUATION_AND_NEW`). ([Anthropic][10])
4. **Инварианты/проверки в инструкции** (сортировка, дедупликация, равенство мощностей множеств). ([Microsoft Learn][6])

Если хотите, под ваши данные соберу мини-конспект с ссылками на конкретные разделы и примерами именно для «продолжение/новое заболевание» (в формате, который вы уже используете).

[1]: https://platform.openai.com/docs/guides/prompt-engineering?utm_source=chatgpt.com "OpenAI - Prompt Engineering"
[2]: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview?utm_source=chatgpt.com "Prompt engineering overview"
[3]: https://python.langchain.com/docs/concepts/structured_outputs/?utm_source=chatgpt.com "Structured outputs"
[4]: https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/?utm_source=chatgpt.com "ChatGPT Prompt Engineering for Developers"
[5]: https://learn.deeplearning.ai/courses/chatgpt-prompt-eng/lesson/dfbds/introduction?utm_source=chatgpt.com "Prompt Engineering for Developers"
[6]: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/prompt-engineering?utm_source=chatgpt.com "Prompt engineering techniques - Azure OpenAI"
[7]: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-engineering-guidelines.html?utm_source=chatgpt.com "Prompt engineering concepts - Amazon Bedrock"
[8]: https://cohere.com/llmu/prompt-engineering-basics?utm_source=chatgpt.com "Prompt Engineering Basics"
[9]: https://python.langchain.com/docs/how_to/structured_output/?utm_source=chatgpt.com "How to return structured data from a model"
[10]: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/multishot-prompting?utm_source=chatgpt.com "Use examples (multishot prompting) to guide Claude's ..."

Best practices

    Be Specific. Leave as little to interpretation as possible. Restrict the operational space.
    Be Descriptive. Use analogies.
    Double Down. Sometimes you might need to repeat yourself to the model. Give instructions before and after your primary content, use an instruction and a cue, etc.
    Order Matters. The order in which you present information to the model might impact the output. Whether you put instructions before your content (“summarize the following…”) or after (“summarize the above…”) can make a difference in output. Even the order of few-shot examples can matter. This is referred to as recency bias.
    Give the model an “out”. It can sometimes be helpful to give the model an alternative path if it's unable to complete the assigned task. For example, when asking a question over a piece of text you might include something like "respond with "not found" if the answer isn't present." This can help the model avoid generating false responses.
