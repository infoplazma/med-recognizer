"""
Пример использования медицинских инструментов
"""
from pprint import pprint

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

DEFAULT_MODEL = "biomedgpt-lm-7b"  # "llama3-med42-8b" "biomedgpt-lm-7b"

# Model
default_llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="not-needed-for-lm-studio",
    model_name=DEFAULT_MODEL,
    temperature=0.0,
)


@tool
def get_diagnosis(fever: float) -> float:
    """Returns a possible diagnosis based on fever"""
    if fever > 37.5:
        return f"At this {fever=}, inflammation is possible"
    else:
        return f"At this {fever=} there is nothing to worry about."


@tool
def classify_bp(pressure: str, age: int) -> str:
    """
    Blood pressure classification by age.

    Arguments:
        pressure (str): blood pressure in the format "systolic/diastolic", e.g. "120/80".
        age (int): age in years.

    Returns:
        str: "low", "normal", or "high".
    """
    try:
        sys, dia = map(int, pressure.split("/"))
    except Exception:
        raise ValueError("pressure должен быть строкой вида '120/80'")

    # Границы по возрастным группам (систолическое, диастолическое)
    ranges = [
        {"min_age": 1, "max_age": 5, "norm": (95, 60), "low": (90, 55), "high": (100, 65)},
        {"min_age": 6, "max_age": 12, "norm": (105, 70), "low": (95, 60), "high": (115, 75)},
        {"min_age": 13, "max_age": 17, "norm": (110, 70), "low": (100, 60), "high": (130, 80)},
        {"min_age": 18, "max_age": 39, "norm": (120, 80), "low": (100, 60), "high": (130, 85)},
        {"min_age": 40, "max_age": 59, "norm": (125, 80), "low": (100, 65), "high": (140, 90)},
        {"min_age": 60, "max_age": 120, "norm": (130, 82), "low": (105, 65), "high": (140, 90)},
    ]

    # Выбираем диапазон по возрасту
    group = next((g for g in ranges if g["min_age"] <= age <= g["max_age"]), None)
    if not group:
        raise ValueError("Возраст вне поддерживаемого диапазона (1–120 лет)")

    # Проверка на пониженное
    if sys < group["low"][0] or dia < group["low"][1]:
        return "low"
    # Проверка на повышенное
    if sys > group["high"][0] or dia > group["high"][1]:
        return "high"
    # Иначе считаем нормой
    return "norm"


llm_with_tools = default_llm.bind_tools([get_diagnosis])

examples = [
    HumanMessage("The patient has a temperature of 39.5 degrees", name="example_user"),
    AIMessage("", name="example_assistant", tool_calls=[{"name": "get_diagnosis", "args": {"fever": 39.5}, "id": "1"}]),
    ToolMessage("At this fever=39.5, inflammation is possible", tool_call_id="1"),

    HumanMessage("A 10-year-old patient has a blood pressure of 95/60", name="example_user"),
    AIMessage("", name="example_assistant",
              tool_calls=[{"name": "classify_bp", "args": {"pressure": "95/60", "age": 10}, "id": "2"}]),
    ToolMessage("norm", tool_call_id="2"),

]
system = """You are a doctor who uses additional diagnostic tools. 
Use past tool usage as an example of how to correctly use the tools."""
few_shot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        *examples,
        ("human", "{query}"),
    ]
)

chain = {"query": RunnablePassthrough()} | few_shot_prompt | llm_with_tools

query = "The patient's temperature was measured to be 41 degrees"
query = "The 65 year old patient's measured blood pressure was 85/55"
ai_msg = chain.invoke(query)
print("tool_calls:", ai_msg.tool_calls)

# Initialize message history
messages = [HumanMessage(query), ai_msg]

# Process tool calls
for tool_call in ai_msg.tool_calls:
    selected_tool = {"get_diagnosis": get_diagnosis, "classify_bp": classify_bp}[tool_call["name"].lower()]
    tool_output = selected_tool.invoke(tool_call["args"])
    messages.append(ToolMessage(str(tool_output), tool_call_id=tool_call["id"]))

# pprint(messages)
print("---")
for message in messages:
    message.pretty_print()
