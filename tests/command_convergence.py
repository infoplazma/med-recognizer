# command_convergence.py
"""
Демо файл-сценарий, где два агента-узла `agent1_node` и `agent2_node` передают свои состояния в
один общий узел `combiner_node` используя подход с `Command`.
"""
import time
import operator

from typing import TypedDict, List, Annotated, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

GRAPH_PNG_FILE_PATH = "graphs/command_graph.png"
GRAPH_MD_FILE_PATH = "graphs/command_graph.md"


# Определение схемы состояния с аннотацией для completed_nodes
class State(TypedDict):
    messages: Annotated[List, add_messages]
    agent1_result: str  # Annotated[str, lambda old, new: new]
    agent2_result: str  # Annotated[str, lambda old, new: new]
    completed_nodes: Annotated[List[str], operator.add]
    completion: str


# Узлы
def agent1_node(state: State) -> Command[Literal["combiner"]]:
    secs = 0.5
    print(f"agent2: Сейчас я зависну на {secs} секунды.")
    time.sleep(secs)  # Имитация извлечения
    return Command(goto="combiner",
                   update={
                       "messages": [AIMessage(content="А1 done")],
                       "agent1_result": "foo",
                       "completed_nodes": ["agent1"],
                   })


def agent2_node(state: State) -> Command[Literal["combiner"]]:
    secs = 1.5
    print(f"agent2: Сейчас я зависну на {secs} секунды.")
    time.sleep(secs)  # Имитация извлечения
    return Command(goto="combiner",
                   update={
                       "messages": [AIMessage(content="А2 done")],
                       "agent2_result": "bar",
                       "completed_nodes": ["agent2"],
                   })


def combiner_node(state: State) -> Command[Literal["completion", END]]:
    if not {"agent1", "agent2"}.issubset(set(state["completed_nodes"])):
        print("ERROR: not found ")
        return Command(goto=END, update={})
    return Command(goto="completion", update={"completion": "Completion node is updated"})


def completion_node(state: State) -> Command[Literal[END]]:
    combined_result = f"{state['agent1_result']} + {state['agent2_result']}"
    completion = state["completion"]
    print(f"{combined_result = }")
    print(f"{completion = }")
    return Command(
        goto=END,
        update={"messages": [AIMessage(content=f"COMBINED: {combined_result}")]})


# Создание графа
graph_builder = StateGraph(State)
graph_builder.add_node("agent1", agent1_node)
graph_builder.add_node("agent2", agent2_node)
graph_builder.add_node("combiner", combiner_node)
graph_builder.add_node("completion", completion_node)

# Настройка входа и выхода графа
graph_builder.set_entry_point("agent1")
graph_builder.set_entry_point("agent2")
graph_builder.set_finish_point("completion")

# Компиляция и выполнение
graph = graph_builder.compile()
initial_state = {
    "messages": [HumanMessage(content="Обработать запрос")],
    "agent1_result": "",
    "agent2_result": "",
    "completed_nodes": []
}
result = graph.invoke(initial_state, config={"recursion_limit": 50})
# print(result)
print("\nConversation:")
for message in result["messages"]:
    message.pretty_print()

print("\n")

# 2. Генерируем PNG-изображение в виде байтов
# Этот метод вызывает Graphviz "под капотом"
png_data = graph.get_graph().draw_mermaid_png()
with open(GRAPH_PNG_FILE_PATH, "wb") as f:
    f.write(png_data)
print(f"Граф успешно сохранен в файл '{GRAPH_PNG_FILE_PATH}'")

mermaid_text = graph.get_graph().draw_mermaid()
with open(GRAPH_MD_FILE_PATH, "w") as f:
    f.write(mermaid_text)
print(f"Граф успешно сохранен в файл '{GRAPH_MD_FILE_PATH}'")
