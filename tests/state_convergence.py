# state_convergence.py
"""
Демо файл-сценарий, где два агента-узла `agent1_node` и `agent2_node` передают свои состояния в
один общий узел `combiner_node` используя подход с `State`.
"""
import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage


GRAPH_PNG_FILE_PATH = "graphs/state_graph.png"
GRAPH_MD_FILE_PATH = "graphs/state_graph.md"


# Определение схемы состояния
class State(TypedDict):
    messages: Annotated[list, add_messages]
    extracted_diseases: str
    metadata: str


# Узлы
def extractor_node(state: State) -> State:
    text = state["messages"][-1].content
    diseases = json.dumps(["COVID-19", "varicella"])  # Имитация извлечения
    state["messages"].append(AIMessage(content=f"Extracted diseases: {diseases}"))
    return {"extracted_diseases": diseases}


def metadata_analyzer_node(state: State) -> State:
    text = state["messages"][-1].content
    metadata = json.dumps({"date": "2023-01-01"})  # Имитация извлечения метаданных
    state["messages"].append(AIMessage(content=f"Extracted metadata: {metadata}"))
    return {"metadata": metadata}


def combiner_node(state: State) -> State:
    diseases = json.loads(state["extracted_diseases"]) if state["extracted_diseases"] else []
    metadata = json.loads(state["metadata"]) if state["metadata"] else {}
    combined = {
        "diseases": diseases,
        "metadata": metadata
    }
    state["messages"].append(AIMessage(content=json.dumps(combined)))
    return state


# Создание графа
graph_builder = StateGraph(State)
graph_builder.add_node("extractor", extractor_node)
graph_builder.add_node("metadata_analyzer", metadata_analyzer_node)
graph_builder.add_node("combiner", combiner_node)

# Настройка рёбер
graph_builder.add_edge(START, "extractor")
graph_builder.add_edge(START, "metadata_analyzer")
graph_builder.add_edge("extractor", "combiner")
graph_builder.add_edge("metadata_analyzer", "combiner")
graph_builder.add_edge("combiner", END)

# Компиляция и выполнение
graph = graph_builder.compile()
initial_state = {"messages": [HumanMessage(content="Patient has COVID-19 and varicella. Date: 2023-01-01")],
                 "extracted_diseases": "", "metadata": ""}
result = graph.invoke(initial_state)
print("Final result:", json.loads(result["messages"][-1].content))
print("\nConversation:")
for message in result["messages"]:
    message.pretty_print()

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
