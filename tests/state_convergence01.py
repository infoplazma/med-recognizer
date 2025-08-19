import random
import operator
from typing import List, TypedDict, Annotated, Literal
from pprint import pprint

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage


GRAPH_PNG_FILE_PATH = "graphs/weather_graph.png"


class WeatherState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    status: Annotated[List[str], operator.add]


# Define a starting node. This node just returns a predefined string.
def weather(state: WeatherState):
    return {"messages": AIMessage(content="Hi! Well.. I have no idea... But... "), "status": ["WEATHER"]}


# Define a starting node. This node just returns a predefined string.
def sunny_weather(state: WeatherState):
    return {"messages": AIMessage(content="Its going to be sunny today. Wear sunscreen."), "status": ["SUNNY"]}


# Define a node that returns rainy weather
def rainy_weather(state: WeatherState):
    return {"messages": AIMessage(content=" Its going to rain today. Carry an umbrella."), "status": ["RAINY"]}


# create a forecast weather function that returns rainy or sunny weather based on a random number.
def forecast_weather(state: WeatherState) -> Literal["rainy", "sunny"]:
    if random.random() < 0.5:
        return "rainy"
    else:
        return "sunny"


workflow = StateGraph(WeatherState)
# Add the nodes
workflow.add_node("weather", weather)
workflow.add_node("rainy", rainy_weather)
workflow.add_node("sunny", sunny_weather)

workflow.add_edge(START, "weather")
workflow.add_conditional_edges("weather", forecast_weather)
workflow.add_edge("rainy", END)
workflow.add_edge("sunny", END)

# Compile the graph
weather_graph = workflow.compile()

# Visualize your graph
png_data = weather_graph.get_graph().draw_mermaid_png()
with open(GRAPH_PNG_FILE_PATH, "wb") as f:
    f.write(png_data)
print(f"Граф успешно сохранен в файл '{GRAPH_PNG_FILE_PATH}'")


init_state = {"messages": HumanMessage(content="Let's do it!"), "status": ["IDEA"]}

result = weather_graph.invoke(init_state)
pprint(result)
# Print the conversation
print("\nConversation:")
for message in result["messages"]:
    message.pretty_print()
