import os
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage
from settings import default_llm, openai_llm

GRAPH_DIR = "./graphs"
GRAPH_PNG_FILE = "my_graph.png"
GRAPH_MD_FILE = "my_graph.md"


@tool
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b


@tool
def sub(a: int, b: int) -> int:
    """Subtracts from number a number b."""
    return a - b


class SimpleResponse(BaseModel):
    result: str = Field(..., description="Human-readable result")


calc_tool = Tool.from_function(lambda x: str(eval(x)), name="Math Calculator", description="for math")

tools = [add, sub]

agent = create_react_agent(
    model=default_llm,
    tools=tools,
    prompt="You are an assistant that can perform math via tools. Use mathematical functions to perform operations with numbers.",
    debug=True,
    version="v2"
)
# Альтернатива
llm_with_tools = default_llm.bind_tools(tools)

# 1. Получаем объект графа
graph = agent.get_graph()

# 2. Генерируем PNG-изображение в виде байтов
# Этот метод вызывает Graphviz "под капотом"
png_data = graph.draw_mermaid_png()
with open("my_graph.png", "wb") as f:
    f.write(png_data)

mermaid_text = agent.get_graph().draw_mermaid()
print(mermaid_text)
# png_data содержит байты, полученные от graph.draw_mermaid_png()

print("Граф успешно сохранен в файл 'my_graph.png'")

max_iterations = 5
recursion_limit = 2 * max_iterations + 1

# messages = [{"role": "user", "content": "What is add(3, 4) and sub(7, 1)? Give your answer in JSON format"}]  # What is add(3, 4) and sub(7, 1)? What is 3 add 2 and then subtract 1
messages = [HumanMessage(content="What is add(3, 4) and sub(7, 1)? Give your answer in JSON format")]

response = agent.invoke({"messages": messages})  # , {"recursion_limit": recursion_limit}
last_message = response["messages"][-1]
if isinstance(last_message, AIMessage):
    output_message = last_message.content
else:
    output_message = str(last_message)

print(f"Agent output: {output_message}")
# print(response["structuredResponse"])
