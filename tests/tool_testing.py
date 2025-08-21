"""
Пример использование функций в необученных для этого LLM
"""

from pprint import pprint

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

load_dotenv()

TEST_MAIN = False
TEST_TOOL_PARSER = False
TEST_FEW_SHOT_PROMPT = True
DEFAULT_MODEL = "llama3-med42-8b"  # "tinyllama-1.1b-chat-v1.0"
# DEFAULT_MODEL = "deepretrieval-pubmed-3b-llama"


query = "determine using 'tools':\nWhat is 3 + 7?\nAnd then determine using 'tools':\nWhat is 10 + 5?"

# Model
default_llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="not-needed-for-lm-studio",
    model_name=DEFAULT_MODEL,
    temperature=0.0,
)


# Функции
@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


# Note that the docstrings here are crucial, as they will be passed along
# to the model along with the class name.
class Add(BaseModel):
    """Add two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class Multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


# model = init_chat_model("gpt-3.5-turbo", temperature=0.0)
llm_with_tools = default_llm.bind_tools([add, multiply])
# llm_with_tools = model.bind_tools([add, multiply])


if TEST_MAIN:
    called_tools = llm_with_tools.invoke(query).tool_calls
    print("Main method:")
    print(called_tools)

if TEST_TOOL_PARSER:
    # llm_with_tools = model.bind_tools([Multiply, Add])
    llm_with_tools = default_llm.bind_tools([Multiply, Add])
    chain = llm_with_tools | PydanticToolsParser(tools=[Multiply, Add])
    called_tools = chain.invoke(query)
    print("\nPydanticToolsParser method:")
    print(called_tools)

if TEST_FEW_SHOT_PROMPT:
    examples = [
        HumanMessage(
            "What's the product of 317253 and 128472 plus four", name="example_user"
        ),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {"name": "Multiply", "args": {"a": 317253, "b": 128472}, "id": "1"}
            ],
        ),
        ToolMessage("16505054784", tool_call_id="1"),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[{"name": "Add", "args": {"a": 16505054784, "b": 4}, "id": "2"}],
        ),
        ToolMessage("16505054788", tool_call_id="2"),
        AIMessage(
            "The product of 317253 and 128472 plus four is 16505054788",
            name="example_assistant",
        ),
    ]

    system = """You are bad at math but are an expert at using a calculator. 
    
    Use past tool usage as an example of how to correctly use the tools."""
    few_shot_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            *examples,
            ("human", "{query}"),
        ]
    )

    print("\nWith few_shot_prompt method:")
    llm_with_tools = default_llm.bind_tools([add, multiply])

    print("First invocation ->")
    chain = {"query": RunnablePassthrough()} | few_shot_prompt | llm_with_tools
    # First invocation
    ai_msg = chain.invoke(query)
    print("First invocation tool calls:", ai_msg.tool_calls)

    # Initialize message history
    messages = [HumanMessage(query), ai_msg]

    # Process tool calls
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        print(f"Using {selected_tool = } with args {tool_call['args']}")
        tool_output = selected_tool.invoke(tool_call["args"])
        print(f"{tool_output = }")
        messages.append(ToolMessage(str(tool_output), tool_call_id=tool_call["id"]))
    pprint(messages)
    print("---")

    print("\nSecond invocation ->")
    messages = [HumanMessage(query)]
    chain = few_shot_prompt | llm_with_tools
    ai_msg = chain.invoke({"query": query})
    print("Second invocation tool calls:", ai_msg.tool_calls)
    messages.append(ai_msg)
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        print(f"Using {selected_tool = } with args {tool_call['args']}")
        tool_output = selected_tool.invoke(tool_call["args"])
        print(f"{tool_output = }")
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    pprint(messages)

    print("\nThird invocation ->")
    # Шаблон, который принимает сообщения
    few_shot_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            *examples,
            ("human", "{query_content}"),  # Извлекаем content из HumanMessage
        ]
    )

    # Функция для извлечения content из последнего сообщения
    def extract_query_content(messages):
        if isinstance(messages, list) and messages and isinstance(messages[-1], HumanMessage):
            return {"query_content": messages[-1].content}
        raise ValueError("Input must be a list with at least one HumanMessage")


    chain = RunnableLambda(extract_query_content) | few_shot_prompt | llm_with_tools
    messages = [HumanMessage(content=query)]
    ai_msg = chain.invoke(messages)
    print("Third invocation tool calls:", ai_msg.tool_calls)
    messages.append(ai_msg)
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        print(f"Using {selected_tool = } with args {tool_call['args']}")
        tool_output = selected_tool.invoke(tool_call["args"])
        print(f"{tool_output = }")
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    print("\nadditional_kwargs:")
    pprint(ai_msg.additional_kwargs["tool_calls"])

    pprint(messages)
