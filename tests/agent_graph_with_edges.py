"""
Пример мульти агента граф с ребрами
Building Multi-Agent Systems with LangGraph
https://medium.com/cwan-engineering/building-multi-agent-systems-with-langgraph-04f90f312b8e
"""
# Import necessary libraries
import os
from typing import Dict, List, TypedDict, Annotated, Sequence, Any, Optional, Literal, Union
import json
from pprint import pprint
from colorama import init, Fore, Style
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# Modern imports for langchain and langgraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from settings import default_llm, openai_llm
from utils.json_response import clean_json_response
from toolkit.prompt_templates.researcher_system_prompts import (RESEARCHER_SYSTEM_PROMPT,
                                                                ENHANCED_RESEARCHER_PROMPT,
                                                                CRITIC_SYSTEM_PROMPT,
                                                                WRITER_SYSTEM_PROMPT,
                                                                COORDINATOR_SYSTEM_PROMPT)


GRAPH_PNG_FILE_PATH = "graphs/researcher_graph.png"
GRAPH_MD_FILE_PATH = "graphs/researcher_graph.md"

init()
load_dotenv()
# Verify that the API key is loaded
if os.getenv("OPENAI_API_KEY") is None:
    print("Warning: OPENAI_API_KEY not found in environment variables.")
else:
    print("OPENAI_API_KEY found in environment variables.")


# Define the state type for our research workflow
class ResearchState(TypedDict):
    """Type definition for our research workflow state."""
    iteration: int
    messages: List[BaseMessage]  # The conversation history
    query: str  # The research query
    agent: Optional[List[str]]  # The research findings
    next: Optional[str]  # Where to go next in the graph


def increase_iteration(state: ResearchState) -> int:
    state["iteration"] = state.get("iteration", 0) + 1
    return state["iteration"]


# Define a COORDINATOR NODE
def coordinator_node(state: ResearchState) -> ResearchState:
    """Coordinator node that decides the workflow path."""
    if "agent" not in state:
        state["agent"] = []
    state["agent"].append("coordinator")
    increase_iteration(state)
    if state["iteration"] == 1:
        query = state["query"]
        print(f"Пользователь задал вопрос '{query}', думаю...")
        return {**state, "messages": [AIMessage(content="Go to Researcher Agent")], "next": "researcher"}
    elif state["iteration"] > 3:
        return {**state, "messages": state["messages"], "next": "output"}
    print("Анализирую ответы агентов...")
    # Extract messages from the state
    messages = state["messages"]
    # Create coordinator messages with the system prompt
    coordinator_messages = [SystemMessage(content=COORDINATOR_SYSTEM_PROMPT)] + messages
    # Initialize the LLM with a lower temperature for consistent decision-making
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    # Get the coordinator's response
    response = llm.invoke(coordinator_messages)
    # Parse the JSON response to determine next steps
    try:
        raw_decision = clean_json_response(response.content)
        decision = json.loads(raw_decision)
        print(Fore.LIGHTGREEN_EX + Style.BRIGHT + f"{decision['reasoning']}" + Style.RESET_ALL)
        next_step = decision.get("next", "researcher")  # Default to researcher if not specified
    except Exception:
        # If there's an error parsing the JSON, default to the researcher
        print(f"ERROR PARSING JSON:{response.content}")
        next_step = "researcher"
    # Return the updated state
    return {**state, "messages": messages, "next": next_step}


# Define a RESEARCHER NODE
def create_researcher_agent(model="gpt-4o", temperature=0.7):
    """Create a researcher agent using the specified LLM."""
    # Initialize the model
    llm = default_llm

    def researcher_function(messages):
        """Function that processes messages and returns a response from the researcher agent."""
        # Add the system prompt if it's not already there
        if not messages or not isinstance(messages[0], SystemMessage) or messages[0].content != ENHANCED_RESEARCHER_PROMPT:
            messages = [SystemMessage(content=ENHANCED_RESEARCHER_PROMPT)] + (messages if isinstance(messages, list) else [])
        # Get response from the LLM
        response = llm.invoke(messages)
        return response

    return researcher_function


# Define an output researcher_node
def researcher_node(state: ResearchState) -> ResearchState:
    """A node in our graph that performs research on the query."""
    # Get the query from the state
    state["agent"].append("user")
    query = state["query"]
    # Create a message specifically for the researcher
    if state["iteration"] == 1:
        research_message = HumanMessage(content=f"Please research the following topic thoroughly: {query}")
    elif state["iteration"] > 1:
        research_message = HumanMessage(content=f"Please research the following topic '{query}' more carefully again "
                                                f"taking into account previous research, reasoning and comments.")
    # Get the researcher agent
    researcher = create_researcher_agent()
    # Get response from the researcher agent
    response = researcher([research_message])
    # Update the state with the research findings
    new_messages = state["messages"] + [research_message, response]
    state["agent"].append("researcher")
    # Return the updated state
    return {
        "messages": new_messages,
        "research": response.content,
        "next": "critic"  # In a multi-agent system, this would go to the next agent
    }


# Define a CRITIC NODE
def critic_node(state: ResearchState) -> ResearchState:
    """Node function for the critic agent node."""
    state["agent"].append("critic")
    # Extract messages from the state
    messages = state["messages"]
    # Create critic messages with the system prompt
    critic_messages = [SystemMessage(content=CRITIC_SYSTEM_PROMPT)] + messages
    # Initialize the LLM with a balance of creativity and accuracy
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
    # llm = openai_llm
    # Get the critic's response
    response = llm.invoke(critic_messages)
    # Return the updated state
    return {"messages": messages + [response], "next": "writer"}


# Define a WRITER NODE
def writer_node(state: ResearchState) -> ResearchState:
    """Node function for the writer agent node."""
    state["agent"].append("writer")
    # Extract messages from the state
    messages = state["messages"]
    # Create writer messages with the system prompt
    writer_messages = [SystemMessage(content=WRITER_SYSTEM_PROMPT)] + messages
    # Initialize the LLM with a balance of creativity and accuracy
    llm = openai_llm
    # Get the writer's response
    response = llm.invoke(writer_messages)
    # Return the updated state
    return {"messages": messages + [response], "next": "output"}


# Define an output node
def output_node(state: ResearchState):
    """Return the final state. This marks the end of the workflow."""
    return {"messages": state["messages"], "next": END}


# Build the graph
def build_research_graph():
    """Build a complete research assistant with researcher, critic, and writer agents."""
    # Create a new graph with our state type
    workflow = StateGraph(ResearchState)
    # Add nodes
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("output", output_node)
    # Add edges
    # Add conditional edges from the coordinator
    workflow.add_conditional_edges(
        "coordinator",
        lambda state: state["next"],
        {
            "researcher": "researcher",
            "done": "output"
        }
    )
    workflow.add_edge("researcher", "critic")
    workflow.add_edge("critic", "writer")
    workflow.add_edge("writer", "coordinator")
    # Set the entry point
    workflow.set_entry_point("coordinator")
    # Compile the graph
    return workflow.compile()


# Create the graph
research_graph = build_research_graph()

# 2. Генерируем PNG-изображение в виде байтов
# Этот метод вызывает Graphviz "под капотом"
png_data = research_graph.get_graph().draw_mermaid_png()
with open(GRAPH_PNG_FILE_PATH, "wb") as f:
    f.write(png_data)
print(f"Граф успешно сохранен в файл '{GRAPH_PNG_FILE_PATH}'")

mermaid_text = research_graph.get_graph().draw_mermaid()
with open(GRAPH_MD_FILE_PATH, "w") as f:
    f.write(mermaid_text)
print(f"Граф успешно сохранен в файл '{GRAPH_MD_FILE_PATH}'")

# Initialize the state with a test message
initial_state = {
    "query": "Explain what a flu is in simple terms.",
    "next": ""
}
# Run the graph
result = research_graph.invoke(initial_state)
# Print the conversation
print("\nConversation:")
for message in result["messages"]:
    message.pretty_print()
    # if isinstance(message, HumanMessage):
    #     print(f"\nHuman ({agent}): {message.content}")
    # elif isinstance(message, AIMessage):
    #     print(f"\nAI ({agent}): {message.content}")

