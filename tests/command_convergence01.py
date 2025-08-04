from typing_extensions import TypedDict, Annotated
from typing import List
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
import operator


class State(TypedDict):
    messages: Annotated[List, add_messages]
    agent1_result: str
    agent2_result: str
    completed_nodes: Annotated[list[str], operator.add]


def agent1(state: State) -> Command:
    state["agent1_result"] = "foo"
    state["completed_nodes"] = state.get("completed_nodes", []) + ["agent1"]
    state["messages"].append(AIMessage(content="agent1 done"))
    return Command(goto="agent2", update=state)


def agent2(state: State) -> Command:
    state["agent2_result"] = "bar"
    state["completed_nodes"] = state.get("completed_nodes", []) + ["agent2"]
    state["messages"].append(AIMessage(content="agent2 done"))
    return Command(goto="combiner", update=state)


def combiner(state: State) -> State:
    agg = f"{state['agent1_result']} + {state['agent2_result']}"
    state["messages"].append(AIMessage(content=f"COMBINED: {agg}"))
    return state


builder = StateGraph(State)
builder.add_node("agent1", agent1)
builder.add_node("agent2", agent2)
builder.add_node("combiner", combiner)

builder.add_edge(START, "agent1")
builder.add_edge("agent1", "agent2")
builder.add_edge("agent2", "combiner")
builder.set_finish_point("combiner")

graph = builder.compile()
initial = {
    "messages": [HumanMessage(content="start")],
    "agent1_result": "",
    "agent2_result": "",
    "completed_nodes": []
}
final = graph.invoke(initial)
print(final["messages"][-1].content)  # → "COMBINED: foo + bar"
