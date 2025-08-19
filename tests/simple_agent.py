from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.tools import Tool
from langchain_core.tools import tool
from langchain.tools import BaseTool
from settings import default_llm, openai_llm
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


# Рабочие подходы
math_tool = Tool(
    name="Calculator",
    func=lambda x: str(eval(x)),
    description="Useful for math reasoning."
)

calc_tool = Tool.from_function(lambda x: str(eval(x)), name="Math Calculator", description="for math")


# Не рабочие подходы
# class CircumTool(BaseTool):
#     name = "Math new calculator"
#     description = "Useful for math reasoning"
#
#     def _run(self, x: str) -> str:
#         return str(eval(x))
#
#
# @tool
# def multiply(x: str) -> str:
#     """Multiply two numbers."""
#     return str(eval(x))


agent = initialize_agent(
    tools=[calc_tool],
    llm=openai_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=2,
    verbose=True
)

result = agent.invoke({"input": "What is 2 + 2?"})
print(result["output"])  # → "4"

