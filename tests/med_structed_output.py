"""
Пример использования медицинских инструментов
"""
from typing import Optional, Sequence

from typing_extensions import Annotated, TypedDict
from utils.custom_print import custom_pretty_print

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

DEFAULT_MODEL = "tinyllama-1.1b-chat-v1.0"  # "llama3-med42-8b" "biomedgpt-lm-7b" "tinyllama-1.1b-chat-v1.0"

# Model
default_llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="not-needed-for-lm-studio",
    model_name=DEFAULT_MODEL,
    temperature=0.1,
)


class Diagnosis(TypedDict):
    name: Annotated[str, ..., "The exact name of the diagnosis according to the icd10"]
    symptoms: Annotated[Sequence[str], ..., "Symptoms list"]
    description: Annotated[str, ..., "Brief description of the disease"]
    links: Annotated[Sequence[str], ..., "Links to medical source"]


structured_llm = default_llm.with_structured_output(Diagnosis, include_raw=False)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a doctor who determines the diagnosis, knows the scientific name of the diagnosis or disease, "
               "according to the icd10 and clearly defines the symptoms associated with a specific disease"),
    ("human", "Tell me a about '{query}'")
])

chain = {"query": RunnablePassthrough()} | prompt | structured_llm

query = "Cryptosporidiosis"
ai_msg = chain.invoke(query)
custom_pretty_print(query, ai_msg)
