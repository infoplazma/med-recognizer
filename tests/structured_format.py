from typing import Optional, TypedDict, Annotated
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field, model_validator


DEFAULT_MODEL = "tinyllama-1.1b-chat-v1.0"

# Model
model = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="not-needed-for-lm-studio",
    model_name=DEFAULT_MODEL,
    temperature=0.0,
)


# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # You can add custom validation logic easily with Pydantic.
    @model_validator(mode="before")
    @classmethod
    def question_ends_with_question_mark(cls, values: dict) -> dict:
        setup = values.get("setup")
        if setup and setup[-1] != "?":
            raise ValueError("Badly formed question!")
        return values


# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

print(prompt.invoke({"query": "Tell me a medical joke."}))
json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {query}"
)
json_parser = SimpleJsonOutputParser()
json_chain = json_prompt | model | json_parser

output = json_chain.invoke({"query": "Tell me a medical joke."})
print(output)


# TypedDict
class DoctorJoke(TypedDict):
    """Doctor Joke to tell user."""

    setup: Annotated[str, ..., "The setup of the doctor joke"]

    # Alternatively, we could have specified setup as:

    # setup: str                    # no default, no description
    # setup: Annotated[str, ...]    # no default, no description
    # setup: Annotated[str, "foo"]  # default, no description

    punchline: Annotated[str, ..., "The punchline of the doctor joke"]
    rating: Annotated[Optional[int], None, "How funny the doctor joke is, from 1 to 10"]


structured_llm = model.with_structured_output(DoctorJoke, include_raw=True)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a doctor with a sense of humor who can deliver jokes in the format JSON"),
    ("human", "Tell me a joke about {query}")
])

print("\nPrompt:")
print(prompt.invoke({"query": "diarrhea"}).to_string())
print()

structured_chain = prompt | structured_llm
output = structured_chain.invoke({"query": "diarrhea"})
print(output['parsed'])

output = structured_llm.invoke("Tell me a joke about runny nose")
print(output)

# And a query intended to prompt a language model to populate the data structure.
# prompt_and_model = prompt | model
# output = prompt_and_model.invoke({"query": "Tell me a medical joke."})
# print(output)
# parser.invoke(output)
