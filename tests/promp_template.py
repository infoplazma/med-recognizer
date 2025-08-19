from utils.custom_print import custom_pretty_print
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage


prompt = PromptTemplate.from_template("Share an interesting fact about {animal}.")  # infers 'animal' as input variable

# Format the template with a specific animal
# filled_prompt = prompt.format(animal="octopus")
filled_prompt = prompt.invoke({"animal": "octopus"})
custom_pretty_print("PromptTemplate:", filled_prompt)

# ******************************************
# Use one of 'human', 'user', 'ai', 'assistant', or 'system'
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a patient tutor who explains things clearly."),
    ("human", "Can you explain {concept} like I'm five?"),
    ("ai", "Yes, I can"),
    ("placeholder", "{examples}"), # Для вставки списка BaseMessages
    HumanMessage(content="Human is human {person}")
])

examples = [ToolMessage(777, tool_call_id=7), AIMessage("BM")]

# Fill in the template with a specific concept
formatted_messages = chat_prompt.format_messages(concept="gravity", person="Alex", examples=examples)

custom_pretty_print("ChatPromptTemplate:", formatted_messages)
