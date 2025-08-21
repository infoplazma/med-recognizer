"""

"""
# **********************************************************************************************************************
#                                          Импорты и предустановки
# **********************************************************************************************************************
import operator
import inspect
from typing import List, Literal, TypedDict, Annotated

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document as LlamaDocument

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send
from langgraph.graph import END, START, StateGraph

from utils.custom_print import custom_pretty_print, pretty_print_json

GRAPH_PNG_FILE_PATH = "graphs/workflow.png"
DEFAULT_MODELS = ("llama3-med42-8b", "biomedgpt-lm-7b", "tinyllama-1.1b-chat-v1.0")
MAX_TOKENS = 1000
CHUNK_SIZE = 300
DEFAULT_MODEL = DEFAULT_MODELS[0]
CONFIG = {"tags": ["env:prod", "feature:qa", "define_disease"]}

print("\rLoading...", end="")

# Инициализация LLM
llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="not-needed-for-lm-studio",
    model_name=DEFAULT_MODEL,
    temperature=0,
)
# Загрузка и разбиение документа
with open("../tests/data/example.txt", "r", encoding="utf-8") as fp:
    sample_text = fp.read()

# Создаем документ
document = LlamaDocument(text=sample_text)

# Инициализируем SentenceSplitter с параметрами
# chunk_size - максимальное количество символов в чанке
# chunk_overlap - перекрытие между чанками в символах
splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=100,
)

# Разбиваем текст на чанки
nodes = splitter.get_nodes_from_documents([document])
contents = [node.text for node in nodes]
print("\rДокументы загружены и разделены", end="")

# Промпты для map и reduce
sys_message = ("system", """You are an experienced doctor who reads and understands medical literature well 
and can explain it to others concisely.""")
map_prompt = ChatPromptTemplate.from_messages([("human", "Write a concise summary of the following:\n\n{context}")])
map_chain = map_prompt | llm | StrOutputParser()

disease_prompt = ChatPromptTemplate.from_messages([
    sys_message,
    ("human", """Determine what a disease or diseases are being discussed in the following:
    ---
    {context}
    ---
    In your answer, give only the name of the disease or diseases separated by commas;
    if the context provided does not refer to any disease, 
    answer in one word 'None'. """)
])
disease_chain = disease_prompt | llm | StrOutputParser()

reduce_template = """The following is a set of summaries:\n{docs}\nTake these and distill it into a final, consolidated summary of the main themes."""
reduce_prompt = ChatPromptTemplate.from_messages([("human", reduce_template)])
reduce_chain = reduce_prompt | llm | StrOutputParser()


# Функция для подсчета токенов в тексте
def count_num_tokens(text: str) -> int:
    return llm.get_num_tokens(text)


# Функция для подсчета всех токенов в списке
def length_function(documents: List[Document]) -> int:
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)


# **********************************************************************************************************************
#                                          Определение состояния и узлов
# **********************************************************************************************************************

# This will be the overall state of the main graph.
# It will contain the input document contents, corresponding
# summaries, and a final summary.
class OverallState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    contents: List[str]
    summaries: Annotated[list, operator.add]
    diseases: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str


# This will be the state of the node that we will "map" all
# documents to in order to generate summaries
class SummaryState(TypedDict):
    content: str
    index: int


# Here we generate a summary, given a document
def generate_summary(state: SummaryState, config: RunnableConfig):
    response = map_chain.invoke(state["content"])
    return {"summaries": [response]}


# Here we define a disease, given a document
def define_disease(state: SummaryState, config: RunnableConfig):
    response = disease_chain.invoke(state["content"])
    tags = config.get("tags", [])
    if inspect.currentframe().f_code.co_name in tags:
        print(f"\r{response=}", end="")
    return {"diseases": [response]}


# Here we define the logic to map out over the documents
# We will use this an edge in the graph
def map_summaries(state: OverallState):
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [
        Send("generate_summary", {"content": content, "index": index}) for index, content in
        enumerate(state["contents"], start=1)
    ] + [
        Send("define_disease", {"content": content, "index": index}) for index, content in
        enumerate(state["contents"], start=1)
    ]


# Construct the graph
# Nodes:
graph = StateGraph(OverallState)
graph.add_node("generate_summary", generate_summary)
graph.add_node("define_disease", define_disease)

# Edges:
graph.add_conditional_edges(START, map_summaries, ["generate_summary", "define_disease"])
graph.add_edge("generate_summary", END)
graph.add_edge("define_disease", END)
# Compile the graph
workflow = graph.compile()

# Visualize your graph
png_data = workflow.get_graph().draw_mermaid_png()
with open(GRAPH_PNG_FILE_PATH, "wb") as f:
    f.write(png_data)
print(f"\rГраф успешно сохранен в файл '{GRAPH_PNG_FILE_PATH}'", end="")

print(f"\rReasoning...")
init_state = {"contents": contents}
result = workflow.invoke(init_state, config=CONFIG)

print(f"Result:")
for i, (summary, disease) in enumerate(zip(result["summaries"], result["diseases"]), start=1):
    num_tokens = count_num_tokens(summary)
    custom_pretty_print(f"{i}. {num_tokens=}  {disease=}", summary)
