"""

"""
# **********************************************************************************************************************
#                                          Импорты и предустановки
# **********************************************************************************************************************
import os
import operator
import inspect
from typing import List, Literal, TypedDict, Annotated

from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document as LlamaDocument

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from utils.custom_print import custom_pretty_print, pretty_print_json
from utils.read_pdf_page import read_page_text_blocks

from toolkit.prompt_templates.chunk_page_reader_prompts import EDITOR_SYSTEM_PROMPT, EDITOR_HUMAN_PROMPT
from toolkit.disease_headings import get_heading_candidates


PDF_SOURCE_FILE = "Guide-to-Common-Childhood-Infections-2023_Final-Approved.pdf"
PAGE_NUM = 14  # Номер страницы (начинается с 0)

DISPLAY_GRAPH = True
GRAPH_PNG_FILE_PATH = "graphs/chunk_page_reader.png"

DEFAULT_MODELS = ("llama3-med42-8b", "openbiollm-llama3-8b")
MAX_TOKENS = 1000
CHUNK_SIZE = 250
DEFAULT_MODEL = DEFAULT_MODELS[0]
CONFIG = RunnableConfig(
    tags=["diseases", "headings", "critic", "editor"],
    metadata={"pdf_file": PDF_SOURCE_FILE, "page": PAGE_NUM},
    run_name="HeadingDiseaseExtraction")

PDF_PATH = os.path.join("../data", "med_sources", PDF_SOURCE_FILE)
EXAMPLE_TEXT_PATH = os.path.join("../tests/data/example.txt")

print("\rLoading...", end="")

# Инициализация LLM
llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="not-needed-for-lm-studio",
    model_name=DEFAULT_MODEL,
    temperature=0,
)

# load_dotenv()
# llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)
# Загрузка и разбиение документа
sample_text = read_page_text_blocks(PDF_PATH, PAGE_NUM)
sample_text = sample_text.strip()
with open(EXAMPLE_TEXT_PATH, "w", encoding="utf-8") as fp:
    fp.write(sample_text)
print(f"Образец текста сохранен: {os.path.abspath(EXAMPLE_TEXT_PATH)}")
# Создаем документ
document = LlamaDocument(text=sample_text)

# Инициализируем SentenceSplitter с параметрами
# chunk_size - максимальное количество символов в чанке
# chunk_overlap - перекрытие между чанками в символах
splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=0,
)
# Разбиваем текст на чанки
nodes = splitter.get_nodes_from_documents([document])
contents = [node.text for node in nodes]
print("Документы загружены и разделены")
print("\rReasoning...", end="")


# **********************************************************************************************************************
#                                          Определение функций и классов
# **********************************************************************************************************************
# Функция для подсчета токенов в тексте
def count_num_tokens(text: str) -> int:
    return llm.get_num_tokens(text)


def check_tokens_overage(text: str, max_tokens: int = 2048, tag: str = None):
    if count_num_tokens(text) > max_tokens:
        raise ValueError(
            f"{tag}:" if tag else "" + f"Превышение количества токенов {count_num_tokens(text)} "
                                       f"при допустимом {max_tokens}")


# **********************************************************************************************************************
#                                          Определение промптов, состояния и узлов
# **********************************************************************************************************************

# Промпты для map и reduce
sys_message = ("system", "You are an experienced doctor who reads and understands medical literature well "
                         "and can explain it to others concisely.And besides that You are an information "
                         "extraction assistant for clinical texts.")

# **************************************** diseases
disease_prompt = ChatPromptTemplate.from_messages([
    sys_message,
    ("human", """
    Analyze the following text to identify the diseases (diagnoses, infections) being discussed:
    ---
    {context}
    ---
    To connect with the previous context, use:
    - Summary of the previous chunk (may be empty):
      ---
      {prev_summary}
      ---
    - List of previously identified diseases (may be empty): {prev_diseases}
    
    **Instructions**:
    1. Identify the diseases (diagnoses, infections) mentioned **only as headings or titles**  in the current chunk.
        A heading or title is defined as:
        - Text formatted as a section or subsection title (e.g., bold, larger font, or structurally separated in the document).
        - Text that introduces a section discussing a specific disease, typically at the start of a paragraph or section.
        - Explicit disease names (e.g., "Diabetes", "Cardiopulmonary Resuscitation (CPR)") appearing as standalone terms in a heading, not within descriptive text or sentences.
        Consider explicit names (e.g., "diabetes") as well as implicit mentions (e.g., symptom descriptions indicating a disease).
    2. Specify the type of the current chunk:
       - "continuation": The chunk continues the description of a disease (diagnoses, infections) from the previous chunk.
       - "continuation_and_new": The chunk continues the description of a previous disease and starts a new disease description.
       - "new": The chunk describes only a new disease.
       - "none": The chunk contains no disease mentions.
    3. If multiple diseases are mentioned, list them separated by commas.
    4. If a disease has synonyms (e.g., "cardiopulmonary resuscitation" and "CPR"), use the primary name.
    5. If the text contains medical terms in English, interpret them correctly.
    
    **Output ONLY IN JSON Format**:
    ```json
    {{
      "diseases": ["disease_name_1", "disease_name_2"],
      "chunk_type": "continuation|continuation_and_new|new|none"
    }}```
    Don't write explanations and don't give comments only JSON !
    """)
])
disease_chain = disease_prompt | llm | JsonOutputParser()

# **************************************** define headings
disease_headings_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are an expert in medical terminology tasked with identifying disease headings from a list of candidates. 
        A disease heading is a term that explicitly refers to a disease, diagnosis, or infection 
        (e.g., "Measles", "Meningitis", "Diabetes"). 
        Exclude terms that refer to symptoms (e.g., "Fever"), medical procedures (e.g., "CPR"), or other medical 
        indicators (e.g., "Incubation period"). 
        Handle synonyms by selecting the primary disease name (e.g., use "Cardiopulmonary Resuscitation" 
        instead of "CPR"). 
        Interpret terms correctly if the text is in English, Russian, or contains mixed medical terminology.
    """),

    ("human", """
        Determine which of the presented candidate headings are disease headings (only diseases, diagnoses, or names 
        of infections, not symptoms, procedures, or other medical indicators).

        **Candidates (JSON) (may be empty)**:
        ```json
        {heading_candidates}
        ```
        **Output ONLY IN JSON Format**:
        ```json
        ["disease_name_1", "disease_name_2"]
        ```
        Don't write explanations and don't give comments only JSON !
    """)
])
disease_headings_chain = disease_headings_prompt | llm | JsonOutputParser()
# **************************************** check headings
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are an expert in medical terminology tasked with identifying contradictions between extracted data and the 
        original text chunk. A contradiction occurs when:
        - A candidate in `Headings` is a valid disease (e.g., "Diarrhea" described as an infectious disease 
            with incubation period, infectious period, or exclusion recommendations) but is not included in `Diseases`.
        - A candidate in `Headings` is not a disease (e.g., a symptom like "Fever" or a procedure like "CPR") 
            and is incorrectly included in `Diseases`.
        - A term in `Medical terms` or `Diseases` is not supported by the chunk text or contradicts `Headings`.
        - There are inconsistencies in terminology (e.g., different names for the same disease without synonym mapping).
        - There are duplicates or conflicting descriptions for the same disease.
        Handle synonyms correctly (e.g., "Cardiopulmonary Resuscitation" and "CPR" are the same).
        Interpret terms correctly if the text is in English, Russian, or contains mixed medical terminology.
        Use the chunk text to verify the validity of disease headings and terms, ensuring that all valid disease 
        headings from `Headings` are included in `Diseases`.
        """),

    ("human", """
        Analyze the chunk text and determine if there are any contradictions between the extracted data and the text.

        **`Headings` (JSON, list of candidate disease names, may be empty)**: 
        ```json
        {disease_headings}
        ```
        **`Medical terms` (JSON, list of terms with definitions, may be empty)**: 
        ```json
        {medical_terms}
        ```
        **`Diseases` (JSON, list of diseases with chunk type, may be empty)**: 
        ```json
        {diseases}
        ```
        **`Chunk text`**: 
        ```
        {context}
        ```
        **Instructions**:
        1. Check for contradictions between `Headings`, `Medical terms`, `Diseases`, and the `Chunk text`.
        2. A contradiction is:
           - A candidate in `Headings` is a valid disease (e.g., "Diarrhea" described as an infectious disease) but is not included in `Diseases`.
           - A term in `Medical terms` or `Diseases` is not supported by the chunk text or contradicts `Headings`.
           - Inconsistent terminology (e.g., using different names for the same disease without synonym mapping).
           - Duplicates or conflicting descriptions for the same disease.
        3. Handle synonyms (e.g., "CPR" and "Cardiopulmonary Resuscitation") as the same disease.
        4. Verify that all valid disease headings from `disease_headings` are included in `Diseases` if supported by the chunk text.
        5. Return "Yes" if contradictions are found, "No" if none, with a detailed explanation in `details`.

        **Output ONLY IN JSON Format**: 
        ```json
        {{ "contradiction": "Yes" | "No", "details": "Explanation of contradictions, if any" }}
        ```
        Don't write explanations and don't give comments only JSON !
    """)
])

critic_chain = critic_prompt | llm | JsonOutputParser()

# **************************************** editor
editor_prompt = ChatPromptTemplate.from_messages([
    ("system", EDITOR_SYSTEM_PROMPT),
    ("human", EDITOR_HUMAN_PROMPT)
])
editor_chain = editor_prompt | llm | JsonOutputParser()


# **************************************** final_diseases
class Diagnoses(TypedDict):
    diseases: Annotated[List[str], ..., "The exact list of the diagnosis according to the icd10"]


final_diseases_prompt = ChatPromptTemplate.from_messages([
    sys_message,
    ("human", """
    From the list provided:
        ---
        {diseases}
        ---
    select the final list of diseases.
    Response only in the JSON format.""")
])
disease_structured_llm = llm.with_structured_output(Diagnoses)
final_diseases_chain = final_diseases_prompt | disease_structured_llm


# **************************************** medical_terms
# Определение структуры
class MedicalTerm(TypedDict):
    term: Annotated[str, ..., "Medical term from the provided context according to the icd10"]
    definition: Annotated[str, ..., "Definition of medical term from the provided context"]


class MedicalTerms(TypedDict):
    medical_terms: Annotated[List[MedicalTerm], ..., "Extracted medical terms and their definitions from the provided text according to the icd10"]


medical_terms_prompt = ChatPromptTemplate.from_messages([
    sys_message,
    ("human", "Extract key medical terms and their definitions from text in JSON format, text:{text}")
])
medical_terms_llm = llm.with_structured_output(MedicalTerms)
medical_terms_chain = medical_terms_prompt | medical_terms_llm


# --- узлы/состояние ---
class OverallState(TypedDict):
    contents: List[str]
    diseases: Annotated[list, operator.add]
    medical_terms: Annotated[list, operator.add]
    headings: Annotated[list, operator.add]
    final_diseases: str
    index: int
    contradiction_details: dict
    after_editing: bool


def process_chunk(state: OverallState, config: RunnableConfig) -> OverallState:
    # берём следующий индекс = текущее число сводок
    index = len(state.get("diseases", []))
    content = state["contents"][index]
    prev_summary = state["contents"][index - 1] if index else ""
    prev_diseases = state["diseases"][index - 1] if index and index <= len(state.get("diseases", [])) else ""

    input_text = disease_prompt.format(context=content, prev_summary=prev_summary, prev_diseases=prev_diseases)
    check_tokens_overage(input_text, tag="process_chunk")

    disease_res = disease_chain.invoke({
        "context": content,
        "prev_summary": prev_summary,
        "prev_diseases": prev_diseases,
    })
    medical_terms_res = medical_terms_chain.invoke(content)

    tags = config.get("tags", [])
    if "diseases" in tags:
        print(f"\rПросмотрел {index+1}-ю часть страницы. Обнаружил что речь идет о {disease_res}.\nReasoning...", end="")
    return {"diseases": [disease_res], "medical_terms": [medical_terms_res], "index": index, "after_editing": False}


def check_headings(state: OverallState, config: RunnableConfig) -> OverallState:
    index = state["index"]
    content = state["contents"][index]
    heading_candidates = get_heading_candidates(content)
    try:
        headings_res = disease_headings_chain.invoke(heading_candidates)
    except ValueError:
        headings_res = []
    if "headings" in config.get("tags", []):
        print(f"\rИзучил возможных кандидатов на заголовки: {headings_res}.\nReasoning...", end="")

    return {"headings": [headings_res]}


def apply_critic(state: OverallState, config: RunnableConfig) -> OverallState:
    # Если было редактирование, тогда критик не делает заключение
    if state.get("after_editing", False) and state["after_editing"]:
        return {}

    content = state["contents"][-1]
    disease_headings = state["headings"][-1]
    medical_terms = [term["term"] for term in state["medical_terms"][-1]["medical_terms"]]
    diseases = state["diseases"][-1]
    input_text = critic_prompt.format(context=content, disease_headings=disease_headings, medical_terms=medical_terms,
                                      diseases=diseases)
    check_tokens_overage(input_text, tag="apply_critic")
    try:
        contradiction_details = critic_chain.invoke({"context": content,
                                                     "disease_headings": disease_headings,
                                                     "medical_terms": medical_terms,
                                                     "diseases": diseases})
    except ValueError:
        contradiction_details = {"contradiction": "No", "details": "Error: Incorrect parsing"}
    if "critic" in config.get("tags", []):
        print(f"\rПроверил возможные противоречия: {contradiction_details}.\nReasoning...", end="")

    return {"contradiction_details": contradiction_details}


# Here we implement logic to either exit the application or refine
# the summary.
def should_continue(state: OverallState, config: RunnableConfig) -> Literal["apply_editor", "process_chunk", "final_diseases"]:
    if not state["after_editing"] and state["contradiction_details"]['contradiction'].lower() == "yes":
        return "apply_editor"

    if state["index"] == len(state["contents"]) - 1:
        return "final_diseases"
    else:
        return "process_chunk"


def apply_editor(state: OverallState, config: RunnableConfig) -> OverallState:
    content = state["contents"][-1]
    disease_headings = state["headings"][-1]
    medical_terms = [term["term"] for term in state["medical_terms"][-1]["medical_terms"]]
    diseases = state["diseases"][-1]
    contradiction_details = state["contradiction_details"]
    input_text = critic_prompt.format(context=content, disease_headings=disease_headings, medical_terms=medical_terms,
                                      diseases=diseases, contradiction_details=contradiction_details)
    check_tokens_overage(input_text, tag="should_continue")

    diseases_res = editor_chain.invoke({"context": content,
                                        "disease_headings": disease_headings,
                                        "medical_terms": medical_terms,
                                        "diseases": diseases,
                                        "contradiction_details": contradiction_details})
    if "editor" in config.get("tags", []):
        print(f"\rОтредактировал список заболеваний: {diseases_res}.\nReasoning...", end="")
    state["diseases"][-1] = diseases_res
    return {"after_editing": True}


def final_diseases(state: OverallState, config: RunnableConfig):
    input_text = final_diseases_prompt.format(diseases=state["diseases"])
    check_tokens_overage(input_text, tag="final_diseases")

    final_diseases_res = final_diseases_chain.invoke(state["diseases"])
    return {"final_diseases": final_diseases_res}


graph = StateGraph(OverallState)
graph.add_node("process_chunk", process_chunk)
graph.add_node("check_headings", check_headings)
graph.add_node("apply_critic", apply_critic)
graph.add_node("apply_editor", apply_editor)
graph.add_node("final_diseases", final_diseases)

graph.add_edge(START, "process_chunk")
graph.add_edge("process_chunk", "check_headings")
graph.add_edge("check_headings", "apply_critic")
graph.add_edge("apply_editor", "apply_critic")
graph.add_conditional_edges("apply_critic", should_continue, {"apply_editor": "apply_editor",
                                                              "process_chunk": "process_chunk",
                                                              "final_diseases": "final_diseases"})
graph.add_edge("final_diseases", END)
workflow = graph.compile()

# визуализация
if DISPLAY_GRAPH:
    png = workflow.get_graph().draw_mermaid_png()
    with open(GRAPH_PNG_FILE_PATH, "wb") as f:
        f.write(png)
    print(f"\rГраф успешно сохранен в файл '{GRAPH_PNG_FILE_PATH}'", end="")

# запуск: инициализируйте списки сразу
init_state = {"contents": contents}
result = workflow.invoke(init_state, config=CONFIG)


print(f"\rЗакончил рассуждать.")

# подсчёт токенов (у ChatOpenAI есть get_num_tokens)
for i, (disease, medical_terms, headings) in enumerate(zip(result["diseases"], result["medical_terms"], result["headings"]), start=1):
    print(f"chunk: {i}")
    pretty_print_json(disease)
    pretty_print_json(medical_terms)
    pretty_print_json(headings)
    print("---")
#     # custom_pretty_print("summary:", summary)


custom_pretty_print("\n\nDiseases:", result["final_diseases"])
pretty_print_json(result["final_diseases"])
