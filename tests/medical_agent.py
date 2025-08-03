import json
import re
from langchain_core.language_models.base import BaseLanguageModel
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage

from settings import default_llm
from utils.json_response import clean_json_response


# --- Preprocess text ---
def preprocess_text(text: str) -> str:
    # Удаляем не-ASCII символы, нормализуем пробелы и разрывы строк
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Удаляем не-ASCII символы
    text = re.sub(r'\s+', ' ', text)  # Заменяем множественные пробелы на одиночные
    text = re.sub(r'\n+', ' ', text)  # Заменяем разрывы строк на пробелы
    return text.strip()


# --- 1. Tool: Disease Extraction ---
def extract_diseases(snippet: str, llm: BaseLanguageModel = default_llm) -> str:
    snippet = preprocess_text(snippet)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a medical analyst. Extract all diseases or diagnoses from the provided medical text. "
                   "Include all conditions mentioned, even if they appear in headings, as short mentions (e.g., 'Covid 19'), "
                   "or are related to viruses (e.g., 'Varicella zoster virus' implies 'varicella' and 'zoster'). "
                   "Exclude symptoms (e.g., 'diarrhea', 'fever') and treatments (e.g., 'antibiotics', 'retinoid'). "
                   "Be thorough and consider all medical terms. Return only a valid JSON array of disease names, or [] if none are found."),
        ("user",
         "Fragment:\n{snippet}\n---\nReturn only a JSON array of disease/diagnosis names, e.g., [\"disease1\", \"disease2\"].")
    ])
    chain = prompt | llm
    raw_response = chain.invoke({"snippet": snippet})
    content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
    result = clean_json_response(content)
    print(f"ExtractDiseases input: {snippet[:100]}...")
    print(f"ExtractDiseases output: {result}")
    return result


extract_diseases_tool = Tool(
    name="ExtractDiseases",
    func=extract_diseases,
    description="Extracts from a medical document fragment a JSON array of disease/diagnosis names, or [] if none."
)


# --- 2. Tool: Disease Verification ---
def verify_diseases(diseases: str, llm: BaseLanguageModel = default_llm) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a medical analyst. Determine which of the names in the provided JSON array are real diseases or medical diagnoses. "
         "Exclude symptoms (e.g., 'diarrhea', 'fever') and treatments (e.g., 'antibiotics', 'retinoid'). "
         "Return only a JSON array of valid names, or [] if none are appropriate. Ensure the response is a valid JSON string."),
        ("user",
         "Disease/diagnosis list:\n{input}\n---\nReturn only a JSON array of valid disease/diagnosis names, e.g., [\"disease1\", \"disease2\"].")
    ])
    chain = prompt | llm
    raw_response = chain.invoke({"input": diseases})
    content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
    result = clean_json_response(content)
    print(f"VerifyDiseases input: {diseases}")
    print(f"VerifyDiseases output: {result}")
    return result


verify_diseases_tool = Tool(
    name="VerifyDiseases",
    func=verify_diseases,
    description="Verifies a JSON array of disease/diagnosis names and returns only valid real diagnoses (or an empty array)."
)

# --- Initialize Agent with LangGraph ---
agent = create_react_agent(
    model=default_llm,
    tools=[extract_diseases_tool, verify_diseases_tool]
)


# --- Post-process agent output to extract JSON ---
def extract_json_from_output(output: str) -> str:
    # Ищем JSON-массив в тексте с помощью регулярного выражения
    json_pattern = r'\[\s*(?:"[^"]*"(?:\s*,\s*"[^"]*")*)*\s*\]'
    match = re.search(json_pattern, output)
    if match:
        json_str = match.group(0)
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            return "[]"
    return "[]"


# --- Test Block ---
if __name__ == "__main__":
    import os
    import glob

    print("\nAgent — comparative test for extraction and verification of diagnoses:")
    print("-" * 100)
    test_dir = "../data/test_chunks"
    for file_path in sorted(glob.glob(os.path.join(test_dir, "*.txt"))):
        file_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as file:
            example = file.read()
            print(f"=== {file_name} ===\n")
            print(f"Input text: {example[:100]}...")
            prompt = (
                "You are a medical analyst. Complete the following task in exactly two steps:\n"
                "1) Use ExtractDiseases to extract all diseases/diagnoses from the following text, including diseases related to mentioned viruses and those in headings.\n"
                "2) Use VerifyDiseases to verify the extracted diseases and return only valid names as a JSON array.\n"
                "Do not repeat any steps. Return only a JSON array of valid disease names, e.g., [\"disease1\", \"disease2\"].\n"
                f"Text: '''{example}'''"
            )
            try:
                # Выполняем агент с использованием langgraph
                result = agent.invoke({
                    "messages": [HumanMessage(content=prompt)]
                })
                # Извлекаем последнее сообщение
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    output_message = last_message.content
                else:
                    output_message = str(last_message)
                print(f"Agent output: {output_message}")
                # Извлекаем JSON из ответа
                json_output = extract_json_from_output(output_message)
                try:
                    diseases_ = json.loads(json_output)
                except json.JSONDecodeError as e:
                    print(
                        f"Failed to parse agent response: {output_message}, extracted JSON: {json_output}, error: {e}")
                    diseases_ = []
                print("Diseases found:", diseases_)
            except Exception as e:
                print(f"Agent execution failed: {e}")
            print("-" * 100)
