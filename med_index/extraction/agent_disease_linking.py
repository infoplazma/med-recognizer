"""
agent_linked_diseases.py

Современный пример: LLMChain как Tool для агента + интеративное увеличение joint_size (wrap tool).
"""
import json
from typing import List, Dict, Any
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import ChatPromptTemplate

from settings import default_llm
from utils.json_response import clean_json_response


# 1. Готовим LLM и Prompt для одной проверки

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ты медицинский аналитик. Определи, есть ли смысловая связь по заболеванию между двумя соседними фрагментами."),
    ("user",
     "Вот два фрагмента медицинского документа.\n"
     "Твоя задача: выявить названия заболеваний/диагнозов, которые обсуждаются в обоих фрагментах, "
     "и связь между ними продолжается. Ответ — JSON-массив названий или [] если нет связи."
     "\n---\nFragment 1:\n{frag1}\n---\nFragment 2:\n{frag2}")
])

chain = prompt | default_llm


# 2. Tool-обёртка: принимает joint_size как аргумент
def check_linked_tool(prev_text: str, next_text: str, joint_size: int) -> List[str]:
    frag1 = prev_text[-joint_size:]
    frag2 = next_text[:joint_size]
    res = chain.invoke({"frag1": frag1, "frag2": frag2})
    try:
        return json.loads(clean_json_response(res["text"]))
    except Exception:
        return []


# 3. Интерактивная функция — автоматизирует подбор joint_size (для удобства)
def find_linked_diseases_iterative_agent(
        prev_text: str,
        next_text: str,
        initial_joint: int = 300,
        max_joint: int = 900,
        max_attempts: int = 4
) -> List[str]:
    joint_size = initial_joint
    attempt = 0
    while attempt < max_attempts and joint_size <= max_joint:
        result = check_linked_tool(prev_text, next_text, joint_size)
        if result:
            print(f"[DEBUG] Связь найдена при joint_size={joint_size}: {result}")
            return result
        joint_size = int(joint_size * 1.6)
        attempt += 1
    print(f"[DEBUG] Связь не найдена (joint_size={joint_size}).")
    return []


# 4. Оформим Tool для LangChain агента (если нужен внешний вызов)
linked_diseases_tool = Tool(
    name="CheckLinkedDiseases",
    func=lambda args: find_linked_diseases_iterative_agent(
        args["prev_text"], args["next_text"],
        initial_joint=args.get("initial_joint", 300)
    ),
    description="Проверяет, есть ли связь по заболеванию между двумя соседними чанками с адаптивным joint_size."
)

# 5. Пример использования с агентом
if __name__ == "__main__":

    prev_text = "Hepatitis B is a viral infection... Patients may develop jaundice..."
    next_text = "Jaundice and tiredness are often seen in hepatitis B. Prevention guidelines are..."
    # Для самостоятельного вызова функции:
    print("Связанные заболевания:", find_linked_diseases_iterative_agent(prev_text, next_text))

    # Для вызова через агента (пример, если хочется автоподбор инструмента):
    agent = initialize_agent(
        tools=[linked_diseases_tool],
        llm=default_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=5,  # максимум 5 reasoning-циклов! max_iterations (или max_steps, иногда max_execution_time)
        verbose=True,
    )
    result = agent.run({
        "prev_text": prev_text,
        "next_text": next_text,
        "initial_joint": 300
    })
    print("Agent result:", result)
