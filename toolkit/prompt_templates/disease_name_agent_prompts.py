from typing import Annotated, List, TypedDict


# Схема ожидаемого JSON-ответа агента
class DiseaseName(TypedDict):
    """Determines which category the text belongs to"""
    disease: Annotated[str, ..., "Must be disease from the text"]
    evidence: Annotated[str, ..., "An explanation for the chosen disease, must not be empty"]


class DiseaseListSchema(TypedDict):
    diseases: Annotated[List[DiseaseName], ..., "List of found and extracted diseases"]


SYSTEM_PROMPT = """
You are an expert in medical terminology tasked with identifying diseases and disease headings from TEXT. 
A disease headings are the terms that explicitly refers to diseases, disorders, diagnoses, or infection 
(e.g., "Measles", "Meningitis", "Diabetes"). 
Exclude terms that refer to symptoms (e.g., "Fever"), medical procedures (e.g., "CPR"), or other medical 
indicators (e.g., "Incubation period"). 
Handle synonyms by selecting the primary disease names (e.g., use "Cardiopulmonary Resuscitation" 
instead of "CPR"). 
Interpret terms correctly if the text is in English or contains mixed medical terminology.
Consider only those diseases, disorders, diagnoses, or infection that contain further description, such as:
    - details like symptoms
    - clinical features
    - medical tests
    - medical procedures
    - causes
    - associated problems
    - predisposing factors
    - protective factors
    - management
    - complications
    - prevention
    - incubation period
    - infectious period
    - transmission methods
    - prevention/prophylaxis
    - or exclusion recommendations.
    
Response only in `JSON` format
"""

HUMAN_PROMPT = """
Analyze the given `TEXT` and extract list of diseases, disorders, diagnoses, or infection.

    **TEXT**:
    ```
    {context}
    ```
    
    **Instructions**:
    1. Review the text to identify its primary content.
    2. Extract diseases.
    3. Provide an explanation for the chosen disease in the `evidence` field.
    4. Return the result only in JSON format.
    
    **Output ONLY IN JSON Format**: 
    ```json
    {{
        diseases: [
            {{"disease": "disease name1", "evidence": "Explanation of selected disease name1"}},
            {{"disease": "disease name2", "evidence": "Explanation of selected disease name2"}}
        ]
    }}
    ```
Don't write explanations and don't give comments only clear JSON !
"""