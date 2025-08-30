from typing import Annotated, Literal, TypedDict


# Схема ожидаемого JSON-ответа агента
class TypeScheme(TypedDict):
    """Determines which category the text belongs to"""
    category: Annotated[Literal[
                            "CONTENT TABLE", "GENERAL", "DISEASES DESCRIPTION", "OTHER"], ...,
                        "Must be one of: CONTENT TABLE, GENERAL, DISEASES DESCRIPTION, OTHER"]
    evidence: Annotated[str, ..., "An explanation for the chosen category, must not be empty"]


TYPE_SYSTEM_PROMPT = """
You are a professional medical editor tasked with classifying the type of medical text provided. Analyze the text and 
assign it to one of the following categories based on its content:

    - CONTENT TABLE: The text contains a table of contents, index, or list of sections for a book or article.
    - GENERAL: The text includes metadata about the book or article, such as author information, publication details, 
        or sources.
    - DISEASES DESCRIPTION: The text provides specific descriptions of diseases, disorders, diagnoses, or infections, 
    including:
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
    - OTHER: The text contains information that does not fit into the above categories, such as general advice, 
        non-specific medical information, or unrelated content.

    Use the context to determine the most appropriate category. Ensure that texts describing diseases, 
    such as "Diarrhea" with details like incubation period or exclusion recommendations, 
    are classified as DISEASES DESCRIPTION.
"""

TYPE_HUMAN_PROMPT = """
Analyze the given text and classify it into one of the specified categories.

    **Text**:
    ```
    {context}
    ```

    **Instructions**:
    1. Review the text to identify its primary content.
    2. Assign the text to one category: CONTENT TABLE, GENERAL, DISEASES DESCRIPTION, or OTHER.
    3. Provide an explanation for the chosen category in the evidence field.
    4. Return the result in JSON format.
    
**Output ONLY IN JSON Format**: 
```json
{{
    "category": "CONTENT TABLE" | "GENERAL" | "DISEASES DESCRIPTION" | "OTHER",
    "evidence": "Explanation of selected category"
}}
```
Don't write explanations and don't give comments only clear JSON !
"""
