"""
pipeline.py

Основной пайплайн для структурирования метаданных медицинских чанков.
"""

from typing import List, Dict, Any
import uuid

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

from settings import MED_SOURCE_DIR

from med_index.extraction.disease import extract_diseases
from med_index.extraction.section_titles import extract_section_titles, get_active_section_titles
from med_index.extraction.summary import extract_chunk_summary


def generate_chunk_id() -> str:
    """Генерирует уникальный идентификатор чанка."""
    return str(uuid.uuid4())


def chunk_documents(docs: List[Any], chunk_size: int = 1024, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Sentence-aware разбиение документов на чанки.
    """
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    chunk_index = 0
    for doc in docs:
        nodes = splitter.get_nodes_from_documents([doc])
        for node in nodes:
            all_chunks.append({
                "text": node.text,
                "file_name": doc.metadata.get("file_name", ""),
                "page": doc.metadata.get("page_label", None),
                "chunk_index": chunk_index,
            })
            chunk_index += 1
    return all_chunks


def enrich_chunks_with_metadata(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Обогащает каждый чанк метаданными: diseases, chunk_summary, section_titles.
    """
    # Сначала — собрать все section titles по документу (регэксп/LLM, см. extraction/section_titles.py)
    full_text = " ".join(chunk["text"] for chunk in chunks)
    all_section_titles = extract_section_titles(full_text)

    for chunk in chunks:
        chunk["id_"] = generate_chunk_id()
        # Diseases extraction (через LLM)
        chunk["diseases"] = extract_diseases(chunk["text"])
        # Chunk summary (через LLM)
        chunk["chunk_summary"] = extract_chunk_summary(chunk["text"])
        # Section titles (по тексту чанка и ближайшим сверху по документу)
        chunk["section_titles"] = get_active_section_titles(chunk["text"], all_section_titles)
    return chunks


def link_chunks_by_disease(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Строит связи между чанками по disease: linked_diagnoses[disease] = id_ соседнего чанка.
    """
    # Соберём для каждого disease -> список (index, id_)
    disease_to_indices = {}
    for idx, chunk in enumerate(chunks):
        for disease in chunk["diseases"]:
            disease_to_indices.setdefault(disease, []).append((idx, chunk["id_"]))

    for idx, chunk in enumerate(chunks):
        chunk["linked_diagnoses"] = {}
        for disease in chunk["diseases"]:
            # ищем соседей (предыдущий и следующий чанк с этим disease)
            indices = [i for i, _ in disease_to_indices[disease]]
            pos = indices.index(idx)
            prev_id = disease_to_indices[disease][pos-1][1] if pos > 0 else None
            next_id = disease_to_indices[disease][pos+1][1] if pos < len(indices)-1 else None
            if prev_id and prev_id != chunk["id_"]:
                chunk["linked_diagnoses"][disease] = prev_id
            elif next_id and next_id != chunk["id_"]:
                chunk["linked_diagnoses"][disease] = next_id
    return chunks


def pipeline() -> List[Dict[str, Any]]:
    """
    Основной orchestrator: читает документы, разбивает на чанки, обогащает метаданными и строит связи.
    """
    # 1. Считываем документы из папки (PDF поддержка!)
    docs = SimpleDirectoryReader(
        input_dir=MED_SOURCE_DIR,
        file_extractor={".pdf": PDFReader()}
    ).load_data()

    # 2. Разбиваем на чанки (sentence-aware)
    chunks = chunk_documents(docs)

    # 3. Извлекаем diseases, section_titles, chunk_summary (через LLM/prompts)
    chunks = enrich_chunks_with_metadata(chunks)

    # 4. Строим связи по disease (linked_diagnoses)
    chunks = link_chunks_by_disease(chunks)

    # 5. Возвращаем готовый массив чанков с метаданными
    return chunks


if __name__ == "__main__":
    import json
    all_chunks = pipeline()
    # Для примера — сохраняем в файл
    with open("med_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Total chunks processed: {len(all_chunks)}")
