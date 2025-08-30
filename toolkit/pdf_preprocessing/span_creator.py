"""
Проверка работы PyMuPDF
Документация:
'PyMuPDF - инструмент парсинга PDF источников' -> в Obsidian 'Dashboard - Сбор мед данных Diagnosis Rules'
"""
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from utilities import get_main_text_properties, get_style


def create_spans(pdf_path: str, page_numbers: Optional[List[int]] = None) -> List[Dict]:
    spans = extract_spans(pdf_path, page_numbers)
    main_text_styles = get_main_text_properties(spans)
    spans = remove_text_hyphenation(spans)
    spans = add_headings(spans, main_text_styles)
    return spans


def extract_spans(pdf_path: str, page_numbers: Optional[List[int]] = None) -> List[Dict]:

    spans = []

    if not Path(pdf_path).is_file():
        print(f"Error: PDF file '{pdf_path}' not found.")
        return spans

    try:
        doc = fitz.open(pdf_path)

        if page_numbers is None:
            page_numbers = list(range(1, doc.page_count + 1))

        total_pages = len(doc)

        for page_num in tqdm(page_numbers, desc="Page"):
            page_index = page_num - 1
            if page_index < 0 or page_index >= total_pages:
                print(f"Warning: Page {page_num} is out of range (1-{total_pages}). Skipping.")
                continue

            page = doc[page_index]

            text_dict = page.get_text("dict", sort=True)
            # text_blocks = page.get_text("blocks")

            # Для сортировки блоков как они появляются на странице page.get_text("dict")
            sorted_dict_blocks = sorted(text_dict.get("blocks", []),
                                        key=lambda block: (block["bbox"][1], block["bbox"][0]))
            # Для сортировки блоков как они появляются на странице page.get_text("blocks")
            # sorted_blocks = sorted(text_blocks, key=lambda block: (block[1], block[0]))
            for block_index, block in enumerate(sorted_dict_blocks):

                if block.get("type") != 0:
                    continue

                for line in block.get("lines", []):
                    # Пропуск вертикального режима письма
                    if line['wmode'] != 0:
                        continue

                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue

                        size = round(span.get("size", 0.0), 2)
                        font = span.get("font", "Unknown")
                        bbox = span.get("bbox", [0, 0, 0, 0])
                        # pretty_print_json(span)

                        # Get text color
                        raw_color = span.get("color")
                        color = raw_color if raw_color is not None else [0, 0, 0]
                        # color_hex = rgb_to_hex(color, is_background=False)

                        span_dict = {
                            "text": text,
                            "size": size,
                            "font": font,
                            # "color": color_hex,
                            "color": color,
                            "bbox": list(bbox),
                            "page_number": page_num,
                            "block_index": block_index,
                        }

                        spans.append(span_dict)

    except Exception as e:
        print(f"Error processing PDF: {e}")

    return spans


def remove_text_hyphenation(spans: List[Dict]) -> List[Dict]:
    """
    Убирает переносы для одинаковых стилей и объединяет реливантный текст
    :param spans:
    :return:
    """
    new_spans = []
    if spans:
        new_spans = spans[0:1]
        new_spans[-1]["origin"] = new_spans[-1]["bbox"][:2]
        del new_spans[-1]["bbox"]
        for index, span in enumerate(spans[1:]):
            last_span = new_spans[-1]
            # Объединяет в одном блоке с одинаковым стилем
            if span["block_index"] == last_span["block_index"] and get_style(span) == get_style(last_span):
                last_span["text"] += " " + span["text"]
                last_span["text"] = normalize_spaces(last_span["text"])
                continue
            # Объединяет в одном блоке с разными стилями, но с последним спаном состоящим из одного не буквенного символа
            elif span["block_index"] == last_span["block_index"] and re.match(r'[^a-zA-Z]', last_span["text"]):
                last_span["text"] += " " + span["text"]
                last_span["text"] = normalize_spaces(last_span["text"])
                continue
            else:
                new_spans.append(span)
                span["origin"] = span["bbox"][:2]
                del span["bbox"]

    return new_spans


def add_headings(spans: List[Dict], main_style: Dict) -> List[Dict]:
    """
    Определяет тип текста, это заглавие или текст
    :param spans:
    :param main_style:
    :return:
    """
    new_spans = list(spans)
    # for index, span in enumerate(new_spans):
    #     next_span = spans[index + 1] if index < len(spans) - 1 else None
    #     # Определяет заглавие в разных блоках
    #     if (next_span and
    #             span["block_index"] != next_span["block_index"] and
    #             get_style(span) != get_style(main_style) and
    #             is_heading_grok(span["text"])):
    #         span["type"] = "heading"
    #     # Определяет заглавие в одном блоке, но с разными стилями
    #     elif (next_span and
    #           get_style(span) != get_style(main_style) and
    #           get_style(span) != get_style(next_span) and
    #           is_heading_grok(span["text"])):
    #         span["type"] = "heading"
    #     else:
    #         span["type"] = "text"

    return new_spans


def normalize_spaces(text: str):
    # Заменяем множественные пробелы на один и убираем пробелы в начале/конце
    text = re.sub(r'\s+', ' ', text.strip())
    # Убираем пробел перед точкой в конце строки
    text = re.sub(r'\s+\.$', '../block_parser', text)
    return text


if __name__ == "__main__":
    import os
    from utils.general import create_local_logger, save_json
    from utils.custom_print import custom_pretty_print

    PAGES = [86]  # [6]
    # PAGES = list(range(261, 271))

    PDF_BOOKS_DIR = "../../data/med_sources"
    SPANS_DIR = "../../tests/data/"

    PDF_FILES = ("Guide-to-Common-Childhood-Infections-2023_Final-Approved.pdf",
                 "Easy Paediatrics.pdf",
                 "Manual_of_childhood_infections.pdf")
    PDF_FILE = PDF_FILES[1]

    pdf_path = os.path.join(PDF_BOOKS_DIR, PDF_FILE)
    block_spans = extract_spans(pdf_path, page_numbers=None)
    # pretty_print_json(block_spans)

    main_text_properties = get_main_text_properties(block_spans)
    custom_pretty_print("Основной стиль:", main_text_properties)

    print("="*180)
    # block_spans = remove_text_hyphenation(block_spans)
    # block_spans = add_headings(block_spans, main_text_properties)
    # pretty_print_json(block_spans)

    # Вывод первых 5 спанов для проверки
    # for span in block_spans[:30]:
    #     print(span)

    spans_path = os.path.join(SPANS_DIR, PDF_FILE.replace(".pdf", "_spans.json"))
    logger = create_local_logger()
    save_json(block_spans, spans_path, logger)
