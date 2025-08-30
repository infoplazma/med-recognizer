import os

from utils.custom_print import custom_pretty_print
from utils.read_pdf_page import read_page_text_blocks
import pymupdf4llm

# Источник для теста
PAGE_NUM = 85  # Номер страницы (начинается с 1)
PDF_SOURCE_FILES = ("Guide-to-Common-Childhood-Infections-2023_Final-Approved.pdf",
                    "Easy Paediatrics.pdf")
PDF_SOURCE_FILE = PDF_SOURCE_FILES[1]
PDF_PATH = os.path.join("../data", "med_sources", PDF_SOURCE_FILE)
EXAMPLE_TEXT_PATH = os.path.join("./data/example.txt")
EXAMPLE_MD_PATH = os.path.join("./data/example.md")

# Загрузка и разбиение документа
print("\rLoading...", end="")
sample_text = read_page_text_blocks(PDF_PATH, PAGE_NUM-1)
sample_text = sample_text.strip()
with open(EXAMPLE_TEXT_PATH, "w", encoding="utf-8") as fp:
    fp.write(sample_text)
print(f"\rОбразец blocks текста сохранен: {os.path.abspath(EXAMPLE_TEXT_PATH)}")

# markdown
pages = [PAGE_NUM-1 + i for i in range(112-PAGE_NUM)]
md_text = pymupdf4llm.to_markdown(PDF_PATH, pages=pages)
with open(EXAMPLE_MD_PATH, "w", encoding="utf-8") as fp:
    fp.write(md_text)
print(f"\rОбразец markdown текста сохранен: {os.path.abspath(EXAMPLE_MD_PATH)}")
