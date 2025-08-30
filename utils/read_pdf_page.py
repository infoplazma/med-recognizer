"""
Пример четырех способов чтения PDF файлов
"""
import fitz  # PyMuPDF


# Вариант 1: Простое извлечение текста с определенной страницы
def read_page_text_simple(pdf_path, page_number):
    try:
        doc = fitz.open(pdf_path)
        if page_number < 0 or page_number >= len(doc):
            raise ValueError(f"Номер страницы {page_number} вне диапазона (0-{len(doc) - 1})")

        page = doc[page_number]
        text = page.get_text("text")
        doc.close()
        return text
    except Exception as e:
        return f"Ошибка: {str(e)}"


# Вариант 2: Извлечение текста с форматированием (включая блоки текста)
def read_page_text_blocks(pdf_path, page_number):
    try:
        doc = fitz.open(pdf_path)
        if page_number < 0 or page_number >= len(doc):
            raise ValueError(f"Номер страницы {page_number} вне диапазона (0-{len(doc) - 1})")

        page = doc[page_number]
        blocks = page.get_text("blocks")  # Возвращает список блоков текста
        result = []
        for block in blocks:
            if block[6] == 0:  # 0 означает текстовый блок
                result.append(block[4])  # Текст блока
        doc.close()
        return "\n".join(result)
    except Exception as e:
        return f"Ошибка: {str(e)}"


# Вариант 3: Извлечение текста в формате словаря (JSON-подобный)
def read_page_text_dict(pdf_path, page_number):
    try:
        doc = fitz.open(pdf_path)
        if page_number < 0 or page_number >= len(doc):
            raise ValueError(f"Номер страницы {page_number} вне диапазона (0-{len(doc) - 1})")

        page = doc[page_number]
        text_dict = page.get_text("dict")  # Возвращает структурированный словарь
        doc.close()
        return text_dict
    except Exception as e:
        return f"Ошибка: {str(e)}"


# Вариант 4: Извлечение текста с координатами слов
def read_page_text_words(pdf_path, page_number):
    try:
        doc = fitz.open(pdf_path)
        if page_number < 0 or page_number >= len(doc):
            raise ValueError(f"Номер страницы {page_number} вне диапазона (0-{len(doc) - 1})")

        page = doc[page_number]
        words = page.get_text("words")  # Возвращает список слов с координатами
        result = []
        for word in words:
            result.append({
                "text": word[4],
                "coordinates": (word[0], word[1], word[2], word[3])  # x0, y0, x1, y1
            })
        doc.close()
        return result
    except Exception as e:
        return f"Ошибка: {str(e)}"


# Пример использования
if __name__ == "__main__":
    import os

    pdf_file = os.path.join("../data", "med_sources", "Guide-to-Common-Childhood-Infections-2023_Final-Approved.pdf")
    page_num = 10  # Номер страницы (начинается с 0)

    # Тестирование каждого варианта
    print("Вариант 1 (Простой текст):")
    print(read_page_text_simple(pdf_file, page_num))
    print("\nВариант 2 (Блоки текста):")
    text = read_page_text_blocks(pdf_file, page_num)
    print(text)
    with open("../tests/data/example.txt", "w", encoding="utf-8") as fp:
        fp.write(text.strip())
    print("\nВариант 3 (Словарь текста):")
    print(read_page_text_dict(pdf_file, page_num))
    print("\nВариант 4 (Слова с координатами):")
    print(read_page_text_words(pdf_file, page_num))
