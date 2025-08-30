import re
import unicodedata
from typing import Any, Tuple, List, Dict, Union, Final
from collections import Counter


# Константа для базового набора ключей стиля
DEFAULT_STYLE_KEYS: Final[Tuple[str, str, str]] = ("color", "font", "size")


def get_style(span: dict, keys: Tuple[str, ...] = DEFAULT_STYLE_KEYS) -> Dict:
    return {k: span[k] for k in keys if k in span}


def sort_spans_by_page_number(spans: List[Dict]) -> List[Dict]:
    """
    Сортирует список спанов по возрастанию номера страницы (page_number).

    :param spans: Список спанов (каждый спан — словарь с ключом "page_number").
    :return: Новый отсортированный список спанов.
    """
    return sorted(spans, key=lambda span: span.get("page_number", 0))


def split_spans_by_page(spans: List[Dict], page_number: int) -> Tuple[List[Dict], List[Dict]]:
    """
    Делит список спанов на две части: до первого появления заданной страницы и начиная с неё.

    :param spans: список спанов (каждый спан — словарь с минимум ключом "page_number").
    :param page_number: номер страницы, с которой начинается разрез.
    :return: кортеж из двух списков:
        - первый список: все спаны до первого спана с page_number,
        - второй список: начиная с первого спана с page_number и до конца.
        Если спаны с такой страницей не найдены — возвращает ([], spans).
    Пример:
        split_spans_by_page(10, spans)
        # -> ([...спаны до 10 страницы...], [...начиная с первой десятой и далее...])
    """
    for i, span in enumerate(spans):
        if span.get("page_number") == page_number:
            # Разбиваем на две части: до и начиная с i-го (включительно)
            return spans[:i], spans[i:]
    # Если не найдено ни одного такого спана
    return [], spans


def normalize_text(text: str) -> str:
    """
    Приводит строку заголовка к нормализованному виду для поиска.
    - Удаляет невидимые символы (\t, \n, \r, \x00-\x1f)
    - Приводит к нижнему регистру
    - Убирает лишние пробелы и повторяющиеся пробелы
    """
    if not text:
        return ""
    # Удалить управляющие символы (все до 0x20), табы, новые строки и т.д.
    text = ''.join(ch if ch >= ' ' else ' ' for ch in text)
    # Удалить Unicode-контрольные символы
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    # Привести к lower-case
    text = text.lower()
    # Убрать лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_main_text_properties(all_spans: List[Dict[str, Any]]) -> Dict[str, any]:
    """
    Определяет размер шрифта и цвет основного текста на основе частоты их появления.
    Возвращает словарь с наиболее частыми размером шрифта и цветом.
    {'color': '#000000', 'font': 'Calibri', 'size': 9.960000038146973}
    """
    font_sizes = []
    fonts = []
    colors = []

    # Собираем все размеры шрифта и цвета из спанов
    for span in all_spans:
        font_sizes.append(span["size"])
        fonts.append(span["font"])
        colors.append(span["color"])

    # Подсчитываем частоту
    size_counter = Counter(font_sizes)
    font_counter = Counter(fonts)
    color_counter = Counter(colors)

    # Определяем наиболее частые значения
    main_font_size = size_counter.most_common(1)[0][0] if size_counter else None
    main_font = font_counter.most_common(1)[0][0] if font_counter else None
    main_color = color_counter.most_common(1)[0][0] if color_counter else None

    return {
        "size": main_font_size,
        "font": main_font,
        "color": main_color
    }


def rgb_to_hex(color: Union[int, List[float], tuple, None], is_background: bool = False) -> str:
    """
    Convert RGB or CMYK color to HEX format (e.g., '#ffffff').

    Args:
        color: RGB/CMYK values as int (packed), list/tuple (0-255 or 0-1 range), or None.
        is_background (bool): If True, defaults to '#FFFFFF' (white) for invalid colors.

    Returns:
        str: HEX color string (e.g., '#000000' or '#FFFFFF' if conversion fails).
    """
    try:
        if color is None:
            return "#FFFFFF" if is_background else "#000000"
        if isinstance(color, int):  # Packed integer
            r = (color >> 16) & 255
            g = (color >> 8) & 255
            b = color & 255
            return f"#{r:02x}{g:02x}{b:02x}"
        if isinstance(color, (list, tuple)):
            if len(color) == 4 and max(color, default=0) <= 1.0:  # CMYK
                r, g, b = cmyk_to_rgb(*color)
                return f"#{r:02x}{g:02x}{b:02x}"
            if len(color) >= 3:
                if is_background and max(color[:3], default=0) <= 1.0:
                    r, g, b = [int(255 * c) for c in color[:3]]
                else:
                    r, g, b = [int(c) for c in color[:3]]
                return f"#{r:02x}{g:02x}{b:02x}"
        raise ValueError(f"Invalid color format: {color}")
    except (TypeError, ValueError):
        return "#FFFFFF" if is_background else "#000000"


def cmyk_to_rgb(c: float, m: float, y: float, k: float) -> List[int]:
    """
    Convert CMYK color values to RGB.

    Args:
        c (float): Cyan component (0.0 to 1.0).
        m (float): Magenta component (0.0 to 1.0).
        y (float): Yellow component (0.0 to 1.0).
        k (float): Black component (0.0 to 1.0).

    Returns:
        List[int]: RGB values as [R, G, B], each in range 0-255.
    """
    r = 255 * (1 - c) * (1 - k)
    g = 255 * (1 - m) * (1 - k)
    b = 255 * (1 - y) * (1 - k)
    return [int(r), int(g), int(b)]
