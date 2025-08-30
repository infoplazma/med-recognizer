from __future__ import annotations
import re
import unicodedata
from typing import Iterable, Dict, Any, List, Tuple, Sequence, Optional
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from toolkit.pdf_preprocessing.utilities import DEFAULT_STYLE_KEYS  # ("color", "font", "size")
from style_frequency import Span, Style


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def match_headings_to_styles(
        spans: Iterable[Span],
        levels_dict: Dict[str, Any],
        llm_headings: List[str],
        *,
        style_keys: Sequence[str] = ("color", "font", "size"),
        use_levels_upto: Optional[int] = None,
        min_cosine: float = 0.9,          # эмбеддинги: разумный дефолт
        size_tol: float = 0.25,
        deduplicate_segments: bool = False,
        merge_before: bool = True,
        line_tol: float = 2.0,
        joiner: str = " ",
) -> List[Tuple[str, Style]]:
    """
    Сопоставляет список LLM-заголовков со «строками» документа (объединённые спаны)
    и возвращает пары (исходный_заголовок, стиль_строки). Вместо мешков слов
    использует эмбеддинги (SentenceTransformer) через `_cosine_sim`.

    :param spans: итерируемая коллекция спанов с полями text, size, font, color, bbox, page_number
    :param levels_dict: словарь уровней от rank_heading_candidates:
                        {
                          "meta": {...},
                          "levels": [
                            {"level": 1, "items": [{"style": {...}, "count": int, "score": float, ...}, ...], "total": int},
                            ...
                          ]
                        }
    :param llm_headings: список строк-заголовков, извлечённых LLM
    :param style_keys: ключи, по которым извлекается стиль из спана/строки
    :param use_levels_upto: если задано (например, 1 или 2), использовать уровни с полем "level"
                            <= этому значению; по умолчанию — все уровни
    :param min_cosine: минимальное косинусное сходство (0..1) для принятия совпадения (эмбеддинги)
    :param size_tol: допуск по сравнению размера шрифта (pt) для сопоставления стилей
    :param deduplicate_segments: если True, одна и та же строка не матчатся на несколько заголовков
    :param merge_before: если True, сначала объединить соседние спаны в строки по стилю и линии
    :param line_tol: допуск по вертикали (pt) для объединения спанов в строку
    :param joiner: строка-соединитель при склейке текста спанов в строку
    :return: список кортежей (llm_heading, style_dict) только для успешно сматченных заголовков
    """
    # 1) Соберём стили-кандидаты из словаря уровней
    if not isinstance(levels_dict, dict):
        return []
    levels: List[Tuple[int, List[Style]]] = []
    for lvl in levels_dict.get("levels", []):
        lvl_no = int(lvl.get("level", 0))
        if use_levels_upto is not None and lvl_no > int(use_levels_upto):
            continue
        styles_in_level: List[Style] = []
        for item in lvl.get("items", []):
            st = item.get("style") if isinstance(item, dict) else None
            if isinstance(st, dict):
                styles_in_level.append(st)
        if styles_in_level:
            levels.append((lvl_no, styles_in_level))
    if not levels or not llm_headings:
        return []
    levels.sort(key=lambda t: t[0])  # 1..N

    def _resolve_level_for_style(seg_style: Style) -> Optional[int]:
        for lvl_no, styles_in_level in levels:
            for cs in styles_in_level:
                if _style_matches(seg_style, cs, size_tol=size_tol):
                    return lvl_no
        return None

    # --- 2) сегменты (строки) ---
    if merge_before:
        segments = merge_spans_by_style_and_line(
            spans, style_keys=style_keys, size_tol=size_tol, line_tol=line_tol, joiner=joiner
        )
    else:
        segments = []
        for sp in spans:
            txt = str(sp.get("text", "") or "").strip()
            if not txt:
                continue
            segments.append({
                "text": txt,
                "style": _canonical_style(sp, style_keys),
                "page_number": sp.get("page_number"),
                "block_index": sp.get("block_index"),
                "bbox": sp.get("bbox"),
            })

    # --- 3) фильтр по кандид. стилям + уровень ---
    eligible: List[Tuple[int, Dict[str, Any], int]] = []  # (seg_idx, segment, level)
    for i, seg in enumerate(segments):
        st = seg.get("style", {})
        level_no = _resolve_level_for_style(st)
        if level_no is not None:
            eligible.append((i, seg, level_no))
    if not eligible:
        return []

    # --- 4) матчинг ---
    results: List[Tuple[str, Style, Dict[str, Optional[int]]]] = []
    used_seg_ids: set[int] = set()  # применяется только при deduplicate_segments=True

    def _size_of(seg: Dict[str, Any]) -> float:
        v = seg.get("style", {}).get("size")
        return float(v) if isinstance(v, (int, float)) else 0.0

    for heading in llm_headings:
        heading = heading.strip()
        if not heading:
            continue

        # 4.1 посчитать скоры со всеми подходящими сегментами
        scored: List[Tuple[float, int, Dict[str, Any], int]] = []  # (score, seg_idx, seg, level)
        for seg_idx, seg, level_no in eligible:
            if deduplicate_segments and seg_idx in used_seg_ids:
                continue
            seg_text = str(seg.get("text", "")).strip()
            if not seg_text:
                continue
            score = _cosine_sim(heading, seg_text)
            if score >= float(min_cosine):
                scored.append((score, seg_idx, seg, level_no))

        if not scored:
            continue

        # сортировка для устойчивого выбора «лучшего»
        # при равном score предпочтём меньший уровень (крупнее иерархия), затем больший size
        scored.sort(key=lambda t: (-t[0], t[3], -_size_of(t[2])))

        if deduplicate_segments:
            # только один лучший
            best_score, best_seg_idx, best_seg, best_level = scored[0]
            results.append((
                heading,
                best_seg.get("style", {}),
                {
                    "level": int(best_level),
                    "page_number": best_seg.get("page_number"),
                    "block_index": best_seg.get("block_index"),
                },
            ))
            used_seg_ids.add(best_seg_idx)
        else:
            # лучший ДЛЯ КАЖДОГО уникального стиля ('color','font','size')
            best_for_style: Dict[Tuple, Tuple[float, int, Dict[str, Any], int]] = {}
            for score, seg_idx, seg, level_no in scored:
                sig = _style_signature(seg.get("style", {}), style_keys)
                prev = best_for_style.get(sig)
                if prev is None:
                    best_for_style[sig] = (score, seg_idx, seg, level_no)
                else:
                    # выбираем лучший по score; при равенстве — меньший уровень, затем больший size
                    p_score, p_idx, p_seg, p_lvl = prev
                    better = (score > p_score) or (
                            score == p_score and (
                                level_no < p_lvl or (level_no == p_lvl and _size_of(seg) > _size_of(p_seg)))
                    )
                    if better:
                        best_for_style[sig] = (score, seg_idx, seg, level_no)

            # добавить в results (порядок — по убыванию score для читабельности)
            selected = sorted(best_for_style.values(), key=lambda t: (-t[0], t[3], -_size_of(t[2])))
            for score, seg_idx, seg, level_no in selected:
                results.append((
                    heading,
                    seg.get("style", {}),
                    {
                        "level": int(level_no),
                        "page_number": seg.get("page_number"),
                        "block_index": seg.get("block_index"),
                    },
                ))
            # здесь НЕ помечаем used_seg_ids — т.к. deduplicate_segments=False означает, что
            # один и тот же сегмент можно использовать и для других заголовков

    return results


# ------------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ -------------------------

def _canonical_style(span: Span, keys: Sequence[str] = ("color", "font", "size")) -> Style:
    """
    Приводит стиль спана к стандартному словарю только с указанными ключами.

    :param span: спан с полями текста и атрибутами от PDF-парсера
    :param keys: какие поля считать стилем (по умолчанию color, font, size)
    :return: словарь стиля
    """
    out: Style = {}
    for k in keys:
        if k in span:
            out[k] = span[k]
    return out


def _style_matches(span_style: Style, cand_style: Style, *, size_tol: float = 0.25) -> bool:
    """
    Сравнивает стилевые словари, разрешая небольшую погрешность по size.

    :param span_style: стиль спана
    :param cand_style: эталонный (кандидат) стиль
    :param size_tol: допустимое абсолютное отклонение по size (pt)
    :return: True, если стили совпадают (с учётом size_tol)
    """
    for k, v in cand_style.items():
        if k == "size":
            sv = span_style.get("size")
            if not isinstance(sv, (int, float)) or not isinstance(v, (int, float)):
                return False
            if abs(float(sv) - float(v)) > float(size_tol):
                return False
        else:
            if span_style.get(k) != v:
                return False
    return True


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _normalize_text(s: str) -> List[str]:
    """
    Нормализует текст: нижний регистр, удаление диакритики, выделение 'слов'.

    :param s: исходная строка
    :return: список токенов
    """
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    return _WORD_RE.findall(s)


@lru_cache(maxsize=1)
def get_st_model_cached(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    print(f"{model_name} loading...", end="", flush=True)
    model = SentenceTransformer(model_name)
    print(f"\r\rLoaded")
    return model


def _cosine_sim(llm_heading: str, spam_text: str, *, model: Optional[SentenceTransformer] = None) -> float:
    """
    Косинусное сходство между строкой-заголовком и текстом спана с помощью SentenceTransformer.

    :param llm_heading: строка заголовка, предложенная LLM
    :param spam_text: текст спана/строки из PDF (опечатка в имени аргумента сохранена по ТЗ)
    :param model: уже созданная модель SentenceTransformer; если None — будет использована/создана дефолтная
    :return: косинусное сходство (-1..1), при L2-нормализации обычно в диапазоне [0..1]
    """
    if not llm_heading.strip() or not spam_text.strip():
        return 0.0

    m = model or get_st_model_cached()
    # Нормализуем эмбеддинги: косинус становится скалярным произведением
    emb = m.encode([llm_heading, spam_text], normalize_embeddings=True, batch_size=2, show_progress_bar=False)
    h, s = emb[0], emb[1]
    return float((h * s).sum())


# ------------------------- ОБЪЕДИНЕНИЕ СПАНОВ В «СТРОКИ» -------------------------

def merge_spans_by_style_and_line(
    spans: Iterable[Span],
    *,
    style_keys: Sequence[str] = ("color", "font", "size"),
    size_tol: float = 0.25,
    line_tol: float = 2.0,
    joiner: str = " ",
) -> List[Dict[str, Any]]:
    """
    Объединяет соседние спаны с одинаковым стилем и близкими координатами по вертикали в «строки».

    :param spans: исходные спаны
    :param style_keys: ключи, образующие стиль
    :param size_tol: допуск по size при сравнении стиля (pt)
    :param line_tol: допуск по вертикали (pt), чтобы считать одну строку
    :param joiner: разделитель при склейке текста
    :return: список сегментов: {"text","style","page_number","block_index","bbox"}
    """
    def _pos(sp: Span) -> Tuple[int, float, float, int]:
        page = int(sp.get("page_number", 0) or 0)
        try:
            x0, y0, x1, y1 = sp.get("bbox", [0, 0, 0, 0])
        except Exception:
            x0 = y0 = 0.0
            x1 = y1 = 0.0
        block = int(sp.get("block_index", 0) or 0)
        return page, float(y0), float(x0), block

    prepared: List[Span] = []
    for sp in spans:
        txt = str(sp.get("text", "") or "")
        if not txt.strip():
            continue
        prepared.append(sp)
    prepared.sort(key=_pos)

    segments: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for sp in prepared:
        txt = str(sp.get("text", ""))
        page = sp.get("page_number", None)
        block = sp.get("block_index", None)
        bbox = sp.get("bbox", [0.0, 0.0, 0.0, 0.0])
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])) if len(bbox) >= 4 else (0.0, 0.0, 0.0, 0.0)
        st = _canonical_style(sp, style_keys)

        if current is not None:
            same_page = (page == current["page_number"])
            same_style = _style_matches(st, current["style"], size_tol=size_tol)
            same_line = abs(y0 - current["line_y"]) <= float(line_tol)

            if same_page and same_style and same_line:
                current["text_parts"].append(txt)
                cb = current["bbox"]
                current["bbox"] = [min(cb[0], x0), min(cb[1], y0), max(cb[2], x1), max(cb[3], y1)]
                # для блока берём минимальный индекс (ранний блок в строке)
                if current["block_index"] is None:
                    current["block_index"] = block
                else:
                    if block is not None and (current["block_index"] is None or block < current["block_index"]):
                        current["block_index"] = block
                continue

            # закрываем сегмент
            segments.append({
                "text": joiner.join(current["text_parts"]).strip(),
                "style": current["style"],
                "page_number": current["page_number"],
                "block_index": current["block_index"],
                "bbox": current["bbox"],
            })
            current = None

        # старт нового сегмента
        current = {
            "text_parts": [txt],
            "style": st,
            "page_number": page,
            "block_index": block,
            "bbox": [x0, y0, x1, y1],
            "line_y": y0,
        }

    if current is not None:
        segments.append({
            "text": joiner.join(current["text_parts"]).strip(),
            "style": current["style"],
            "page_number": current["page_number"],
            "block_index": current["block_index"],
            "bbox": current["bbox"],
        })

    return segments


# ------------------------- УБИРАЕМ ДУБЛИРОВАНИЕ -------------------------

def _style_signature(style: Style, keys: Sequence[str] = ("color", "font", "size"), *, size_round: int = 2) -> Tuple:
    """Хэшируемый «отпечаток» стиля для де-дупа."""
    sig: List[Tuple[str, Any]] = []
    for k in keys:
        v = style.get(k)
        if k == "size" and isinstance(v, (int, float)):
            v = round(float(v), size_round)
        sig.append((k, v))
    return tuple(sig)


# ===== Пример использования =====
if __name__ == "__main__":
    import os
    from utils.general import load_json  # create_local_logger,
    from utils.custom_print import custom_pretty_print

    LLM_HEADINGS = ["PAEDIATRIC HISTORY", "INFANT AND CHILD", "MYOCARDITIS", "Clinical features", "DILATED CARDIOMYOPATHY"]
    SPANS_DIR = "../../tests/data/"
    PDF_FILES = ("Guide-to-Common-Childhood-Infections-2023_Final-Approved.pdf",
                 "Easy Paediatrics.pdf",
                 "Manual_of_childhood_infections.pdf")
    PDF_FILE = PDF_FILES[1]

    spans_path = os.path.join(SPANS_DIR, PDF_FILE.replace(".pdf", "_spans.json"))
    heading_levels_path = os.path.join(SPANS_DIR, PDF_FILE.replace(".pdf", "_heading_levels.json"))
    # logger = create_local_logger()

    spans_ = load_json(spans_path)
    heading_levels_ = load_json(heading_levels_path)
    matched_headings = match_headings_to_styles(spans_, heading_levels_, LLM_HEADINGS, min_cosine=0.9, deduplicate_segments=False)
    custom_pretty_print("matched_headings:", matched_headings)
