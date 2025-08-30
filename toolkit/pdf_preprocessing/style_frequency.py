from collections import Counter
from typing import Iterable, Dict, Tuple, List, Any, Literal, Optional

from toolkit.pdf_preprocessing.utilities import DEFAULT_STYLE_KEYS  # ("color", "font", "size")

Style = Dict[str, Any]
Span = Dict[str, Any]
StyleCount = Tuple[Style, int]


def canonical_style(
        span: Span,
        keys: Tuple[str, ...] = DEFAULT_STYLE_KEYS,
        *,
        size_round: int | None = None
) -> Style:
    """
    Извлекает стиль из спана и канонизирует его.

    :param span: словарь с информацией о тексте (включая font, size, color и др.)
    :param keys: список ключей, которые определяют стиль
    :param size_round: если указано, округляет размер шрифта до указанного числа знаков
    :return: словарь стиля (только указанные ключи)
    """
    style: Style = {}
    for k in keys:
        if k in span:
            v = span[k]
            if k == "size" and isinstance(v, (float, int)) and size_round is not None:
                v = round(float(v), size_round)
            style[k] = v
    return style


def count_styles(
        spans: Iterable[Span],
        keys: Tuple[str, ...] = DEFAULT_STYLE_KEYS,
        *,
        size_round: int | None = None,
        skip_empty_text: bool = True
) -> List[StyleCount]:
    """
    Подсчитывает частоты уникальных стилей в списке спанов.

    :param spans: список/итерируемый объект спанов (dict)
    :param keys: поля, которые определяют стиль (по умолчанию color, font, size)
    :param size_round: если задано, округляет size до указанного числа знаков
    :param skip_empty_text: если True, спаны с пустым или пробельным текстом пропускаются
    :return: список пар (стиль, количество), отсортированный по частоте
    """
    counter: Counter = Counter()

    for s in spans:
        if skip_empty_text and not str(s.get("text", "")).strip():
            continue

        style = canonical_style(s, keys, size_round=size_round)
        key = tuple((k, style.get(k)) for k in keys)
        counter[key] += 1

    items: List[Tuple[Style, int]] = []
    for key, cnt in counter.items():
        style_dict = {k: v for k, v in key}
        items.append((style_dict, cnt))

    def sort_key(item: Tuple[Style, int]):
        style, cnt = item
        return -cnt, str(style.get("font")), float(style.get("size", 0)), style.get("color")

    items.sort(key=sort_key)
    return items


def rank_heading_candidates(
    style_counts: List[StyleCount],
    levels: int = 3,
    *,
    min_size_diff: float = 0.5,
    size_weight: float = 0.6,
    freq_weight: float = 0.4,
    bucketing: Literal["quantile", "equal"] = "quantile",
    per_level_limit: Optional[int] = None,
    include_scores: bool = False,
) -> Tuple[Style, List[List[Any]]]:
    """
    Функция, которая берёт результат count_styles, находит основной стиль (самый частый) и раскладывает кандидатов на
    заголовки по уровням (например, 3 уровня) с учётом двух факторов: размера шрифта и частоты появления.
    Кандидаты отбираются только если их size больше size основного стиля на заданный порог.

    Определяет основной стиль (самый частый) и раскладывает стили-кандидаты на заголовки
    по уровням важности (1 — самый высокий) с учётом размера шрифта и частоты появления.

    :param style_counts: результат `count_styles`: список кортежей (стиль, количество),
                         отсортированный по убыванию частоты
    :param levels: число уровней для группировки кандидатов (например, 3)
    :param min_size_diff: минимальная разница в размере шрифта (пт) между кандидатом и основным стилем:
                          кандидат учитывается, если size > main_size + min_size_diff
    :param size_weight: вес вклада размера шрифта в итоговый скор (0..1)
    :param freq_weight: вес вклада частоты в итоговый скор (0..1);
                        если сумма весов ≠ 1, они будут нормализованы
    :param bucketing: способ разбиения по уровням:
                      - "quantile": равное число элементов в каждом уровне (по скору)
                      - "equal": равные интервалы по значению скора (min..max делится на levels)
    :param per_level_limit: максимум элементов в каждом уровне (None — без ограничения)
    :param include_scores: если True, в результат уровня кладутся тройки (style, count, score),
                           иначе — только style

    :return: кортеж (main_style, levels_list), где:
             - main_style: словарь основного стиля (самый частый)
             - levels_list: список уровней от 1 до `levels`;
               каждый уровень — список стилей-кандидатов (или кортежей (style, count, score),
               если include_scores=True), отсортированных по убыванию score

    Примечания:
    - Нормализация признаков:
        * size нормализуется по min..max среди кандидатов;
        * freq нормализуется по max частоте среди кандидатов.
      Если разброс нулевой (все одинаковые), признак даёт 0 вклад.
    - Итоговый score = w_size * size_norm + w_freq * freq_norm.
    - Пример, когда стиль с size=15 и count=1 и стиль с size=12 и count=20
      могут оказаться в одном уровне: высокий freq «компенсирует» меньший size.
    """
    if levels < 1:
        raise ValueError("levels must be >= 1")

    if not style_counts:
        return {}, {"meta": _build_meta(levels, bucketing, min_size_diff, size_weight, freq_weight, per_level_limit),
                    "levels": [{"level": i + 1, "items": [], "total": 0} for i in range(levels)]}

        # Основной стиль — самый частый (первый в style_counts).
    main_style, _ = style_counts[0]
    main_size_val = main_style.get("size")
    if not isinstance(main_size_val, (int, float)):
        # Без числового размера невозможно корректно фильтровать «крупнее основного».
        return main_style, {
            "meta": _build_meta(levels, bucketing, min_size_diff, size_weight, freq_weight, per_level_limit),
            "levels": [{"level": i + 1, "items": [], "total": 0} for i in range(levels)]}
    main_size = float(main_size_val)

    # Соберём кандидатов: строго крупнее основного.
    candidates: List[Tuple[Style, int]] = []
    for style, cnt in style_counts[1:]:
        sz = style.get("size")
        if isinstance(sz, (int, float)) and float(sz) > main_size + float(min_size_diff):
            candidates.append((style, int(cnt)))

    if not candidates:
        return main_style, {
            "meta": _build_meta(levels, bucketing, min_size_diff, size_weight, freq_weight, per_level_limit),
            "levels": [{"level": i + 1, "items": [], "total": 0} for i in range(levels)]}

    # Нормализация весов.
    w_sum = float(size_weight) + float(freq_weight)
    size_w = float(size_weight) / w_sum if w_sum > 0 else 0.5
    freq_w = float(freq_weight) / w_sum if w_sum > 0 else 0.5

    # Ряды для нормализации.
    sizes = [float(s["size"]) for s, _ in candidates]
    freqs = [int(c) for _, c in candidates]

    size_min, size_max = min(sizes), max(sizes)
    size_range = size_max - size_min
    freq_max = max(freqs)
    freq_range = float(freq_max)

    def norm_size(x: float) -> float:
        if size_range <= 0:
            return 0.0
        return (x - size_min) / size_range

    def norm_freq(x: int) -> float:
        if freq_range <= 0:
            return 0.0
        return x / freq_range

    # Посчитаем score для каждого кандидата.
    scored: List[Tuple[Style, int, float]] = []
    for style, cnt in candidates:
        s_norm = norm_size(float(style["size"]))
        f_norm = norm_freq(int(cnt))
        score = size_w * s_norm + freq_w * f_norm
        scored.append((style, cnt, float(score)))

    # Сортировка по score (desc), затем по size (desc), затем по font.
    scored.sort(key=lambda t: (t[2], float(t[0]["size"]), str(t[0].get("font", ""))), reverse=True)

    # Первичное разбиение по уровням.
    provisional: List[Tuple[Style, int, float, int]] = []  # (style, count, score, level_idx)
    if bucketing == "equal":
        scores = [sc for *_rest, sc in scored]
        s_min, s_max = min(scores), max(scores)
        s_range = (s_max - s_min) or 1.0
        for item in scored:
            sc = item[2]
            rel = (sc - s_min) / s_range  # 0..1
            idx = min(levels - 1, int((1.0 - rel) * levels))  # top score -> 0
            provisional.append((*item, idx))
    else:  # quantile
        n = len(scored)
        for i, item in enumerate(scored):
            idx = int(i * levels / max(1, n))  # 0..levels-1
            provisional.append((*item, idx))

    # Постобработка монотонности по size:
    # если size_A > size_B, то level(A) <= level(B). Демотируем меньшие, если нужно.
    provisional.sort(key=lambda t: float(t[0]["size"]), reverse=True)  # по size desc
    max_level_so_far = -1
    corrected: List[Tuple[Style, int, float, int]] = []
    for style, cnt, score, lvl in provisional:
        if lvl < max_level_so_far:
            lvl = max_level_so_far
        if lvl > max_level_so_far:
            max_level_so_far = lvl
        corrected.append((style, cnt, score, lvl))

    # Разложим по уровням и отсортируем внутри уровней по score (desc).
    buckets: List[List[Tuple[Style, int, float]]] = [[] for _ in range(levels)]
    for style, cnt, score, lvl in corrected:
        if 0 <= lvl < levels:
            buckets[lvl].append((style, cnt, score))

    for b in buckets:
        b.sort(key=lambda t: t[2], reverse=True)

    # Ограничим per_level_limit и соберём словарь уровней с понятными полями.
    levels_dict: Dict[str, Any] = {
        "meta": _build_meta(levels, bucketing, min_size_diff, size_weight, freq_weight, per_level_limit),
        "levels": []
    }

    for i, bucket in enumerate(buckets):
        if per_level_limit is not None and per_level_limit >= 0:
            bucket = bucket[:per_level_limit]

        items: List[Dict[str, Any]] = []
        for style, cnt, score in bucket:
            item: Dict[str, Any] = {
                "style": style,
                "count": int(cnt),
                "size": float(style.get("size")) if isinstance(style.get("size"), (int, float)) else None,
                "font": style.get("font"),
            }
            if include_scores:
                item["score"] = float(score)
            items.append(item)

        levels_dict["levels"].append({
            "level": i + 1,  # человеко-читаемая нумерация
            "items": items,
            "total": len(items),
        })

    return main_style, levels_dict


def _build_meta(
        levels: int,
        bucketing: str,
        min_size_diff: float,
        size_weight: float,
        freq_weight: float,
        per_level_limit: Optional[int],
) -> Dict[str, Any]:
    """Внутренний помощник для сборки блока meta результата."""
    return {
        "levels_count": levels,
        "bucketing": bucketing,
        "min_size_diff": float(min_size_diff),
        "weights": {"size": float(size_weight), "frequency": float(freq_weight)},
        "per_level_limit": per_level_limit,
        "monotonic_by_size": True,
    }


# ===== Пример использования =====
if __name__ == "__main__":
    import os
    from utils.general import create_local_logger, load_json, save_json

    SPANS_DIR = "../../tests/data/"
    PDF_FILES = ("Guide-to-Common-Childhood-Infections-2023_Final-Approved.pdf",
                 "Easy Paediatrics.pdf",
                 "Manual_of_childhood_infections.pdf")
    PDF_FILE = PDF_FILES[1]

    spans_path = os.path.join(SPANS_DIR, PDF_FILE.replace(".pdf", "_spans.json"))
    heading_levels_path = os.path.join(SPANS_DIR, PDF_FILE.replace(".pdf", "_heading_levels.json"))
    logger = create_local_logger()

    spans = load_json(spans_path, logger=logger)  # ваш список спанов
    ###########################################################################
    #               count_styles - testing
    ###########################################################################
    print("\nTest: count_styles")
    # Базовый подсчет по (color, font, size) с округлением размера до 1 знака
    style_counts_ = count_styles(spans, ("color", "font", "size"), size_round=1)

    # Красивый вывод результата
    for style_, cnt_ in style_counts_:
        print(f"{cnt_:>5} × {style_}")

    # Если хотите считать стиль только по (font, size), игнорируя цвет:
    # style_counts_no_color = count_styles(spans, ("font", "size"), size_round=1)
    ###########################################################################
    #               rank_heading_candidates - testing
    ###########################################################################
    print("\nTest: rank_heading_candidates")
    main_style, heading_levels = rank_heading_candidates(
        style_counts_,
        levels=5,  # три уровня
        min_size_diff=0.5,  # заголовки строго крупнее основного на 0.5pt
        size_weight=0.55,  # вес размера
        freq_weight=0.45,  # вес частоты
        bucketing="quantile",
        per_level_limit=5,  # максимум 5 вариантов на уровень
        include_scores=True  # включить score для отладки
    )

    print("Основной стиль:", main_style)

    # heading_levels — теперь словарь с ключами "meta" и "levels"
    for level in heading_levels["levels"]:
        print(f"Уровень {level['level']} (total={level['total']}):")
        for item in level["items"]:
            style = item["style"]
            cnt = item.get("count")
            score = item.get("score")  # может отсутствовать, если include_scores=False
            size = item.get("size", style.get("size"))
            font = item.get("font", style.get("font"))

            if score is None:
                print(f"  size={size}, font={font}, count={cnt}")
            else:
                print(f"  size={size}, font={font}, count={cnt}, score={score:.3f}")

    # Если нужно увидеть целиком как JSON:
    # import json
    # print(json.dumps(heading_levels, ensure_ascii=False, indent=2))

    # Сохраняем результат
    save_json(heading_levels, heading_levels_path, logger=logger)
