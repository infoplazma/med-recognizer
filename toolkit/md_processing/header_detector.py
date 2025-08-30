from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional


class HeaderDetector:
    """
    Детектор заголовков для PyMuPDF4LLM: объект с методом get_header_id(span, page=None) -> str,
    который возвращает "#", "##", "###", ... или "" (если не заголовок).

    Поддерживает два формата правил:
      1) 'style_to_header' (dict): { (font, size, color): "#"/"##"/... }  — быстрый точный матч
      2) 'rules' (list[dict]): [{"font": str, "size": float, "color": int, "header": str}, ...]
         — «читаемый» формат; используется как fallback с допуском по size.
    """

    DEFAULT_LEVEL_TO_HEADER = {1: "#", 2: "##", 3: "###", 4: "####", 5: "#####", 6: "######"}

    def __init__(
            self,
            rules: Optional[List[Dict[str, Any]]] = None,
            style_to_header: Optional[Dict[Tuple[str, float, int], str]] = None,
            size_tol: float = 0.15,
    ) -> None:
        self.rules: List[Dict[str, Any]] = rules or []
        self.style_to_header: Dict[Tuple[str, float, int], str] = style_to_header or {}
        self.size_tol = size_tol

    # ───────────────────────────── статические утилиты ─────────────────────────────

    @staticmethod
    def _norm_color(color: Any) -> int:
        """Нормализует цвет: (r,g,b) -> 0xRRGGBB, int -> int."""
        if isinstance(color, (tuple, list)) and len(color) == 3:
            r, g, b = color
            return (int(r) << 16) + (int(g) << 8) + int(b)
        return int(color)

    @staticmethod
    def _norm_size(size: Any) -> float:
        return float(size)

    @classmethod
    def to_style_to_header(
            cls,
            levels_json: Dict[str, Any],
            level_to_header: Optional[Dict[int, str]] = None,
            per_level_limit: Optional[int] = None,
    ) -> Dict[Tuple[str, float, int], str]:
        """
        Преобразует ваш JSON (levels/items) в dict формата:
            { (font, size, color): "#"/"##"/... }

        :param levels_json: ваш словарь с ключом "levels"
        :param level_to_header: соответствие уровня → префикс (по умолчанию 1→#, 2→##, ...)
        :param per_level_limit: ограничение кол-ва стилей на уровень (по умолчанию из meta.per_level_limit)
        """
        l2h = level_to_header or cls.DEFAULT_LEVEL_TO_HEADER
        if per_level_limit is None:
            per_level_limit = int(levels_json.get("meta", {}).get("per_level_limit", 999999))

        out: Dict[Tuple[str, float, int], str] = {}
        for level_block in levels_json.get("levels", []):
            level = int(level_block.get("level", 0))
            header = l2h.get(level)
            if not header:
                continue
            for i, item in enumerate(level_block.get("items", [])):
                if i >= per_level_limit:
                    break
                st = item.get("style", {})
                font = st.get("font")
                size = cls._norm_size(st.get("size"))
                color = cls._norm_color(st.get("color"))
                if font is None or size is None or color is None:
                    continue
                out[(str(font), float(size), int(color))] = header
        return out

    @classmethod
    def to_rule_list(
            cls,
            levels_json: Dict[str, Any],
            level_to_header: Optional[Dict[int, str]] = None,
            per_level_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Преобразует levels JSON (levels/items) в список «читаемых» правил:
            [{"font": "...", "size": 17.0, "color": 2301728, "header": "#"}, ...]
        """
        style_map = cls.to_style_to_header(levels_json, level_to_header, per_level_limit)
        rules: List[Dict[str, Any]] = []
        for (font, size, color), header in style_map.items():
            rules.append({"font": font, "size": size, "color": color, "header": header})
        return rules

    @classmethod
    def from_levels(
            cls,
            levels_json: Dict[str, Any],
            level_to_header: Optional[Dict[int, str]] = None,
            per_level_limit: Optional[int] = None,
            size_tol: float = 0.15,
    ) -> "HeaderDetector":
        """
        Быстрый конструктор детектора из levels JSON.
        Создаёт и точное отображение (tuple-ключ), и «читаемые» правила.
        """
        style_map = cls.to_style_to_header(levels_json, level_to_header, per_level_limit)
        rules = cls.to_rule_list(levels_json, level_to_header, per_level_limit)
        return cls(rules=rules, style_to_header=style_map, size_tol=size_tol)

    # ───────────────────────────── основной метод для PyMuPDF4LLM ─────────────────────────────

    def get_header_id(self, span: dict, page=None) -> str:
        """
        Возвращает "#"/"##"/... или "" для данного span (PyMuPDF4LLM вызовет это на каждый span).
        Сначала пробуем точное совпадение, затем «мягкую» проверку по правилам (с допуском по size).
        """
        font = str(span.get("font"))
        size = self._norm_size(span.get("size"))
        color = self._norm_color(span.get("color"))

        # 1) точный ключ
        exact = self.style_to_header.get((font, float(size), int(color)))
        if exact:
            return exact

        # 2) fallback: «читаемые» правила с допуском по size
        for rule in self.rules:
            if rule["font"] != font:
                continue
            if rule["color"] != color:
                continue
            if abs(rule["size"] - size) <= self.size_tol:
                return rule["header"]

        return ""
