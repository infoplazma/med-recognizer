from typing import Any, Dict, Optional
from langchain_core.runnables import RunnableConfig


def log_chain_step(input_dict: Dict[str, Any], config: RunnableConfig, tag: Optional[str] = None, msg: str = "") -> Dict:
    """
    Внутреннее логирование шагов цепочки. Печатает сообщение, только если `tag`
    присутствует среди `config["tags"]`.

    :param input_dict: произвольный вход (прозрачно прокидывается дальше)
    :param config: текущий RunnableConfig или None
    :param tag: метка, по которой фильтруется лог
    :param msg: текст сообщения
    :return: исходный input_dict без изменений
    """
    if tag and config and tag in (config.get("tags") or []):
        print(f"\r[{tag}] {msg}.\nReasoning...", end="")
    return input_dict
