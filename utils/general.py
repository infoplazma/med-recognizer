import os
import logging
import json
from typing import List, Dict, Optional, Union


def create_local_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)
    return logger


def save_json(data: Union[List, Dict], filepath: str, logger: Optional[logging.Logger] = None) -> bool:
    """
    Сохраняет данные в JSON-файл.

    Args:
        data (Union[List, Dict]): Данные для сохранения (список или словарь).
        filepath (str): Путь к файлу для сохранения.
        logger (Optional[logging.Logger]): Логгер для записи ошибок. Если None, ошибки не логируются.

    Returns:
        bool: True, если сохранение прошло успешно, False в случае ошибки.
    """
    # Проверяем, существует ли директория, если нет — создаем
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if logger:
                logger.error(f"❌ Не удалось создать директорию {directory}: {str(e)}")
            else:
                print(f"❌ Не удалось создать директорию {directory}: {str(e)}")
            return False

    try:
        with open(filepath, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=4)
        if logger:
            logger.info(f"✅ Данные успешно сохранены в файл: {os.path.abspath(filepath)}")
        else:
            print(f"✅ Данные успешно сохранены в файл: {os.path.abspath(filepath)}")
        return True
    except json.JSONEncodeError as e:
        if logger:
            logger.error(f"❌ Ошибка кодирования JSON для файла {filepath}: {str(e)}")
        else:
            print(f"❌ Ошибка кодирования JSON для файла {filepath}: {str(e)}")
        return False
    except IOError as e:
        if logger:
            logger.error(f"❌ Ошибка записи в файл {filepath}: {str(e)}")
        else:
            print(f"❌ Ошибка записи в файл {filepath}: {str(e)}")
        return False


def load_json(filepath: str, logger: Optional[logging.Logger] = None) -> Optional[Union[Dict, List]]:
    """
    Загружает JSON-данные из файла.

    Args:
        filepath (str): Путь к JSON-файлу.
        logger (Optional[logging.Logger]): Логгер для записи ошибок. Если None, ошибки не логируются.

    Returns:
        Optional[Union[Dict, List]]: Загруженные данные (словарь или список) или None в случае ошибки.
    """
    if not os.path.isfile(filepath):
        if logger:
            logger.error(f"❌ Файл не найден: {filepath}")
        else:
            print(f"❌ Файл не найден: {filepath}")
        return None
    try:
        with open(filepath, "r", encoding='utf-8') as fp:
            return json.load(fp)
    except json.JSONDecodeError as e:
        if logger:
            logger.error(f"❌ Ошибка декодирования JSON из файла {filepath}: {str(e)}")
        else:
            print(f"❌ Ошибка декодирования JSON из файла {filepath}: {str(e)}")
        return None
    except IOError as e:
        if logger:
            logger.error(f"❌ Ошибка чтения файла {filepath}: {str(e)}")
        else:
            print(f"❌ Ошибка чтения файла {filepath}: {str(e)}")
        return None