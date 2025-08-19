from typing import Annotated, get_args, Callable
import operator

# 1. Создаем аннотированный тип
# Здесь 'int' - это базовый тип, а остальное - метаданные.
DbInfo = Annotated[int, "primary_key", "autoincrement", operator.add]


def process_type(some_type):
    """Функция для извлечения и печати метаданных."""

    # 2. Извлекаем аргументы из аннотации
    type_args = get_args(some_type)

    # 3. get_args возвращает кортеж. Если это не Annotated, кортеж будет пустым.
    if not type_args:
        print(f"Тип '{some_type}' не содержит метаданных.")
        return

    # 4. Распаковываем кортеж
    base_type = type_args[0]
    metadata = type_args[1:]  # все остальные элементы - метаданные

    print(f"Исходный тип: {base_type}")
    print(f"Метаданные: {metadata}")

    meta_funcs = []
    for meta_fun in metadata:
        if callable(meta_fun):
            meta_funcs.append(meta_fun)

    print(meta_funcs)


# --- Проверяем ---
process_type(DbInfo)
print("-" * 20)
process_type(str)  # Пример с обычным типом