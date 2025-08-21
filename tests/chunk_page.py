from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

# Пример текста (примерно страница)

with open("data/example.txt", "r", encoding="utf-8") as fp:
    sample_text = fp.read()

# Создаем документ
document = Document(text=sample_text)

# Инициализируем SentenceSplitter с параметрами
# chunk_size - максимальное количество символов в чанке
# chunk_overlap - перекрытие между чанками в символах
splitter = SentenceSplitter(
    chunk_size=200,
    chunk_overlap=100,
)

# Разбиваем текст на чанки
nodes = splitter.get_nodes_from_documents([document])

# Выводим чанки
for i, node in enumerate(nodes):
    print(f"Чанк {i + 1}:")
    print(node.text)
    print("-" * 50)
