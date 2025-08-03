# index_creator.py

import os
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core import Settings

from settings import MED_SOURCE_DIR, PERSISTED_INDEX_DIR, INDEXED_FILES_PATH

# Временно отключаем
Settings.llm = None


# Функция для получения уже индексированных файлов
def get_indexed_files():
    if not os.path.exists(INDEXED_FILES_PATH):
        return set()
    with open(INDEXED_FILES_PATH, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f.readlines())


# Функция для добавления новых файлов в список
def add_indexed_files(files):
    with open(INDEXED_FILES_PATH, "a", encoding="utf-8") as f:
        for file in files:
            f.write(file + "\n")


indexed_files = get_indexed_files()

# 1. Считываем документы
docs = SimpleDirectoryReader(
    input_dir=MED_SOURCE_DIR,
    file_extractor={".pdf": PDFReader()}
).load_data()

# 2. Оставляем только новые документы (по file_name в metadata)
new_docs = [doc for doc in docs if doc.metadata.get("file_name") not in indexed_files]

if not new_docs:
    print("Нет новых документов для индексации!")
else:
    # 3. Создаём splitter (разбивка на чанки)
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

    # 4. Разбиваем на чанки каждый новый документ
    all_nodes = []
    for doc in new_docs:
        nodes = splitter.get_nodes_from_documents([doc])
        all_nodes.extend(nodes)

    # 5. Добавляем инфу о файле и странице
    for node in all_nodes:
        page_num = node.metadata.get("page_label", "unknown")
        source = node.metadata.get("file_name", "unknown")
        node.text += f"\nsource:'{source}' page:{page_num}"
        # print(node.text[-100:])
        # print("---")

    # 6. Индексация новых документов
    if os.path.exists(PERSISTED_INDEX_DIR):
        # Если индекс уже существует — догружаем его и дополняем
        storage_context = StorageContext.from_defaults(persist_dir=PERSISTED_INDEX_DIR)
        index = load_index_from_storage(storage_context)
        index.insert_nodes(all_nodes)
        index.storage_context.persist(persist_dir=PERSISTED_INDEX_DIR)
    else:
        # Создаём новый индекс
        index = VectorStoreIndex.from_documents(all_nodes)
        index.storage_context.persist(persist_dir=PERSISTED_INDEX_DIR)

    # 7. Добавляем имена новых файлов в учёт
    add_indexed_files([doc.metadata.get("file_name", "unknown") for doc in new_docs])

# --- Теперь, для запросов, загружаем индекс:
storage_context = StorageContext.from_defaults(persist_dir=PERSISTED_INDEX_DIR)
loaded_index = load_index_from_storage(storage_context)

query_engine = loaded_index.as_query_engine()
response = query_engine.query("Cryptosporidiosis*")
print(response)

