# ✅ Правильный пакет для PDF в LlamaIndex

### **`llama-index-readers-file`**

* Он устанавливается стандартно вместе с `pip install llama-index` (core bundle) или может быть установлен самостоятельно:

  ```bash
  pip install llama-index llama-index-readers-file
  ```
* Именно он предоставляет `PDFReader`, `DocxReader`, `MarkdownReader` и др. ([PyPI][1], [docs.llamaindex.ai][2], [GitHub][3])

**Пример использования:**

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader

documents = SimpleDirectoryReader(
    input_dir="data",
    file_extractor={".pdf": PDFReader()}
).load_data()
```

Это официальная поддержка PDF в LlamaIndex. ([PyPI][1])

---

## 🧠 Более продвинутые PDF-ридеры

Если тебе нужно **лучшее chunking по структуре документа** — заголовки, таблицы, списки, разделы, то есть специальные интеграции:

### **`llama-index-readers-smart-pdf-loader`**

* Использует API от **llmsherpa**: автоматически извлекает структуру PDF и разбивает его на логические части (секции, таблицы и др.).
* Подходит для полноценной chunk-разбивки по смыслу. ([PyPI][4])

**Пример:**

```python
from llama_index.readers.smart_pdf_loader import SmartPDFLoader

pdf_loader = SmartPDFLoader(llmsherpa_api_url="https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all")
documents = pdf_loader.load_data("path_or_url_to_pdf.pdf")
```

* Далее можно индексировать и выдавать результаты в LlamaIndex. ([PyPI][4])

---

### 📦 Варианты: `llama-index-readers-pdf-marker`, `PDFTableReader`

* `llama-index-readers-pdf-marker` — парсит PDF → markdown. Быстро и точно, но менее мощно, чем smart loader. ([llamahub.ai][5])
* `llama-index-readers-pdf-table` — извлекает таблицы с указанной страницы. Подходит, если таблицы важны. ([PyPI][6])

---

## 📋 Обзор по пакетам и их назначениям

| Пакет                                  | Назначение                                               |
| -------------------------------------- | -------------------------------------------------------- |
| `llama-index-readers-file`             | Стандартный rич загрузчик PDFReader и др.                |
| `llama-index-readers-smart-pdf-loader` | Smart chunking по структурам (разделы, списки, таблицы). |
| `llama-index-readers-pdf-marker`       | Преобразование PDF → Markdown, сохраняется структура.    |
| `llama-index-readers-pdf-table`        | Извлечение таблиц из PDF-файлов.                         |

---

## 🛠 Что делать, чтобы исправить ошибку

1. Установить нужный пакет:

   ```bash
   pip install llama-index llama-index-readers-file
   ```

2. Если нужен SmartPDFLoader:

   ```bash
   pip install llama-index-readers-smart-pdf-loader llmsherpa
   ```

3. Импорт и использование:

```python
from llama_index.readers.file import PDFReader
from llama_index.core import SimpleDirectoryReader

docs = SimpleDirectoryReader(
    input_dir="data",
    file_extractor={".pdf": PDFReader()}
).load_data()
```

или для smart:

```python
from llama_index.readers.smart_pdf_loader import SmartPDFLoader

loader = SmartPDFLoader(llmsherpa_api_url="https://readers.llmsherpa.com/api/...")  # endpoint разработчика
docs = loader.load_data("your.pdf")
```

---

## 🚨 Почему ты получил ошибку

* Пакет `llama-index-readers-pdf` **не существует**, поэтому pip его не нашёл.
* Нужен **правильный modular loader (`readers-file`)**, или сторонние расширения (smart loader, marker, table).

---

## ✏️ Резюме

* Для базовой работы с PDF в LlamaIndex используй **`llama-index-readers-file`**, он содержит `PDFReader`. ([PyPI][1], [PyDigger][7], [docs.llamaindex.ai][8], [GitHub][9], [docs.llamaindex.ai][10])
* Для сложных PDF с layout-структурой рекомендуются:

  * **SmartPDFLoader**, если нужен автоматический структурный chunking. ([PyPI][4])
  * **pdf-marker или PDFTableReader**, если важно markdown-конверсия или таблицы. ([llamahub.ai][5])

---

Если хочешь — могу показать конкретный пример загрузки PDF с `SmartPDFLoader` или `PDFReader`, и как дальше разбить его на чанки с метаданными страницы и раздела.

[1]: https://pypi.org/project/llama-index-readers-file/?utm_source=chatgpt.com "llama-index-readers-file - PyPI"
[2]: https://docs.llamaindex.ai/en/stable/getting_started/installation/?utm_source=chatgpt.com "Installation and Setup - LlamaIndex"
[3]: https://github.com/run-llama/llama_index/issues/11681?utm_source=chatgpt.com "ImportError: cannot import name 'PDFReader' from 'llama_index ..."
[4]: https://pypi.org/project/llama-index-readers-smart-pdf-loader/?utm_source=chatgpt.com "llama-index-readers-smart-pdf-loader - PyPI"
[5]: https://llamahub.ai/l/readers/llama-index-readers-pdf-marker?from=&utm_source=chatgpt.com "LlamaIndex Readers Integration: Pdf-Marker - Llama Hub"
[6]: https://pypi.org/project/llama-index-readers-pdf-table/?utm_source=chatgpt.com "llama-index-readers-pdf-table - PyPI"
[7]: https://pydigger.com/pypi/llama-index-readers-smart-pdf-loader?utm_source=chatgpt.com "llama-index-readers-smart-pdf-loader - PyDigger"
[8]: https://docs.llamaindex.ai/en/stable/api_reference/readers/smart_pdf_loader/?utm_source=chatgpt.com "Smart pdf loader - LlamaIndex"
[9]: https://github.com/run-llama/llama_index/issues/15243?utm_source=chatgpt.com "How to handle complex PDFs · Issue #15243 · run-llama/llama_index"
[10]: https://docs.llamaindex.ai/en/stable/module_guides/loading/?utm_source=chatgpt.com "Loading Data - LlamaIndex"


# 📚 SmartPDFLoader

### **Нужна ли регистрация для SmartPDFLoader?**

* **Нет**, ты не регистрируешься на llama-index — всё, что нужно, это установить модуль:

  ```bash
  pip install llama-index-readers-smart-pdf-loader
  ```
* **Да**, тебе потребуется доступ к **API-серверу LayoutPDFReader от LLMSherpa**:
  SMART PDF Loader использует `LayoutPDFReader`, который отправляет PDF в API (на `llmsherpa_api_url`) для структурного разбора.([PyPI][1])

---

### ✅ Возможности и требования

* **LayoutPDFReader** понимает структуру документа: заголовки, списки, таблицы, подразделы.
* Он автоматически разбивает PDF на логические чанки (разделы, таблицы и т. д.) — это как раз «smart chunking».([GitHub][2])
* API-сервер может быть:

  * публичным от LLMSherpa (не требуется регистрация) — хотя иногда бывают недоступности.([Llama Hub][3], [GitHub][4])
  * **самостоятельно хоститься** (через Docker), чтобы обеспечить надёжность и контроль.([GoPenAI][5])

---

### ⚙ Пример работы с SmartPDFLoader

```python
from llama_index.readers.smart_pdf_loader import SmartPDFLoader
from llama_index.core import VectorStoreIndex

llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
pdf_loader = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url)
documents = pdf_loader.load_data("path_or_url_to_pdf.pdf")

index = VectorStoreIndex.from_documents(documents)
response = index.as_query_engine().query("Что такое BART?")
print(response)
```

* `SmartPDFLoader` отправляет PDF на указанный API, получает структурированные блоки (`doc.chunks()`) и преобразует их в объекты `Document` с `chunk_type` и другими метаданными.([LlamaIndex][6])

---

### 📌 Итак:

* **Регистрация на llama-index не нужна.**
* **Но**: для работы SmartPDFLoader потребуется доступ к **LLMSherpa API** (либо публичному, либо self-hosted).
* Если публичный API недоступен — можешь развернуть его локально (через Docker) и использовать `LayoutPDFReader` локально.([Stack Overflow][7], [GoPenAI][5])

---

### 👣 Резюме:

| Этап                                               | Нужна регистрация                              |
| -------------------------------------------------- | ---------------------------------------------- |
| `pip install llama-index-readers-smart-pdf-loader` | ❌ Нет                                          |
| Обращение к стороннему LLMSherpa API               | ✅ Да (но публичный, регистрация необязательна) |
| Self-hosted LLMSherpa сервер (Docker)              | 👍 Нет                                         |

---

Если хочешь, могу показать как запустить локальный сервер LLMSherpa с Docker и протестировать SmartPDFLoader на твоём документе.

[1]: https://pypi.org/project/llama-index-readers-smart-pdf-loader/?utm_source=chatgpt.com "llama-index-readers-smart-pdf-loader - PyPI"
[2]: https://github.com/nlmatics/llmsherpa?utm_source=chatgpt.com "nlmatics/llmsherpa: Developer APIs to Accelerate LLM Projects"
[3]: https://llamahub.ai/l/readers/llama-index-readers-smart-pdf-loader?from=readers&utm_source=chatgpt.com "Smart PDF Loader - Llama Hub"
[4]: https://github.com/run-llama/llama_index/issues/14902?utm_source=chatgpt.com "[Bug]: Smart PDF Loader - Failed to establish a new connection ..."
[5]: https://blog.gopenai.com/running-pdf-parsers-in-docker-containers-5e7a7ed829c8?utm_source=chatgpt.com "Running PDF Parsers in Docker Containers - GoPenAI"
[6]: https://docs.llamaindex.ai/en/stable/api_reference/readers/smart_pdf_loader/?utm_source=chatgpt.com "Smart pdf loader - LlamaIndex"
[7]: https://stackoverflow.com/questions/78125440/loading-data-with-llama-index-after-upgrading-package-cant-find-readers-file?utm_source=chatgpt.com "loading data with llama-index after upgrading package can't find ..."
