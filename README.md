## ✅ Правильный пакет для PDF в LlamaIndex

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

## 🛠 Что делать

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

