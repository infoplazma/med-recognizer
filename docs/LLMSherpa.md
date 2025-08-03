Конечно! Вот **подробная инструкция, как быстро запустить LLMSherpa локально через Docker** и использовать SmartPDFLoader для структурного парсинга PDF на твоём сервере или ноутбуке.
Это лучший способ, если ты не хочешь зависеть от публичного API или у тебя корпоративные данные.

---

# 🚀 **1. Запуск LLMSherpa (LayoutPDFReader) локально через Docker**

### **A. Установи Docker, если он ещё не установлен**

* [Docker для Windows](https://www.docker.com/products/docker-desktop/)
* [Docker для Linux](https://docs.docker.com/engine/install/)

---

### **B. Запусти LLMSherpa контейнер**

```bash
docker pull nlmatics/llmsherpa:latest
docker run -d -p 8080:80 --name llmsherpa nlmatics/llmsherpa:latest
```

* После этого сервис будет доступен по адресу:
  **[http://localhost:8080/api/document/developer/parseDocument?renderFormat=all](http://localhost:8080/api/document/developer/parseDocument?renderFormat=all)**

---

### **C. Проверь работу API**

Открой в браузере:

```
http://localhost:8080/
```

или попробуй запрос через curl:

```bash
curl http://localhost:8080/api/
```

---

# 🐍 **2. Используй SmartPDFLoader с локальным LLMSherpa API**

```python
from llama_index.readers.smart_pdf_loader import SmartPDFLoader

llmsherpa_api_url = "http://localhost:8080/api/document/developer/parseDocument?renderFormat=all"
pdf_loader = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url)

documents = pdf_loader.load_data("your_pdf_file.pdf")
```

* `your_pdf_file.pdf` — путь к твоему файлу (или можно указать URL).

---

# 📋 **3. Пример полного рабочего кода**

```python
from llama_index.readers.smart_pdf_loader import SmartPDFLoader
from llama_index.core import VectorStoreIndex

# Используй локальный сервер LLMSherpa
llmsherpa_api_url = "http://localhost:8080/api/document/developer/parseDocument?renderFormat=all"

# Парсим PDF
loader = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url)
documents = loader.load_data("your_pdf_file.pdf")

# Индексация и простой запрос
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("Каковы начальные симптомы гриппа?")
print(response)
```

---

# ⚡️ **Что ты получаешь:**

* Полный контроль, нет внешних API-лимитов.
* Чанки PDF разбиты по структуре: разделы, подзаголовки, списки, таблицы.
* Можно строить гибкий RAG-поиск и генерацию на своих данных.

---

## **Если появятся вопросы по Docker или по LLMSherpa API — пиши!**

Могу объяснить логику работы, рассказать, как получить более структурированные чанки или объединять таблицы и разделы.
Очень хороший вопрос!
**Нет, каждый раз полный `docker run ...` делать не нужно.**
Это нужно только при первом запуске, чтобы создать контейнер.
Дальше для управления — используй команды `docker start` и `docker stop`.

---

## **Пояснение**

* **`docker run`** — создаёт **новый контейнер** из образа (если такого контейнера с этим именем ещё нет).
* **`docker start llmsherpa`** — запускает **уже существующий** (созданный ранее) контейнер.
* **`docker stop llmsherpa`** — останавливает работающий контейнер.

---

## **Стандартный рабочий цикл**

1. **Первый запуск (создание):**

   ```bash
   docker run -d -p 8080:80 --name llmsherpa nlmatics/llmsherpa:latest
   ```
2. **Дальше, когда нужно включить сервер:**

   ```bash
   docker start llmsherpa
   ```
3. **Когда нужно остановить сервер:**

   ```bash
   docker stop llmsherpa
   ```
4. **Если вдруг контейнер был удалён (`docker rm`), то только тогда снова делай полный `docker run ...`**

---

## **Проверка контейнеров**

* Список всех контейнеров (включая остановленные):

  ```bash
  docker ps -a
  ```

---

## **Если изменил параметры или хочешь пересоздать контейнер**

1. Останови контейнер:

   ```bash
   docker stop llmsherpa
   ```
2. Удали контейнер:

   ```bash
   docker rm llmsherpa
   ```
3. И только теперь снова создай с `docker run ...` (например, если меняешь порты, имя, переменные окружения и т.д.).

---

**Резюме:**

> **В повседневной работе достаточно делать только `docker start llmsherpa` и `docker stop llmsherpa`. Полный `docker run ...` нужен только для создания нового контейнера!**

---

Если что-то не запускается или есть вопросы по Docker — спрашивай!
