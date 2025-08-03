# retriever_adapter.py

from typing import Any, List

from pydantic import Field
from langchain.schema import BaseRetriever, Document


class LlamaRetrieverForLangChain(BaseRetriever):
    llama_retriever: Any = Field()  # это теперь валидное поле для pydantic!

    def _get_relevant_documents(self, query: str) -> List[Document]:
        nodes = self.llama_retriever.retrieve(query)
        docs = [
            Document(
                page_content=node.text,
                metadata=node.metadata
            )
            for node in nodes
        ]
        return docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        # Если не используешь асинхронно, просто повторяем sync-метод
        return self._get_relevant_documents(query)
