from abc import ABC, abstractmethod, abstractproperty

from langchain_core.documents import Document


class VectorStore(ABC):
    """Abstract base class for vector store implementations.

    Defines interface for document storage, retrieval, and management using
    vector embeddings. Supports both sync and async operations.
    """

    @abstractproperty
    def vector_store(self):
        """Vector store"""

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete collection"""

    @abstractmethod
    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of Document objects to add.
            ids: Corresponding unique IDs for each document.
        """

    @abstractmethod
    async def aadd_documents(self, documents: list[Document], ids: list[str]) -> None:
        """Async add documents to the vector store.

        Args:
            documents: List of Document objects to add.
            ids: Corresponding unique IDs for each document.
        """

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents from the vector store.

        Args:
            ids: List of document IDs to delete.
        """

    @abstractmethod
    async def adelete(self, ids: list[str]) -> None:
        """Async delete documents from the vector store.

        Args:
            ids: List of document IDs to delete.
        """

    @abstractmethod
    def save(self) -> None:
        """Save vector store content"""
