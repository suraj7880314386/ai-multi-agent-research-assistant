"""ChromaDB vector store for document indexing and semantic retrieval."""

import logging
from typing import List, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.schema import Document

from app.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages ChromaDB operations for semantic search."""

    def __init__(self):
        self._client = None
        self._collection = None
        self._embedding_fn = None

    def _get_embedding_function(self):
        """Lazy-load the embedding model."""
        if self._embedding_fn is None:
            from chromadb.utils import embedding_functions

            self._embedding_fn = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=settings.embedding_model
                )
            )
        return self._embedding_fn

    def _get_client(self) -> chromadb.ClientAPI:
        """Get or create the ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=settings.chroma_persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self):
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=settings.chroma_collection_name,
                embedding_function=self._get_embedding_function(),
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_documents(self, documents: List[Document]) -> int:
        """
        Add LangChain Document objects to the vector store.

        Returns:
            Number of documents added.
        """
        collection = self._get_collection()

        ids = []
        texts = []
        metadatas = []

        for doc in documents:
            doc_id = f"{doc.metadata['doc_id']}_chunk_{doc.metadata['chunk_index']}"
            ids.append(doc_id)
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)

        collection.add(ids=ids, documents=texts, metadatas=metadatas)

        logger.info(f"Added {len(documents)} chunks to vector store")
        return len(documents)

    def search(
        self, query: str, top_k: int = 5, filter_dict: dict = None
    ) -> List[Tuple[Document, float]]:
        """
        Semantic search over the vector store.

        Returns:
            List of (Document, relevance_score) tuples sorted by relevance.
        """
        collection = self._get_collection()

        search_params = {
            "query_texts": [query],
            "n_results": min(top_k, collection.count()) if collection.count() > 0 else top_k,
        }

        if filter_dict:
            search_params["where"] = filter_dict

        results = collection.query(**search_params)

        documents_with_scores = []
        if results["documents"] and results["documents"][0]:
            for i, doc_text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0

                # Convert cosine distance to similarity score (0-1)
                relevance_score = 1 - distance

                doc = Document(page_content=doc_text, metadata=metadata)
                documents_with_scores.append((doc, relevance_score))

        logger.info(f"Search returned {len(documents_with_scores)} results for: {query[:50]}...")
        return documents_with_scores

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks belonging to a specific document."""
        collection = self._get_collection()

        # Get IDs matching this doc_id
        results = collection.get(where={"doc_id": doc_id})

        if results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for doc_id={doc_id}")
            return len(results["ids"])

        return 0

    def get_stats(self) -> dict:
        """Get vector store statistics."""
        try:
            collection = self._get_collection()
            return {
                "status": "connected",
                "total_chunks": collection.count(),
                "collection_name": settings.chroma_collection_name,
            }
        except Exception as e:
            return {"status": f"error: {str(e)}", "total_chunks": 0}

    def reset(self):
        """Clear all data from the vector store."""
        client = self._get_client()
        try:
            client.delete_collection(settings.chroma_collection_name)
            self._collection = None
            logger.info("Vector store reset complete")
        except Exception:
            pass


# Singleton
vector_store = VectorStore()
