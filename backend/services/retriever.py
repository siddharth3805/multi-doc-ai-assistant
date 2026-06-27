# backend/services/retriever.py

"""
Vector Store Service

Stores and retrieves text chunks using ChromaDB.

ChromaDB is a vector database that:
- Stores text + embeddings + metadata together
- Searches by semantic similarity
- Persists data to disk
- Supports metadata filtering

Think of it as a special database where you search
by MEANING instead of exact keywords.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Optional
from rag.chunker import TextChunk
from backend.services.embeddings import EmbeddingsService
from config.settings import CHROMA_PERSIST_DIR
from utils.logger import get_logger

logger = get_logger(__name__)




class VectorStoreService:
    """
    Manages ChromaDB vector storage and retrieval.

    Usage:
        store = VectorStoreService()
        store.add_chunks(chunks)
        results = store.search("payment terms", top_k=5)
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_provider: str = "openai"
    ):
        """
        Args:
            collection_name:    ChromaDB collection (like a table)
            embedding_provider: "openai" or "local"
        """
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PERSIST_DIR)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        # Initialize embeddings service
        self.embeddings = EmbeddingsService(provider=embedding_provider)

        logger.info(
            f"VectorStoreService initialized: "
            f"collection='{collection_name}', "
            f"provider='{embedding_provider}'"
        )

    def add_chunks(self, chunks: List[TextChunk]) -> int:
        """
        Adds text chunks to ChromaDB.

        Args:
            chunks: List of TextChunk objects

        Returns:
            Number of chunks added
        """
        if not chunks:
            logger.warning("No chunks to add")
            return 0

        logger.info(f"Adding {len(chunks)} chunks to ChromaDB")

        # Extract components for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Generate embeddings for all chunks
        embeddings = self.embeddings.embed_texts(texts)

        # Store in ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        logger.info(f"Successfully added {len(chunks)} chunks")
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None
    ) -> List[dict]:
        """
        Searches for chunks similar to the query.

        Args:
            query:         User's question
            top_k:         Number of results to return
            source_filter: Only search in specific file

        Returns:
            List of dicts with text, metadata, and similarity score
        """
        logger.info(f"Searching: '{query[:50]}...' top_k={top_k}")

        # Embed the query
        query_embedding = self.embeddings.embed_text(query)

        # Build filter if source specified
        where_filter = None
        if source_filter:
            where_filter = {"source": source_filter}

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity score
                # ChromaDB cosine distance: 0=identical, 2=opposite
                similarity = 1 - (dist / 2)

                formatted.append({
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", 0),
                    "file_type": meta.get("file_type", "unknown"),
                    "similarity": round(similarity, 4),
                    "rank": i + 1
                })

        logger.info(f"Found {len(formatted)} results")
        return formatted

    def get_collection_stats(self) -> dict:
        """Returns info about stored documents."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection.name
        }

    def delete_document(self, filename: str) -> int:
        """
        Removes all chunks from a specific file.

        Args:
            filename: Name of file to delete

        Returns:
            Number of chunks deleted
        """
        # Find chunks from this file
        results = self.collection.get(
            where={"source": filename}
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            count = len(results["ids"])
            logger.info(f"Deleted {count} chunks from {filename}")
            return count

        logger.warning(f"No chunks found for {filename}")
        return 0

    def clear_all(self) -> None:
        """Clears entire collection. Use with caution."""
        self.client.delete_collection(self.collection.name)
        logger.warning("Entire vector store cleared")