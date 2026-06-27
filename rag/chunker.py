# rag/chunker.py

"""
Text Chunking Service

Splits large documents into smaller overlapping chunks
for storage in the vector database.

Three strategies:
1. Recursive Character Splitter  ← Default, works for everything
2. Semantic Chunker              ← Splits on meaning (advanced)
3. Fixed Size Chunker            ← Simple, predictable

Industry standard: Recursive + overlap is used by
LangChain, LlamaIndex, and most production RAG systems.
"""

from dataclasses import dataclass, field
from typing import List
# TO THIS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.services.document_processor import ProcessedDocument
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """
    Represents one chunk of text ready for embedding.

    chunk_id:      Unique identifier
    text:          The actual text content
    source_file:   Which file this came from
    page_number:   Which page (for citations)
    chunk_index:   Position in document (0, 1, 2...)
    char_count:    Length of this chunk
    metadata:      Extra info for filtering
    """
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    char_count: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.char_count = len(self.text)


# ─────────────────────────────────────────────────────────────
# CHUNKER CLASS
# ─────────────────────────────────────────────────────────────

class DocumentChunker:
    """
    Splits ProcessedDocument into TextChunks.

    Usage:
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(processed_doc)
        print(f"Created {len(chunks)} chunks")
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Args:
            chunk_size:    Max characters per chunk (default 1000)
            chunk_overlap: Characters shared between chunks (default 200)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # RecursiveCharacterTextSplitter tries to split on:
        # 1. Paragraphs (\n\n)  ← preferred
        # 2. Lines (\n)
        # 3. Sentences (. )
        # 4. Words ( )
        # 5. Characters         ← last resort
        # This preserves natural language boundaries
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        logger.info(
            f"DocumentChunker initialized: "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

    def chunk_document(self, doc: ProcessedDocument) -> List[TextChunk]:
        """
        Splits a ProcessedDocument into TextChunks.

        Strategy:
        - Process each page separately
        - Track which page each chunk came from
        - Assign unique IDs for vector database storage

        Args:
            doc: ProcessedDocument from DocumentProcessor

        Returns:
            List of TextChunk objects
        """
        all_chunks = []
        chunk_index = 0

        logger.info(f"Chunking document: {doc.filename}")

        for page in doc.pages:
            # Skip empty pages
            if not page.text.strip():
                continue

            # Split this page's text into chunks
            page_chunks = self.splitter.split_text(page.text)

            for chunk_text in page_chunks:
                # Skip tiny chunks (less than 50 chars)
                if len(chunk_text.strip()) < 50:
                    continue

                # Create unique ID: filename_page_chunkindex
                chunk_id = (
                    f"{doc.filename}_p{page.page_number}_c{chunk_index}"
                )

                chunk = TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text.strip(),
                    source_file=doc.filename,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    metadata={
                        "source": doc.filename,
                        "page": page.page_number,
                        "file_type": doc.file_type,
                        "file_path": doc.file_path,
                        "chunk_index": chunk_index
                    }
                )

                all_chunks.append(chunk)
                chunk_index += 1

        logger.info(
            f"Chunking complete: {doc.filename} → "
            f"{len(all_chunks)} chunks from {doc.total_pages} pages"
        )

        return all_chunks

    def chunk_multiple_documents(
        self,
        docs: List[ProcessedDocument]
    ) -> List[TextChunk]:
        """
        Chunks multiple documents at once.

        Args:
            docs: List of ProcessedDocuments

        Returns:
            Combined list of all chunks from all documents
        """
        all_chunks = []

        for doc in docs:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            logger.info(f"Added {len(chunks)} chunks from {doc.filename}")

        logger.info(f"Total chunks across all documents: {len(all_chunks)}")
        return all_chunks

    def get_chunk_stats(self, chunks: List[TextChunk]) -> dict:
        """
        Returns statistics about the chunks.
        Useful for debugging and optimization.
        """
        if not chunks:
            return {"total": 0}

        sizes = [c.char_count for c in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(sizes) // len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "total_characters": sum(sizes),
            "unique_sources": len(set(c.source_file for c in chunks))
        }