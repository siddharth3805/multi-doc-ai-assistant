# backend/services/embeddings.py

"""
Embeddings Service

Converts text into vector representations using:
- OpenAI text-embedding-3-small (primary, best quality)
- Sentence Transformers (free, local fallback)

Industry practice:
OpenAI embeddings are used by most production RAG systems
because they have the best quality/cost ratio.
Sentence Transformers are used when privacy matters
(data never leaves your machine).
"""

from typing import List
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from config.settings import OPENAI_API_KEY
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingsService:
    """
    Generates embeddings for text using OpenAI or local models.

    Usage:
        service = EmbeddingsService(provider="openai")
        vectors = service.embed_texts(["hello world", "how are you"])
        print(len(vectors[0]))  # 1536 dimensions
    """

    # Available embedding models
    OPENAI_MODEL = "text-embedding-3-small"  # 1536 dims, cheap, fast
    LOCAL_MODEL = "all-MiniLM-L6-v2"         # 384 dims, free, offline

    def __init__(self, provider: str = "openai"):
        """
        Args:
            provider: "openai" or "local"
        """
        self.provider = provider

        if provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is missing in .env file"
                )
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = self.OPENAI_MODEL
            self.dimensions = 1536
            logger.info(f"EmbeddingsService: OpenAI ({self.model})")

        elif provider == "local":
            logger.info("Loading local embedding model (first time is slow)...")
            self.model_instance = SentenceTransformer(self.LOCAL_MODEL)
            self.dimensions = 384
            logger.info(f"EmbeddingsService: Local ({self.LOCAL_MODEL})")

        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'local'")

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single text string.

        Args:
            text: Text to embed

        Returns:
            List of floats (the vector)
        """
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds multiple texts in one API call (more efficient).

        Args:
            texts: List of texts to embed

        Returns:
            List of vectors (one per text)
        """
        if not texts:
            return []

        # Clean texts — embedding models don't like empty strings
        cleaned = [t.strip().replace("\n", " ") for t in texts]
        cleaned = [t for t in cleaned if t]

        if self.provider == "openai":
            return self._embed_openai(cleaned)
        else:
            return self._embed_local(cleaned)

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """Calls OpenAI Embeddings API."""
        try:
            logger.info(f"Embedding {len(texts)} texts with OpenAI")

            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )

            # Extract vectors from response
            vectors = [item.embedding for item in response.data]

            logger.info(
                f"Embeddings created: {len(vectors)} vectors, "
                f"{len(vectors[0])} dimensions each"
            )

            return vectors

        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """Uses local SentenceTransformer model."""
        try:
            logger.info(f"Embedding {len(texts)} texts locally")

            vectors = self.model_instance.encode(
                texts,
                convert_to_numpy=True
            ).tolist()

            logger.info(f"Local embeddings created: {len(vectors)} vectors")
            return vectors

        except Exception as e:
            logger.error(f"Local embedding error: {e}")
            raise