# config/settings.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ── BASE PATH ──────────────────────────────────────────
# Path(__file__) → config/settings.py
# .parent        → config/
# .parent        → project root
BASE_DIR = Path(__file__).parent.parent

# ── APP SETTINGS ───────────────────────────────────────
APP_NAME: str = os.getenv("APP_NAME", "Multi-Doc AI Assistant")
APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── API KEYS ───────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ── FILE UPLOAD ────────────────────────────────────────
UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS: set = {
    ext.strip().lower()
    for ext in os.getenv("ALLOWED_EXTENSIONS", "pdf,docx,txt,csv,md").split(",")
}

# ── VECTOR DATABASE ────────────────────────────────────
CHROMA_PERSIST_DIR: Path = BASE_DIR / "vectorstore" / "chroma"
FAISS_INDEX_DIR: Path = BASE_DIR / "vectorstore" / "faiss"

# ── DATABASE ───────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{BASE_DIR / 'database' / 'app.db'}"
)

# ── SECURITY ───────────────────────────────────────────
SECRET_KEY: str = os.getenv("SECRET_KEY", "change-this-in-production")

# ── RAG SETTINGS ───────────────────────────────────────
TOP_K_RESULTS: int = 5
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# ── LLM SETTINGS ──────────────────────────────────────
DEFAULT_LLM_PROVIDER: str = "openai"
DEFAULT_MODEL: str = "gpt-4o-mini"
MAX_TOKENS: int = 2048
TEMPERATURE: float = 0.1

# ── CREATE DIRECTORIES IF MISSING ─────────────────────
for _dir in [UPLOAD_DIR, PROCESSED_DIR, CHROMA_PERSIST_DIR, FAISS_INDEX_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)