# settings.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env as early as possible (works for uvicorn reload too)
load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    upload_dir: str = os.getenv("UPLOAD_DIR", "uploaded_pdfs")
    max_evidence_chunks: int = int(os.getenv("MAX_EVIDENCE_CHUNKS", "6"))
    max_answer_tokens: int = int(os.getenv("MAX_ANSWER_TOKENS", "500"))


settings = Settings()
