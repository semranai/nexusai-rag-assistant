# embeddings.py
from __future__ import annotations

from typing import List
from openai import OpenAI

from settings import settings


class ProfessionalEmbeddingGenerator:
    """
    Embeddings generator that does NOT crash at import-time.
    It creates the OpenAI client only when actually needed.
    """

    def __init__(self, model: str):
        self.model = model
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client

        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Put it in .env (OPENAI_API_KEY=...) "
                "or set it as an environment variable."
            )

        self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()

        # OpenAI embeddings API: pass list of strings
        resp = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


# ✅ IMPORTANT:
# No OpenAI() created here at import time.
# We only create an embedder object (safe).
embedder = ProfessionalEmbeddingGenerator(model=settings.embedding_model)
print(f"🔬 Using model: {settings.embedding_model} (embeddings)")
