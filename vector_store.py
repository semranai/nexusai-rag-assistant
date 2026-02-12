# vector_store.py
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def _cosine_sim_matrix(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """
    query_vec: (d,)
    mat: (n, d)
    returns similarity (n,)
    """
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return m @ q


def _sanitize_doc_ids(doc_ids: List[str]) -> List[str]:
    out: List[str] = []
    for d in doc_ids or []:
        if d is None:
            continue
        s = str(d).strip()
        if not s:
            continue
        out.append(s)
    # dedupe while preserving order
    seen = set()
    final: List[str] = []
    for d in out:
        if d in seen:
            continue
        seen.add(d)
        final.append(d)
    return final


def _normalize_text_for_dedupe(text: str) -> str:
    """
    Normalize chunk text to dedupe across duplicate PDFs or repeated slides.
    Keeps it simple & fast:
    - lowercase
    - collapse whitespace
    - strip
    """
    if not text:
        return ""
    t = text.lower()
    t = t.replace("\x00", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _chunk_fingerprint(text: str, max_chars: int = 220) -> str:
    """
    Hash-like fingerprint (string) for dedupe without importing hashlib.
    Uses normalized prefix which is usually enough for slide chunks.
    """
    norm = _normalize_text_for_dedupe(text)
    if not norm:
        return ""
    return norm[:max_chars]


@dataclass
class StoredChunk:
    chunk_id: str
    text: str
    pages: List[int]
    document_id: str
    document_title: str
    document_author: str
    document_year: str
    source_locations: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class StoredDocument:
    document_id: str
    title: str
    author: str
    year: str
    filename: str
    file_path: str
    pages: int


class VectorStore:
    """
    Simple in-memory vector store with disk persistence.
    Phase 3 adds document-aware retrieval with:
      - balanced top_k_per_doc
      - optional prefer_first_pages boost (pages 1-2)
      - cross-doc dedupe
      - doc_id sanitization
    """

    def __init__(self, persist_dir: str = "data"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self._chunks: List[StoredChunk] = []
        self._embeddings: Optional[np.ndarray] = None  # shape (n, d)
        self._documents: Dict[str, StoredDocument] = {}

        self._chunks_path = os.path.join(self.persist_dir, "chunks.json")
        self._emb_path = os.path.join(self.persist_dir, "embeddings.npy")
        self._docs_path = os.path.join(self.persist_dir, "documents.json")

    # -----------------------------
    # Persistence
    # -----------------------------
    def load(self) -> None:
        if os.path.exists(self._chunks_path):
            with open(self._chunks_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._chunks = [StoredChunk(**x) for x in raw]
        else:
            self._chunks = []

        if os.path.exists(self._emb_path):
            self._embeddings = np.load(self._emb_path)
        else:
            self._embeddings = None

        if os.path.exists(self._docs_path):
            with open(self._docs_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._documents = {k: StoredDocument(**v) for k, v in raw.items()}
        else:
            self._documents = {}

        # ✅ Repair mode: if documents.json is missing/broken but chunks exist
        if (not self._documents) and self._chunks:
            self._rebuild_documents_from_chunks()
            self.persist()

    def persist(self) -> None:
        with open(self._chunks_path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self._chunks], f, ensure_ascii=False, indent=2)

        if self._embeddings is not None:
            np.save(self._emb_path, self._embeddings)

        with open(self._docs_path, "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in self._documents.items()}, f, ensure_ascii=False, indent=2)

    def _rebuild_documents_from_chunks(self) -> None:
        docs: Dict[str, Dict[str, Any]] = {}

        for ch in self._chunks:
            doc_id = ch.document_id
            md = ch.metadata or {}
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "title": md.get("title") or ch.document_title or "Unknown Title",
                    "author": md.get("author") or ch.document_author or "Unknown Author",
                    "year": md.get("year") or ch.document_year or "",
                    "filename": md.get("filename") or "unknown.pdf",
                    "file_path": md.get("file_path") or "",
                    "pages": int(md.get("total_pages") or 0),
                }

        self._documents = {k: StoredDocument(**v) for k, v in docs.items()}

    # -----------------------------
    # Add / List
    # -----------------------------
    def add_document(
        self,
        doc: StoredDocument,
        chunks: List[StoredChunk],
        embeddings: List[List[float]],
    ) -> None:
        self._documents[doc.document_id] = doc
        self._chunks.extend(chunks)

        emb_np = np.array(embeddings, dtype=np.float32)
        if self._embeddings is None or self._embeddings.size == 0:
            self._embeddings = emb_np
        else:
            self._embeddings = np.vstack([self._embeddings, emb_np])

        self.persist()
        print(f"✅ Added document {doc.document_id} with {len(chunks)} chunks (total chunks: {len(self._chunks)})")

    def list_documents(self) -> List[StoredDocument]:
        return list(self._documents.values())

    def get_document(self, document_id: str) -> Optional[StoredDocument]:
        return self._documents.get(document_id)

    def count_chunks_by_doc(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for ch in self._chunks:
            counts[ch.document_id] = counts.get(ch.document_id, 0) + 1
        return counts

    # -----------------------------
    # Search (global)
    # -----------------------------
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[StoredChunk, float]]:
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        sims = _cosine_sim_matrix(q, self._embeddings)
        idx = np.argsort(-sims)[:top_k]

        out: List[Tuple[StoredChunk, float]] = []
        for i in idx:
            out.append((self._chunks[int(i)], float(sims[int(i)])))
        return out

    # -----------------------------
    # Phase 3: Search within ONE document
    # -----------------------------
    def search_in_document(
        self, query_embedding: List[float], document_id: str, top_k: int = 5
    ) -> List[Tuple[StoredChunk, float]]:
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        document_id = str(document_id).strip()
        if not document_id:
            return []

        doc_indexes = [i for i, ch in enumerate(self._chunks) if ch.document_id == document_id]
        if not doc_indexes:
            return []

        mat = self._embeddings[doc_indexes, :]
        q = np.array(query_embedding, dtype=np.float32)
        sims = _cosine_sim_matrix(q, mat)

        local_idx = np.argsort(-sims)[:top_k]
        out: List[Tuple[StoredChunk, float]] = []
        for j in local_idx:
            global_i = doc_indexes[int(j)]
            out.append((self._chunks[global_i], float(sims[int(j)])))
        return out

    # -----------------------------
    # Phase 3 Helper: get chunks from first pages (1-2)
    # -----------------------------
    def get_first_page_chunks(
        self,
        document_id: str,
        pages: Tuple[int, ...] = (1, 2),
        max_chunks: int = 4,
    ) -> List[Tuple[StoredChunk, float]]:
        """
        Returns chunks from the first pages of a document.
        Score is set to 0.0 (these are "boost" evidence, not similarity-ranked).
        """
        document_id = str(document_id).strip()
        if not document_id:
            return []

        hits: List[Tuple[StoredChunk, float]] = []
        for ch in self._chunks:
            if ch.document_id != document_id:
                continue
            if not ch.pages:
                continue
            if any(p in pages for p in ch.pages):
                hits.append((ch, 0.0))

        # Keep stable + limited
        return hits[:max_chunks]

    # -----------------------------
    # Phase 3 Helper: dedupe results by text fingerprint
    # -----------------------------
    def dedupe_results(
        self,
        results: List[Tuple[StoredChunk, float]],
        keep: str = "highest_score",
    ) -> List[Tuple[StoredChunk, float]]:
        """
        Dedupe across documents when the same slide/chunk content appears multiple times.
        keep:
          - "highest_score": keep the highest similarity item
          - "first": keep first occurrence
        """
        if not results:
            return []

        buckets: Dict[str, List[Tuple[StoredChunk, float]]] = {}
        order: List[str] = []

        for ch, score in results:
            fp = _chunk_fingerprint(ch.text)
            if not fp:
                # if empty, treat as unique
                fp = f"__empty__::{ch.document_id}::{ch.chunk_id}"
            if fp not in buckets:
                buckets[fp] = []
                order.append(fp)
            buckets[fp].append((ch, score))

        deduped: List[Tuple[StoredChunk, float]] = []
        for fp in order:
            items = buckets[fp]
            if not items:
                continue
            if keep == "first":
                deduped.append(items[0])
            else:
                # highest_score
                best = max(items, key=lambda x: x[1])
                deduped.append(best)

        return deduped

    # -----------------------------
    # Phase 3: Search across multiple docs (balanced + boosted + deduped)
    # -----------------------------
    def search_across_documents(
        self,
        query_embedding: List[float],
        document_ids: List[str],
        top_k_per_doc: int = 3,
        prefer_first_pages: bool = False,
        first_pages: Tuple[int, ...] = (1, 2),
        first_page_extra_per_doc: int = 3,
        dedupe: bool = True,
        return_used_doc_ids: bool = False,
    ) -> Union[List[Tuple[StoredChunk, float]], Tuple[List[Tuple[StoredChunk, float]], List[str]]]:
        """
        Balanced retrieval across docs:
          - pulls top_k_per_doc from each document
          - optionally boosts identity by adding some chunks from pages 1-2 per doc
          - optional dedupe to avoid repeated evidence across docs
          - optional return_used_doc_ids for debugging/reporting
        """
        doc_ids = _sanitize_doc_ids(document_ids)

        results: List[Tuple[StoredChunk, float]] = []
        used_doc_ids: List[str] = []

        for doc_id in doc_ids:
            doc_hits = self.search_in_document(query_embedding, doc_id, top_k=top_k_per_doc)
            if doc_hits:
                used_doc_ids.append(doc_id)
                results.extend(doc_hits)

            if prefer_first_pages:
                first_hits = self.get_first_page_chunks(
                    document_id=doc_id,
                    pages=first_pages,
                    max_chunks=first_page_extra_per_doc,
                )
                if first_hits:
                    # even if similarity had nothing, we might still include first pages
                    if doc_id not in used_doc_ids:
                        used_doc_ids.append(doc_id)
                    results.extend(first_hits)

        # Sort by similarity score first (boost chunks score=0.0 will naturally float lower)
        results.sort(key=lambda x: x[1], reverse=True)

        # Dedupe repeated chunks across docs (same text)
        if dedupe:
            results = self.dedupe_results(results, keep="highest_score")
            results.sort(key=lambda x: x[1], reverse=True)

        if return_used_doc_ids:
            return results, used_doc_ids
        return results


# singleton
vector_store = VectorStore(persist_dir=os.getenv("VECTOR_DB_DIR", "data"))
