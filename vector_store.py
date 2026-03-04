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
    if not text:
        return ""
    t = text.lower()
    t = t.replace("\x00", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _chunk_fingerprint(text: str, max_chars: int = 220) -> str:
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
    Also supports delete-by-doc and delete-by-filename safely (keeps embeddings aligned).
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

        # Repair: if docs missing but chunks exist
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
    # Delete helpers (keeps embeddings aligned)
    # -----------------------------
    def _delete_chunk_indexes(self, indexes: List[int]) -> None:
        if not indexes:
            return

        idx_set = set(int(i) for i in indexes)
        if not idx_set:
            return

        # Remove chunks
        new_chunks: List[StoredChunk] = []
        for i, ch in enumerate(self._chunks):
            if i in idx_set:
                continue
            new_chunks.append(ch)
        self._chunks = new_chunks

        # Remove embedding rows
        if self._embeddings is not None and self._embeddings.shape[0] > 0:
            n = self._embeddings.shape[0]
            keep_mask = np.ones(n, dtype=bool)
            for i in idx_set:
                if 0 <= i < n:
                    keep_mask[i] = False
            self._embeddings = self._embeddings[keep_mask, :]

    def remove_document(self, document_id: str) -> bool:
        document_id = str(document_id).strip()
        if not document_id:
            return False

        idxs = [i for i, ch in enumerate(self._chunks) if ch.document_id == document_id]
        removed_any = bool(idxs)

        self._delete_chunk_indexes(idxs)

        if document_id in self._documents:
            del self._documents[document_id]
            removed_any = True

        if removed_any:
            self.persist()
        return removed_any

    def remove_document_by_filename(self, filename: str) -> List[str]:
        fn = (filename or "").strip()
        if not fn:
            return []

        to_remove = [d.document_id for d in self._documents.values() if d.filename == fn]
        removed: List[str] = []
        for doc_id in to_remove:
            if self.remove_document(doc_id):
                removed.append(doc_id)
        return removed

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
        print(f"✅ Added document {doc.document_id} with {len(chunks)} chunks (total: {len(self._chunks)})")

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

        return [(self._chunks[int(i)], float(sims[int(i)])) for i in idx]

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
    # Cross-doc search
    # -----------------------------
    def dedupe_results(self, results: List[Tuple[StoredChunk, float]]) -> List[Tuple[StoredChunk, float]]:
        if not results:
            return []

        buckets: Dict[str, Tuple[StoredChunk, float]] = {}
        order: List[str] = []

        for ch, score in results:
            fp = _chunk_fingerprint(ch.text)
            if not fp:
                fp = f"__empty__::{ch.document_id}::{ch.chunk_id}"
            if fp not in buckets:
                order.append(fp)
                buckets[fp] = (ch, score)
            else:
                # keep highest score
                if score > buckets[fp][1]:
                    buckets[fp] = (ch, score)

        return [buckets[k] for k in order]

    def search_across_documents(
        self,
        query_embedding: List[float],
        document_ids: List[str],
        top_k_per_doc: int = 3,
        dedupe: bool = True,
    ) -> List[Tuple[StoredChunk, float]]:
        doc_ids = _sanitize_doc_ids(document_ids)
        results: List[Tuple[StoredChunk, float]] = []

        for doc_id in doc_ids:
            results.extend(self.search_in_document(query_embedding, doc_id, top_k=top_k_per_doc))

        results.sort(key=lambda x: x[1], reverse=True)

        if dedupe:
            results = self.dedupe_results(results)
            results.sort(key=lambda x: x[1], reverse=True)

        return results


# singleton
vector_store = VectorStore(persist_dir=os.getenv("VECTOR_DB_DIR", "data"))