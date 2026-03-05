# main.py
import os
import re
import uuid
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdf_loader import extract_metadata, read_pdf_chunks
from embeddings import embedder
from vector_store import StoredChunk, StoredDocument, vector_store
from answer_generator import (
    generate_answer,
    generate_comparison_answer,
    INSUFFICIENT_MSG,
)

ENABLE_OCR = os.getenv("ENABLE_OCR", "false").strip().lower() == "true"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where PDFs are stored
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(BASE_DIR, "uploaded_pdfs"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Where vector DB is stored (same env var you use in vector_store.py)
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "data")


# -----------------------------
# Request Schemas
# -----------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryDocRequest(BaseModel):
    question: str
    document_id: str
    top_k: int = 5


class QueryDocsRequest(BaseModel):
    question: str
    doc_ids: List[str]
    top_k_per_doc: int = 3


class CompareRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None  # if None -> compare across ALL docs
    top_k_per_doc: int = 3


class SummarizeDocRequest(BaseModel):
    document_id: str
    max_chars: int = 1200


app = FastAPI(title="NexusAI RAG Assistant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok for local dev; tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "NexusAI RAG Assistant",
        "status": "ok",
        "docs_url": "/docs",
        "system_url": "/system",
    }


_last_ingest_error: Optional[str] = None


# -----------------------------
# Metadata override helpers (slide-text based)
# -----------------------------
def _extract_presenter(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"Presented\s*by\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if not m:
        return ""
    name = m.group(1).strip()
    name = name.split("|")[0].strip()
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name


def _extract_chapter_title_line(text: str) -> str:
    if not text:
        return ""

    m = re.search(r"(Chapter\s*\d+\s*:\s*[^\n\r]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m2 = re.search(r"(Chapter\s*\d+\s*:\s*)\s*[\r\n]+([^\n\r]+)", text, flags=re.IGNORECASE)
    if m2:
        return f"{m2.group(1).strip()} {m2.group(2).strip()}".strip()

    return ""


def _looks_generic_or_wrong_title(title: str) -> bool:
    t = (title or "").strip().lower()
    if not t:
        return True
    bad = [
        "web services",
        "xml processing",
        "powerpoint",
        "untitled",
        "microsoft",
        "presentation",
        "slide",
    ]
    return any(b in t for b in bad)


def _override_meta_from_first_page(meta: Dict[str, str], chunks_raw: List[Dict[str, Any]], filename: str) -> Dict[str, str]:
    if not meta:
        meta = {"title": "", "author": "", "year": ""}

    first_text = ""
    for c in chunks_raw or []:
        md = c.get("metadata") or {}
        if md.get("page") == 1:
            first_text = (c.get("text") or "").strip()
            break
    if not first_text and chunks_raw:
        first_text = (chunks_raw[0].get("text") or "").strip()

    if not first_text:
        meta["title"] = meta.get("title") or filename
        meta["author"] = meta.get("author") or "Unknown Author"
        return meta

    presenter = _extract_presenter(first_text)
    chap_line = _extract_chapter_title_line(first_text)

    if presenter:
        meta["author"] = presenter

    cur_title = meta.get("title") or ""
    if chap_line and (_looks_generic_or_wrong_title(cur_title) or cur_title.strip() == ""):
        meta["title"] = chap_line

    meta["title"] = meta.get("title") or filename
    meta["author"] = meta.get("author") or "Unknown Author"
    meta["year"] = meta.get("year") or ""

    return meta


# -----------------------------
# Utilities
# -----------------------------
def _make_document_id(filename: str) -> str:
    return "doc_" + uuid.uuid4().hex[:10]


def _pdf_page_count(pdf_path: str) -> int:
    try:
        doc = fitz.open(pdf_path)
        n = int(doc.page_count)
        doc.close()
        return n
    except Exception:
        return 0


def _safe_remove_file(path: str) -> bool:
    """
    Best-effort file delete. Returns True if deleted, False if not.
    """
    try:
        if path and os.path.exists(path) and os.path.isfile(path):
            os.remove(path)
            return True
    except Exception:
        pass
    return False


def _ingest_one_pdf(pdf_path: str, force_reingest: bool = False) -> None:
    global _last_ingest_error

    try:
        filename = os.path.basename(pdf_path)

        # ✅ If force_reingest, delete any existing doc(s) with the same filename (vector store only)
        if force_reingest:
            removed_ids = vector_store.remove_document_by_filename(filename)
            if removed_ids:
                print(f"🧹 Removed existing docs for {filename}: {removed_ids}")

        # If not forcing, keep the skip behavior
        if not force_reingest:
            existing = [d for d in vector_store.list_documents() if d.filename == filename]
            if existing:
                return

        doc_id = _make_document_id(filename)
        total_pages = _pdf_page_count(pdf_path)

        chunks_raw = read_pdf_chunks(file_path=pdf_path)

        if not chunks_raw:
            print(f"⚠️ Ingestion: PDF produced 0 chunks: {filename}")
            return

        meta = extract_metadata(pdf_path)
        meta = _override_meta_from_first_page(meta, chunks_raw, filename)

        texts: List[str] = []
        stored_chunks: List[StoredChunk] = []

        for c in chunks_raw:
            text = c.get("text", "") or ""
            md = c.get("metadata") or {}
            page = md.get("page")
            pages = [page] if isinstance(page, int) else []

            texts.append(text)

            raw_chunk_id = c.get("chunk_id", "unknown_chunk")
            unique_chunk_id = f"{doc_id}_{raw_chunk_id}"

            stored_chunks.append(
                StoredChunk(
                    chunk_id=unique_chunk_id,
                    text=text,
                    pages=pages,
                    document_id=doc_id,
                    document_title=meta.get("title") or filename,
                    document_author=meta.get("author") or "Unknown Author",
                    document_year=meta.get("year") or "",
                    source_locations=[{"page": page, "location": md.get("location", "")}],
                    metadata={
                        **md,
                        "page": page,
                        "document_id": doc_id,
                        "filename": filename,
                        "title": meta.get("title") or filename,
                        "author": meta.get("author") or "Unknown Author",
                        "year": meta.get("year") or "",
                        "file_path": pdf_path,
                        "total_pages": total_pages,
                    },
                )
            )

        embeddings = embedder.embed_texts(texts)

        doc = StoredDocument(
            document_id=doc_id,
            title=meta.get("title") or filename,
            author=meta.get("author") or "Unknown Author",
            year=meta.get("year") or "",
            filename=filename,
            file_path=pdf_path,
            pages=total_pages,
        )

        vector_store.add_document(doc, stored_chunks, embeddings)
        _last_ingest_error = None

    except Exception as e:
        _last_ingest_error = f"{os.path.basename(pdf_path)}: {e}"
        print(f"⚠️ Ingestion warning: {_last_ingest_error}")


def ingest_pdfs(force_reingest: bool = False) -> None:
    vector_store.load()

    for name in os.listdir(UPLOAD_DIR):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(UPLOAD_DIR, name)
        _ingest_one_pdf(path, force_reingest=force_reingest)


@app.on_event("startup")
def startup_init():
    print(f"📁 UPLOAD_DIR resolved to: {UPLOAD_DIR}")
    ingest_pdfs(force_reingest=False)


# -----------------------------
# Vector search helpers
# -----------------------------
def _vector_search(question: str, top_k: int) -> List[Dict[str, Any]]:
    q_emb = embedder.embed_text(question)
    results = vector_store.search(q_emb, top_k=top_k)

    out: List[Dict[str, Any]] = []
    for ch, score in results:
        out.append(
            {
                "chunk_id": ch.chunk_id,
                "text": ch.text,
                "score": score,
                "pages": ch.pages,
                "document_title": ch.document_title,
                "document_author": ch.document_author,
                "document_year": ch.document_year,
                "metadata": ch.metadata,
            }
        )
    return out


def _vector_search_in_doc(question: str, document_id: str, top_k: int) -> List[Dict[str, Any]]:
    q_emb = embedder.embed_text(question)
    results = vector_store.search_in_document(q_emb, document_id=document_id, top_k=top_k)

    out: List[Dict[str, Any]] = []
    for ch, score in results:
        out.append(
            {
                "chunk_id": ch.chunk_id,
                "text": ch.text,
                "score": score,
                "pages": ch.pages,
                "document_title": ch.document_title,
                "document_author": ch.document_author,
                "document_year": ch.document_year,
                "metadata": ch.metadata,
            }
        )
    return out


def _vector_search_compare(question: str, doc_ids: List[str], top_k_per_doc: int) -> List[Dict[str, Any]]:
    q_emb = embedder.embed_text(question)
    results = vector_store.search_across_documents(q_emb, doc_ids, top_k_per_doc=top_k_per_doc)

    out: List[Dict[str, Any]] = []
    for ch, score in results:
        out.append(
            {
                "chunk_id": ch.chunk_id,
                "text": ch.text,
                "score": score,
                "pages": ch.pages,
                "document_title": ch.document_title,
                "document_author": ch.document_author,
                "document_year": ch.document_year,
                "metadata": ch.metadata,
            }
        )
    return out


def _chunks_indexed_count() -> int:
    try:
        if getattr(vector_store, "_embeddings", None) is not None:
            return int(vector_store._embeddings.shape[0])  # type: ignore
    except Exception:
        pass
    return 0


def _get_doc_by_id(doc_id: str) -> Optional[StoredDocument]:
    for d in vector_store.list_documents():
        if d.document_id == doc_id:
            return d
    return None


# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/system")
def system_status() -> Dict[str, Any]:
    docs = vector_store.list_documents()

    return {
        "service": "NexusAI RAG Assistant",
        "version": "0.1.0",
        "openai_key_present": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        "chat_model": os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        "base_dir": BASE_DIR,
        "upload_dir": UPLOAD_DIR,
        "docs_ingested": len(docs),
        "chunks_indexed": _chunks_indexed_count(),
        "last_ingest_error": _last_ingest_error,
        "vector_db_dir": VECTOR_DB_DIR,
    }


# ✅ Add query param: /reload?force=true
@app.post("/reload")
def reload_index(force: bool = Query(default=False)) -> Dict[str, Any]:
    ingest_pdfs(force_reingest=force)
    docs = vector_store.list_documents()

    return {
        "ok": True,
        "force_reingest": force,
        "upload_dir": UPLOAD_DIR,
        "docs_ingested": len(docs),
        "chunks_indexed": _chunks_indexed_count(),
        "last_ingest_error": _last_ingest_error,
    }


@app.get("/documents")
def list_documents() -> List[Dict[str, Any]]:
    docs = vector_store.list_documents()
    counts = vector_store.count_chunks_by_doc()

    out: List[Dict[str, Any]] = []
    for d in docs:
        out.append(
            {
                "document_id": d.document_id,
                "filename": d.filename,
                "title": d.title,
                "author": d.author,
                "year": d.year,
                "num_chunks": counts.get(d.document_id, 0),
                "pages": d.pages,
            }
        )
    return out


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), force: bool = Query(default=True)) -> Dict[str, Any]:
    """
    Default force=True: if user uploads a PDF with same filename, replace it (vector store doc with same filename).
    NOTE: This does NOT delete other PDFs with different names.
    """
    filename = file.filename or "uploaded.pdf"
    save_path = os.path.join(UPLOAD_DIR, filename)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    _ingest_one_pdf(save_path, force_reingest=force)

    return {"ok": True, "saved_as": filename, "force_reingest": force}


# ✅ NEW: Delete a single document (vector + pdf file)
@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> Dict[str, Any]:
    doc = _get_doc_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="document_id not found")

    # 1) remove from vector store (also persists)
    removed = vector_store.remove_document(document_id)

    # 2) remove the file from uploaded_pdfs (best-effort)
    pdf_deleted = False
    pdf_path = doc.file_path

    # If file_path is missing or wrong, fallback to UPLOAD_DIR/filename
    if (not pdf_path) or (not os.path.exists(pdf_path)):
        pdf_path = os.path.join(UPLOAD_DIR, doc.filename)

    pdf_deleted = _safe_remove_file(pdf_path)

    return {
        "ok": True,
        "document_id": document_id,
        "removed_from_vector_store": bool(removed),
        "pdf_deleted": bool(pdf_deleted),
        "pdf_path": pdf_path,
        "filename": doc.filename,
    }


@app.post("/query")
def query_docs(req: QueryRequest) -> Dict[str, Any]:
    chunks = _vector_search(req.question, top_k=max(1, int(req.top_k)))
    result = generate_answer(req.question, chunks, chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    return {
        "question": req.question,
        "answer": result.get("answer", INSUFFICIENT_MSG),
        "citations": result.get("citations", []),
        "evidence": result.get("evidence", []),
    }


@app.post("/query_doc")
def query_one_doc(req: QueryDocRequest) -> Dict[str, Any]:
    docs = vector_store.list_documents()
    all_ids = {d.document_id for d in docs}
    if req.document_id not in all_ids:
        return {
            "question": req.question,
            "answer": INSUFFICIENT_MSG,
            "citations": [],
            "evidence": [],
            "used_doc_id": req.document_id,
            "error": "document_id not found",
        }

    chunks = _vector_search_in_doc(req.question, req.document_id, top_k=max(1, int(req.top_k)))
    result = generate_answer(req.question, chunks, chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    return {
        "question": req.question,
        "answer": result.get("answer", INSUFFICIENT_MSG),
        "citations": result.get("citations", []),
        "evidence": result.get("evidence", []),
        "used_doc_id": req.document_id,
    }


@app.post("/query_docs")
def query_multiple_docs(req: QueryDocsRequest) -> Dict[str, Any]:
    docs = vector_store.list_documents()
    all_ids = {d.document_id for d in docs}
    doc_ids = [d for d in req.doc_ids if d in all_ids]

    if not doc_ids:
        return {
            "question": req.question,
            "answer": INSUFFICIENT_MSG,
            "citations": [],
            "evidence": [],
            "used_doc_ids": doc_ids,
        }

    chunks = _vector_search_compare(req.question, doc_ids, top_k_per_doc=max(1, int(req.top_k_per_doc)))
    result = generate_answer(req.question, chunks, chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    return {
        "question": req.question,
        "answer": result.get("answer", INSUFFICIENT_MSG),
        "citations": result.get("citations", []),
        "evidence": result.get("evidence", []),
        "used_doc_ids": doc_ids,
    }


@app.post("/compare")
def compare_docs(req: CompareRequest) -> Dict[str, Any]:
    docs = vector_store.list_documents()
    all_ids = [d.document_id for d in docs]

    doc_ids = req.doc_ids if req.doc_ids else all_ids
    doc_ids = [d for d in doc_ids if d in all_ids]

    if len(doc_ids) < 2:
        return {
            "question": req.question,
            "answer": INSUFFICIENT_MSG,
            "citations": [],
            "evidence": [],
            "used_doc_ids": doc_ids,
        }

    chunks = _vector_search_compare(req.question, doc_ids, top_k_per_doc=max(1, int(req.top_k_per_doc)))
    result = generate_comparison_answer(req.question, chunks, chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"))

    return {
        "question": req.question,
        "answer": result.get("answer", INSUFFICIENT_MSG),
        "citations": result.get("citations", []),
        "evidence": result.get("evidence", []),
        "used_doc_ids": doc_ids,
    }


@app.post("/summarize_doc")
def summarize_one_doc(req: SummarizeDocRequest) -> Dict[str, Any]:
    doc = _get_doc_by_id(req.document_id)
    if not doc:
        return {
            "summary": INSUFFICIENT_MSG,
            "citations": [],
            "evidence": [],
            "used_doc_id": req.document_id,
            "error": "document_id not found",
        }

    summary_prompt = (
        "Summarize this document using ONLY the provided evidence.\n"
        "Format:\n"
        "1) One-line topic\n"
        "2) Key points (5–10 bullets)\n"
        "3) Definitions / steps (if any)\n"
        "4) Important formulas or terms (if present)\n"
        "Keep it concise and cite pages."
    )

    chunks = _vector_search_in_doc(summary_prompt, req.document_id, top_k=20)

    if not chunks:
        anchor_q = f"{doc.title} {doc.filename} main topics key concepts"
        chunks = _vector_search_in_doc(anchor_q, req.document_id, top_k=25)

    if not chunks:
        anchor_q2 = "Presented by Chapter Transformer attention encoder decoder"
        chunks = _vector_search_in_doc(anchor_q2, req.document_id, top_k=25)

    result = generate_answer(summary_prompt, chunks, chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    summary = result.get("answer", INSUFFICIENT_MSG) or INSUFFICIENT_MSG

    max_chars = max(250, int(req.max_chars or 1200))
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rstrip() + "..."

    return {
        "document_id": req.document_id,
        "summary": summary,
        "citations": result.get("citations", []),
        "evidence": result.get("evidence", []),
    }


# ✅ NEW: Clear ALL docs + PDFs (DEV reset button)
@app.post("/clear")
def clear_all(
    confirm: bool = Query(default=False, description="Must be true to clear everything."),
    delete_vectors: bool = Query(default=True, description="Also delete vector DB files on disk."),
) -> Dict[str, Any]:
    """
    Clears all documents + PDFs. Useful for online/dev when storage gets messy.

    confirm=true is required so you don't wipe things accidentally.
    """
    if not confirm:
        raise HTTPException(status_code=400, detail="Pass confirm=true to clear all data.")

    # 1) Remove docs from vector store
    docs = vector_store.list_documents()
    removed_doc_ids: List[str] = []
    for d in docs:
        if vector_store.remove_document(d.document_id):
            removed_doc_ids.append(d.document_id)

    # 2) Delete all PDFs in upload dir
    deleted_pdfs: List[str] = []
    failed_pdfs: List[str] = []
    try:
        for name in os.listdir(UPLOAD_DIR):
            if not name.lower().endswith(".pdf"):
                continue
            p = os.path.join(UPLOAD_DIR, name)
            if _safe_remove_file(p):
                deleted_pdfs.append(name)
            else:
                failed_pdfs.append(name)
    except Exception as e:
        failed_pdfs.append(f"UPLOAD_DIR_ERROR: {e}")

    # 3) Optionally delete vector DB files on disk (hard reset)
    deleted_vector_files: List[str] = []
    if delete_vectors:
        candidates = [
            os.path.join(VECTOR_DB_DIR, "chunks.json"),
            os.path.join(VECTOR_DB_DIR, "documents.json"),
            os.path.join(VECTOR_DB_DIR, "embeddings.npy"),
        ]
        for fp in candidates:
            if _safe_remove_file(fp):
                deleted_vector_files.append(fp)

        # Reload empty store after deleting files
        vector_store.load()

    return {
        "ok": True,
        "removed_doc_ids": removed_doc_ids,
        "deleted_pdfs": deleted_pdfs,
        "failed_pdfs": failed_pdfs,
        "deleted_vector_files": deleted_vector_files,
    }