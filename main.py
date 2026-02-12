# main.py
import os
import uuid
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdf_loader import extract_metadata, read_pdf_chunks
from embeddings import embedder
from vector_store import StoredChunk, StoredDocument, vector_store
from answer_generator import generate_answer, generate_comparison_answer, INSUFFICIENT_MSG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(BASE_DIR, "uploaded_pdfs"))
os.makedirs(UPLOAD_DIR, exist_ok=True)


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


app = FastAPI(title="NexusAI RAG Assistant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        "system_url": "/system"
    }

_last_ingest_error: Optional[str] = None


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


def _ingest_one_pdf(pdf_path: str) -> None:
    global _last_ingest_error

    try:
        meta = extract_metadata(pdf_path)
        filename = os.path.basename(pdf_path)

        existing = [d for d in vector_store.list_documents() if d.filename == filename]
        if existing:
            return

        doc_id = _make_document_id(filename)
        total_pages = _pdf_page_count(pdf_path)

        chunks_raw = read_pdf_chunks(file_path=pdf_path)

        if not chunks_raw:
            print(f"⚠️ Ingestion: PDF produced 0 chunks: {filename}")
            return

        texts: List[str] = []
        stored_chunks: List[StoredChunk] = []

        for c in chunks_raw:
            text = c.get("text", "") or ""
            md = c.get("metadata") or {}
            page = md.get("page")
            pages = [page] if isinstance(page, int) else []

            texts.append(text)

            raw_chunk_id = c.get("chunk_id", "unknown_chunk")
            unique_chunk_id = f"{doc_id}_{raw_chunk_id}"  # ✅ prevent collisions

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


def ingest_pdfs() -> None:
    vector_store.load()

    for name in os.listdir(UPLOAD_DIR):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(UPLOAD_DIR, name)
        _ingest_one_pdf(path)


@app.on_event("startup")
def startup_init():
    print(f"📁 UPLOAD_DIR resolved to: {UPLOAD_DIR}")
    ingest_pdfs()


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




@app.get("/system")
def system_status() -> Dict[str, Any]:
    docs = vector_store.list_documents()
    chunks_count = 0
    try:
        if getattr(vector_store, "_embeddings", None) is not None:
            chunks_count = int(vector_store._embeddings.shape[0])  # type: ignore
    except Exception:
        chunks_count = 0

    return {
        "service": "NexusAI RAG Assistant",
        "version": "0.1.0",
        "openai_key_present": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        "chat_model": os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        "base_dir": BASE_DIR,
        "upload_dir": UPLOAD_DIR,
        "docs_ingested": len(docs),
        "chunks_indexed": chunks_count,
        "last_ingest_error": _last_ingest_error,
    }


@app.post("/reload")
def reload_index() -> Dict[str, Any]:
    ingest_pdfs()
    docs = vector_store.list_documents()

    chunks_count = 0
    try:
        if getattr(vector_store, "_embeddings", None) is not None:
            chunks_count = int(vector_store._embeddings.shape[0])  # type: ignore
    except Exception:
        chunks_count = 0

    return {
        "ok": True,
        "upload_dir": UPLOAD_DIR,
        "docs_ingested": len(docs),
        "chunks_indexed": chunks_count,
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
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    filename = file.filename or "uploaded.pdf"
    save_path = os.path.join(UPLOAD_DIR, filename)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    _ingest_one_pdf(save_path)

    return {"ok": True, "saved_as": filename}


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
    # Validate doc exists
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
