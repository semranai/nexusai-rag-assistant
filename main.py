# main.py
import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import fitz
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

from answer_generator import (
    INSUFFICIENT_MSG,
    NOT_MENTIONED_MSG,
    SCANNED_OR_LOW_TEXT_MSG,
    generate_answer,
)
from embeddings import embedder
from pdf_loader import extract_metadata, read_pdf_chunks
from vector_store import StoredChunk, StoredDocument, vector_store

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(BASE_DIR, "uploaded_pdfs"))
os.makedirs(UPLOAD_DIR, exist_ok=True)


# -----------------------------
# Request Schemas
# -----------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 8


class QueryDocRequest(BaseModel):
    question: str
    document_id: str
    top_k: int = 8


class QueryDocsRequest(BaseModel):
    question: str
    doc_ids: List[str]
    top_k_per_doc: int = 8


class CompareRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None
    top_k_per_doc: int = 8


class SummarizeDocRequest(BaseModel):
    document_id: str
    max_chars: int = 1200


app = FastAPI(title="NexusAI RAG Assistant", version="CHECK136")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_last_ingest_error: Optional[str] = None


@app.get("/")
def root():
    return {
        "service": "NexusAI RAG Assistant",
        "status": "ok",
        "docs_url": "/docs",
        "system_url": "/system",
    }


# -----------------------------
# Metadata override helpers
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

    m2 = re.search(
        r"(Chapter\s*\d+\s*:\s*)\s*[\r\n]+([^\n\r]+)",
        text,
        flags=re.IGNORECASE,
    )
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


def _override_meta_from_first_page(
    meta: Dict[str, str], chunks_raw: List[Dict[str, Any]], filename: str
) -> Dict[str, str]:
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
    if chap_line and (_looks_generic_or_wrong_title(cur_title) or not cur_title.strip()):
        meta["title"] = chap_line

    meta["title"] = meta.get("title") or filename
    meta["author"] = meta.get("author") or "Unknown Author"
    meta["year"] = meta.get("year") or ""
    return meta


# -----------------------------
# General helpers
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


def _make_metadata_chunk_text(meta: Dict[str, str], filename: str) -> str:
    title = (meta.get("title") or "").strip() or filename
    author = (meta.get("author") or "").strip() or "Unknown Author"
    year = (meta.get("year") or "").strip() or "n.d."
    return (
        "DOCUMENT METADATA\n"
        f"Title: {title}\n"
        f"Author/Presenter: {author}\n"
        f"Year: {year}\n"
        f"Filename: {filename}\n"
    )


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


def _best_effort_persist_vector_store() -> None:
    for meth in ("save", "persist", "dump", "flush"):
        fn = getattr(vector_store, meth, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def _normalize_for_match(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace(".pdf", " ")
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _filename_stem(filename: str) -> str:
    return os.path.splitext(filename or "")[0]


def _clean_text_for_extract(text: str) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    t = re.sub(r"^[^A-Za-z0-9]+", "", t)
    return t


def _split_sentences(text: str) -> List[str]:
    t = _clean_text_for_extract(text)
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.strip() for p in parts if p.strip()]


def _read_pdf_text(pdf_path: str, max_pages: int = 6) -> str:
    if not pdf_path or not os.path.exists(pdf_path):
        return ""
    try:
        doc = fitz.open(pdf_path)
        limit = min(doc.page_count, max_pages)
        parts: List[str] = []
        for i in range(limit):
            page = doc.load_page(i)
            parts.append(page.get_text("text") or "")
        doc.close()
        return "\n".join(parts)
    except Exception:
        return ""


def _trim_text(text: str, max_chars: int = 7000) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:max_chars]


def _get_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _chat_completion(system_msg: str, user_msg: str, temperature: float = 0.1) -> str:
    client = _get_openai_client()
    if client is None:
        return ""

    try:
        resp = client.chat.completions.create(
            model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"OpenAI chat error: {e}")
        return ""


def _doc_citation(doc: StoredDocument, page: int = 1) -> Dict[str, Any]:
    author = doc.author or "Unknown Author"
    year = doc.year or "n.d."
    title = doc.title or doc.filename
    return {
        "author": author,
        "year": year,
        "title": title,
        "pages": [page],
        "citation": f"({author.split(',')[0]}, {year}, p. {page})",
        "page": page,
    }


def _dedupe_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for c in citations:
        key = (
            str(c.get("citation", "")),
            str(c.get("title", "")),
            tuple(c.get("pages") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _dedupe_results_keep_best(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_chunk: Dict[str, Dict[str, Any]] = {}
    for item in results:
        chunk_id = item.get("chunk_id") or ""
        if not chunk_id:
            continue
        existing = best_by_chunk.get(chunk_id)
        if existing is None or float(item.get("score", 0.0)) > float(existing.get("score", 0.0)):
            best_by_chunk[chunk_id] = item

    deduped = list(best_by_chunk.values())
    deduped.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return deduped


# -----------------------------
# Ingestion
# -----------------------------
def _ingest_one_pdf(pdf_path: str, force_reingest: bool = False) -> None:
    global _last_ingest_error

    try:
        filename = os.path.basename(pdf_path)

        if force_reingest:
            removed_ids = vector_store.remove_document_by_filename(filename)
            if removed_ids:
                print(f"Removed existing docs for {filename}: {removed_ids}")

        if not force_reingest:
            existing = [d for d in vector_store.list_documents() if d.filename == filename]
            if existing:
                return

        doc_id = _make_document_id(filename)
        total_pages = _pdf_page_count(pdf_path)

        chunks_raw = read_pdf_chunks(file_path=pdf_path)
        if not chunks_raw:
            print(f"Ingestion: PDF produced 0 chunks: {filename}")
            return

        meta = extract_metadata(pdf_path)
        meta = _override_meta_from_first_page(meta, chunks_raw, filename)

        texts: List[str] = []
        stored_chunks: List[StoredChunk] = []

        meta_text = _make_metadata_chunk_text(meta, filename)
        texts.append(meta_text)
        stored_chunks.append(
            StoredChunk(
                chunk_id=f"{doc_id}_meta",
                text=meta_text,
                pages=[1],
                document_id=doc_id,
                document_title=meta.get("title") or filename,
                document_author=meta.get("author") or "Unknown Author",
                document_year=meta.get("year") or "",
                source_locations=[{"page": 1, "location": "Metadata"}],
                metadata={
                    "page": 1,
                    "document_id": doc_id,
                    "filename": filename,
                    "title": meta.get("title") or filename,
                    "author": meta.get("author") or "Unknown Author",
                    "year": meta.get("year") or "",
                    "file_path": pdf_path,
                    "total_pages": total_pages,
                    "chunk_type": "document_metadata",
                },
            )
        )

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
        print(f"Ingestion warning: {_last_ingest_error}")


def ingest_pdfs(force_reingest: bool = False) -> None:
    vector_store.load()

    for name in os.listdir(UPLOAD_DIR):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(UPLOAD_DIR, name)
        _ingest_one_pdf(path, force_reingest=force_reingest)


@app.on_event("startup")
def startup_init():
    print(f"UPLOAD_DIR resolved to: {UPLOAD_DIR}")
    ingest_pdfs(force_reingest=False)


# -----------------------------
# Brief-first, topic-agnostic reasoning
# -----------------------------
def _extract_keywords_naive(text: str, top_n: int = 6) -> List[str]:
    stop = {
        "the", "and", "for", "with", "from", "into", "about", "this", "that",
        "these", "those", "document", "chapter", "paper", "article", "study",
        "discusses", "main", "ideas", "presented", "using", "used", "their",
        "there", "which", "what", "when", "where", "have", "has", "been",
        "are", "was", "were", "will", "would", "could", "should", "pdf",
        "author", "presented", "page", "title", "filename", "year",
    }
    toks = re.findall(r"[a-z][a-z\-]{2,}", (text or "").lower())
    counts: Dict[str, int] = {}
    for tok in toks:
        if tok in stop:
            continue
        counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in ranked[:top_n]]


def _build_general_doc_brief(doc: StoredDocument) -> Dict[str, Any]:
    title = doc.title or doc.filename
    author = doc.author or "Unknown Author"
    year = doc.year or "n.d."
    citation = f"({author.split(',')[0]}, {year}, p. 1)"

    text = _trim_text(_read_pdf_text(doc.file_path, max_pages=6), max_chars=7000)

    system_msg = (
        "You are creating a compact, topic-agnostic document brief from extracted PDF text. "
        "Return strict JSON only with keys: summary, keywords. "
        "Rules: summary must be one sentence, 18 to 40 words, general and grounded in the text; "
        "keywords must be 3 to 6 short topical keywords; "
        "do not assume a domain; do not copy large chunks verbatim; do not mention missing evidence."
    )
    user_msg = (
        f"Title: {title}\n"
        f"Author: {author}\n"
        f"Year: {year}\n\n"
        f"Extracted text:\n{text}\n\n"
        'Return JSON like {"summary":"...","keywords":["...","..."]}'
    )

    raw = _chat_completion(system_msg, user_msg, temperature=0.0)

    summary = ""
    keywords: List[str] = []

    if raw:
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                payload = json.loads(raw[start : end + 1])
                summary = str(payload.get("summary", "")).strip()
                kws = payload.get("keywords", [])
                if isinstance(kws, list):
                    keywords = [str(x).strip() for x in kws if str(x).strip()]
        except Exception:
            pass

    if not summary:
        title_clean = re.sub(r"\s+", " ", title).strip()
        summary = f"This document presents the main ideas, topics, or arguments covered in {title_clean}."

    if len(summary) > 280:
        summary = summary[:277].rstrip() + "..."

    if not keywords:
        keywords = _extract_keywords_naive(f"{title} {text}", top_n=5)

    return {
        "document_id": doc.document_id,
        "filename": doc.filename,
        "title": title,
        "author": author,
        "year": year,
        "pages": [1],
        "summary": summary,
        "keywords": keywords[:6],
        "citation": citation,
    }


def _build_briefs_for_docs(docs: List[StoredDocument]) -> List[Dict[str, Any]]:
    return [_build_general_doc_brief(doc) for doc in docs]


def _direct_doc_line(doc: StoredDocument) -> Dict[str, Any]:
    brief = _build_general_doc_brief(doc)
    cite = {
        "author": brief["author"],
        "year": brief["year"],
        "title": brief["title"],
        "pages": [1],
        "citation": brief["citation"],
        "page": 1,
    }
    return {
        "line": f"**{brief['title']}**: {brief['summary']} {brief['citation']}",
        "citations": [cite],
        "evidence": [],
        "status": "answered",
    }


def _top_keywords_from_briefs(briefs: List[Dict[str, Any]], top_n: int = 8) -> List[str]:
    counter: Dict[str, int] = {}
    for b in briefs:
        for kw in b.get("keywords", []) or []:
            kw_clean = str(kw).strip().lower()
            if not kw_clean:
                continue
            counter[kw_clean] = counter.get(kw_clean, 0) + 1
    ranked = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in ranked[:top_n]]


def _general_compare_from_briefs(question: str, docs: List[StoredDocument]) -> Dict[str, Any]:
    briefs = _build_briefs_for_docs(docs)
    if not briefs:
        return {
            "answer": "No documents available to compare.",
            "citations": [],
            "evidence": [],
            "status": "not_mentioned",
        }

    citations: List[Dict[str, Any]] = []
    evidence_lines: List[str] = []

    for idx, b in enumerate(briefs, start=1):
        citations.append(
            {
                "author": b["author"],
                "year": b["year"],
                "title": b["title"],
                "pages": [1],
                "citation": b["citation"],
                "document_id": b["document_id"],
                "filename": b["filename"],
            }
        )
        evidence_lines.append(
            f"Document {idx}\n"
            f"Title: {b['title']}\n"
            f"Summary: {b['summary']}\n"
            f"Keywords: {', '.join(b.get('keywords', []))}\n"
            f"Citation: {b['citation']}\n"
        )

    system_msg = (
        "You are a grounded multi-document analyst. "
        "Use ONLY the supplied document briefs. "
        "Answer the user's comparison/themes/conclusions question in a general, topic-agnostic way. "
        "Do not assume any specific domain. "
        "Include inline citations exactly as supplied."
    )
    user_msg = (
        f"Question: {question}\n\n"
        f"Document briefs:\n{chr(10).join(evidence_lines)}\n\n"
        "Write a grounded response with sections when useful."
    )

    answer = _chat_completion(system_msg, user_msg, temperature=0.1)

    if not answer:
        keywords = _top_keywords_from_briefs(briefs, top_n=6)
        lines: List[str] = []
        for idx, b in enumerate(briefs, start=1):
            lines.append(f"**Document {idx}: {b['title']}**")
            lines.append(f"{b['summary']} {b['citation']}")
            lines.append("")
        lines.append("**Overall connection**")
        if keywords:
            lines.append("These documents are connected through themes such as " + ", ".join(keywords) + ".")
        else:
            lines.append("These documents are connected through broader overlapping themes, while each one emphasizes a different angle.")
        answer = "\n".join(lines).strip()

    return {
        "answer": answer,
        "citations": _dedupe_citations(citations),
        "evidence": briefs,
        "status": "answered",
    }


# -----------------------------
# Vector search helpers
# -----------------------------
def _chunk_to_result_dict(ch: StoredChunk, score: float) -> Dict[str, Any]:
    return {
        "chunk_id": ch.chunk_id,
        "text": ch.text,
        "score": score,
        "pages": ch.pages,
        "document_title": ch.document_title,
        "document_author": ch.document_author,
        "document_year": ch.document_year,
        "metadata": ch.metadata,
    }


def _vector_search(question: str, top_k: int) -> List[Dict[str, Any]]:
    q_emb = embedder.embed_text(question)
    results = vector_store.search(q_emb, top_k=top_k)
    return [_chunk_to_result_dict(ch, score) for ch, score in results]


def _vector_search_in_doc(question: str, document_id: str, top_k: int) -> List[Dict[str, Any]]:
    q_emb = embedder.embed_text(question)
    results = vector_store.search_in_document(q_emb, document_id=document_id, top_k=top_k)
    return [_chunk_to_result_dict(ch, score) for ch, score in results]


def _vector_search_compare(question: str, doc_ids: List[str], top_k_per_doc: int) -> List[Dict[str, Any]]:
    q_emb = embedder.embed_text(question)
    results = vector_store.search_across_documents(q_emb, doc_ids, top_k_per_doc=top_k_per_doc)
    return [_chunk_to_result_dict(ch, score) for ch, score in results]


def _decompose_question(question: str) -> List[str]:
    q = (question or "").strip()
    if not q:
        return []

    work = q
    work = re.sub(
        r"(?i)\b(explain|define|what is|what are|tell me about|give me|in one line|briefly)\b",
        "",
        work,
    )
    work = work.replace(";", ",")
    work = re.sub(r"(?i)\band\b", ",", work)
    parts = [p.strip(" .:?-\n\t") for p in work.split(",") if p.strip(" .:?-\n\t")]

    cleaned: List[str] = []
    seen = set()
    for p in parts:
        p = re.sub(r"\s{2,}", " ", p).strip()
        if len(p) < 3:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(p)

    return cleaned if len(cleaned) >= 2 else []


def _multi_query_vector_search(question: str, top_k: int) -> List[Dict[str, Any]]:
    subqueries = _decompose_question(question)
    if not subqueries:
        return _vector_search(question, top_k)

    combined: List[Dict[str, Any]] = []
    per_subquery_k = max(4, top_k)

    for sq in subqueries:
        combined.extend(_vector_search(sq, per_subquery_k))

    combined.extend(_vector_search(question, max(4, top_k)))

    deduped = _dedupe_results_keep_best(combined)
    return deduped[: max(24, len(subqueries) * 6)]


def _multi_query_vector_search_compare(question: str, doc_ids: List[str], top_k_per_doc: int) -> List[Dict[str, Any]]:
    subqueries = _decompose_question(question)
    if not subqueries:
        return _vector_search_compare(question, doc_ids, top_k_per_doc)

    combined: List[Dict[str, Any]] = []

    for sq in subqueries:
        q_emb = embedder.embed_text(sq)
        results = vector_store.search_across_documents(
            q_emb,
            doc_ids,
            top_k_per_doc=max(10, top_k_per_doc),
        )
        for ch, score in results:
            combined.append(_chunk_to_result_dict(ch, score))

    combined.extend(_vector_search_compare(question, doc_ids, max(10, top_k_per_doc)))

    deduped = _dedupe_results_keep_best(combined)
    return deduped[: max(60, len(subqueries) * len(doc_ids) * 5)]


def _answer_one_concept_across_docs(term: str, doc_ids: List[str], top_k_per_doc: int) -> Dict[str, Any]:
    chunks = _multi_query_vector_search_compare(term, doc_ids, top_k_per_doc=max(10, top_k_per_doc))
    prompt = f"Define {term} in exactly one sentence using only the provided evidence."
    result = generate_answer(prompt, chunks, chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"))

    answer = result.get("answer", INSUFFICIENT_MSG)
    citations = result.get("citations", [])
    evidence = result.get("evidence", [])

    if answer in {INSUFFICIENT_MSG, NOT_MENTIONED_MSG, SCANNED_OR_LOW_TEXT_MSG}:
        return {
            "line": f"**{term}**: {answer}",
            "citations": [],
            "evidence": evidence,
        }

    return {
        "line": f"**{term}**: {answer}",
        "citations": citations,
        "evidence": evidence,
    }


# -----------------------------
# Intent helpers
# -----------------------------
def _is_summarize_all_question(question: str, docs_count: int) -> bool:
    if docs_count < 1:
        return False
    q = (question or "").lower().strip()
    signals = [
        "summarize each",
        "summarise each",
        "summarize all",
        "summarise all",
        "each uploaded pdf",
        "each uploaded document",
        "all uploaded pdf",
        "all uploaded document",
        "summarize the uploaded pdfs",
        "summarize uploaded pdfs",
        "summarize uploaded documents",
        "one sentence with its title",
        "one line with the title",
        "what does each pdf discuss",
        "what each pdf discusses",
        "main topic of each",
        "summarize the pdf in one sentence with its title",
        "summarize each uploaded pdf in one sentence with its title",
    ]
    return any(s in q for s in signals)


def _is_compare_question(question: str, docs_count: int) -> bool:
    if docs_count < 2:
        return False
    q = (question or "").lower().strip()
    signals = [
        "compare",
        "differences",
        "similarities",
        "similarity",
        "difference",
        "vs ",
        "versus",
        "compare the uploaded documents",
        "compare the uploaded pdfs",
        "compare uploaded documents",
        "compare uploaded pdfs",
        "how are these documents different",
        "how are these documents similar",
        "cross reference",
        "cross-reference",
        "connections between",
        "connections across",
        "draw conclusions",
        "draw conclusion",
        "contrast the documents",
        "relate the documents",
        "themes across",
        "common themes",
        "link between",
        "relationship between",
        "what themes are common across the documents",
        "what is common in these pdf",
        "what is common in these documents",
        "what common in these pdf",
        "what common in these documents",
    ]
    return any(s in q for s in signals)


def _is_multi_concept_question(question: str, docs_count: int) -> bool:
    if docs_count < 2:
        return False
    q = (question or "").lower().strip()
    comma_count = q.count(",")
    and_count = len(re.findall(r"\band\b", q))
    prompt_signals = [
        "explain",
        "define",
        "what is",
        "what are",
        "meaning of",
        "in one line",
        "briefly",
    ]
    has_prompt = any(s in q for s in prompt_signals)
    return has_prompt and (comma_count >= 1 or and_count >= 1)


def _is_summaryish_question(question: str) -> bool:
    q = (question or "").lower().strip()
    patterns = [
        "summarize",
        "summarise",
        "summary",
        "what is the first pdf about",
        "what is the second pdf about",
        "what is the third pdf about",
        "what is the fourth pdf about",
        "what is the fifth pdf about",
        "what is this pdf about",
        "what is the pdf about",
        "what is this document about",
        "what is the document about",
        "main idea",
        "main topic",
        "overview",
    ]
    return any(p in q for p in patterns)


def _ordinal_to_index(question: str) -> Optional[int]:
    q = (question or "").lower()

    patterns = [
        (r"\bfirst pdf\b|\bfirst document\b|\b1st pdf\b|\b1st document\b", 0),
        (r"\bsecond pdf\b|\bsecond document\b|\b2nd pdf\b|\b2nd document\b", 1),
        (r"\bthird pdf\b|\bthird document\b|\b3rd pdf\b|\b3rd document\b", 2),
        (r"\bfourth pdf\b|\bfourth document\b|\b4th pdf\b|\b4th document\b", 3),
        (r"\bfifth pdf\b|\bfifth document\b|\b5th pdf\b|\b5th document\b", 4),
    ]

    for pat, idx in patterns:
        if re.search(pat, q):
            return idx

    return None


def _score_doc_match(question: str, doc: StoredDocument) -> float:
    q_norm = _normalize_for_match(question)
    filename = doc.filename or ""
    title = doc.title or filename
    stem = _filename_stem(filename)

    candidates = [
        _normalize_for_match(filename),
        _normalize_for_match(stem),
        _normalize_for_match(title),
    ]

    score = 0.0
    for cand in candidates:
        if not cand:
            continue
        if cand and cand in q_norm:
            score += 15.0
    return score


def _match_document_from_question(question: str, docs: List[StoredDocument]) -> Optional[StoredDocument]:
    if not docs:
        return None

    ordinal_idx = _ordinal_to_index(question)
    if ordinal_idx is not None and 0 <= ordinal_idx < len(docs):
        return docs[ordinal_idx]

    scored: List[Tuple[float, StoredDocument]] = []
    for doc in docs:
        scored.append((_score_doc_match(question, doc), doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return None

    best_score, best_doc = scored[0]
    if best_score >= 4.0:
        return best_doc

    return None


def _strip_doc_reference_from_question(question: str, doc: StoredDocument) -> str:
    q = question or ""

    patterns = [
        re.escape(doc.filename or ""),
        re.escape(_filename_stem(doc.filename or "")),
        re.escape(doc.title or ""),
    ]

    cleaned = q
    for pat in patterns:
        if pat:
            cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"(?i)\b(pdf|document|file|uploaded)\b", " ", cleaned)
    cleaned = re.sub(
        r"(?i)\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;?-")
    return cleaned or question


def _find_doc_by_keyword(question: str, docs: List[StoredDocument]) -> Optional[StoredDocument]:
    q = (question or "").lower()
    q_tokens = set(re.findall(r"[a-z0-9]+", q))

    best_doc: Optional[StoredDocument] = None
    best_overlap = 0

    for doc in docs:
        title = (doc.title or "").lower()
        filename = (doc.filename or "").lower()
        cand_tokens = set(re.findall(r"[a-z0-9]+", f"{title} {filename}"))
        overlap = len(q_tokens.intersection(cand_tokens))
        if overlap > best_overlap:
            best_overlap = overlap
            best_doc = doc

    return best_doc if best_overlap > 0 else None


# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/system")
def system_status() -> Dict[str, Any]:
    docs = vector_store.list_documents()
    return {
        "service": "NexusAI RAG Assistant",
        "version": "CHECK136",
        "openai_key_present": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        "chat_model": os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        "base_dir": BASE_DIR,
        "upload_dir": UPLOAD_DIR,
        "docs_ingested": len(docs),
        "chunks_indexed": _chunks_indexed_count(),
        "last_ingest_error": _last_ingest_error,
    }


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
    filename = file.filename or "uploaded.pdf"
    save_path = os.path.join(UPLOAD_DIR, filename)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    _ingest_one_pdf(save_path, force_reingest=force)
    return {"ok": True, "saved_as": filename, "force_reingest": force}


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> Dict[str, Any]:
    doc = _get_doc_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"document_id not found: {document_id}")

    removed = bool(vector_store.remove_document(document_id))
    _best_effort_persist_vector_store()

    file_deleted = False
    file_path = getattr(doc, "file_path", "") or ""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            file_deleted = True
    except Exception:
        pass

    return {
        "ok": True,
        "deleted": removed,
        "document_id": document_id,
        "filename": doc.filename,
        "file_deleted": file_deleted,
        "file_path": file_path,
    }


@app.post("/clear")
def clear_all() -> Dict[str, Any]:
    doc_ids = [d.document_id for d in vector_store.list_documents()]
    removed_count = 0
    for did in doc_ids:
        if vector_store.remove_document(did):
            removed_count += 1
    _best_effort_persist_vector_store()

    deleted_files = 0
    for name in os.listdir(UPLOAD_DIR):
        if not name.lower().endswith(".pdf"):
            continue
        try:
            os.remove(os.path.join(UPLOAD_DIR, name))
            deleted_files += 1
        except Exception:
            pass

    return {
        "ok": True,
        "documents_removed": removed_count,
        "pdf_files_deleted": deleted_files,
        "upload_dir": UPLOAD_DIR,
        "docs_ingested": len(vector_store.list_documents()),
        "chunks_indexed": _chunks_indexed_count(),
    }


@app.post("/query")
def query_docs(req: QueryRequest) -> Dict[str, Any]:
    q = (req.question or "").strip()

    if q == "PING123":
        return {
            "question": q,
            "answer": "MAIN_FILE_IS_RUNNING",
            "citations": [],
            "evidence": [],
            "used_doc_ids": [],
            "status": "answered",
        }

    print("QUERY ROUTE CHECK136:", req.question)

    docs = vector_store.list_documents()
    doc_ids = [d.document_id for d in docs]
    docs_count = len(doc_ids)
    question = (req.question or "").strip()

    if docs_count == 0:
        return {
            "question": question,
            "answer": INSUFFICIENT_MSG,
            "citations": [],
            "evidence": [],
            "used_doc_ids": [],
            "status": "no_documents",
        }

    if _is_summarize_all_question(question, docs_count):
        lines: List[str] = []
        all_citations: List[Dict[str, Any]] = []
        for doc in docs:
            item = _direct_doc_line(doc)
            lines.append(item["line"])
            all_citations.extend(item["citations"])
        return {
            "question": question,
            "answer": "\n\n".join(lines),
            "citations": _dedupe_citations(all_citations),
            "evidence": [],
            "used_doc_ids": doc_ids,
            "status": "answered",
        }

    if _is_compare_question(question, docs_count):
        result = _general_compare_from_briefs(question, docs)
        return {
            "question": question,
            "answer": result["answer"],
            "citations": result["citations"],
            "evidence": result["evidence"],
            "used_doc_ids": doc_ids,
            "status": result["status"],
        }

    matched_doc = _match_document_from_question(question, docs)
    if matched_doc is None:
        matched_doc = _find_doc_by_keyword(question, docs)

    if matched_doc and _is_summaryish_question(question):
        item = _direct_doc_line(matched_doc)
        return {
            "question": question,
            "answer": item["line"],
            "citations": item["citations"],
            "evidence": item["evidence"],
            "used_doc_ids": [matched_doc.document_id],
            "matched_document_id": matched_doc.document_id,
            "matched_filename": matched_doc.filename,
            "status": item["status"],
        }

    ordinal_idx = _ordinal_to_index(question)
    if ordinal_idx is not None and 0 <= ordinal_idx < len(docs):
        item = _direct_doc_line(docs[ordinal_idx])
        return {
            "question": question,
            "answer": item["line"],
            "citations": item["citations"],
            "evidence": item["evidence"],
            "used_doc_ids": [docs[ordinal_idx].document_id],
            "matched_document_id": docs[ordinal_idx].document_id,
            "matched_filename": docs[ordinal_idx].filename,
            "status": item["status"],
        }

    if matched_doc and any(x in question.lower() for x in ["pdf", "document", "file", "about", "explain"]):
        item = _direct_doc_line(matched_doc)
        return {
            "question": question,
            "answer": item["line"],
            "citations": item["citations"],
            "evidence": item["evidence"],
            "used_doc_ids": [matched_doc.document_id],
            "matched_document_id": matched_doc.document_id,
            "matched_filename": matched_doc.filename,
            "status": item["status"],
        }

    if _is_multi_concept_question(question, docs_count):
        concepts = _decompose_question(question)
        if concepts:
            lines: List[str] = []
            all_citations: List[Dict[str, Any]] = []
            all_evidence: List[Dict[str, Any]] = []

            for concept in concepts:
                item = _answer_one_concept_across_docs(
                    concept,
                    doc_ids,
                    top_k_per_doc=max(10, req.top_k),
                )
                lines.append(item["line"])
                all_citations.extend(item["citations"])
                all_evidence.extend(item["evidence"])

            return {
                "question": question,
                "answer": "\n\n".join(lines),
                "citations": _dedupe_citations(all_citations),
                "evidence": _dedupe_results_keep_best(all_evidence),
                "used_doc_ids": doc_ids,
                "concepts": concepts,
                "status": "answered",
            }

    if matched_doc:
        narrowed_question = _strip_doc_reference_from_question(question, matched_doc)
        chunks = _vector_search_in_doc(
            narrowed_question,
            matched_doc.document_id,
            top_k=max(1, int(req.top_k)),
        )
        result = generate_answer(
            narrowed_question,
            chunks,
            chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        )

        if result.get("status") in {"not_mentioned", "insufficient_evidence"}:
            item = _direct_doc_line(matched_doc)
            return {
                "question": question,
                "answer": item["line"],
                "citations": item["citations"],
                "evidence": item["evidence"],
                "used_doc_ids": [matched_doc.document_id],
                "matched_document_id": matched_doc.document_id,
                "matched_filename": matched_doc.filename,
                "status": item["status"],
            }

        return {
            "question": question,
            "answer": result.get("answer", INSUFFICIENT_MSG),
            "citations": result.get("citations", []),
            "evidence": result.get("evidence", []),
            "used_doc_ids": [matched_doc.document_id],
            "matched_document_id": matched_doc.document_id,
            "matched_filename": matched_doc.filename,
            "status": result.get("status", "unknown"),
        }

    chunks = _multi_query_vector_search(question, top_k=max(1, int(req.top_k)))
    result = generate_answer(
        question,
        chunks,
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
    )
    return {
        "question": question,
        "answer": result.get("answer", INSUFFICIENT_MSG),
        "citations": result.get("citations", []),
        "evidence": result.get("evidence", []),
        "used_doc_ids": doc_ids,
        "status": result.get("status", "unknown"),
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
            "status": "doc_not_found",
        }

    chunks = _vector_search_in_doc(
        req.question,
        req.document_id,
        top_k=max(1, int(req.top_k)),
    )
    result = generate_answer(req.question, chunks, chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    return {
        "question": req.question,
        "answer": result.get("answer", INSUFFICIENT_MSG),
        "citations": result.get("citations", []),
        "evidence": result.get("evidence", []),
        "used_doc_id": req.document_id,
        "status": result.get("status", "unknown"),
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
            "status": "doc_not_found",
        }

    selected_docs = [d for d in docs if d.document_id in doc_ids]
    result = _general_compare_from_briefs(req.question, selected_docs)
    return {
        "question": req.question,
        "answer": result["answer"],
        "citations": result["citations"],
        "evidence": result["evidence"],
        "used_doc_ids": doc_ids,
        "status": result["status"],
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
            "status": "insufficient_documents",
        }

    selected_docs = [d for d in docs if d.document_id in doc_ids]
    result = _general_compare_from_briefs(req.question, selected_docs)
    return {
        "question": req.question,
        "answer": result["answer"],
        "citations": result["citations"],
        "evidence": result["evidence"],
        "used_doc_ids": doc_ids,
        "status": result["status"],
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
            "status": "doc_not_found",
        }

    item = _direct_doc_line(doc)
    summary = item["line"]

    max_chars = max(250, int(req.max_chars or 1200))
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rstrip() + "..."

    return {
        "document_id": req.document_id,
        "summary": summary,
        "citations": item["citations"],
        "evidence": item["evidence"],
        "status": item["status"],
    }