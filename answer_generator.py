# answer_generator.py
import os
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INSUFFICIENT_MSG = "Insufficient evidence in the provided documents."
NOT_MENTIONED_MSG = "The answer does not appear to be mentioned in the retrieved document evidence."
SCANNED_OR_LOW_TEXT_MSG = (
    "Insufficient evidence in the provided documents. "
    "This may happen if the PDF is scanned or the extracted text is too weak."
)


# -----------------------------
# Helpers
# -----------------------------
def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _author_short(author: str) -> str:
    author = _safe_str(author).strip()
    if not author:
        return "Unknown"
    if "," in author:
        return author.split(",")[0].strip() or author.strip()
    parts = [p for p in author.split() if p.strip()]
    return parts[-1] if parts else author


def _format_pages(pages: List[int]) -> str:
    if not pages:
        return "p. ?"
    pages_sorted = sorted(set(pages))
    if len(pages_sorted) == 1:
        return f"p. {pages_sorted[0]}"
    return f"pp. {pages_sorted[0]}–{pages_sorted[-1]}"


def _make_citation(author_short: str, year: str, pages: List[int]) -> str:
    a = author_short if author_short else "Unknown"
    y = year if year else "n.d."
    return f"({a}, {y}, {_format_pages(pages)})"


def _make_secondary_citation(
    secondary_author: str,
    primary_author_short: str,
    primary_year: str,
    pages: List[int],
) -> str:
    y = primary_year if primary_year else "n.d."
    return f"({_author_short(secondary_author)}, as cited in {primary_author_short}, {y}, {_format_pages(pages)})"


def _citation_key(c: Dict[str, Any]) -> Tuple[str, str, str, Tuple[int, ...], str]:
    return (
        _safe_str(c.get("author", "")),
        _safe_str(c.get("year", "")),
        _safe_str(c.get("title", "")),
        tuple(c.get("pages") or []),
        _safe_str(c.get("citation", "")),
    )


def _dedupe_and_sort_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []

    for c in citations:
        pages = c.get("pages") or []
        pages = [p for p in pages if isinstance(p, int)]
        c["pages"] = pages
        c["page"] = pages[0] if pages else 0

        key = _citation_key(c)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    def sort_key(x: Dict[str, Any]):
        pages = x.get("pages") or []
        first_page = pages[0] if pages else 10**9
        return (
            _safe_str(x.get("title", "")).lower(),
            _safe_str(x.get("author", "")).lower(),
            _safe_str(x.get("year", "")).lower(),
            first_page,
            _safe_str(x.get("citation", "")).lower(),
        )

    unique.sort(key=sort_key)
    return unique


def _contains_inline_citation(answer: str) -> bool:
    pat = r"\([^)]+,\s*[^)]+,\s*p{1,2}\.\s*[^)]+\)"
    return bool(re.search(pat, answer))


# -----------------------------
# Metadata overrides from evidence text
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

    m2 = re.search(r"(Chapter\s*\d+\s*:\s*)\n+([^\n\r]+)", text, flags=re.IGNORECASE)
    if m2:
        return f"{m2.group(1).strip()} {m2.group(2).strip()}".strip()

    m3 = re.search(r"\b(Chapter\s*\d+)\b", text, flags=re.IGNORECASE)
    if m3:
        return m3.group(1).strip()

    return ""


# -----------------------------
# Name plausibility checks
# -----------------------------
def _is_plausible_author_name(author: str) -> bool:
    a = _safe_str(author).strip()
    if not a:
        return False

    low = a.lower()

    if "log" in low:
        return False
    if any(ch in a for ch in ["=", "(", ")", "{", "}", "\\", "^", "_", "∑", "∫", "θ"]):
        return False

    digits = sum(ch.isdigit() for ch in a)
    if digits >= 2:
        return False

    letters = sum(ch.isalpha() for ch in a)
    if letters < 3:
        return False

    if letters / max(1, len(a)) < 0.45:
        return False

    if len(a) <= 2:
        return False

    return True


def _normalize_retrieved_item(item: Dict[str, Any]) -> Dict[str, Any]:
    meta = item.get("metadata") or {}

    text = item.get("text") or meta.get("text") or ""
    chunk_id = item.get("chunk_id") or meta.get("chunk_id") or meta.get("id") or "unknown_chunk"

    author = item.get("document_author") or meta.get("author") or meta.get("document_author") or "Unknown Author"
    year = item.get("document_year") or meta.get("year") or meta.get("document_year") or ""
    title = (
        item.get("document_title")
        or meta.get("title")
        or meta.get("document_title")
        or meta.get("filename")
        or "Unknown Title"
    )

    document_id = item.get("document_id") or meta.get("document_id") or meta.get("doc_id") or ""

    pages = item.get("pages") or meta.get("pages")
    page = item.get("page") or meta.get("page")

    if pages is None:
        pages = [page] if isinstance(page, int) else []
    if isinstance(pages, int):
        pages = [pages]
    if not isinstance(pages, list):
        pages = []
    pages = [p for p in pages if isinstance(p, int)]

    score = item.get("relevance_score")
    if score is None:
        score = item.get("score")
    if score is None:
        score = 0.0

    presenter = _extract_presenter(text)
    if presenter:
        author = presenter

    chap_line = _extract_chapter_title_line(text)
    if chap_line:
        tl = (title or "").lower()
        if ("web services" in tl) or ("xml processing" in tl) or ("powerpoint" in tl) or ("untitled" in tl):
            title = chap_line

    return {
        "chunk_id": _safe_str(chunk_id),
        "text": _safe_str(text).strip(),
        "score": float(score) if isinstance(score, (int, float)) else 0.0,
        "author_full": _safe_str(author),
        "author_short": _author_short(author),
        "year": _safe_str(year),
        "title": _safe_str(title),
        "pages": pages,
        "document_id": _safe_str(document_id),
        "metadata": meta,
    }


# -----------------------------
# Secondary-citation detection (STRICT)
# -----------------------------
_BAD_BRACKET_TOKENS = {
    "CLS", "SEP", "SOS", "EOS", "PAD", "UNK", "MASK",
    "GPT", "BERT", "LLM",
    "WQ", "WK", "WV", "Q", "K", "V",
}

_NAME_LIKE_BRACKET = re.compile(r"\[[A-Z][A-Za-z.\s'’\-]{1,35}\]")


def _looks_like_reference_context(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if "ref:" in t or "reference" in t or "references" in t:
        return True
    if _NAME_LIKE_BRACKET.search(text):
        return True
    return False


def _extract_secondary_authors(text: str) -> List[str]:
    if not text:
        return []

    if not _looks_like_reference_context(text):
        return []

    hits = re.findall(r"\[([^\[\]\n]{2,60})\]", text)
    out: List[str] = []

    for raw in hits:
        raw_stripped = raw.strip()
        low_raw = raw_stripped.lower()

        if "log" in low_raw:
            continue
        if any(ch in raw_stripped for ch in ["=", "(", ")", "{", "}", "\\", "^", "_", "θ"]):
            continue
        if "p(" in low_raw or "p{" in low_raw:
            continue

        cand = raw_stripped
        cand = re.sub(r"[^A-Za-z.\s'’\-]", "", cand).strip()
        cand = re.sub(r"\s{2,}", " ", cand).strip()
        if not cand:
            continue

        upper_token = re.sub(r"[^A-Z]", "", cand.upper()).strip()

        if upper_token in _BAD_BRACKET_TOKENS:
            continue

        letters = sum(ch.isalpha() for ch in cand)
        if letters < 3:
            continue

        if not any(ch.isupper() for ch in cand):
            continue

        if cand.upper() == cand and len(cand) <= 6:
            continue

        if not _is_plausible_author_name(cand):
            continue

        out.append(cand)

    seen = set()
    final: List[str] = []
    for x in out:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        final.append(x)
    return final


# -----------------------------
# Doc-level author fallback
# -----------------------------
def _apply_document_author_fallback(normalized: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_doc: Dict[str, List[str]] = {}
    for n in normalized:
        did = n.get("document_id", "") or ""
        a = n.get("author_full", "") or ""
        if did not in by_doc:
            by_doc[did] = []
        if _is_plausible_author_name(a):
            by_doc[did].append(a.strip())

    best_author: Dict[str, str] = {}
    for did, authors in by_doc.items():
        if not authors:
            continue
        best_author[did] = Counter(authors).most_common(1)[0][0]

    out: List[Dict[str, Any]] = []
    for n in normalized:
        did = n.get("document_id", "") or ""
        a = n.get("author_full", "") or ""
        if (not _is_plausible_author_name(a)) and (did in best_author):
            n = dict(n)
            n["author_full"] = best_author[did]
            n["author_short"] = _author_short(best_author[did])
        out.append(n)

    return out


# -----------------------------
# Failure diagnosis
# -----------------------------
def _is_metadata_chunk(n: Dict[str, Any]) -> bool:
    meta = n.get("metadata") or {}
    return (meta.get("chunk_type") or "") == "document_metadata"


def _is_metadata_question(question: str) -> bool:
    q = _safe_str(question).lower()
    patterns = [
        "author", "presenter", "who wrote", "who is the author", "who is the presenter",
        "title", "year", "filename", "what is the title", "what year"
    ]
    return any(p in q for p in patterns)


def _diagnose_evidence_quality(question: str, used_normalized: List[Dict[str, Any]]) -> str:
    if not used_normalized:
        return "low_text"

    non_meta = [n for n in used_normalized if not _is_metadata_chunk(n)]
    non_meta_text = [n["text"] for n in non_meta if _safe_str(n.get("text")).strip()]
    total_non_meta_chars = sum(len(t.strip()) for t in non_meta_text)

    if _is_metadata_question(question):
        if any(_is_metadata_chunk(n) for n in used_normalized):
            return "usable"

    if not non_meta:
        return "metadata_only"

    if total_non_meta_chars < 250:
        return "low_text"

    return "usable"


def _fallback_failure_message(question: str, used_normalized: List[Dict[str, Any]]) -> str:
    quality = _diagnose_evidence_quality(question, used_normalized)

    if quality in {"metadata_only", "low_text"}:
        return SCANNED_OR_LOW_TEXT_MSG

    return NOT_MENTIONED_MSG


# -----------------------------
# Evidence builders
# -----------------------------
def _build_evidence_block(
    normalized: List[Dict[str, Any]],
    max_evidence_chars: int,
    include_doc_header: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    evidence_lines: List[str] = []
    total_chars = 0
    used: List[Dict[str, Any]] = []

    for idx, n in enumerate(normalized, start=1):
        cite = _make_citation(n["author_short"], n["year"], n["pages"])

        doc_hdr = ""
        if include_doc_header:
            doc_hdr = (
                f"DocumentID: {n.get('document_id', '')}\n"
                f"Filename: {n.get('metadata', {}).get('filename', '')}\n"
            )

        block = (
            f"[E{idx}] {cite}\n"
            f"{doc_hdr}"
            f"Title: {n['title']}\n"
            f"ChunkID: {n['chunk_id']}\n"
            f"Text: {n['text']}\n"
        )

        if total_chars + len(block) > max_evidence_chars:
            break

        evidence_lines.append(block)
        used.append(n)
        total_chars += len(block)

    return ("\n---\n".join(evidence_lines), used)


def _group_chunks_by_document(normalized: List[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []

    for n in normalized:
        did = n.get("document_id") or n.get("metadata", {}).get("filename") or "unknown_document"
        if did not in grouped:
            grouped[did] = []
            order.append(did)
        grouped[did].append(n)

    return [(did, grouped[did]) for did in order]


def _build_grouped_comparison_evidence_block(
    normalized: List[Dict[str, Any]],
    max_evidence_chars: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    grouped = _group_chunks_by_document(normalized)

    lines: List[str] = []
    used: List[Dict[str, Any]] = []
    total_chars = 0

    for doc_index, (doc_id, chunks) in enumerate(grouped, start=1):
        if not chunks:
            continue

        first = chunks[0]
        filename = _safe_str(first.get("metadata", {}).get("filename", ""))
        title = _safe_str(first.get("title", ""))
        author = _safe_str(first.get("author_full", ""))
        year = _safe_str(first.get("year", ""))

        header = (
            f"=== DOCUMENT {doc_index} ===\n"
            f"DocumentID: {doc_id}\n"
            f"Filename: {filename}\n"
            f"Title: {title}\n"
            f"Author: {author}\n"
            f"Year: {year}\n"
        )

        if total_chars + len(header) > max_evidence_chars:
            break

        lines.append(header)
        total_chars += len(header)

        for chunk_index, n in enumerate(chunks, start=1):
            cite = _make_citation(n["author_short"], n["year"], n["pages"])
            block = (
                f"[D{doc_index}-E{chunk_index}] {cite}\n"
                f"ChunkID: {n['chunk_id']}\n"
                f"Text: {n['text']}\n"
            )

            if total_chars + len(block) > max_evidence_chars:
                break

            lines.append(block)
            used.append(n)
            total_chars += len(block)

    return ("\n---\n".join(lines), used)


def _build_citations_from_used(used_normalized: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []

    for n in used_normalized:
        citations.append(
            {
                "author": n["author_full"],
                "year": n["year"],
                "title": n["title"],
                "pages": n["pages"],
                "citation": _make_citation(n["author_short"], n["year"], n["pages"]),
                "document_id": n.get("document_id", ""),
                "filename": n.get("metadata", {}).get("filename", ""),
            }
        )

        secondary_authors = _extract_secondary_authors(n["text"])
        for sec in secondary_authors:
            citations.append(
                {
                    "author": sec,
                    "year": "",
                    "title": n["title"],
                    "pages": n["pages"],
                    "citation": _make_secondary_citation(sec, n["author_short"], n["year"], n["pages"]),
                    "document_id": n.get("document_id", ""),
                    "filename": n.get("metadata", {}).get("filename", ""),
                }
            )

    return _dedupe_and_sort_citations(citations)


# -----------------------------
# Core answer generators
# -----------------------------
def generate_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    chat_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_evidence_chars: int = 12000,
) -> Dict[str, Any]:
    question = _safe_str(question).strip()
    if not question:
        return {
            "answer": "No question provided.",
            "citations": [],
            "evidence": [],
            "status": "no_question",
        }

    if not retrieved_chunks:
        return {
            "answer": SCANNED_OR_LOW_TEXT_MSG,
            "citations": [],
            "evidence": [],
            "status": "insufficient_evidence",
        }

    normalized = [_normalize_retrieved_item(x) for x in retrieved_chunks]
    normalized = [x for x in normalized if x["text"]]

    if not normalized:
        return {
            "answer": SCANNED_OR_LOW_TEXT_MSG,
            "citations": [],
            "evidence": [],
            "status": "insufficient_evidence",
        }

    normalized = _apply_document_author_fallback(normalized)

    evidence_block, used_normalized = _build_evidence_block(
        normalized=normalized,
        max_evidence_chars=max_evidence_chars,
        include_doc_header=False,
    )

    if not used_normalized:
        return {
            "answer": SCANNED_OR_LOW_TEXT_MSG,
            "citations": [],
            "evidence": [],
            "status": "insufficient_evidence",
        }

    evidence_out = [
        {"chunk_id": n["chunk_id"], "text": n["text"], "score": n["score"], "metadata": n["metadata"]}
        for n in used_normalized
    ]

    citations = _build_citations_from_used(used_normalized)

    system_msg = (
        "You are a strictly grounded QA assistant.\n"
        "RULES:\n"
        "1) Use ONLY the evidence provided.\n"
        "2) Do NOT use outside knowledge.\n"
        "3) Every key claim MUST include an inline citation:\n"
        "   (AuthorLastName, Year, p. X) or (AuthorLastName, Year, pp. X–Y)\n"
        f"4) If the evidence is present but the answer is not found, output EXACTLY:\n   {NOT_MENTIONED_MSG}\n"
        f"5) If the evidence is too weak/unusable, output EXACTLY:\n   {INSUFFICIENT_MSG}\n"
        "6) Be concise and clear.\n"
    )

    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer using ONLY EVIDENCE.\n"
        "- Add citations inline using (AuthorLastName, Year, p. X).\n"
        f"- If evidence exists but the answer is not stated, output exactly: {NOT_MENTIONED_MSG}\n"
        f"- If the evidence is too weak to answer, output exactly: {INSUFFICIENT_MSG}\n"
    )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {
            "answer": "OpenAI API key missing. Set OPENAI_API_KEY in your environment or .env.",
            "citations": citations,
            "evidence": evidence_out,
            "status": "config_error",
        }

    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=chat_model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    answer = (resp.choices[0].message.content or "").strip()
    if not answer:
        answer = _fallback_failure_message(question, used_normalized)

    if answer not in {NOT_MENTIONED_MSG, INSUFFICIENT_MSG} and not _contains_inline_citation(answer):
        answer = _fallback_failure_message(question, used_normalized)

    if answer == INSUFFICIENT_MSG:
        answer = _fallback_failure_message(question, used_normalized)

    if answer == NOT_MENTIONED_MSG:
        return {
            "answer": answer,
            "citations": [],
            "evidence": evidence_out,
            "status": "not_mentioned",
        }

    if answer == SCANNED_OR_LOW_TEXT_MSG:
        return {
            "answer": answer,
            "citations": [],
            "evidence": evidence_out,
            "status": "insufficient_evidence",
        }

    return {
        "answer": answer,
        "citations": citations,
        "evidence": evidence_out,
        "status": "answered",
    }


def generate_comparison_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    chat_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_evidence_chars: int = 14000,
) -> Dict[str, Any]:
    question = _safe_str(question).strip()
    if not question:
        return {
            "answer": "No question provided.",
            "citations": [],
            "evidence": [],
            "status": "no_question",
        }

    if not retrieved_chunks:
        return {
            "answer": SCANNED_OR_LOW_TEXT_MSG,
            "citations": [],
            "evidence": [],
            "status": "insufficient_evidence",
        }

    normalized = [_normalize_retrieved_item(x) for x in retrieved_chunks]
    normalized = [x for x in normalized if x["text"]]

    if not normalized:
        return {
            "answer": SCANNED_OR_LOW_TEXT_MSG,
            "citations": [],
            "evidence": [],
            "status": "insufficient_evidence",
        }

    normalized = _apply_document_author_fallback(normalized)

    evidence_block, used_normalized = _build_grouped_comparison_evidence_block(
        normalized=normalized,
        max_evidence_chars=max_evidence_chars,
    )

    if not used_normalized:
        return {
            "answer": SCANNED_OR_LOW_TEXT_MSG,
            "citations": [],
            "evidence": [],
            "status": "insufficient_evidence",
        }

    evidence_out = [
        {"chunk_id": n["chunk_id"], "text": n["text"], "score": n["score"], "metadata": n["metadata"]}
        for n in used_normalized
    ]
    citations = _build_citations_from_used(used_normalized)

    grouped = _group_chunks_by_document(used_normalized)
    expected_doc_count = len(grouped)

    system_msg = (
        "You are a strictly grounded comparison assistant.\n"
        "RULES:\n"
        "1) Use ONLY the evidence provided.\n"
        "2) Do NOT use outside knowledge.\n"
        "3) Every document section MUST contain at least one inline citation from that same document.\n"
        "4) Do NOT collapse all citations into one source.\n"
        "5) Keep documents separate first, then synthesize.\n"
        f"6) If evidence exists but the requested comparison is not stated, output EXACTLY:\n   {NOT_MENTIONED_MSG}\n"
        f"7) If the evidence is too weak/unusable, output EXACTLY:\n   {INSUFFICIENT_MSG}\n"
        "8) Output MUST be structured exactly as:\n"
        "   Document 1\n"
        "   Document 2\n"
        "   ...\n"
        "   Similarities\n"
        "   Differences\n"
        "   Short takeaway\n"
    )

    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        "INSTRUCTIONS:\n"
        f"- There are {expected_doc_count} documents in the evidence. You MUST include all of them.\n"
        "- First, create one section per document in order: Document 1, Document 2, etc.\n"
        "- In each document section:\n"
        "  * describe what that specific document says\n"
        "  * include at least one inline citation from that document section\n"
        "- Then write:\n"
        "  Similarities\n"
        "  Differences\n"
        "  Short takeaway\n"
        "- Use ONLY the provided evidence.\n"
        f"- If evidence exists but the answer is not stated, output exactly: {NOT_MENTIONED_MSG}\n"
        f"- If the evidence is too weak to answer, output exactly: {INSUFFICIENT_MSG}\n"
    )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {
            "answer": "OpenAI API key missing. Set OPENAI_API_KEY in your environment or .env.",
            "citations": citations,
            "evidence": evidence_out,
            "status": "config_error",
        }

    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=chat_model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    answer = (resp.choices[0].message.content or "").strip()
    if not answer:
        answer = _fallback_failure_message(question, used_normalized)

    if answer not in {NOT_MENTIONED_MSG, INSUFFICIENT_MSG} and not _contains_inline_citation(answer):
        answer = _fallback_failure_message(question, used_normalized)

    if answer == INSUFFICIENT_MSG:
        answer = _fallback_failure_message(question, used_normalized)

    if answer == NOT_MENTIONED_MSG:
        return {
            "answer": answer,
            "citations": [],
            "evidence": evidence_out,
            "status": "not_mentioned",
        }

    if answer == SCANNED_OR_LOW_TEXT_MSG:
        return {
            "answer": answer,
            "citations": [],
            "evidence": evidence_out,
            "status": "insufficient_evidence",
        }

    return {
        "answer": answer,
        "citations": citations,
        "evidence": evidence_out,
        "status": "answered",
    }