# answer_generator.py
import os
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INSUFFICIENT_MSG = "Insufficient evidence in the provided documents."


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
    # APA-ish: (Bishop, as cited in Azadani, 2025, p. 12)
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

    # Also allow plain "Chapter 3" (common in your slides)
    m3 = re.search(r"\b(Chapter\s*\d+)\b", text, flags=re.IGNORECASE)
    if m3:
        return m3.group(1).strip()

    return ""


# -----------------------------
# Name plausibility checks (critical for OpenEvidence-like behavior)
# -----------------------------
def _is_plausible_author_name(author: str) -> bool:
    """
    Reject garbage authors like:
      - "log pX Z"
      - "PZX8"
      - math fragments
    Accept typical slide authors:
      - "Salar N. Azadani"
      - "Azadani"
      - "Kenneth Fogel"
    """
    a = _safe_str(author).strip()
    if not a:
        return False

    low = a.lower()

    # Hard reject math / formula-y patterns
    if "log" in low:
        return False
    if any(ch in a for ch in ["=", "(", ")", "{", "}", "\\", "^", "_", "∑", "∫", "θ"]):
        return False

    # Too many digits means it's not a person name
    digits = sum(ch.isdigit() for ch in a)
    if digits >= 2:
        return False

    letters = sum(ch.isalpha() for ch in a)
    if letters < 3:
        return False

    # If it's mostly non-letters, reject
    if letters / max(1, len(a)) < 0.45:
        return False

    # Single-letter or tiny token is never a real author
    if len(a) <= 2:
        return False

    return True


def _normalize_retrieved_item(item: Dict[str, Any]) -> Dict[str, Any]:
    meta = item.get("metadata") or {}

    text = item.get("text") or meta.get("text") or ""
    chunk_id = item.get("chunk_id") or meta.get("chunk_id") or meta.get("id") or "unknown_chunk"

    # Base doc metadata from store
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

    # ---- Override author using slide text if present ----
    presenter = _extract_presenter(text)
    if presenter:
        author = presenter

    # ---- Override title hint from slide text if present ----
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

# Name-like bracket: [Bishop], [Goodfellow], [Salar Azadani]
_NAME_LIKE_BRACKET = re.compile(r"\[[A-Z][A-Za-z.\s'’\-]{1,35}\]")

def _looks_like_reference_context(text: str) -> bool:
    """
    Only allow secondary citations if the chunk looks like references.
    Allow:
      - 'ref:' / 'reference(s)'
      - OR a name-like bracket [Bishop]
    Block:
      - math brackets like [log P(Z|X,θ)]
    """
    if not text:
        return False
    t = text.lower()
    if "ref:" in t or "reference" in t or "references" in t:
        return True
    if _NAME_LIKE_BRACKET.search(text):
        return True
    return False


def _extract_secondary_authors(text: str) -> List[str]:
    """
    Detect bracketed references like: [Bishop]
    Strictly reject math: [log P(Z,X|θ)]
    """
    if not text:
        return []

    if not _looks_like_reference_context(text):
        return []

    hits = re.findall(r"\[([^\[\]\n]{2,60})\]", text)
    out: List[str] = []

    for raw in hits:
        raw_stripped = raw.strip()
        low_raw = raw_stripped.lower()

        # Hard reject common math / log bracket patterns
        if "log" in low_raw:
            continue
        if any(ch in raw_stripped for ch in ["=", "(", ")", "{", "}", "\\", "^", "_", "θ"]):
            continue
        # "P(" often shows probability expressions
        if "p(" in low_raw or "p{" in low_raw:
            continue

        cand = raw_stripped

        # Keep only name-ish characters
        cand = re.sub(r"[^A-Za-z.\s'’\-]", "", cand).strip()
        cand = re.sub(r"\s{2,}", " ", cand).strip()
        if not cand:
            continue

        upper_token = re.sub(r"[^A-Z]", "", cand.upper()).strip()

        if upper_token in _BAD_BRACKET_TOKENS:
            continue

        # Must have at least 3 letters
        letters = sum(ch.isalpha() for ch in cand)
        if letters < 3:
            continue

        # Must contain at least one capital letter (author names)
        if not any(ch.isupper() for ch in cand):
            continue

        # Reject all-caps short tokens
        if cand.upper() == cand and len(cand) <= 6:
            continue

        # Finally: must look like an author string
        if not _is_plausible_author_name(cand):
            continue

        out.append(cand)

    # dedupe preserve order
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
# Doc-level author fallback (fixes "one chunk has garbage author")
# -----------------------------
def _apply_document_author_fallback(normalized: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each document_id, find the most common plausible author among the used chunks.
    If a chunk author is not plausible, replace it with the doc's best author.
    """
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

    # apply
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
# Core answer generators
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
            doc_hdr = f"DocumentID: {n.get('document_id','')}\nFilename: {n.get('metadata',{}).get('filename','')}\n"

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


def _build_citations_from_used(used_normalized: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []

    for n in used_normalized:
        # Primary citation (ALWAYS from document metadata, never from chunk text)
        citations.append(
            {
                "author": n["author_full"],
                "year": n["year"],
                "title": n["title"],
                "pages": n["pages"],
                "citation": _make_citation(n["author_short"], n["year"], n["pages"]),
            }
        )

        # Secondary citations (strict: only real author-looking brackets)
        secondary_authors = _extract_secondary_authors(n["text"])
        for sec in secondary_authors:
            citations.append(
                {
                    "author": sec,
                    "year": "",  # unknown from slides
                    "title": n["title"],
                    "pages": n["pages"],
                    "citation": _make_secondary_citation(sec, n["author_short"], n["year"], n["pages"]),
                }
            )

    return _dedupe_and_sort_citations(citations)


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    chat_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_evidence_chars: int = 12000,
) -> Dict[str, Any]:
    question = _safe_str(question).strip()
    if not question:
        return {"answer": "No question provided.", "citations": [], "evidence": []}

    if not retrieved_chunks:
        return {"answer": INSUFFICIENT_MSG, "citations": [], "evidence": []}

    normalized = [_normalize_retrieved_item(x) for x in retrieved_chunks]
    normalized = [x for x in normalized if x["text"]]

    if not normalized:
        return {"answer": INSUFFICIENT_MSG, "citations": [], "evidence": []}

    # ✅ Apply doc-level author fallback BEFORE building evidence/citations
    normalized = _apply_document_author_fallback(normalized)

    evidence_block, used_normalized = _build_evidence_block(
        normalized=normalized,
        max_evidence_chars=max_evidence_chars,
        include_doc_header=False,
    )

    if not used_normalized:
        return {"answer": INSUFFICIENT_MSG, "citations": [], "evidence": []}

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
        f"4) If evidence is insufficient, output EXACTLY:\n   {INSUFFICIENT_MSG}\n"
        "5) Be concise and clear.\n"
    )

    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer using ONLY EVIDENCE.\n"
        "- Add citations inline using (AuthorLastName, Year, p. X).\n"
        f"- If you cannot answer from evidence, output exactly: {INSUFFICIENT_MSG}\n"
    )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {
            "answer": "OpenAI API key missing. Set OPENAI_API_KEY in your environment or .env.",
            "citations": citations,
            "evidence": evidence_out,
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
        answer = INSUFFICIENT_MSG

    if answer != INSUFFICIENT_MSG and not _contains_inline_citation(answer):
        answer = INSUFFICIENT_MSG

    if answer == INSUFFICIENT_MSG:
        return {"answer": answer, "citations": [], "evidence": []}

    return {"answer": answer, "citations": citations, "evidence": evidence_out}


def generate_comparison_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    chat_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_evidence_chars: int = 14000,
) -> Dict[str, Any]:
    question = _safe_str(question).strip()
    if not question:
        return {"answer": "No question provided.", "citations": [], "evidence": []}

    if not retrieved_chunks:
        return {"answer": INSUFFICIENT_MSG, "citations": [], "evidence": []}

    normalized = [_normalize_retrieved_item(x) for x in retrieved_chunks]
    normalized = [x for x in normalized if x["text"]]

    if not normalized:
        return {"answer": INSUFFICIENT_MSG, "citations": [], "evidence": []}

    # ✅ Apply doc-level author fallback
    normalized = _apply_document_author_fallback(normalized)

    evidence_block, used_normalized = _build_evidence_block(
        normalized=normalized,
        max_evidence_chars=max_evidence_chars,
        include_doc_header=True,
    )

    if not used_normalized:
        return {"answer": INSUFFICIENT_MSG, "citations": [], "evidence": []}

    evidence_out = [
        {"chunk_id": n["chunk_id"], "text": n["text"], "score": n["score"], "metadata": n["metadata"]}
        for n in used_normalized
    ]
    citations = _build_citations_from_used(used_normalized)

    system_msg = (
        "You are a strictly grounded comparison assistant.\n"
        "RULES:\n"
        "1) Use ONLY the evidence provided.\n"
        "2) Do NOT use outside knowledge.\n"
        "3) Every key claim MUST include an inline citation:\n"
        "   (AuthorLastName, Year, p. X) or (AuthorLastName, Year, pp. X–Y)\n"
        f"4) If evidence is insufficient, output EXACTLY:\n   {INSUFFICIENT_MSG}\n"
        "5) Output MUST be structured per document, then similarities/differences.\n"
    )

    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        "INSTRUCTIONS:\n"
        "- First: list each document separately (Doc A, Doc B, ...).\n"
        "  For each doc: what it is about + any requested attributes (title/presenter/etc).\n"
        "- Then: Similarities\n"
        "- Then: Differences\n"
        "- Then: Short takeaway\n"
        "- Use ONLY evidence and cite every key claim inline.\n"
        f"- If you cannot answer from evidence, output exactly: {INSUFFICIENT_MSG}\n"
    )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {
            "answer": "OpenAI API key missing. Set OPENAI_API_KEY in your environment or .env.",
            "citations": citations,
            "evidence": evidence_out,
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
        answer = INSUFFICIENT_MSG

    if answer != INSUFFICIENT_MSG and not _contains_inline_citation(answer):
        answer = INSUFFICIENT_MSG

    if answer == INSUFFICIENT_MSG:
        return {"answer": answer, "citations": [], "evidence": []}

    return {"answer": answer, "citations": citations, "evidence": evidence_out}