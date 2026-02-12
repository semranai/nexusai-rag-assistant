# pdf_loader.py
from __future__ import annotations

import os
import re
import uuid
from typing import Any, Dict, List

import fitz  # PyMuPDF

# OCR is optional (only used if a page has almost no text)
try:
    from PIL import Image
    import pytesseract
    _OCR_AVAILABLE = True
except Exception:
    _OCR_AVAILABLE = False


# -----------------------------
# Helpers
# -----------------------------
def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple sliding-window chunker on characters.
    """
    text = _clean_text(text)
    if not text:
        return []

    if chunk_size < 200:
        chunk_size = 200
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def _ocr_page_to_text(doc: fitz.Document, page_index: int) -> str:
    """
    OCR fallback for pages where text extraction fails.
    Only used if pytesseract + PIL are installed and available.
    """
    if not _OCR_AVAILABLE:
        return ""

    page = doc[page_index]
    zoom = 2.0  # higher = better OCR, slower
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img)


def _looks_suspicious_title(title: str) -> bool:
    """
    Heuristic: slide PDFs often have wrong embedded metadata.
    We treat a title as suspicious if it's empty or looks unrelated/generic.
    """
    t = (title or "").strip()
    if not t:
        return True
    tl = t.lower()
    bad_markers = [
        "microsoft",
        "powerpoint",
        "presentation",
        "untitled",
        "slide",
        "document",
        "web services",  # your example wrong title
        "xml processing",  # your example wrong title
    ]
    return any(b in tl for b in bad_markers)


def _looks_suspicious_author(author: str) -> bool:
    a = (author or "").strip()
    if not a or a.lower() in {"unknown", "unknown author"}:
        return True
    # Often embedded author can be like a random username; we keep it unless empty/unknown.
    return False


def _parse_first_page_fallback(doc: fitz.Document) -> Dict[str, str]:
    """
    Extract title/author/course/chapter from first page text like:
      Recurrent Neural Network
      420-A21-AS
      Chapter 9:
      Transformer
      Presented by : Salar Azadani
    """
    out: Dict[str, str] = {"title": "", "author": "", "year": ""}

    if doc.page_count <= 0:
        return out

    page0 = doc[0]
    text = _clean_text(page0.get_text("text") or "")
    if not text:
        return out

    # Normalize spaces but keep newlines for pattern matching across lines
    # We'll search with DOTALL where needed.
    t = text

    # 1) Author: Presented by : X
    m = re.search(r"Presented\s*by\s*:\s*(.+)", t, flags=re.IGNORECASE)
    if m:
        author = m.group(1).strip()
        # stop at line break if it captured too much
        author = author.split("\n")[0].strip()
        out["author"] = author

    # 2) Course code like 420-A21-AS
    m = re.search(r"\b\d{3}-[A-Z]\d{2}-[A-Z]{2}\b", t)
    course = m.group(0).strip() if m else ""

    # 3) Chapter number + title:
    # Sometimes appears as:
    #   Chapter 9:
    #   Transformer
    # or "Chapter 9: Transformer"
    chap_num = ""
    chap_title = ""

    m = re.search(r"Chapter\s*(\d+)\s*:\s*([^\n]+)", t, flags=re.IGNORECASE)
    if m:
        chap_num = m.group(1).strip()
        chap_title = m.group(2).strip()
    else:
        # Two-line case: "Chapter 9:" then next non-empty line is title
        m2 = re.search(r"Chapter\s*(\d+)\s*:\s*\n+([^\n]+)", t, flags=re.IGNORECASE)
        if m2:
            chap_num = m2.group(1).strip()
            chap_title = m2.group(2).strip()

    # 4) Course/Deck headline (first non-empty line)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    headline = lines[0] if lines else ""

    # Build best-effort title
    # Prefer "Headline - Chapter X: Y" when available
    parts: List[str] = []
    if headline:
        parts.append(headline)
    if chap_num and chap_title:
        parts.append(f"Chapter {chap_num}: {chap_title}")
    elif chap_num and not chap_title:
        parts.append(f"Chapter {chap_num}")
    elif chap_title and not chap_num:
        parts.append(chap_title)

    # If still nothing, try to use any line containing "Transformer"
    if not parts:
        for ln in lines[:10]:
            if "transformer" in ln.lower():
                parts.append(ln)
                break

    title = " - ".join(parts).strip()

    # If course exists and title exists, we can optionally append course,
    # but keep it clean for citations (course can be in metadata if needed).
    out["title"] = title

    # Put course into title only if title is empty and course exists
    if not out["title"] and course:
        out["title"] = course

    return out


# -----------------------------
# Public API
# -----------------------------
def extract_metadata(file_path: str) -> Dict[str, str]:
    """
    Extract basic PDF metadata for citations.
    - Try embedded PDF metadata first
    - Fallback to Page 1 text parsing (common for lecture slides)
    """
    meta: Dict[str, str] = {"title": "", "author": "", "year": ""}

    try:
        doc = fitz.open(file_path)

        # --- Embedded metadata ---
        md = doc.metadata or {}
        title = (md.get("title") or "").strip()
        author = (md.get("author") or "").strip()

        # Attempt year from metadata dates like "D:20251026..."
        year = ""
        for k in ("creationDate", "modDate"):
            raw = md.get(k) or ""
            m = re.search(r"(19|20)\d{2}", raw)
            if m:
                year = m.group(0)
                break

        meta["title"] = title
        meta["author"] = author
        meta["year"] = year

        # --- Fallback using first page ---
        fallback = _parse_first_page_fallback(doc)

        # Replace suspicious/missing title/author with better first-page values
        if _looks_suspicious_title(meta["title"]) and fallback.get("title"):
            meta["title"] = fallback["title"]

        if _looks_suspicious_author(meta["author"]) and fallback.get("author"):
            meta["author"] = fallback["author"]

        # If year still missing, keep empty (your pipeline handles "")
        doc.close()

    except Exception:
        pass

    # Final fallbacks
    if not meta["title"]:
        meta["title"] = os.path.basename(file_path)

    if not meta["author"]:
        meta["author"] = "Unknown Author"

    return meta


def read_pdf_chunks(
    file_path: str,
    chunk_size: int = 1200,
    overlap: int = 200,
    min_page_chars_for_text: int = 30,
    use_ocr_if_empty: bool = True,
) -> List[Dict[str, Any]]:
    """
    Read a PDF and return a list of chunk dicts like:

    {
      "chunk_id": "chunk_1",
      "text": "...",
      "metadata": {
          "page": 18,
          "paragraph_id": "...",
          "location": "Page 18, Paragraph 1"
      }
    }

    IMPORTANT: This function does NOT require document_id.
    document_id should be injected later by your ingestion pipeline.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(file_path)
    all_chunks: List[Dict[str, Any]] = []
    chunk_counter = 0

    for page_index in range(doc.page_count):
        page_number = page_index + 1
        page = doc[page_index]

        raw_text = page.get_text("text") or ""
        raw_text = _clean_text(raw_text)

        # OCR fallback only if almost empty
        if use_ocr_if_empty and len(raw_text) < min_page_chars_for_text:
            ocr_text = _clean_text(_ocr_page_to_text(doc, page_index))
            if len(ocr_text) > len(raw_text):
                raw_text = ocr_text

        page_chunks = _chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)

        for j, ch_text in enumerate(page_chunks, start=1):
            chunk_counter += 1
            paragraph_id = uuid.uuid4().hex[:8]
            all_chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_counter}",
                    "text": ch_text,
                    "metadata": {
                        "page": page_number,
                        "paragraph_id": paragraph_id,
                        "location": f"Page {page_number}, Paragraph {j}",
                    },
                }
            )

    doc.close()
    return all_chunks
