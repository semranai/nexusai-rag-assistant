# pdf_loader.py
import os
import re
import uuid
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF


def _env_true(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


# OCR flags
DISABLE_OCR = _env_true("DISABLE_OCR", "1")  # default: disabled
ENABLE_OCR = _env_true("ENABLE_OCR", "0") and (not DISABLE_OCR)

# Optional: Windows path to tesseract.exe
TESSERACT_CMD = (os.getenv("TESSERACT_CMD") or "").strip()


def _looks_like_low_signal_text(text: str) -> bool:
    """
    Detect pages where extracted text exists but is basically useless
    (headers/course codes/page numbers) while main content is likely an image.

    Heuristics:
    - very low alphanumeric content
    - mostly short tokens / numbers
    - looks like slide footer/header only
    """
    if not text:
        return True

    t = re.sub(r"\s+", " ", text).strip()
    if len(t) < 40:
        return True

    # Ratio of alphanumeric chars
    alnum = sum(ch.isalnum() for ch in t)
    ratio = alnum / max(1, len(t))

    # If mostly non-alphanumeric -> low signal
    if ratio < 0.35:
        return True

    # If it looks like just course/page markers
    footer_signals = ["420-", "page", "chapter", "presented by"]
    footer_hits = sum(1 for s in footer_signals if s in t.lower())

    if len(t) < 120 and footer_hits >= 1:
        return True

    return False


def _try_ocr_page(page: "fitz.Page", dpi: int = 250) -> str:
    """
    Best-effort OCR. If tesseract isn't installed or misconfigured, return "".
    """
    try:
        import pytesseract
        from PIL import Image

        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(img, config=config)

        return (text or "").strip()
    except Exception:
        return ""


def _read_first_page_text(file_path: str, max_chars: int = 3000) -> str:
    """
    Reads first page text only. Safe and fast.
    """
    try:
        doc = fitz.open(file_path)
        if doc.page_count < 1:
            doc.close()
            return ""
        page = doc.load_page(0)
        text = (page.get_text("text") or "").strip()
        doc.close()
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""


def _extract_presenter_or_author(text: str) -> str:
    """
    Tries to extract presenter/author from common cover-page patterns.
    """
    if not text:
        return ""

    # Presented by: Name
    m = re.search(r"Presented\s*by\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        name = name.split("|")[0].strip()
        return re.sub(r"\s{2,}", " ", name).strip()

    # Author: Name
    m = re.search(r"Author\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if m:
        return re.sub(r"\s{2,}", " ", m.group(1).strip())

    # By Name (common papers)
    m = re.search(r"\bBy\s+([A-Z][A-Za-z\-\s]{2,80})", text)
    if m:
        return re.sub(r"\s{2,}", " ", m.group(1).strip())

    return ""


def _extract_year_from_text(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"(19|20)\d{2}", text)
    return m.group(0) if m else ""


def extract_metadata(file_path: str) -> Dict[str, str]:
    """
    Safe metadata extraction. Never crashes ingestion.
    Also tries to improve author/year using first-page text as fallback.
    """
    meta = {"title": "", "author": "", "year": ""}

    try:
        doc = fitz.open(file_path)
        md = doc.metadata or {}

        meta["title"] = (md.get("title") or "").strip()
        meta["author"] = (md.get("author") or "").strip()

        embedded_year = ""
        for k in ["creationDate", "modDate"]:
            v = (md.get(k) or "").strip()
            m = re.search(r"(19|20)\d{2}", v)
            if m:
                embedded_year = m.group(0)
                break
        meta["year"] = embedded_year

        doc.close()
    except Exception:
        pass

    # --- Fallback enhancement from first page text ---
    # (Do not overwrite good metadata; only fill missing fields)
    first_text = _read_first_page_text(file_path)
    if first_text:
        if not meta["author"]:
            author = _extract_presenter_or_author(first_text)
            if author:
                meta["author"] = author

        if not meta["year"]:
            year = _extract_year_from_text(first_text)
            if year:
                meta["year"] = year

        # If metadata title is missing or generic, at least set something stable
        if not meta["title"]:
            # Use filename as safe fallback; title refinement happens in main.py override
            meta["title"] = os.path.basename(file_path)

    return meta


def read_pdf_chunks(
    file_path: str,
    max_pages: Optional[int] = None,
    min_text_chars_per_page: int = 30,
    ocr_dpi: int = 250,
) -> List[Dict[str, Any]]:
    """
    Reads PDF and returns chunks like:
      { "chunk_id": "chunk_1", "text": "...", "metadata": {"page": 1, ...} }

    Strategy:
    1) Extract normal text
    2) If OCR enabled and text is short OR low-signal -> OCR the page
    """
    chunks: List[Dict[str, Any]] = []
    doc = fitz.open(file_path)

    total_pages = doc.page_count
    limit = total_pages if max_pages is None else min(total_pages, max_pages)

    for page_index in range(limit):
        page = doc.load_page(page_index)
        page_number = page_index + 1

        text = (page.get_text("text") or "").strip()

        should_ocr = False
        if ENABLE_OCR:
            if len(text) < min_text_chars_per_page:
                should_ocr = True
            elif _looks_like_low_signal_text(text):
                should_ocr = True

        if should_ocr:
            ocr_text = _try_ocr_page(page, dpi=ocr_dpi)
            if ocr_text and len(ocr_text) > len(text):
                text = ocr_text
            elif ocr_text and len(text) < 80:
                text = ocr_text

        if not text:
            continue

        chunk_id = f"chunk_{page_number}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "metadata": {
                    "page": page_number,
                    "location": f"Page {page_number}",
                    "paragraph_id": uuid.uuid4().hex[:8],
                },
            }
        )

    doc.close()
    return chunks