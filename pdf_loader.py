# pdf_loader.py
import os
import re
import uuid
from typing import Any, Dict, List, Optional
import fitz  # PyMuPDF

ENABLE_OCR = os.getenv("ENABLE_OCR", "false").strip().lower() == "true"

def _env_true(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


# OCR is OFF by default on servers like Render (no tesseract binary).
DISABLE_OCR = _env_true("DISABLE_OCR", "1")  # default: disabled
ENABLE_OCR = _env_true("ENABLE_OCR", "0") and (not DISABLE_OCR)


def extract_metadata(file_path: str) -> Dict[str, str]:
    """
    Safe metadata extraction. Never crashes ingestion.
    """
    meta = {"title": "", "author": "", "year": ""}
    try:
        doc = fitz.open(file_path)
        md = doc.metadata or {}
        doc.close()

        title = (md.get("title") or "").strip()
        author = (md.get("author") or "").strip()

        # Try to find a year in metadata if present
        year = ""
        for k in ["creationDate", "modDate"]:
            v = (md.get(k) or "").strip()
            # PDF date looks like: D:20250101120000
            m = re.search(r"(19|20)\d{2}", v)
            if m:
                year = m.group(0)
                break

        meta.update({"title": title, "author": author, "year": year})
    except Exception:
        # keep empty defaults
        pass
    return meta


def _try_ocr_page(pix: "fitz.Pixmap") -> str:
    """
    Best-effort OCR. If tesseract isn't installed, return "" and DO NOT crash.
    """
    try:
        # Import only when needed (prevents hard failure on Render)
        import pytesseract
        from PIL import Image
        from pytesseract import TesseractNotFoundError

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img)

        return (text or "").strip()
    except Exception as e:
        # Common case on Render: TesseractNotFoundError
        # We swallow it so ingestion continues with normal text extraction.
        return ""


def read_pdf_chunks(
    file_path: str,
    max_pages: Optional[int] = None,
    min_text_chars_per_page: int = 30,
) -> List[Dict[str, Any]]:
    """
    Reads PDF and returns chunks like:
      { "chunk_id": "chunk_1", "text": "...", "metadata": {"page": 1, ...} }

    Uses PyMuPDF text extraction.
    OCR is optional and will NEVER crash the pipeline.
    """
    chunks: List[Dict[str, Any]] = []
    doc = fitz.open(file_path)

    total_pages = doc.page_count
    limit = total_pages if max_pages is None else min(total_pages, max_pages)

    for page_index in range(limit):
        page = doc.load_page(page_index)
        page_number = page_index + 1

        text = (page.get_text("text") or "").strip()

        # If OCR enabled AND page has too little text, attempt OCR (best-effort)
        text = (text or "").strip()

        if ENABLE_OCR and len(text) < min_text_chars_per_page:
            pix = page.get_pixmap(dpi=200)
            ocr_text = _try_ocr_page(pix)
            if ocr_text:
                text = ocr_text


        if not text:
            # skip empty pages
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
