import os
import re

import fitz


def normalize_text(text):
    """Clean text while preserving important technical terms."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf(pdf_source):
    """Extract text from a PDF file or uploaded file-like object."""
    try:
        if hasattr(pdf_source, "read"):
            pdf_source.seek(0)
            pdf_bytes = pdf_source.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        else:
            file_path = os.fspath(pdf_source)
            if not os.path.exists(file_path):
                raise FileNotFoundError(file_path)
            doc = fitz.open(file_path)

        pages = []
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                pages.append(page_text)
        doc.close()

        extracted_text = "\n".join(pages)
        cleaned = normalize_text(extracted_text)
        if not cleaned:
            raise ValueError(
                "The uploaded PDF has no extractable text. Scanned PDFs may require OCR."
            )
        return cleaned
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"PDF file not found: {pdf_source}") from exc
    except (fitz.FileDataError, RuntimeError, ValueError, TypeError):
        raise ValueError(
            "Invalid or unreadable PDF file. Please upload a valid PDF or text-based resume."
        )
