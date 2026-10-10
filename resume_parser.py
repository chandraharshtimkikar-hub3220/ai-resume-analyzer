import os
import re
from pathlib import Path

import fitz


def normalize_text(text):
    """Clean and normalize text without losing technical terms."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def extract_text_from_pdf(pdf_path_or_file):
    """Extract text from a PDF and raise a useful error on invalid files."""
    try:
        if hasattr(pdf_path_or_file, "read"):
            uploaded_file = pdf_path_or_file
            uploaded_file.seek(0)
            pdf_bytes = uploaded_file.read()
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        else:
            file_path = str(pdf_path_or_file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            pdf_doc = fitz.open(file_path)

        extracted_pages = []
        for page in pdf_doc:
            page_text = page.get_text("text")
            if page_text:
                extracted_pages.append(page_text)

        pdf_doc.close()

        text = "\n".join(extracted_pages)
        cleaned_text = normalize_text(text)

        if not cleaned_text:
            raise ValueError(
                "The uploaded PDF has no extractable text. Scanned PDFs may require OCR."
            )

        return cleaned_text

    except FileNotFoundError:
        raise
    except (fitz.FileDataError, ValueError, RuntimeError, TypeError):
        raise ValueError(
            "Invalid or unreadable PDF file. Please upload a valid PDF or try a text-based resume."
        )


if __name__ == "__main__":
    sample_path = Path("data") / "sample.pdf"
    if sample_path.exists():
        print(extract_text_from_pdf(sample_path)[:200])
