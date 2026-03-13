import io
import logging
from typing import Dict, List

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def load_pdf_document(uploaded_file) -> str:
    try:
        pdf_bytes = uploaded_file.getvalue()
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()
    except Exception as exc:
        logger.exception("Failed to load PDF %s: %s", getattr(uploaded_file, "name", "unknown"), exc)
        return ""


def load_text_document(uploaded_file) -> str:
    try:
        return uploaded_file.getvalue().decode("utf-8", errors="ignore").strip()
    except Exception as exc:
        logger.exception("Failed to load text file %s: %s", getattr(uploaded_file, "name", "unknown"), exc)
        return ""


def load_documents(uploaded_files) -> List[Dict[str, str]]:
    documents = []
    for uploaded_file in uploaded_files or []:
        try:
            file_name = uploaded_file.name
            lower_name = file_name.lower()

            if lower_name.endswith(".pdf"):
                text = load_pdf_document(uploaded_file)
            elif lower_name.endswith((".txt", ".md")):
                text = load_text_document(uploaded_file)
            else:
                text = ""

            if text:
                documents.append({"name": file_name, "text": text})
        except Exception as exc:
            logger.exception("Error while processing uploaded file: %s", exc)
    return documents
