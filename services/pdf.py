"""
PDF Extraction Service
Handles PDF file processing, page-by-page text extraction, and metadata gathering.
"""

from typing import List, Dict, Any, Tuple
import io
import pypdf
from utils.helpers import clean_text


class PDFProcessingError(Exception):
    """Custom exception raised when PDF processing fails."""
    pass


def extract_text_from_pdf(
    file_bytes_or_path: Any,
    filename: str = "document.pdf"
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract text from a PDF file page by page.

    Args:
        file_bytes_or_path: Bytes, file-like object, or file path.
        filename: Name of the PDF file.

    Returns:
        Tuple containing:
        - pages: List of dicts with 'page_number' and 'text'.
        - stats: Dict with summary statistics (page_count, char_count, word_count, filename).

    Raises:
        PDFProcessingError: If the PDF is corrupted, empty, or unreadable.
    """
    try:
        if isinstance(file_bytes_or_path, (bytes, bytearray)):
            stream = io.BytesIO(file_bytes_or_path)
        elif hasattr(file_bytes_or_path, "read"):
            stream = file_bytes_or_path
        else:
            stream = open(str(file_bytes_or_path), "rb")

        reader = pypdf.PdfReader(stream)
        num_pages = len(reader.pages)

        if num_pages == 0:
            raise PDFProcessingError(f"PDF '{filename}' contains 0 pages.")

        pages_data: List[Dict[str, Any]] = []
        total_chars = 0
        total_words = 0

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            raw_text = page.extract_text() or ""
            cleaned = clean_text(raw_text)

            pages_data.append({
                "page_number": page_num,
                "text": cleaned,
                "char_count": len(cleaned),
                "word_count": len(cleaned.split()) if cleaned else 0
            })

            total_chars += len(cleaned)
            total_words += len(cleaned.split()) if cleaned else 0

        # Check if PDF is scanned or has no extractable text
        if total_chars == 0:
            raise PDFProcessingError(
                f"No extractable text found in '{filename}'. "
                "The PDF might be scanned/image-only or encrypted."
            )

        stats = {
            "filename": filename,
            "page_count": num_pages,
            "char_count": total_chars,
            "word_count": total_words,
            "has_text": total_chars > 0
        }

        return pages_data, stats

    except PDFProcessingError:
        raise
    except Exception as e:
        raise PDFProcessingError(f"Failed to process PDF '{filename}': {str(e)}")
