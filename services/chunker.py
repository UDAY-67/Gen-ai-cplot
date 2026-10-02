"""
Text Chunking Service
Splits document text into manageable, overlapping chunks with rich metadata.
Built from first principles without LangChain.
"""

from typing import List, Dict, Any
from utils.helpers import estimate_tokens


def split_text_into_chunks(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[str]:
    """
    Split a long string of text into overlapping segments using paragraph and sentence boundaries.

    Args:
        text: Input string to chunk.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a clean breaking point (paragraph -> sentence -> word)
        # Search backwards from target end to avoid breaking words or sentences mid-way
        search_window = text[max(start, end - 100):min(text_length, end + 50)]
        
        # Look for double newline (paragraph break)
        para_break = text.rfind("\n\n", start, end)
        if para_break != -1 and para_break > start + (chunk_size // 2):
            split_pos = para_break + 2
        else:
            # Look for sentence boundary (. ! ?)
            sentence_break = -1
            for punct in [". ", "? ", "! ", ".\n"]:
                pos = text.rfind(punct, start, end)
                if pos > sentence_break:
                    sentence_break = pos + len(punct)

            if sentence_break != -1 and sentence_break > start + (chunk_size // 2):
                split_pos = sentence_break
            else:
                # Fallback to whitespace boundary
                space_pos = text.rfind(" ", start, end)
                if space_pos != -1 and space_pos > start:
                    split_pos = space_pos + 1
                else:
                    split_pos = end

        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)

        # Advance start position with overlap
        start = split_pos - chunk_overlap
        if start < 0 or start >= split_pos:
            start = split_pos

    return chunks


def chunk_document_pages(
    pages_data: List[Dict[str, Any]],
    filename: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Process page-by-page extracted PDF data and generate structured chunks with metadata.

    Args:
        pages_data: List of dicts with 'page_number' and 'text'.
        filename: Name of the source PDF.
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.

    Returns:
        List of chunk dicts with {chunk_id, doc_name, page_number, text, char_count, token_estimate}.
    """
    all_chunks: List[Dict[str, Any]] = []
    global_chunk_idx = 1

    for page in pages_data:
        page_num = page["page_number"]
        page_text = page["text"]

        if not page_text:
            continue

        raw_chunks = split_text_into_chunks(page_text, chunk_size, chunk_overlap)

        for chunk_text in raw_chunks:
            if not chunk_text.strip():
                continue

            all_chunks.append({
                "chunk_id": global_chunk_idx,
                "doc_name": filename,
                "page_number": page_num,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "token_estimate": estimate_tokens(chunk_text)
            })
            global_chunk_idx += 1

    return all_chunks
