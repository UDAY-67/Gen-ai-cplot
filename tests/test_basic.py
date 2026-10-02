"""
Basic Unit Tests for GenAI Study Copilot
Validates database, text chunker, PDF processor, helper utilities, and vector store.
"""

import os
import io
import pytest
import numpy as np
from pathlib import Path

from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from database.db import (
    init_db,
    create_conversation,
    get_conversations,
    add_message,
    get_messages,
    delete_conversation
)
from services.chunker import split_text_into_chunks, chunk_document_pages
from utils.helpers import clean_text, extract_json, estimate_tokens
from services.vector_store import FAISSVectorStore
from services.pdf import extract_text_from_pdf
from utils.generate_sample_pdf import generate_minimal_pdf


def test_helpers():
    # Test clean text
    dirty = "Hello \r\n\r\n\r\n  World! \t Test  "
    assert clean_text(dirty) == "Hello\n\nWorld! Test"

    # Test token estimator
    assert estimate_tokens("Hello world, this is a test string.") > 0

    # Test extract_json with markdown fences
    md_json = '```json\n{"status": "ok", "count": 5}\n```'
    parsed = extract_json(md_json)
    assert parsed == {"status": "ok", "count": 5}


def test_chunker():
    sample_text = (
        "Gradient descent is an optimization algorithm used to minimize a cost function. "
        "It iteratively moves toward the minimum value of the function. "
        "The learning rate determines the size of the steps taken toward the minimum. "
        "If the learning rate is too large, the algorithm might overshoot the minimum. "
        "If the learning rate is too small, convergence will be slow."
    )
    chunks = split_text_into_chunks(sample_text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2
    assert all(len(c) > 0 for c in chunks)

    # Test document page chunking
    pages_data = [
        {"page_number": 1, "text": "Page one text about Machine Learning fundamentals."},
        {"page_number": 2, "text": "Page two text about Deep Learning and Backpropagation."}
    ]
    doc_chunks = chunk_document_pages(pages_data, "notes.pdf", chunk_size=80, chunk_overlap=15)
    assert len(doc_chunks) >= 2
    assert doc_chunks[0]["doc_name"] == "notes.pdf"
    assert doc_chunks[0]["page_number"] == 1


def test_database_crud(tmp_path):
    test_db = tmp_path / "test.db"
    init_db(test_db)

    conv_id = create_conversation("Test Session", mode="general", db_path=test_db)
    assert conv_id is not None

    add_message(conv_id, "user", "What is backpropagation?", db_path=test_db)
    add_message(conv_id, "assistant", "Backpropagation is the reverse pass of calculating gradients.", db_path=test_db)

    msgs = get_messages(conv_id, db_path=test_db)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"

    delete_conversation(conv_id, db_path=test_db)
    assert len(get_messages(conv_id, db_path=test_db)) == 0


def test_faiss_vector_store():
    store = FAISSVectorStore(dimension=4)
    # 3 dummy chunks
    chunks = [
        {"chunk_id": 1, "doc_name": "doc1.pdf", "page_number": 1, "text": "Linear Regression"},
        {"chunk_id": 2, "doc_name": "doc1.pdf", "page_number": 2, "text": "Logistic Regression"},
        {"chunk_id": 3, "doc_name": "doc2.pdf", "page_number": 1, "text": "Convolutional Neural Networks"}
    ]
    # 3 normalized dummy vectors
    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype="float32")

    store.add_chunks(chunks, vectors)
    assert store.get_total_chunks() == 3

    # Query vector close to vector 0 and 1
    query = np.array([[0.95, 0.05, 0.0, 0.0]], dtype="float32")
    results = store.similarity_search(query, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] in (1, 2)
    assert "score" in results[0]


def test_pdf_extraction_and_chunking(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    generate_minimal_pdf(str(pdf_path))

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pages, stats = extract_text_from_pdf(pdf_bytes, "sample.pdf")
    assert stats["page_count"] == 3
    assert stats["char_count"] > 0
    assert len(pages) == 3
    assert "Gradient Descent" in pages[0]["text"]
