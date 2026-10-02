"""
RAG (Retrieval-Augmented Generation) Service
Orchestrates vector search, context construction, and grounded Mistral LLM generation.
"""

from typing import List, Dict, Any, Tuple, Generator, Optional
from prompts.prompts import RAG_SYSTEM_PROMPT
from services.embeddings import generate_query_embedding
from services.vector_store import FAISSVectorStore
from services.llm import generate_chat_response, stream_chat_response, LLMError
from config import DEFAULT_TOP_K, DEFAULT_MISTRAL_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS


def build_rag_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a clear, numbered context block with source citations.

    Args:
        retrieved_chunks: List of chunk dicts returned by vector store similarity search.

    Returns:
        Formatted context string.
    """
    if not retrieved_chunks:
        return "No relevant context found in uploaded documents."

    context_parts = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        doc = chunk.get("doc_name", "Document")
        page = chunk.get("page_number", "Unknown")
        cid = chunk.get("chunk_id", idx)
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "").strip()

        header = f"--- [Excerpt #{idx} | Source: {doc} (Page {page}, Chunk #{cid}) | Relevance Score: {score:.2f}] ---"
        context_parts.append(f"{header}\n{text}")

    return "\n\n".join(context_parts)


def answer_question_with_rag(
    question: str,
    vector_store: FAISSVectorStore,
    model: str = DEFAULT_MISTRAL_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    top_k: int = DEFAULT_TOP_K,
    api_key: Optional[str] = None,
    stream: bool = True
) -> Tuple[Any, List[Dict[str, Any]], str]:
    """
    Execute full RAG pipeline:
    1. Embed user question.
    2. Retrieve top-K chunks from FAISS.
    3. Construct grounded prompt.
    4. Call Mistral AI (streaming or non-streaming).

    Args:
        question: User query.
        vector_store: Initialized FAISSVectorStore.
        model: Mistral model name.
        temperature: Generation temperature.
        max_tokens: Max output tokens.
        top_k: Number of chunks to retrieve.
        api_key: Mistral API key.
        stream: Whether to stream response tokens.

    Returns:
        Tuple of (response_stream_or_str, retrieved_chunks, context_text).
    """
    if vector_store.get_total_chunks() == 0:
        raise ValueError("No study documents have been indexed yet. Please upload a PDF first.")

    # Step 1: Embed question
    query_vector = generate_query_embedding(question)

    # Step 2: Retrieve top-K chunks
    retrieved_chunks = vector_store.similarity_search(query_vector, top_k=top_k)

    if not retrieved_chunks:
        raise ValueError("Vector search returned 0 matching results for your query.")

    # Step 3: Build grounded context
    context_text = build_rag_context(retrieved_chunks)

    # Step 4: Construct messages
    system_content = RAG_SYSTEM_PROMPT.format(context=context_text, question=question)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question}
    ]

    # Step 5: Generate response via Mistral AI
    if stream:
        response_generator = stream_chat_response(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )
        return response_generator, retrieved_chunks, context_text
    else:
        answer_text = generate_chat_response(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )
        return answer_text, retrieved_chunks, context_text
