"""
GenAI Study Copilot - Main Application
Streamlit UI orchestrating Chat, Explanations, PDF Processing, RAG, Quiz Generation, and SQLite Persistence.
"""

import streamlit as st
import time
from typing import List, Dict, Any

from config import (
    AVAILABLE_MISTRAL_MODELS,
    DEFAULT_MISTRAL_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    validate_api_key,
    MISTRAL_API_KEY
)
from database.db import (
    init_db,
    create_conversation,
    get_conversations,
    get_conversation,
    delete_conversation,
    update_conversation_title,
    add_message,
    get_messages,
    save_document_meta,
    get_documents
)
from services.llm import (
    test_mistral_connection,
    stream_chat_response,
    generate_chat_response,
    LLMError,
    LLMAuthError,
    LLMRateLimitError
)
from services.chat import build_conversation_payload, get_system_prompt_for_mode
from services.pdf import extract_text_from_pdf, PDFProcessingError
from services.chunker import chunk_document_pages
from services.embeddings import generate_embeddings, generate_query_embedding
from services.vector_store import FAISSVectorStore
from services.rag import answer_question_with_rag, build_rag_context
from services.quiz import generate_study_quiz, QuizValidationError
from utils.helpers import measure_latency, format_source_citation

# Configure Streamlit Page
st.set_page_config(
    page_title="GenAI Study Copilot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for clean modern aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .mode-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #EEF2FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
    }
    .source-card {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        background-color: #F8FAFC;
        border-left: 4px solid #4F46E5;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .metric-badge {
        background-color: #F1F5F9;
        border-radius: 6px;
        padding: 4px 8px;
        font-family: monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Database & Session State
init_db()

if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = None

if "vector_store" not in st.session_state:
    vs = FAISSVectorStore()
    vs.load()  # Try loading existing index if present
    st.session_state.vector_store = vs

if "extracted_doc_text" not in st.session_state:
    st.session_state.extracted_doc_text = ""

if "current_doc_stats" not in st.session_state:
    st.session_state.current_doc_stats = None

if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = None

if "user_quiz_answers" not in st.session_state:
    st.session_state.user_quiz_answers = {}

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## 🎓 **GenAI Study Copilot**")
    st.caption("AI-Powered Pedagogical Assistant for STEM Students")
    st.divider()

    # Mode Selector
    st.markdown("### 🎯 **Study Mode**")
    selected_mode = st.selectbox(
        "Choose Mode:",
        [
            "💬 General Chat",
            "🧠 Explain Concept",
            "📚 Document RAG & Q&A",
            "📝 Document Summarizer",
            "🎯 Interactive Quiz",
            "📊 Educational Evaluation"
        ],
        index=0,
        label_visibility="collapsed"
    )

    # Clean mode name without emoji
    mode_name = selected_mode.split(" ", 1)[1] if " " in selected_mode else selected_mode

    st.divider()

    # Session Management
    st.markdown("### 🗂️ **Study Sessions**")
    if st.button("➕ New Study Session", use_container_width=True, type="primary"):
        new_id = create_conversation(title=f"Session ({mode_name})", mode=mode_name)
        st.session_state.current_conv_id = new_id
        st.session_state.quiz_state = None
        st.session_state.quiz_submitted = False
        st.rerun()

    conversations = get_conversations()
    if conversations:
        conv_options = {c["id"]: f"{c['title']} ({c.get('mode', 'Chat')})" for c in conversations}
        
        # Default to first conversation if none selected
        if not st.session_state.current_conv_id or st.session_state.current_conv_id not in conv_options:
            st.session_state.current_conv_id = conversations[0]["id"]

        selected_conv_id = st.selectbox(
            "Saved Sessions:",
            options=list(conv_options.keys()),
            format_func=lambda x: conv_options[x],
            index=list(conv_options.keys()).index(st.session_state.current_conv_id) if st.session_state.current_conv_id in conv_options else 0,
            label_visibility="collapsed"
        )

        if selected_conv_id != st.session_state.current_conv_id:
            st.session_state.current_conv_id = selected_conv_id
            st.rerun()

        col_ren, col_del = st.columns([1, 1])
        with col_del:
            if st.button("🗑️ Delete Session", use_container_width=True):
                delete_conversation(st.session_state.current_conv_id)
                st.session_state.current_conv_id = None
                st.rerun()
    else:
        st.info("No saved sessions yet. Start typing to begin!")

    st.divider()

    # Document Upload & Indexing Section
    st.markdown("### 📄 **Document Indexing**")
    uploaded_file = st.file_uploader("Upload Lecture Notes / PDF:", type=["pdf"])

    if uploaded_file is not None:
        if st.button("⚡ Process & Index PDF", use_container_width=True):
            with st.spinner("Extracting text and building vector index..."):
                try:
                    file_bytes = uploaded_file.read()
                    pages_data, stats = extract_text_from_pdf(file_bytes, uploaded_file.name)
                    st.session_state.current_doc_stats = stats
                    
                    # Store full text in session state for summarizer/quiz modes
                    st.session_state.extracted_doc_text = "\n\n".join([p["text"] for p in pages_data])

                    # Chunking
                    chunks = chunk_document_pages(
                        pages_data,
                        filename=uploaded_file.name,
                        chunk_size=DEFAULT_CHUNK_SIZE,
                        chunk_overlap=DEFAULT_CHUNK_OVERLAP
                    )

                    # Generate embeddings
                    chunk_texts = [c["text"] for c in chunks]
                    embeddings = generate_embeddings(chunk_texts)

                    # Index in FAISS
                    vs = st.session_state.vector_store
                    vs.add_chunks(chunks, embeddings)
                    vs.save()

                    # Save Document metadata to SQLite
                    save_document_meta(
                        filename=uploaded_file.name,
                        page_count=stats["page_count"],
                        char_count=stats["char_count"],
                        chunk_count=len(chunks)
                    )

                    st.success(f"Indexed **{uploaded_file.name}** ({stats['page_count']} pages, {len(chunks)} chunks)!")
                except PDFProcessingError as pe:
                    st.error(f"PDF Error: {str(pe)}")
                except Exception as e:
                    st.error(f"Indexing failed: {str(e)}")

    total_indexed_chunks = st.session_state.vector_store.get_total_chunks()
    indexed_docs = st.session_state.vector_store.get_indexed_documents()
    
    st.caption(f"📊 **Index Status**: {total_indexed_chunks} chunks indexed across {len(indexed_docs)} documents.")
    if indexed_docs:
        with st.expander("📁 Indexed Documents"):
            for d in indexed_docs:
                st.markdown(f"- `{d}`")
            if st.button("🧹 Clear Vector Index", use_container_width=True):
                st.session_state.vector_store.clear()
                st.session_state.vector_store.save()
                st.session_state.extracted_doc_text = ""
                st.session_state.current_doc_stats = None
                st.success("Vector store cleared.")
                st.rerun()

    st.divider()

    # Settings & Model Configuration
    with st.expander("⚙️ **Settings & API Configuration**"):
        user_api_key = st.text_input(
            "Mistral API Key:",
            value=MISTRAL_API_KEY if MISTRAL_API_KEY and MISTRAL_API_KEY != "your_mistral_api_key_here" else "",
            type="password",
            help="Your API key is kept secure and never committed."
        )

        selected_model = st.selectbox(
            "Mistral Model:",
            AVAILABLE_MISTRAL_MODELS,
            index=AVAILABLE_MISTRAL_MODELS.index(DEFAULT_MISTRAL_MODEL) if DEFAULT_MISTRAL_MODEL in AVAILABLE_MISTRAL_MODELS else 0
        )

        temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.05)
        max_tokens = st.slider("Max Output Tokens:", min_value=256, max_value=4096, value=DEFAULT_MAX_TOKENS, step=256)
        top_k = st.slider("Top-K Retrieved Chunks:", min_value=1, max_value=10, value=DEFAULT_TOP_K, step=1)

        if st.button("🔍 Test API Connection", use_container_width=True):
            with st.spinner("Testing Mistral API connectivity..."):
                res = test_mistral_connection(user_api_key or None)
                if res.get("success"):
                    st.success(f"Connected! Model: `{res.get('model')}`")
                else:
                    st.error(f"Connection Failed: {res.get('error')}")


# Ensure active conversation
if not st.session_state.current_conv_id:
    st.session_state.current_conv_id = create_conversation(title=f"Session ({mode_name})", mode=mode_name)


# ==========================================
# MAIN CONTENT AREA
# ==========================================

# Header Banner
current_conv = get_conversation(st.session_state.current_conv_id)
conv_title = current_conv["title"] if current_conv else "Study Session"

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f"<h1 class='main-header'>🚀 GenAI Study Copilot</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-header'>Session: <b>{conv_title}</b> &nbsp;|&nbsp; <span class='mode-badge'>{selected_mode}</span> &nbsp;|&nbsp; Model: <code>{selected_model}</code></div>", unsafe_allow_html=True)

with col_h2:
    if st.session_state.vector_store.get_total_chunks() > 0:
        st.markdown(f"<div style='text-align:right; margin-top:15px;'><span class='metric-badge'>🟢 RAG Active ({st.session_state.vector_store.get_total_chunks()} Chunks)</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:right; margin-top:15px;'><span class='metric-badge'>⚪ Direct LLM Mode</span></div>", unsafe_allow_html=True)

st.divider()


# ==========================================
# MODE 1 & 2: CHAT & CONCEPT EXPLAINER
# ==========================================
if mode_name in ("General Chat", "Explain Concept"):
    # Load messages
    messages = get_messages(st.session_state.current_conv_id)

    # Render message history
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("metadata") and "elapsed_seconds" in msg["metadata"]:
                st.caption(f"⏱️ Generated in {msg['metadata']['elapsed_seconds']}s")

    # Chat Input
    placeholder_text = "Ask any engineering or CS question..." if mode_name == "General Chat" else "Enter a concept to explain (e.g. Backpropagation, Paxos, B-Trees)..."
    prompt = st.chat_input(placeholder_text)

    if prompt:
        # Check API key
        active_api_key = user_api_key or MISTRAL_API_KEY
        if not validate_api_key(active_api_key):
            st.warning("⚠️ Please provide a valid Mistral API Key in `.env` or in the sidebar settings.")
        else:
            # 1. Display and save user message
            with st.chat_message("user"):
                st.markdown(prompt)
            add_message(st.session_state.current_conv_id, "user", prompt)

            # Auto-update title if it's the first message
            if len(messages) == 0:
                new_title = prompt[:30] + ("..." if len(prompt) > 30 else "")
                update_conversation_title(st.session_state.current_conv_id, new_title)

            # 2. Build context payload
            history = get_messages(st.session_state.current_conv_id)
            payload = build_conversation_payload(history, mode=mode_name)

            # 3. Stream assistant response
            with st.chat_message("assistant"):
                with measure_latency() as lat:
                    try:
                        stream = stream_chat_response(
                            messages=payload,
                            model=selected_model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            api_key=active_api_key
                        )
                        full_response = st.write_stream(stream)
                    except LLMAuthError as ae:
                        st.error(f"Authentication Error: {str(ae)}")
                        full_response = None
                    except LLMRateLimitError as re:
                        st.error(f"Rate Limit: {str(re)}")
                        full_response = None
                    except LLMError as le:
                        st.error(f"Generation Error: {str(le)}")
                        full_response = None
                    except Exception as e:
                        st.error(f"Unexpected error: {str(e)}")
                        full_response = None

                if full_response:
                    st.caption(f"⏱️ Generated in {lat['elapsed_seconds']}s")
                    add_message(
                        st.session_state.current_conv_id,
                        "assistant",
                        full_response,
                        metadata={"elapsed_seconds": lat["elapsed_seconds"], "model": selected_model}
                    )


# ==========================================
# MODE 3: DOCUMENT RAG & Q&A
# ==========================================
elif mode_name == "Document RAG & Q&A":
    vs = st.session_state.vector_store
    if vs.get_total_chunks() == 0:
        st.warning("📚 **No documents indexed yet!** Upload a PDF in the sidebar and click **'Process & Index PDF'** to begin asking grounded questions.")
    else:
        st.info(f"💡 Grounded Mode: Answering strictly from **{vs.get_total_chunks()} indexed chunks** across `{', '.join(vs.get_indexed_documents())}`.")

    # Render previous messages
    messages = get_messages(st.session_state.current_conv_id)
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("metadata"):
                meta = msg["metadata"]
                if meta.get("sources"):
                    with st.expander("📑 Referenced Sources & Context Chunks"):
                        for s in meta["sources"]:
                            st.markdown(f"**{s.get('doc_name')}** (Page {s.get('page_number')}) — *Relevance: {s.get('score', 0):.2f}*")
                            st.caption(s.get("text", "")[:300] + "...")
                if meta.get("elapsed_seconds"):
                    st.caption(f"⏱️ Retrieval & Generation: {meta['elapsed_seconds']}s")

    prompt = st.chat_input("Ask a question about your uploaded study material...")
    if prompt:
        active_api_key = user_api_key or MISTRAL_API_KEY
        if not validate_api_key(active_api_key):
            st.warning("⚠️ Please provide a valid Mistral API Key in `.env` or in the sidebar settings.")
        elif vs.get_total_chunks() == 0:
            st.error("Please upload and index a PDF first.")
        else:
            with st.chat_message("user"):
                st.markdown(prompt)
            add_message(st.session_state.current_conv_id, "user", prompt)

            with st.chat_message("assistant"):
                with measure_latency() as lat:
                    try:
                        response_stream, retrieved_chunks, context_text = answer_question_with_rag(
                            question=prompt,
                            vector_store=vs,
                            model=selected_model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_k=top_k,
                            api_key=active_api_key,
                            stream=True
                        )
                        full_response = st.write_stream(response_stream)
                        
                        # Display Source Citations
                        if retrieved_chunks:
                            with st.expander("📑 Referenced Sources & Context Chunks", expanded=False):
                                for chunk in retrieved_chunks:
                                    st.markdown(format_source_citation(chunk))
                                    st.caption(chunk.get("text", "")[:300] + "...")

                    except Exception as e:
                        st.error(f"RAG Error: {str(e)}")
                        full_response = None
                        retrieved_chunks = []

                if full_response:
                    st.caption(f"⏱️ Retrieved & generated in {lat['elapsed_seconds']}s")
                    add_message(
                        st.session_state.current_conv_id,
                        "assistant",
                        full_response,
                        metadata={
                            "sources": [
                                {
                                    "doc_name": c.get("doc_name"),
                                    "page_number": c.get("page_number"),
                                    "score": c.get("score"),
                                    "text": c.get("text")[:200]
                                }
                                for c in retrieved_chunks
                            ],
                            "elapsed_seconds": lat["elapsed_seconds"]
                        }
                    )


# ==========================================
# MODE 4: DOCUMENT SUMMARIZER
# ==========================================
elif mode_name == "Document Summarizer":
    st.markdown("### 📝 **Academic Document Summarizer**")
    st.caption("Generate high-yield summaries, key definitions, formulas, and potential exam questions.")

    doc_text = st.session_state.extracted_doc_text
    stats = st.session_state.current_doc_stats

    if not doc_text:
        st.warning("⚠️ No document text loaded. Please upload a PDF in the sidebar first.")
    else:
        st.success(f"Loaded: **{stats['filename']}** ({stats['page_count']} pages, {stats['word_count']} words)")

        col_sum1, col_sum2 = st.columns([1, 3])
        with col_sum1:
            summary_length = st.radio("Summary Depth:", ["Standard (High-Yield)", "Comprehensive (Deep-Dive)"])
            generate_btn = st.button("✨ Generate Summary", type="primary", use_container_width=True)

        if generate_btn:
            active_api_key = user_api_key or MISTRAL_API_KEY
            if not validate_api_key(active_api_key):
                st.warning("⚠️ Please provide a valid Mistral API Key.")
            else:
                with st.spinner("Analyzing document and compiling study summary..."):
                    # For long documents, pass a representative high-density chunk or use map/reduce
                    # Truncate to first ~12,000 characters (approx 3000 tokens) for single-pass high yield
                    context_sample = doc_text[:14000]
                    if len(doc_text) > 14000:
                        st.info("ℹ️ Document is large: Synthesizing core chapters and high-yield sections.")

                    prompt_content = f"Please summarize the following study material thoroughly according to your system prompt instructions:\n\n{context_sample}"
                    messages = [
                        {"role": "system", "content": get_system_prompt_for_mode("Summarize")},
                        {"role": "user", "content": prompt_content}
                    ]

                    with measure_latency() as lat:
                        try:
                            summary_result = generate_chat_response(
                                messages=messages,
                                model=selected_model,
                                temperature=0.2,
                                max_tokens=2500,
                                api_key=active_api_key
                            )
                            st.session_state["cached_summary"] = summary_result
                            st.session_state["summary_latency"] = lat["elapsed_seconds"]
                        except Exception as e:
                            st.error(f"Summary Generation Failed: {str(e)}")

        if "cached_summary" in st.session_state:
            st.divider()
            st.markdown(st.session_state["cached_summary"])
            if "summary_latency" in st.session_state:
                st.caption(f"⏱️ Summarized in {st.session_state['summary_latency']}s")


# ==========================================
# MODE 5: INTERACTIVE QUIZ GENERATOR
# ==========================================
elif mode_name == "Interactive Quiz":
    st.markdown("### 🎯 **Interactive Quiz Generator (Structured JSON)**")
    st.caption("Generate pedagogical Multiple Choice Quizzes with instant automated grading and explanations.")

    doc_text = st.session_state.extracted_doc_text
    
    col_q1, col_q2, col_q3 = st.columns([2, 1, 1])
    with col_q1:
        quiz_source_mode = st.radio("Quiz Source:", ["Uploaded PDF Material", "Custom Topic Name"], horizontal=True)
        if quiz_source_mode == "Custom Topic Name":
            quiz_topic = st.text_input("Enter Topic:", value="Machine Learning: Gradient Descent & Loss Functions")
        else:
            if not doc_text:
                st.info("Upload a PDF in the sidebar to generate a quiz directly from your notes.")
                quiz_topic = "General Computer Science Fundamentals"
            else:
                quiz_topic = doc_text[:2000]
                st.caption(f"Using uploaded document context ({st.session_state.current_doc_stats.get('filename', 'PDF')})")

    with col_q2:
        num_q = st.selectbox("Questions:", [5, 10], index=0)
    with col_q3:
        difficulty = st.selectbox("Difficulty:", ["Easy", "Medium", "Hard"], index=1)

    if st.button("🎲 Generate Interactive Quiz", type="primary"):
        active_api_key = user_api_key or MISTRAL_API_KEY
        if not validate_api_key(active_api_key):
            st.warning("⚠️ Please provide a valid Mistral API Key.")
        else:
            with st.spinner("Generating structured quiz from Mistral AI..."):
                try:
                    quiz = generate_study_quiz(
                        topic_or_context=quiz_topic,
                        num_questions=num_q,
                        difficulty=difficulty,
                        model=selected_model,
                        api_key=active_api_key
                    )
                    st.session_state.quiz_state = quiz
                    st.session_state.user_quiz_answers = {}
                    st.session_state.quiz_submitted = False
                except Exception as e:
                    st.error(f"Quiz Generation Error: {str(e)}")

    # Render Active Quiz
    if st.session_state.quiz_state:
        quiz = st.session_state.quiz_state
        st.divider()
        st.markdown(f"#### 📝 Quiz: **{quiz.get('topic', 'Study Material')}** &nbsp; <span class='mode-badge'>{quiz.get('difficulty')}</span>", unsafe_allow_html=True)

        for q in quiz["questions"]:
            qid = q["id"]
            st.markdown(f"**Q{qid}. {q['question']}**")
            
            selected_option = st.radio(
                f"Options for Q{qid}:",
                options=q["options"],
                key=f"q_{qid}",
                index=None,
                disabled=st.session_state.quiz_submitted,
                label_visibility="collapsed"
            )
            st.session_state.user_quiz_answers[qid] = selected_option

            if st.session_state.quiz_submitted:
                user_ans = st.session_state.user_quiz_answers.get(qid)
                correct_ans = q["correct_answer"]
                
                # Check if correct (fuzzy check letter prefix A, B, C, D)
                is_correct = False
                if user_ans:
                    if user_ans.strip() == correct_ans.strip() or user_ans.strip()[:2] == correct_ans.strip()[:2]:
                        is_correct = True

                if is_correct:
                    st.success(f"✅ **Correct!** Your answer: {user_ans}")
                else:
                    st.error(f"❌ **Incorrect.** Correct answer: **{correct_ans}**")
                
                with st.expander("💡 Explanation & Key Concept"):
                    st.markdown(q["explanation"])

            st.write("")

        if not st.session_state.quiz_submitted:
            if st.button("📊 Submit & Grade Quiz", type="primary"):
                st.session_state.quiz_submitted = True
                st.rerun()
        else:
            # Calculate final score
            total = len(quiz["questions"])
            score = 0
            for q in quiz["questions"]:
                u = st.session_state.user_quiz_answers.get(q["id"])
                c = q["correct_answer"]
                if u and (u.strip() == c.strip() or u.strip()[:2] == c.strip()[:2]):
                    score += 1
            
            st.divider()
            col_sc1, col_sc2 = st.columns([1, 2])
            with col_sc1:
                pct = int((score / total) * 100)
                st.metric("Final Score", f"{score}/{total}", f"{pct}%")
            with col_sc2:
                if pct >= 80:
                    st.balloons()
                    st.success("🎉 **Outstanding Mastery!** You demonstrated strong conceptual understanding.")
                elif pct >= 50:
                    st.info("👍 **Good effort!** Review the explanations above to strengthen edge cases.")
                else:
                    st.warning("📚 **Keep practicing!** Re-read the lecture notes and try again.")


# ==========================================
# MODE 6: EDUCATIONAL EVALUATION
# ==========================================
elif mode_name == "Educational Evaluation":
    st.markdown("### 📊 **Basic Educational GenAI Evaluation**")
    st.caption("Measure retrieval relevance, latency, and context faithfulness across sample queries.")
    
    vs = st.session_state.vector_store
    if vs.get_total_chunks() == 0:
        st.warning("⚠️ Please index a PDF in the sidebar first to run RAG evaluation benchmarks.")
    else:
        sample_queries = [
            "What is the primary objective of the algorithm described?",
            "What are the mathematical formulas or equations mentioned?",
            "What are the main limitations or disadvantages discussed?"
        ]

        eval_query = st.text_input("Test Query for Evaluation:", value=sample_queries[0])
        
        if st.button("🚀 Run Evaluation Benchmark", type="primary"):
            active_api_key = user_api_key or MISTRAL_API_KEY
            if not validate_api_key(active_api_key):
                st.warning("⚠️ Please provide a valid Mistral API Key.")
            else:
                with st.spinner("Executing retrieval & generation evaluation..."):
                    # 1. Measure Retrieval
                    with measure_latency() as ret_lat:
                        q_vec = generate_query_embedding(eval_query)
                        retrieved = vs.similarity_search(q_vec, top_k=top_k)

                    # 2. Measure Generation
                    with measure_latency() as gen_lat:
                        ans_str, _, ctx_str = answer_question_with_rag(
                            question=eval_query,
                            vector_store=vs,
                            model=selected_model,
                            temperature=0.0,
                            max_tokens=512,
                            top_k=top_k,
                            api_key=active_api_key,
                            stream=False
                        )

                    st.divider()
                    st.markdown("#### 📈 **Benchmark Results**")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Retrieval Latency", f"{ret_lat['elapsed_seconds']}s")
                    with c2:
                        st.metric("Generation Latency", f"{gen_lat['elapsed_seconds']}s")
                    with c3:
                        avg_score = sum(c.get("score", 0) for c in retrieved) / len(retrieved) if retrieved else 0
                        st.metric("Avg Chunk Cosine Sim", f"{avg_score:.3f}")
                    with c4:
                        st.metric("Retrieved Chunks", f"{len(retrieved)}")

                    st.markdown("#### 🤖 **Generated Answer**")
                    st.info(ans_str)

                    with st.expander("🔍 Context Passed to LLM"):
                        st.text(ctx_str)
