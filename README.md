# 🎓 GenAI Study Copilot

An AI-powered pedagogical study assistant built for Engineering, Computer Science, and STEM students. The application implements an end-to-end GenAI pipeline—including direct **Mistral AI** integration, conversation memory, text chunking, local sentence embeddings, **FAISS** vector search, grounded Retrieval-Augmented Generation (RAG) with source citations, structured JSON quiz generation, and SQLite persistence.

---

## 📌 Problem & Motivation

Engineering and Computer Science students frequently deal with dense technical textbooks, research papers, and lecture slides. Traditional study approaches have key challenges:
1. **Generic LLM Hallucinations**: Standard chatbots often invent unsupported facts when asked about specialized course material.
2. **Context Limits**: Entire 100+ page textbooks cannot be fed in one prompt due to token limits, latency, and cost.
3. **Passive Reading vs. Active Recall**: Students read notes passively rather than testing themselves with structured quizzes and conceptual breakdowns.

---

## 💡 Solution

**GenAI Study Copilot** solves this by combining:
- **Retrieval-Augmented Generation (RAG)**: Chunks study materials, generates dense semantic embeddings, and retrieves the most relevant context snippets using FAISS before invoking Mistral AI.
- **Strict Grounding & Source Citations**: The assistant answers directly from the provided lecture notes and cites document names and page numbers.
- **Multi-Mode Study Support**: From intuitive 6-step concept explainers to structured MCQ generation and automated scoring.
- **Local Persistence & Privacy**: Conversations and index metadata are stored locally in SQLite and FAISS.

---

## 🚀 Key Features

* **💬 Conversational Study Chat**: Stateful multi-turn chat with conversation history and latency tracking.
* **⚡ Real-Time Streaming**: Token-by-token streaming using the official Mistral AI SDK and Streamlit.
* **🧠 Concept Explainer Mode**: Pedagogy-focused 6-step breakdowns (Intuition → Formal Math → Analogy → Steps → Pitfalls → Self-Check).
* **📄 PDF Processing & Chunking**: Page-by-page text extraction with recursive sliding-window chunking and token estimation.
* **🔍 Semantic Search with FAISS**: Dense vector search powered by `sentence-transformers` and FAISS IndexFlatIP (cosine similarity).
* **📚 Grounded RAG with Citations**: Accurate answers strictly derived from uploaded documents with page number references.
* **🎯 Interactive Quiz Generator**: Generates schema-validated JSON MCQs with instant scoring, feedback, and in-depth explanations.
* **📝 Document Summarizer**: Formats high-yield summaries, core definitions, formulas, and potential exam questions.
* **💾 SQLite Persistence**: Full local conversation history with session switching, title updates, and deletion.
* **📊 Educational Evaluation**: Evaluates retrieval cosine similarity, latency, and context relevance on benchmark queries.

---

## 🏗️ Architecture & Pipeline

```
                         STUDENT / USER
                               │
                               ▼
                        STREAMLIT UI
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
      CHAT / QUERY INTERFACE            DOCUMENT UPLOADER (PDF)
              │                                 │
              │                                 ▼
              │                         PAGE TEXT EXTRACTION (pypdf)
              │                                 │
              │                                 ▼
              │                         SLIDING-WINDOW CHUNKER
              │                          (Chunk Size & Overlap)
              │                                 │
              │                                 ▼
              │                         EMBEDDINGS (sentence-transformers)
              │                                 │
              │                                 ▼
              │                         FAISS VECTOR INDEX
              │                                 │
              └───────────────┬─────────────────┘
                              ▼
                   COSINE SIMILARITY SEARCH
                              │
                              ▼
                     TOP-K RELEVANT CHUNKS
                              │
                              ▼
                   GROUNDED CONTEXT PROMPT
                              │
                              ▼
                    MISTRAL AI (LLM API)
                              │
                              ▼
                    REAL-TIME STREAMING
                              │
                              ▼
              STRUCTURED ANSWER + SOURCE CITATIONS
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Responsive, interactive web application |
| **LLM Provider** | Mistral AI (`mistralai` SDK) | Core reasoning, chat, explanations, and JSON generation |
| **PDF Extraction** | `pypdf` | Page-level text and metadata extraction |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Local 384-dimensional dense vector embeddings |
| **Vector Database** | `faiss-cpu` (IndexFlatIP) | High-speed semantic similarity retrieval |
| **Database** | SQLite (`sqlite3`) | Persistent conversation and document metadata storage |
| **Configuration** | `python-dotenv` | Secure environment variable handling |
| **Testing** | `pytest` | Unit testing for chunker, vector store, DB, and utilities |

---

## 📁 Project Structure

```text
genai-study-copilot/
├── app.py                  # Streamlit application UI and workflows
├── config.py               # Centralized configuration and defaults
├── requirements.txt        # Project dependencies
├── README.md               # Comprehensive documentation
├── .env                    # Environment variables (API keys - gitignored)
├── .env.example            # Environment template
├── .gitignore              # Git ignore rules (.env, .venv, database)
│
├── services/
│   ├── llm.py              # Mistral SDK client, completion, streaming, JSON
│   ├── chat.py             # Conversation payload and prompt routing
│   ├── pdf.py              # PDF extraction and error handling
│   ├── chunker.py          # Sliding-window text chunker with metadata
│   ├── embeddings.py       # SentenceTransformer embedding generation
│   ├── vector_store.py     # FAISS vector store and persistence
│   ├── rag.py              # Retrieval-Augmented Generation pipeline
│   └── quiz.py             # Structured MCQ generator and schema validator
│
├── database/
│   └── db.py               # SQLite database schemas and CRUD operations
│
├── prompts/
│   └── prompts.py          # System prompts for all study modes
│
├── utils/
│   └── helpers.py          # Cleaners, JSON extractors, latency timers
│
├── data/
│   └── .gitkeep            # Data directory for SQLite DB and FAISS index
│
└── tests/
    └── test_basic.py       # Automated unit tests
```

---

## ⚙️ Installation & Virtual Environment Setup

### 1. Clone & Navigate to the Project

```bash
cd simple-projects
```

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Configuration

Create a `.env` file in the project root:

```env
# Mistral AI API Key
MISTRAL_API_KEY=your_mistral_api_key_here

# Model Selection
MISTRAL_MODEL=mistral-small-latest

# Embeddings & Vector Search
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=800
CHUNK_OVERLAP=150
TOP_K=4

# Storage
DB_PATH=data/study_copilot.db
FAISS_INDEX_DIR=data/faiss_index
```

> **Security Note:** Never commit your `.env` file or hardcode your API key. `.env` is protected via `.gitignore`.

---

## 🚀 Running the Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests

To run the automated test suite:

```bash
pytest tests/test_basic.py -v
```

---

## 🧠 Core GenAI Concepts Explained

### 1. Stateless LLM & Conversation Memory
LLMs do not retain memory across API calls. To build a conversational agent, the application stores message history in SQLite and sends a curated window of previous turns (`[{"role": "user", ...}, {"role": "assistant", ...}]`) with each new prompt.

### 2. Why Chunking is Essential
Textbooks contain tens of thousands of tokens. Chunking splits documents into semantically coherent pieces (e.g. 800 characters) with overlap (e.g. 150 characters) to ensure concepts split across chunk boundaries remain intact.

### 3. Vector Embeddings & Cosine Similarity
An embedding model maps text strings to numerical vectors in high-dimensional space where semantically related concepts are positioned close together. FAISS computes the inner product between normalized query vectors and document vectors to find top-K nearest neighbors.

### 4. Grounded RAG & Hallucination Prevention
Rather than asking the LLM to answer from internal training memory, RAG injects retrieved context snippets directly into the system prompt with strict instructions: *"Answer strictly using the provided context. If the information is absent, state that it cannot be found."*

---

## ⚖️ Limitations & Educational Scope

* **Scanned/Image-Only PDFs**: Text extraction relies on digital text; scanned documents require OCR.
* **Context Window**: Extremely large multi-document comparisons may require hierarchical summarization.
* **Evaluation**: Evaluation metrics in this application are designed for educational insight (latency, cosine similarity) rather than formal LLM-as-a-Judge benchmarking.
