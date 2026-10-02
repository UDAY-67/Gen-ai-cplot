"""
SQLite Database Module
Provides local persistent storage for conversations, chat messages, and document metadata.
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from config import DB_PATH


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create and return a database connection with dictionary-like row access."""
    target_path = Path(db_path) if db_path is not None else DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables if they do not exist."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                mode TEXT DEFAULT 'general',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)

        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                char_count INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def create_conversation(title: str = "New Study Session", mode: str = "general", db_path: Optional[Path] = None) -> str:
    """Create a new conversation entry and return its ID."""
    init_db(db_path)
    conv_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (id, title, mode, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (conv_id, title, mode, now, now)
        )
        conn.commit()
    return conv_id


def get_conversations(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve all conversations ordered by recent update."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, mode, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_conversation(conv_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a single conversation by ID."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, mode, created_at, updated_at FROM conversations WHERE id = ?",
            (conv_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def update_conversation_title(conv_id: str, title: str, db_path: Optional[Path] = None) -> None:
    """Update conversation title."""
    init_db(db_path)
    now = datetime.now().isoformat()
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, conv_id)
        )
        conn.commit()


def delete_conversation(conv_id: str, db_path: Optional[Path] = None) -> None:
    """Delete a conversation and all its messages."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()


def add_message(
    conv_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None
) -> str:
    """Add a message to a conversation and bump updated_at."""
    init_db(db_path)
    msg_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    meta_json = json.dumps(metadata) if metadata else None

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (id, conversation_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (msg_id, conv_id, role, content, meta_json, now)
        )
        cursor.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conv_id)
        )
        conn.commit()
    return msg_id


def get_messages(conv_id: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve all messages for a specific conversation ordered chronologically."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, role, content, metadata, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conv_id,)
        )
        rows = cursor.fetchall()
        result = []
        for row in rows:
            item = dict(row)
            if item.get("metadata"):
                try:
                    item["metadata"] = json.loads(item["metadata"])
                except Exception:
                    pass
            result.append(item)
        return result


def save_document_meta(
    filename: str,
    page_count: int,
    char_count: int,
    chunk_count: int,
    db_path: Optional[Path] = None
) -> str:
    """Record metadata for an indexed PDF document."""
    init_db(db_path)
    doc_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents (id, filename, page_count, char_count, chunk_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, filename, page_count, char_count, chunk_count, now)
        )
        conn.commit()
    return doc_id


def get_documents(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Get list of all indexed documents."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]
