import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "rag.db"


def init_chunks_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            filename TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def add_document(username: str, filename: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO documents (username, filename) VALUES (?, ?)",
        (username, filename)
    )
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return doc_id


def add_chunk(document_id: int, chunk_text: str, embedding_blob: bytes):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chunks (document_id, chunk_text, embedding) VALUES (?, ?, ?)",
        (document_id, chunk_text, embedding_blob)
    )
    conn.commit()
    conn.close()


def get_all_chunks(username: str):
    """Récupère tous les chunks + embeddings appartenant aux documents d'un utilisateur."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT chunks.id, chunks.chunk_text, chunks.embedding
        FROM chunks
        JOIN documents ON chunks.document_id = documents.id
        WHERE documents.username = ?
    """, (username,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_user_documents(username: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT filename, uploaded_at FROM documents WHERE username = ? ORDER BY uploaded_at DESC",
        (username,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def save_message(username: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (username, role, content) VALUES (?, ?, ?)",
        (username, role, content)
    )
    conn.commit()
    conn.close()


def get_chat_history(username: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM chat_history WHERE username = ? ORDER BY created_at ASC",
        (username,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows