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
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT 'Nouvelle conversation',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    # Ajoute la colonne project_id à sessions si elle n'existe pas encore
    # (évite de perdre tes documents/conversations déjà en base)
    cursor.execute("PRAGMA table_info(sessions)")
    columns = [col[1] for col in cursor.fetchall()]
    if "project_id" not in columns:
        cursor.execute("ALTER TABLE sessions ADD COLUMN project_id INTEGER")

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


# ---------- Projets ----------

def create_project(username: str, name: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO projects (username, name) VALUES (?, ?)",
        (username, name)
    )
    project_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return project_id


def get_projects(username: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, name FROM projects WHERE username = ? ORDER BY created_at DESC",
        (username,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def delete_project(project_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Les conversations du projet redeviennent "sans projet" plutôt que d'être supprimées
    cursor.execute("UPDATE sessions SET project_id = NULL WHERE project_id = ?", (project_id,))
    cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()


# ---------- Sessions de chat ----------

def create_session(username: str, title: str = "Nouvelle conversation", project_id: int = None) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (username, title, project_id) VALUES (?, ?, ?)",
        (username, title, project_id)
    )
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id


def get_sessions(username: str):
    """Retourne toutes les sessions, avec leur project_id (None si pas de projet)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title, created_at, project_id FROM sessions WHERE username = ? ORDER BY created_at DESC",
        (username,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def update_session_title(session_id: int, title: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
    conn.commit()
    conn.close()


def delete_session(session_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()


def save_message(session_id: int, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()


def get_session_messages(session_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows