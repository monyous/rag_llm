import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model():
    """Charge le modèle une seule fois et le garde en mémoire."""
    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> np.ndarray:
    """Transforme un texte en vecteur numérique."""
    model = get_model()
    return model.encode(text, convert_to_numpy=True)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Transforme plusieurs textes d'un coup (plus rapide qu'un par un)."""
    model = get_model()
    return model.encode(texts, convert_to_numpy=True)


def vector_to_blob(vector: np.ndarray) -> bytes:
    """Convertit un vecteur en binaire pour le stocker dans SQLite."""
    return vector.astype(np.float32).tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    """Reconvertit le binaire stocké en vecteur numérique."""
    return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Mesure à quel point deux vecteurs sont similaires (entre -1 et 1)."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 30) -> list[str]:
    """Découpe un texte long en morceaux (chunks) avec un léger chevauchement."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def find_relevant_chunks(query: str, chunks_with_embeddings: list, top_k: int = 3) -> list[str]:
    """
    Trouve les chunks les plus pertinents par rapport à la question.
    chunks_with_embeddings = liste de (id, texte_du_chunk, embedding_en_bytes)
    """
    query_vector = embed_text(query)
    scored = []
    for chunk_id, chunk_content, embedding_blob in chunks_with_embeddings:
        chunk_vector = blob_to_vector(embedding_blob)
        score = cosine_similarity(query_vector, chunk_vector)
        scored.append((score, chunk_content))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for score, text in scored[:top_k]]