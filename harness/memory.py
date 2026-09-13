import json
import sqlite3
import struct
from typing import Any, Dict, List, Optional, Union

try:
    import sqlite_vec
    HAS_SQLITE_VEC_PKG = True
except ImportError:
    HAS_SQLITE_VEC_PKG = False


def serialize_float_vector(vector: List[float]) -> bytes:
    """Pack a list of floats into little-endian bytes format for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


class MemoryStore:
    """
    Embedded RAG memory module using sqlite3 and sqlite-vec extension
    with graceful fallback to text/keyword queries if sqlite-vec is unavailable.
    """

    def __init__(self, db_path: str = ":memory:", dim: int = 384):
        self.db_path = db_path
        self.dim = dim
        self.vec_enabled = False

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()

        # Always create metadata table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT,
                metadata JSON
            )
            """
        )

        # Attempt to load sqlite-vec extension if available and supported
        if HAS_SQLITE_VEC_PKG and hasattr(self.conn, "enable_load_extension"):
            try:
                self.conn.enable_load_extension(True)
                sqlite_vec.load(self.conn)
                self.conn.enable_load_extension(False)

                cursor.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_knowledge USING vec0(
                        id INTEGER PRIMARY KEY,
                        embedding float[{self.dim}]
                    )
                    """
                )
                self.vec_enabled = True
            except Exception:
                self.vec_enabled = False
        else:
            self.vec_enabled = False

        self.conn.commit()

    def store_memory(
        self,
        title: str,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Stores a memory entry in knowledge_meta and vec_knowledge (if embedding provided & vec active).
        """
        meta_json = json.dumps(metadata or {})
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO knowledge_meta (title, content, metadata) VALUES (?, ?, ?)",
            (title, content, meta_json),
        )
        row_id = cursor.lastrowid

        if self.vec_enabled and embedding is not None and len(embedding) == self.dim:
            try:
                blob_vec = serialize_float_vector(embedding)
                cursor.execute(
                    "INSERT INTO vec_knowledge(id, embedding) VALUES (?, ?)",
                    (row_id, blob_vec),
                )
            except Exception:
                pass

        self.conn.commit()
        return row_id

    def retrieve_relevant(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves relevant memories using vector search if available, falling back to
        keyword/exact match search on knowledge_meta.
        """
        cursor = self.conn.cursor()

        if self.vec_enabled and query_embedding is not None and len(query_embedding) == self.dim:
            try:
                blob_vec = serialize_float_vector(query_embedding)
                cursor.execute(
                    """
                    SELECT v.id, v.distance, k.title, k.content, k.metadata
                    FROM vec_knowledge v
                    JOIN knowledge_meta k ON v.id = k.id
                    WHERE v.embedding MATCH ? AND k = ?
                    ORDER BY v.distance
                    """,
                    (blob_vec, top_k),
                )
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "id": row["id"],
                        "title": row["title"],
                        "content": row["content"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                        "distance": row["distance"],
                    })
                return results
            except Exception:
                pass  # Fall through to fallback text search

        # Fallback text/keyword query search
        pattern = f"%{query}%"
        cursor.execute(
            """
            SELECT id, title, content, metadata
            FROM knowledge_meta
            WHERE title LIKE ? OR content LIKE ?
            LIMIT ?
            """,
            (pattern, pattern, top_k),
        )
        rows = cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "title": row["title"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            })

        # If keyword search returned empty results, return all up to top_k as a secondary fallback
        if not results:
            cursor.execute(
                "SELECT id, title, content, metadata FROM knowledge_meta LIMIT ?",
                (top_k,),
            )
            rows = cursor.fetchall()
            for row in rows:
                results.append({
                    "id": row["id"],
                    "title": row["title"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                })

        return results

    def close(self):
        self.conn.close()
