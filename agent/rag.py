# agent/rag.py
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple

from agent.chunking import SmartChunker
from agent.embeddings import embed_dim, embed_query, embed_texts
from agent.loaders import load_text


PathLike = Union[str, Path]
ProgressCallback = Callable[[str, Dict[str, Any]], None]


class RagStore:
    """
    Neon/Postgres + pgvector-backed RAG store with:
      - file registry (rag_files) for dedupe + update detection
      - chunk storage (rag_chunks)
      - cleanup of deleted files (with optional dry-run)
      - single-file ingest helper for progress reporting

    Env:
      DATABASE_URL=postgresql://... (Neon connection string)
      (or NEON_DATABASE_URL)
    """

    def __init__(self, persist_dir: Union[str, Path] = "data/rag"):
        # persist_dir kept for compatibility with existing code; not used for storage
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.db_url = os.environ.get("DATABASE_URL") or os.environ.get("NEON_DATABASE_URL")
        if not self.db_url:
            raise RuntimeError(
                "RAG requires DATABASE_URL (Neon Postgres connection string). "
                "Set DATABASE_URL in your environment."
            )

        self._conn = None
        self._ensure_schema()

    # -------------------------
    # DB helpers
    # -------------------------
    def _connect(self):
        if self._conn is not None:
            return self._conn
        try:
            import psycopg
        except Exception as e:
            raise RuntimeError("Missing dependency: psycopg. Install: pip install psycopg[binary]") from e

        self._conn = psycopg.connect(self.db_url)
        return self._conn

    def _ensure_schema(self):
        conn = self._connect()
        dim = embed_dim()

        with conn.cursor() as cur:
            # pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # file registry for dedupe/update/cleanup
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_files (
                    namespace TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    mtime_utc TIMESTAMPTZ NOT NULL,
                    size_bytes BIGINT NOT NULL,
                    ingested_at_utc TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (namespace, path)
                );
                """
            )

            # chunk store
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    path TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding VECTOR({dim}) NOT NULL
                );
                """
            )

            # helpful indexes
            cur.execute("CREATE INDEX IF NOT EXISTS rag_chunks_ns_idx ON rag_chunks(namespace);")
            cur.execute("CREATE INDEX IF NOT EXISTS rag_chunks_path_idx ON rag_chunks(path);")

        conn.commit()

    # -------------------------
    # File fingerprint helpers
    # -------------------------
    def _file_fingerprint(self, path: Path) -> Dict[str, Any]:
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        st = path.stat()
        mtime_utc = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
        return {"sha256": sha, "size_bytes": int(st.st_size), "mtime_utc": mtime_utc}

    def _get_existing_file_record(self, namespace: str, path_str: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT sha256, mtime_utc, size_bytes, ingested_at_utc
                FROM rag_files
                WHERE namespace=%s AND path=%s
                """,
                (namespace, path_str),
            )
            row = cur.fetchone()

        if not row:
            return None

        sha256, mtime_utc, size_bytes, ingested_at_utc = row
        return {
            "sha256": sha256,
            "mtime_utc": mtime_utc,
            "size_bytes": int(size_bytes),
            "ingested_at_utc": ingested_at_utc,
        }

    def _upsert_file_record(self, namespace: str, path_str: str, sha256: str, mtime_utc, size_bytes: int):
        conn = self._connect()
        now = datetime.now(tz=timezone.utc)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_files(namespace, path, sha256, mtime_utc, size_bytes, ingested_at_utc)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (namespace, path) DO UPDATE
                SET sha256=EXCLUDED.sha256,
                    mtime_utc=EXCLUDED.mtime_utc,
                    size_bytes=EXCLUDED.size_bytes,
                    ingested_at_utc=EXCLUDED.ingested_at_utc
                """,
                (namespace, path_str, sha256, mtime_utc, size_bytes, now),
            )
        conn.commit()

    def _list_db_paths(self, namespace: str) -> Set[str]:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("SELECT path FROM rag_files WHERE namespace=%s", (namespace,))
            rows = cur.fetchall()
        return set(r[0] for r in rows if r and r[0])

    def _delete_chunks_for_file(self, namespace: str, path_str: str):
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM rag_chunks WHERE namespace=%s AND path=%s", (namespace, path_str))
        conn.commit()

    def _delete_file_record(self, namespace: str, path_str: str):
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM rag_files WHERE namespace=%s AND path=%s", (namespace, path_str))
        conn.commit()

    # -------------------------
    # Cleanup deleted files
    # -------------------------
    def cleanup_deleted_files(
        self,
        *,
        namespace: str,
        current_paths: Set[str],
        dry_run: bool = False,
        cap_list: int = 500,
        on_delete: Optional[Callable[[str], None]] = None,
        on_deleted: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Remove DB entries for any files (rag_files) that no longer exist locally.

        current_paths should be absolute resolved paths (same format used during ingest).
        If dry_run=True, does not delete anything; only reports what *would* be deleted.

        Callbacks:
          on_delete(path)  -> called right before deletion
          on_deleted(path) -> called right after deletion finished
        """
        db_paths = self._list_db_paths(namespace)
        to_delete = sorted(list(db_paths - current_paths))

        if dry_run:
            return {
                "namespace": namespace,
                "db_paths": len(db_paths),
                "current_paths": len(current_paths),
                "files_to_delete": len(to_delete),
                "to_delete": to_delete[:cap_list],
                "dry_run": True,
            }

        deleted: List[str] = []
        for path_str in to_delete:
            if on_delete:
                on_delete(path_str)
            self._delete_chunks_for_file(namespace, path_str)
            self._delete_file_record(namespace, path_str)
            if on_deleted:
                on_deleted(path_str)
            deleted.append(path_str)

        return {
            "namespace": namespace,
            "db_paths": len(db_paths),
            "current_paths": len(current_paths),
            "files_deleted": len(deleted),
            "deleted_paths": deleted[:cap_list],
            "dry_run": False,
        }

    # -------------------------
    # Single-file ingest (for progress UI)
    # -------------------------
    def ingest_file(
        self,
        path: PathLike,
        *,
        namespace: str = "knowledge",
        chunk_size: int = 1200,
        overlap: int = 150,
    ) -> Dict[str, Any]:
        """
        Ingest or update ONE file. Returns a detailed report:
          status: one of {"missing","skipped_unchanged","new","updated"}
          chunks_added: int
          chunk_count: int
          file_sha256, file_mtime_utc, file_size_bytes
        """
        p = Path(path)
        if not p.exists() or not p.is_file():
            return {"status": "missing", "chunks_added": 0, "chunk_count": 0, "path": str(p)}

        p = p.resolve()
        path_str = str(p)

        fp = self._file_fingerprint(p)
        existing = self._get_existing_file_record(namespace, path_str)

        # unchanged
        if existing and existing["sha256"] == fp["sha256"]:
            return {
                "status": "skipped_unchanged",
                "chunks_added": 0,
                "chunk_count": 0,
                "path": path_str,
                "file_sha256": fp["sha256"],
                "file_mtime_utc": fp["mtime_utc"].isoformat(),
                "file_size_bytes": fp["size_bytes"],
            }

        # updated: delete old chunks for that file
        status = "new"
        if existing and existing["sha256"] != fp["sha256"]:
            self._delete_chunks_for_file(namespace, path_str)
            status = "updated"

        # load -> chunk -> embed -> upload
        chunker = SmartChunker(max_chars=chunk_size, overlap=overlap)
        text = load_text(p)
        chunks = chunker.chunk(text, source=path_str, section="")

        if not chunks:
            self._upsert_file_record(namespace, path_str, fp["sha256"], fp["mtime_utc"], fp["size_bytes"])
            return {
                "status": status,
                "chunks_added": 0,
                "chunk_count": 0,
                "path": path_str,
                "file_sha256": fp["sha256"],
                "file_mtime_utc": fp["mtime_utc"].isoformat(),
                "file_size_bytes": fp["size_bytes"],
                "note": "no_chunks_extracted",
            }

        docs = [c.text for c in chunks]
        metas: List[dict] = []
        for i, c in enumerate(chunks):
            metas.append(
                {
                    **(c.meta or {}),
                    "namespace": namespace,
                    "path": path_str,
                    "chunk_index": i,
                    "file_sha256": fp["sha256"],
                    "file_mtime_utc": fp["mtime_utc"].isoformat(),
                    "file_size_bytes": fp["size_bytes"],
                }
            )

        embs = embed_texts(docs)

        conn = self._connect()
        rows = []
        for i in range(len(docs)):
            cid = f"{namespace}::{p.name}::chunk::{i}"
            rows.append((cid, namespace, path_str, i, docs[i], json.dumps(metas[i]), embs[i]))

        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO rag_chunks (id, namespace, path, chunk_index, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (id) DO UPDATE
                SET content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    namespace = EXCLUDED.namespace,
                    path = EXCLUDED.path,
                    chunk_index = EXCLUDED.chunk_index
                """,
                rows,
            )
        conn.commit()

        self._upsert_file_record(namespace, path_str, fp["sha256"], fp["mtime_utc"], fp["size_bytes"])

        return {
            "status": status,
            "chunks_added": len(docs),
            "chunk_count": len(docs),
            "path": path_str,
            "file_sha256": fp["sha256"],
            "file_mtime_utc": fp["mtime_utc"].isoformat(),
            "file_size_bytes": fp["size_bytes"],
        }

    # -------------------------
    # Ingest many files (kept for compatibility)
    # -------------------------
    def ingest_files(
        self,
        paths: List[PathLike],
        *,
        chunk_size: int = 1200,
        overlap: int = 150,
        namespace: str = "knowledge",
        cleanup_deleted: bool = False,
    ) -> Dict[str, Any]:
        """
        Batch ingest with dedupe/update and optional cleanup.
        Uses ingest_file(...) internally.
        """
        resolved_paths: List[Path] = []
        current_set: Set[str] = set()
        missing = 0

        for p in paths:
            pp = Path(p)
            if pp.exists() and pp.is_file():
                rp = pp.resolve()
                resolved_paths.append(rp)
                current_set.add(str(rp))
            else:
                missing += 1

        cleanup_report = None
        if cleanup_deleted:
            cleanup_report = self.cleanup_deleted_files(namespace=namespace, current_paths=current_set, dry_run=False)

        files_new = 0
        files_updated = 0
        files_skipped_unchanged = 0
        chunks_added = 0

        for p in resolved_paths:
            r = self.ingest_file(p, namespace=namespace, chunk_size=chunk_size, overlap=overlap)
            st = r.get("status")
            if st == "new":
                files_new += 1
            elif st == "updated":
                files_updated += 1
            elif st == "skipped_unchanged":
                files_skipped_unchanged += 1
            chunks_added += int(r.get("chunks_added") or 0)

        report: Dict[str, Any] = {
            "backend": "neon_pgvector",
            "namespace": namespace,
            "chunks_added": chunks_added,
            "files_new": files_new,
            "files_updated": files_updated,
            "files_skipped_unchanged": files_skipped_unchanged,
            "files_missing": missing,
        }
        if cleanup_report is not None:
            report["cleanup"] = cleanup_report
        return report

    # compatibility alias used by older code
    def add_files(self, paths: List[PathLike], *, chunk_size: int = 1200, overlap: int = 150) -> Dict[str, Any]:
        return self.ingest_files(paths, chunk_size=chunk_size, overlap=overlap)

    # -------------------------
    # Retrieve
    # -------------------------
    @staticmethod
    def _to_pgvector_literal(vec: Any) -> str:
        """
        Convert an embedding (list/tuple/numpy array) to pgvector literal string:
            "[0.1,0.2,0.3]"
        This avoids Postgres interpreting it as double precision[].
        """
        # Numpy arrays have .tolist()
        if hasattr(vec, "tolist"):
            vec = vec.tolist()

        if not isinstance(vec, (list, tuple)):
            raise TypeError(f"Embedding must be list/tuple/ndarray, got: {type(vec).__name__}")

        # Use repr-safe float conversion
        return "[" + ",".join(f"{float(x):.10f}" for x in vec) + "]"

    def retrieve(self, query: str, *, k: int = 6, namespace: str = "knowledge") -> List[Dict[str, Any]]:
        """
        Uses cosine distance operator (<=>) from pgvector.
        Lower distance = more similar.

        IMPORTANT:
        - Postgres column is pgvector `vector`
        - Python embedding must be sent as a pgvector literal and cast to ::vector
          to avoid: operator does not exist: vector <=> double precision[]
        """
        qemb = embed_query(query)
        qemb_vec = self._to_pgvector_literal(qemb)

        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content, metadata, (embedding <=> %s::vector) AS distance
                FROM rag_chunks
                WHERE namespace = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (qemb_vec, namespace, qemb_vec, k),
            )
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for rid, content, meta, dist in rows:
            out.append(
                {
                    "id": rid,
                    "text": content,
                    "meta": meta if isinstance(meta, dict) else (json.loads(meta) if isinstance(meta, str) else meta),
                    "distance": float(dist) if dist is not None else None,
                }
            )
        return out
