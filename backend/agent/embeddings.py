# agent/embeddings.py
from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import List

import numpy as np


def _quiet_hf_logging() -> None:
    """
    HuggingFace hub prints retry/protocol messages at INFO/WARNING.
    We reduce that noise so dev runs stay clean.
    """
    for name in ("huggingface_hub", "transformers", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)


@lru_cache(maxsize=1)
def _model():
    """
    Load the SentenceTransformer embedding model.

    Strategy:
      1) Try local cache only (no network calls). This prevents HEAD requests
         and avoids protocol errors on restricted networks.
      2) If not available locally, fall back to online download once.
    """
    _quiet_hf_logging()

    model_name = os.environ.get("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")

    offline = os.environ.get("RAG_OFFLINE", "").strip().lower() in {"1","true","yes","y"}

    # Optional: allow explicit cache folder
    cache_folder = os.environ.get("HF_HOME") or os.environ.get("SENTENCE_TRANSFORMERS_HOME")

    from sentence_transformers import SentenceTransformer

    # 1) Prefer local files only (prevents the HEAD request in most situations)
    try:
        return SentenceTransformer(
            model_name,
            cache_folder=cache_folder,
            local_files_only=True,   # ✅ key change: avoids remote HEAD checks
        )
    except Exception as e:
        # 2) If not cached, allow online download once (unless offline mode)
        if offline:
            raise RuntimeError(
                f"Embedding model '{model_name}' is not available locally and RAG_OFFLINE=1 is set. "
                "Pre-download the model to the HuggingFace cache (HF_HOME / SENTENCE_TRANSFORMERS_HOME) "
                "or unset RAG_OFFLINE."
            ) from e
        return SentenceTransformer(
            model_name,
            cache_folder=cache_folder,
            local_files_only=False,
        )


def embed_dim() -> int:
    """
    Returns embedding dimensionality for the current SentenceTransformer model.
    Needed for pgvector VECTOR(n) schema.
    """
    m = _model()
    try:
        return int(m.get_sentence_embedding_dimension())
    except Exception:
        v = m.encode(["dim_probe"], normalize_embeddings=True, show_progress_bar=False)[0]
        return int(len(v))


def embed_texts(texts: List[str]) -> List[List[float]]:
    m = _model()
    vecs = m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vecs.tolist()


def embed_query(query: str) -> List[float]:
    m = _model()
    vec = m.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    return vec.tolist()
