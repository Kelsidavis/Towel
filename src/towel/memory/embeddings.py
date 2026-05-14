"""Optional vector embeddings for the memory store.

Wraps a small local sentence-transformer model so the memory store
can do paraphrase recall — "database performance" finding "N+1 query
fix" even though no token overlaps. Gracefully no-ops when the
``[embeddings]`` extra isn't installed: callers see ``is_available()``
go False and fall back to BM25 + graph fusion.

Design constraints:

* **Lazy model load.** The model lives behind ``_get_model()`` so
  importing this module costs nothing. First call to encode() pays
  the warm-up; subsequent calls are sub-millisecond per short string.
* **Single model per process.** Loading ``all-MiniLM-L6-v2`` (~90MB,
  384-dim) takes ~2s + ~80MB RAM; we cache the handle in module
  state so a long-lived coordinator amortizes that across all calls.
* **Float32 little-endian BLOB on disk.** Stored directly in the
  pre-reserved ``embedding BLOB`` column on memories. Cosine
  similarity is computed in numpy when both query and stored vectors
  are unit-normalized at write time, so search() becomes a fast dot
  product over a single np.stack.

If you need a different model, set the ``TOWEL_EMBED_MODEL`` env var
to any sentence-transformers identifier. Dimension must be consistent
across the corpus — switching models invalidates old vectors.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("towel.memory.embeddings")

DEFAULT_MODEL = os.environ.get(
    "TOWEL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
_EMBED_DTYPE = "float32"

# Module-state cache. Held weakly so subsequent imports in the same
# process reuse the warm-up. None until the first encode() call.
_model: Any | None = None
_model_load_failed: bool = False


def is_available() -> bool:
    """True iff the embeddings extra is installed AND a model loads.

    Cheap to call repeatedly — caches the negative result so a missing
    dep doesn't keep retrying the import on every retrieval.
    """
    global _model_load_failed
    if _model_load_failed:
        return False
    try:
        import sentence_transformers  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        _model_load_failed = True
        return False
    return True


def _get_model() -> Any:
    """Return the cached sentence-transformers model, loading on first call."""
    global _model, _model_load_failed
    if _model is not None:
        return _model
    if _model_load_failed:
        raise RuntimeError("embeddings unavailable (prior load failure)")
    try:
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model %r (first call, may take a few seconds)", DEFAULT_MODEL)
        _model = SentenceTransformer(DEFAULT_MODEL)
        return _model
    except Exception as exc:
        log.warning("Embedding model failed to load (%s); vector search disabled", exc)
        _model_load_failed = True
        raise


def encode(text: str) -> bytes | None:
    """Return a unit-normalized float32 vector as raw bytes, or None.

    None is the correct value on disk for "no embedding available";
    pairs with the nullable embedding BLOB column. Bytes are little-
    endian float32 so a future reader can mmap or np.frombuffer
    without copying.
    """
    if not text:
        return None
    if not is_available():
        return None
    try:
        import numpy as np

        model = _get_model()
        vec = model.encode(text, normalize_embeddings=True)
        arr = np.asarray(vec, dtype=np.float32)
        return arr.tobytes()
    except Exception as exc:
        log.debug("encode(%r…) failed: %s", text[:40], exc)
        return None


def cosine_topk(
    query_blob: bytes,
    candidates: list[tuple[str, bytes]],
    k: int = 5,
) -> list[tuple[str, float]]:
    """Rank ``candidates`` by cosine similarity to ``query_blob``.

    Both query and stored vectors are unit-normalized at encode time,
    so cosine reduces to a dot product. ``candidates`` is a list of
    (key, blob) — entries missing a stored embedding are skipped, not
    a crash. Returns a list of (key, score) sorted high-to-low.
    """
    if not query_blob or not candidates:
        return []
    try:
        import numpy as np
    except ImportError:
        return []
    q = np.frombuffer(query_blob, dtype=np.float32)
    rows: list[tuple[str, np.ndarray]] = []
    for key, blob in candidates:
        if not blob:
            continue
        v = np.frombuffer(blob, dtype=np.float32)
        # Guard against dimension drift if the embedding model changed
        # mid-life. Skip mismatched rows; the operator can re-encode
        # by editing the entry, or we add a `memory reembed` command
        # later.
        if v.shape != q.shape:
            continue
        rows.append((key, v))
    if not rows:
        return []
    keys = [k for k, _ in rows]
    matrix = np.stack([v for _, v in rows])
    scores = matrix @ q  # (N,) dot products; both pre-normalized → cosine
    # argsort descending without copying the full array.
    top = np.argsort(-scores)[:k]
    return [(keys[i], float(scores[i])) for i in top]
