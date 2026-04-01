"""RAG — Retrieval-Augmented Generation for conversations and files.

Indexes text chunks with TF-IDF scoring for fast local retrieval.
No vector database or embedding model needed — pure Python.

Usage:
    rag = RAGIndex()
    rag.add("doc1", "Python is a programming language...")
    rag.add("doc2", "JavaScript runs in the browser...")
    results = rag.search("what language runs in browser", top_k=3)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RAGChunk:
    """A chunk of indexed text."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResult:
    """A search result."""

    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer."""
    return re.findall(r"\b\w{2,}\b", text.lower())


class RAGIndex:
    """TF-IDF based retrieval index. No external dependencies."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunks: list[RAGChunk] = []
        self._doc_freq: Counter[str] = Counter()
        self._chunk_tokens: list[Counter[str]] = []

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> int:
        """Add a document to the index. Returns number of chunks created."""
        chunks = self._split(text)
        added = 0
        for chunk_text in chunks:
            chunk = RAGChunk(doc_id=doc_id, text=chunk_text, metadata=metadata or {})
            self._chunks.append(chunk)

            tokens = _tokenize(chunk_text)
            token_counts = Counter(tokens)
            self._chunk_tokens.append(token_counts)

            for token in set(tokens):
                self._doc_freq[token] += 1

            added += 1
        return added

    def add_file(self, path: str, metadata: dict[str, Any] | None = None) -> int:
        """Index a file."""
        from pathlib import Path

        p = Path(path).expanduser()
        if not p.is_file():
            return 0
        text = p.read_text(encoding="utf-8", errors="replace")
        return self.add(p.name, text, metadata={"path": str(p), **(metadata or {})})

    def search(self, query: str, top_k: int = 5) -> list[RAGResult]:
        """Search the index using TF-IDF scoring."""
        if not self._chunks:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        n_docs = len(self._chunks)
        scores: list[tuple[int, float]] = []

        for i, chunk_tokens in enumerate(self._chunk_tokens):
            score = 0.0
            chunk_len = sum(chunk_tokens.values()) or 1

            for token in query_tokens:
                tf = chunk_tokens.get(token, 0) / chunk_len
                df = self._doc_freq.get(token, 0)
                if df > 0:
                    idf = math.log(n_docs / df)
                    score += tf * idf

            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            chunk = self._chunks[idx]
            results.append(
                RAGResult(
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=round(score, 4),
                    metadata=chunk.metadata,
                )
            )
        return results

    def clear(self) -> None:
        self._chunks.clear()
        self._doc_freq.clear()
        self._chunk_tokens.clear()

    def _split(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap

        return chunks
