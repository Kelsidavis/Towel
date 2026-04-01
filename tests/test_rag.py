"""Tests for RAG retrieval system."""

import pytest

from towel.agent.rag import RAGIndex


class TestRAGIndex:
    @pytest.fixture
    def idx(self):
        rag = RAGIndex(chunk_size=50)
        rag.add(
            "python",
            "Python is a high-level programming language known for "
            "readability. It supports multiple paradigms including "
            "object-oriented and functional programming.",
        )
        rag.add(
            "javascript",
            "JavaScript is the language of the web. It runs in "
            "browsers and on servers via Node.js. It is dynamically "
            "typed and supports async programming.",
        )
        rag.add(
            "rust",
            "Rust is a systems programming language focused on "
            "safety and performance. It prevents memory errors at "
            "compile time through its ownership system.",
        )
        return rag

    def test_basic_search(self, idx):
        results = idx.search("web browser language")
        assert len(results) > 0
        assert results[0].doc_id == "javascript"

    def test_search_relevance(self, idx):
        results = idx.search("memory safety systems")
        assert len(results) > 0
        assert results[0].doc_id == "rust"

    def test_search_python(self, idx):
        results = idx.search("readability object oriented")
        assert results[0].doc_id == "python"

    def test_empty_search(self):
        rag = RAGIndex()
        assert rag.search("anything") == []

    def test_no_match(self, idx):
        results = idx.search("quantum physics black holes")
        # May return low-score results or empty
        if results:
            assert results[0].score < 1.0

    def test_add_returns_chunk_count(self):
        rag = RAGIndex(chunk_size=10)
        count = rag.add("big", " ".join(["word"] * 100))
        assert count > 1

    def test_size(self, idx):
        assert idx.size >= 3

    def test_clear(self, idx):
        idx.clear()
        assert idx.size == 0

    def test_add_file(self, idx, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("This is a test document about databases and SQL queries.")
        count = idx.add_file(str(f))
        assert count >= 1
        results = idx.search("database SQL")
        assert any(r.doc_id == "test.txt" for r in results)

    def test_result_has_score(self, idx):
        results = idx.search("programming")
        for r in results:
            assert isinstance(r.score, float)
            assert r.score > 0
