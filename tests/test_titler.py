"""Tests for auto-title generation."""

from towel.agent.titler import generate_title


class TestGenerateTitle:
    def test_basic(self):
        title = generate_title("How do I deploy a Python app to AWS?")
        assert title
        assert len(title.split()) <= 6
        # Should contain meaningful words
        lower = title.lower()
        assert "deploy" in lower or "python" in lower or "aws" in lower

    def test_strips_filler(self):
        title = generate_title("Can you please help me understand the database architecture?")
        lower = title.lower()
        assert "please" not in lower
        assert "database" in lower

    def test_code_question(self):
        title = generate_title("What does this function do? def factorial(n): return n * factorial(n-1)")
        assert title
        assert len(title) > 0

    def test_file_ref_stripped(self):
        title = generate_title("explain @src/main.py and @utils/helpers.py")
        assert "@" not in title

    def test_url_stripped(self):
        title = generate_title("summarize https://example.com/article")
        assert "https" not in title
        assert title

    def test_code_block_stripped(self):
        title = generate_title("fix this:\n```python\nprint('hello')\n```")
        assert "```" not in title

    def test_short_input(self):
        title = generate_title("hi")
        # Even very short input should produce something
        assert isinstance(title, str)

    def test_empty_input(self):
        title = generate_title("")
        assert title == ""

    def test_max_words(self):
        title = generate_title(
            "explain the difference between merge sort quick sort "
            "bubble sort insertion sort heap sort radix sort bucket sort"
        )
        assert len(title.split()) <= 6

    def test_title_case(self):
        title = generate_title("how do kubernetes pods communicate")
        words = title.split()
        for w in words:
            assert w[0].isupper(), f"'{w}' should be capitalized"

    def test_with_assistant_message(self):
        title = generate_title(
            "What is Docker?",
            "Docker is a containerization platform..."
        )
        assert title
        assert "docker" in title.lower()

    def test_practical_examples(self):
        cases = [
            ("Write a Python script to parse CSV files", "python"),
            ("How do I fix a segfault in my C program", "segfault"),
            ("Explain React hooks and state management", "react"),
            ("What's the best way to handle errors in Rust", "rust"),
            ("Create a REST API with FastAPI", "rest"),
        ]
        for prompt, expected_word in cases:
            title = generate_title(prompt)
            assert title, f"No title for: {prompt}"
            assert expected_word in title.lower(), f"'{expected_word}' not in title '{title}' for: {prompt}"
