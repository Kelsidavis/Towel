"""Tests for @file reference expansion."""

from pathlib import Path

import pytest

from towel.agent.refs import parse_refs, expand_refs, FileRef, _ext_to_lang


class TestParseRefs:
    def test_simple_file(self):
        refs = parse_refs("explain @src/main.py please")
        assert len(refs) == 1
        assert refs[0].path == "src/main.py"
        assert refs[0].line_start is None

    def test_relative_path(self):
        refs = parse_refs("look at @./config.toml")
        assert len(refs) == 1
        assert refs[0].path == "./config.toml"

    def test_home_path(self):
        refs = parse_refs("check @~/.towel/config.toml")
        assert len(refs) == 1
        assert refs[0].path == "~/.towel/config.toml"

    def test_line_range(self):
        refs = parse_refs("explain @main.py:10-20")
        assert len(refs) == 1
        assert refs[0].path == "main.py"
        assert refs[0].line_start == 10
        assert refs[0].line_end == 20

    def test_single_line(self):
        refs = parse_refs("what does @main.py:42 do")
        assert len(refs) == 1
        assert refs[0].line_start == 42
        assert refs[0].line_end is None

    def test_glob_pattern(self):
        refs = parse_refs("review @src/*.py")
        assert len(refs) == 1
        assert refs[0].path == "src/*.py"

    def test_multiple_refs(self):
        refs = parse_refs("compare @file_a.py and @file_b.py")
        assert len(refs) == 2
        assert refs[0].path == "file_a.py"
        assert refs[1].path == "file_b.py"

    def test_no_refs(self):
        refs = parse_refs("just a normal message")
        assert len(refs) == 0

    def test_email_not_matched(self):
        refs = parse_refs("email me at user@example.com")
        assert len(refs) == 0

    def test_bare_word_without_extension_ignored(self):
        refs = parse_refs("@someone said hello")
        assert len(refs) == 0


class TestExpandRefs:
    def test_expands_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "hello.py").write_text("print('hello')")

        result = expand_refs("explain @hello.py")
        assert "print('hello')" in result
        assert "```python" in result
        assert "hello.py" in result

    def test_preserves_surrounding_text(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "x.py").write_text("code")

        result = expand_refs("before @x.py after")
        assert result.startswith("before ")
        assert result.endswith(" after")
        assert "code" in result

    def test_nonexistent_file_left_alone(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = expand_refs("look at @nonexistent.py")
        assert result == "look at @nonexistent.py"

    def test_line_range(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "lines.txt").write_text("line1\nline2\nline3\nline4\nline5")

        result = expand_refs("show @lines.txt:2-4")
        assert "line2" in result
        assert "line3" in result
        assert "line4" in result
        assert "line1" not in result
        assert "line5" not in result

    def test_glob_expansion(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a.py").write_text("aaa")
        (tmp_path / "b.py").write_text("bbb")
        (tmp_path / "c.txt").write_text("ccc")

        result = expand_refs("review @*.py")
        assert "aaa" in result
        assert "bbb" in result
        assert "ccc" not in result

    def test_no_refs_passthrough(self):
        text = "just a normal message"
        assert expand_refs(text) == text

    def test_large_file_warning(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "huge.txt").write_text("x" * 600_000)

        result = expand_refs("check @huge.txt")
        assert "too large" in result

    def test_multiple_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "one.py").write_text("first")
        (tmp_path / "two.py").write_text("second")

        result = expand_refs("compare @one.py and @two.py")
        assert "first" in result
        assert "second" in result


class TestUrlRefs:
    def test_url_pattern_parsed(self):
        from towel.agent.refs import _URL_PATTERN
        m = _URL_PATTERN.findall("check @https://example.com/api.json please")
        assert len(m) == 1
        assert m[0] == "https://example.com/api.json"

    def test_http_pattern(self):
        from towel.agent.refs import _URL_PATTERN
        m = _URL_PATTERN.findall("see @http://localhost:8080/health")
        assert len(m) == 1

    def test_url_not_confused_with_file(self):
        # URL should not appear in file refs
        refs = parse_refs("check @https://example.com/file.py")
        # File parser shouldn't match URLs (they start with https://)
        for r in refs:
            assert not r.path.startswith("https://")

    def test_expand_url_fetch(self):
        """Test URL expansion with a mocked httpx response."""
        from unittest.mock import patch, MagicMock

        mock_resp = MagicMock()
        mock_resp.text = '{"status": "ok"}'
        mock_resp.headers = {"content-type": "application/json"}

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp

        with patch("httpx.Client", return_value=mock_client):
            result = expand_refs("check @https://api.example.com/status.json")
        assert '{"status": "ok"}' in result
        assert "```json" in result

    def test_expand_url_failure(self):
        """Failed URL fetch should show error inline."""
        from unittest.mock import patch, MagicMock

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = ConnectionError("refused")

        with patch("httpx.Client", return_value=mock_client):
            result = expand_refs("check @https://down.example.com/api")
        assert "Failed to fetch" in result


class TestExtToLang:
    def test_python(self):
        assert _ext_to_lang("py") == "python"

    def test_javascript(self):
        assert _ext_to_lang("js") == "javascript"

    def test_rust(self):
        assert _ext_to_lang("rs") == "rust"

    def test_unknown(self):
        assert _ext_to_lang("xyz") == "xyz"

    def test_toml(self):
        assert _ext_to_lang("toml") == "toml"
