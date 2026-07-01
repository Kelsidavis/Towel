"""Tests for the REPL key-value argument parser."""

from towel.cli.repl import parse_kv_args


class TestParseKvArgs:
    def test_simple_string(self):
        assert parse_kv_args("text=hello") == {"text": "hello"}

    def test_multiple_pairs(self):
        result = parse_kv_args("name=Alice age=30")
        assert result == {"name": "Alice", "age": 30}

    def test_json_array(self):
        result = parse_kv_args('options=["red","green","blue"]')
        assert result == {"options": ["red", "green", "blue"]}

    def test_json_object(self):
        result = parse_kv_args('data={"key":"value"}')
        assert result == {"data": {"key": "value"}}

    def test_numeric_value(self):
        result = parse_kv_args("count=42")
        assert result == {"count": 42}

    def test_boolean_value(self):
        result = parse_kv_args("verbose=true")
        assert result == {"verbose": True}

    def test_quoted_string(self):
        result = parse_kv_args('text="hello world"')
        assert result == {"text": "hello world"}

    def test_empty_string(self):
        assert parse_kv_args("") == {}

    def test_mixed_types(self):
        result = parse_kv_args('name=test count=5 items=["a","b"]')
        assert result == {"name": "test", "count": 5, "items": ["a", "b"]}
