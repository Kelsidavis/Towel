"""Tests for the data skill."""

import pytest

from towel.skills.builtin.data import DataSkill


@pytest.fixture
def skill():
    return DataSkill()


class TestTools:
    def test_tools_defined(self, skill):
        names = {t.name for t in skill.tools()}
        assert names == {"parse_json", "parse_csv", "format_json", "calculate"}


class TestParseJSON:
    @pytest.mark.asyncio
    async def test_basic(self, skill):
        result = await skill.execute("parse_json", {"data": '{"name": "Towel"}'})
        assert "Towel" in result

    @pytest.mark.asyncio
    async def test_path_query(self, skill):
        data = '{"users": [{"name": "Alice"}, {"name": "Bob"}]}'
        result = await skill.execute("parse_json", {"data": data, "path": "users.0.name"})
        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_wildcard(self, skill):
        data = '{"items": [{"id": 1}, {"id": 2}, {"id": 3}]}'
        result = await skill.execute("parse_json", {"data": data, "path": "items.*.id"})
        assert "1" in result
        assert "2" in result
        assert "3" in result

    @pytest.mark.asyncio
    async def test_invalid_json(self, skill):
        result = await skill.execute("parse_json", {"data": "not json"})
        assert "Invalid" in result


class TestParseCSV:
    @pytest.mark.asyncio
    async def test_basic(self, skill):
        csv = "name,age\nAlice,30\nBob,25"
        result = await skill.execute("parse_csv", {"data": csv})
        assert "Alice" in result
        assert "Bob" in result
        assert "columns" in result

    @pytest.mark.asyncio
    async def test_custom_delimiter(self, skill):
        tsv = "name\tage\nAlice\t30"
        result = await skill.execute("parse_csv", {"data": tsv, "delimiter": "\t"})
        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_limit(self, skill):
        csv = "x\n" + "\n".join(str(i) for i in range(100))
        result = await skill.execute("parse_csv", {"data": csv, "limit": 5})
        assert '"row_count": 5' in result


class TestFormatJSON:
    @pytest.mark.asyncio
    async def test_pretty(self, skill):
        result = await skill.execute("format_json", {"data": '{"a":1,"b":2}'})
        assert "\n" in result  # pretty printed

    @pytest.mark.asyncio
    async def test_compact(self, skill):
        result = await skill.execute("format_json", {"data": '{"a": 1, "b": 2}', "compact": True})
        assert " " not in result.strip() or result == '{"a":1,"b":2}'

    @pytest.mark.asyncio
    async def test_invalid(self, skill):
        result = await skill.execute("format_json", {"data": "nope"})
        assert "Invalid" in result


class TestCalculate:
    @pytest.mark.asyncio
    async def test_basic_math(self, skill):
        result = await skill.execute("calculate", {"expression": "2 + 3 * 4"})
        assert result == "14"

    @pytest.mark.asyncio
    async def test_power(self, skill):
        result = await skill.execute("calculate", {"expression": "2 ** 10"})
        assert result == "1024"

    @pytest.mark.asyncio
    async def test_functions(self, skill):
        result = await skill.execute("calculate", {"expression": "abs(-42)"})
        assert result == "42"

    @pytest.mark.asyncio
    async def test_sum(self, skill):
        result = await skill.execute("calculate", {"expression": "sum([1, 2, 3, 4, 5])"})
        assert result == "15"

    @pytest.mark.asyncio
    async def test_error(self, skill):
        result = await skill.execute("calculate", {"expression": "1/0"})
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_no_builtins_access(self, skill):
        result = await skill.execute(
            "calculate", {"expression": "__import__('os').system('echo hi')"}
        )
        assert "error" in result.lower()
