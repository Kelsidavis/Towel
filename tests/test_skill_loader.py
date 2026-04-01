"""Tests for the skill auto-loader."""

import textwrap

import pytest

from towel.skills.loader import SkillLoader
from towel.skills.registry import SkillRegistry

SINGLE_FILE_SKILL = textwrap.dedent("""\
    from typing import Any
    from towel.skills.base import Skill, ToolDefinition

    class GreeterSkill(Skill):
        @property
        def name(self) -> str:
            return "greeter"

        @property
        def description(self) -> str:
            return "A friendly greeter"

        def tools(self) -> list[ToolDefinition]:
            return [
                ToolDefinition(name="greet", description="Say hello"),
            ]

        async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
            return f"Hello, {arguments.get('name', 'world')}!"
""")

PACKAGE_INIT_SKILL = textwrap.dedent("""\
    from typing import Any
    from towel.skills.base import Skill, ToolDefinition

    class MathSkill(Skill):
        @property
        def name(self) -> str:
            return "math"

        @property
        def description(self) -> str:
            return "Basic math"

        def tools(self) -> list[ToolDefinition]:
            return [ToolDefinition(name="add", description="Add numbers")]

        async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
            return arguments.get("a", 0) + arguments.get("b", 0)
""")

BAD_SKILL = "raise ImportError('intentional test error')"


class TestSkillLoader:
    def test_load_single_file_skill(self, tmp_path):
        (tmp_path / "greeter.py").write_text(SINGLE_FILE_SKILL)
        reg = SkillRegistry()
        loader = SkillLoader(reg)
        loaded = loader.load_from_dirs([str(tmp_path)])
        assert loaded == 1
        assert "greeter" in reg.list_skills()

    def test_load_package_skill(self, tmp_path):
        pkg = tmp_path / "mathpkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text(PACKAGE_INIT_SKILL)
        reg = SkillRegistry()
        loader = SkillLoader(reg)
        loaded = loader.load_from_dirs([str(tmp_path)])
        assert loaded == 1
        assert "math" in reg.list_skills()

    def test_skip_nonexistent_dir(self):
        reg = SkillRegistry()
        loader = SkillLoader(reg)
        loaded = loader.load_from_dirs(["/nonexistent/path"])
        assert loaded == 0
        assert len(loader.errors) == 0

    def test_bad_skill_records_error(self, tmp_path):
        (tmp_path / "bad.py").write_text(BAD_SKILL)
        reg = SkillRegistry()
        loader = SkillLoader(reg)
        loaded = loader.load_from_dirs([str(tmp_path)])
        assert loaded == 0
        assert len(loader.errors) == 1
        assert "intentional" in str(loader.errors[0].error)

    def test_skip_hidden_and_dunder(self, tmp_path):
        (tmp_path / ".hidden.py").write_text(SINGLE_FILE_SKILL)
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cache.py").write_text(SINGLE_FILE_SKILL)
        reg = SkillRegistry()
        loader = SkillLoader(reg)
        loaded = loader.load_from_dirs([str(tmp_path)])
        assert loaded == 0

    def test_no_duplicate_registration(self, tmp_path):
        (tmp_path / "greeter.py").write_text(SINGLE_FILE_SKILL)
        reg = SkillRegistry()
        loader = SkillLoader(reg)
        # Load once
        loader.load_from_dirs([str(tmp_path)])
        # Load again — should skip duplicate
        loaded = loader.load_from_dirs([str(tmp_path)])
        assert loaded == 0

    def test_multiple_dirs(self, tmp_path):
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        (dir1 / "greeter.py").write_text(SINGLE_FILE_SKILL)
        (dir2 / "mathpkg").mkdir()
        (dir2 / "mathpkg" / "__init__.py").write_text(PACKAGE_INIT_SKILL)

        reg = SkillRegistry()
        loader = SkillLoader(reg)
        loaded = loader.load_from_dirs([str(dir1), str(dir2)])
        assert loaded == 2
        assert set(reg.list_skills()) == {"greeter", "math"}

    @pytest.mark.asyncio
    async def test_loaded_skill_executes(self, tmp_path):
        (tmp_path / "greeter.py").write_text(SINGLE_FILE_SKILL)
        reg = SkillRegistry()
        loader = SkillLoader(reg)
        loader.load_from_dirs([str(tmp_path)])
        result = await reg.execute_tool("greet", {"name": "Towel"})
        assert result == "Hello, Towel!"
