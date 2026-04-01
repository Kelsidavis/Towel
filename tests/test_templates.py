"""Tests for prompt templates."""

import pytest

from towel.templates.engine import BUILTIN_TEMPLATES, TemplateEngine


@pytest.fixture
def engine(tmp_path):
    return TemplateEngine(templates_dir=tmp_path)


class TestBuiltinTemplates:
    def test_builtins_exist(self):
        assert "review" in BUILTIN_TEMPLATES
        assert "explain" in BUILTIN_TEMPLATES
        assert "summarize" in BUILTIN_TEMPLATES
        assert "translate" in BUILTIN_TEMPLATES
        assert "fix" in BUILTIN_TEMPLATES
        assert "test" in BUILTIN_TEMPLATES
        assert "commit" in BUILTIN_TEMPLATES
        assert "refactor" in BUILTIN_TEMPLATES

    def test_all_have_descriptions(self):
        engine = TemplateEngine()
        tpls = engine.list_templates()
        for name, desc in tpls.items():
            assert desc, f"Template '{name}' has no description"


class TestListTemplates:
    def test_lists_builtins(self, engine):
        tpls = engine.list_templates()
        assert "review" in tpls
        assert "Code review" in tpls["review"]

    def test_user_templates(self, engine, tmp_path):
        (tmp_path / "custom.txt").write_text("# My custom template\nDo something with {{input}}")
        tpls = engine.list_templates()
        assert "custom" in tpls
        assert "My custom template" in tpls["custom"]

    def test_user_overrides_builtin(self, engine, tmp_path):
        (tmp_path / "review.txt").write_text("# My review\nCustom review: {{input}}")
        tpls = engine.list_templates()
        assert "My review" in tpls["review"]


class TestRender:
    def test_basic_render(self, engine):
        result = engine.render("review", input_text="def foo(): pass")
        assert result is not None
        assert "def foo(): pass" in result
        assert "Review" in result or "review" in result

    def test_input_replacement(self, engine):
        result = engine.render("explain", input_text="quantum entanglement")
        assert result is not None
        assert "quantum entanglement" in result

    def test_variable_with_default(self, engine):
        result = engine.render("translate", input_text="hello")
        assert result is not None
        assert "English" in result  # default

    def test_variable_override(self, engine):
        result = engine.render("translate", input_text="hello", variables={"lang": "Spanish"})
        assert result is not None
        assert "Spanish" in result
        assert "English" not in result

    def test_nonexistent_template(self, engine):
        assert engine.render("nonexistent") is None

    def test_user_template(self, engine, tmp_path):
        (tmp_path / "greet.txt").write_text("# Greeting\nSay hello to {{name|World}}: {{input}}")
        result = engine.render("greet", input_text="Hi there")
        assert result is not None
        assert "World" in result
        assert "Hi there" in result

    def test_user_template_with_var(self, engine, tmp_path):
        (tmp_path / "greet.txt").write_text("# Greeting\nSay hello to {{name|World}}: {{input}}")
        result = engine.render("greet", input_text="Hi", variables={"name": "Kelsi"})
        assert "Kelsi" in result

    def test_description_stripped(self, engine):
        result = engine.render("review", input_text="code")
        assert result is not None
        assert not result.startswith("#")

    def test_empty_input(self, engine):
        result = engine.render("summarize", input_text="")
        assert result is not None


class TestCLI:
    def test_templates_command(self):
        from click.testing import CliRunner

        from towel.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["templates"])
        assert result.exit_code == 0
        assert "review" in result.output
        assert "summarize" in result.output
