"""Tests for the plugin system."""

from towel.skills.plugin import (
    PluginManifest,
    create_plugin_scaffold,
    discover_plugins,
    validate_plugin,
)


class TestPluginManifest:
    def test_from_toml(self, tmp_path):
        d = tmp_path / "my-plugin"
        d.mkdir()
        (d / "towel-plugin.toml").write_text("""
[plugin]
name = "test-plugin"
version = "1.2.3"
description = "A test"
author = "Tester"
tags = ["test"]
""")
        manifest = PluginManifest.from_toml(d / "towel-plugin.toml")
        assert manifest is not None
        assert manifest.name == "test-plugin"
        assert manifest.version == "1.2.3"
        assert manifest.author == "Tester"
        assert "test" in manifest.tags

    def test_to_dict(self):
        m = PluginManifest(name="x", version="1.0.0", description="test")
        d = m.to_dict()
        assert d["name"] == "x"
        assert d["version"] == "1.0.0"


class TestDiscoverPlugins:
    def test_empty_dir(self, tmp_path):
        assert discover_plugins([tmp_path]) == []

    def test_finds_plugin(self, tmp_path):
        d = tmp_path / "my-plugin"
        d.mkdir()
        (d / "towel-plugin.toml").write_text('[plugin]\nname = "found"\nversion = "1.0.0"')
        found = discover_plugins([tmp_path])
        assert len(found) == 1
        assert found[0].name == "found"


class TestValidatePlugin:
    def test_valid(self, tmp_path):
        d = tmp_path / "good"
        d.mkdir()
        (d / "towel-plugin.toml").write_text('[plugin]\nname = "good"\nversion = "1.0.0"')
        (d / "skill.py").write_text("pass")
        issues = validate_plugin(d)
        assert issues == []

    def test_missing_manifest(self, tmp_path):
        d = tmp_path / "bad"
        d.mkdir()
        issues = validate_plugin(d)
        assert any("Missing" in i for i in issues)

    def test_missing_skill(self, tmp_path):
        d = tmp_path / "no-skill"
        d.mkdir()
        (d / "towel-plugin.toml").write_text('[plugin]\nname = "no-skill"\nversion = "1.0.0"')
        issues = validate_plugin(d)
        assert any("skill file" in i.lower() for i in issues)


class TestCreateScaffold:
    def test_creates_files(self, tmp_path):
        path = create_plugin_scaffold("my-cool-plugin", output_dir=tmp_path)
        assert (path / "towel-plugin.toml").exists()
        assert (path / "skill.py").exists()
        manifest_text = (path / "towel-plugin.toml").read_text()
        assert "my-cool-plugin" in manifest_text
        skill_text = (path / "skill.py").read_text()
        assert "MyCoolPluginSkill" in skill_text
