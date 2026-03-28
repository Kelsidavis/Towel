"""Plugin system — package, validate, and manage skill plugins.

A plugin is a directory with a towel-plugin.toml manifest:

    my-plugin/
        towel-plugin.toml
        skill.py            (or __init__.py)

Manifest format (towel-plugin.toml):
    [plugin]
    name = "weather"
    version = "1.0.0"
    description = "Get weather forecasts"
    author = "Jane Doe"
    license = "MIT"
    min_towel = "0.3.0"
    tags = ["weather", "api"]

    [plugin.dependencies]
    httpx = ">=0.28"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from towel.config import TOWEL_HOME

log = logging.getLogger("towel.skills.plugin")

PLUGINS_DIR = TOWEL_HOME / "plugins"


@dataclass
class PluginManifest:
    """Parsed plugin manifest."""
    name: str
    version: str
    description: str = ""
    author: str = ""
    license: str = ""
    min_towel: str = ""
    tags: list[str] = field(default_factory=list)
    dependencies: dict[str, str] = field(default_factory=dict)
    path: Path = field(default_factory=lambda: Path("."))

    @classmethod
    def from_toml(cls, path: Path) -> PluginManifest | None:
        """Parse a towel-plugin.toml file."""
        try:
            import toml
            data = toml.loads(path.read_text(encoding="utf-8"))
            plugin = data.get("plugin", {})
            return cls(
                name=plugin.get("name", path.parent.name),
                version=plugin.get("version", "0.0.0"),
                description=plugin.get("description", ""),
                author=plugin.get("author", ""),
                license=plugin.get("license", ""),
                min_towel=plugin.get("min_towel", ""),
                tags=plugin.get("tags", []),
                dependencies=plugin.get("dependencies", {}),
                path=path.parent,
            )
        except Exception as e:
            log.warning(f"Failed to parse {path}: {e}")
            return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "tags": self.tags,
            "path": str(self.path),
        }


def discover_plugins(dirs: list[Path] | None = None) -> list[PluginManifest]:
    """Discover plugins in the plugins directory."""
    search_dirs = dirs or [PLUGINS_DIR]
    plugins: list[PluginManifest] = []

    for d in search_dirs:
        if not d.exists():
            continue
        for entry in sorted(d.iterdir()):
            if entry.is_dir():
                manifest_path = entry / "towel-plugin.toml"
                if manifest_path.exists():
                    manifest = PluginManifest.from_toml(manifest_path)
                    if manifest:
                        plugins.append(manifest)

    return plugins


def validate_plugin(path: Path) -> list[str]:
    """Validate a plugin directory. Returns list of issues (empty = valid)."""
    issues: list[str] = []

    if not path.is_dir():
        return [f"Not a directory: {path}"]

    manifest_path = path / "towel-plugin.toml"
    if not manifest_path.exists():
        issues.append("Missing towel-plugin.toml")
        return issues

    manifest = PluginManifest.from_toml(manifest_path)
    if not manifest:
        issues.append("Failed to parse towel-plugin.toml")
        return issues

    if not manifest.name:
        issues.append("Missing plugin name")
    if not manifest.version:
        issues.append("Missing version")

    # Check for skill file
    has_skill = (
        (path / "skill.py").exists() or
        (path / "__init__.py").exists() or
        any(path.glob("*_skill.py"))
    )
    if not has_skill:
        issues.append("No skill file found (need skill.py, __init__.py, or *_skill.py)")

    # Check dependencies
    for dep, ver in manifest.dependencies.items():
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            issues.append(f"Missing dependency: {dep} {ver}")

    return issues


def create_plugin_scaffold(name: str, output_dir: Path | None = None) -> Path:
    """Create a plugin directory with manifest and skeleton skill."""
    target = (output_dir or PLUGINS_DIR) / name
    target.mkdir(parents=True, exist_ok=True)

    class_name = "".join(w.capitalize() for w in name.replace("-", "_").split("_")) + "Skill"

    manifest = f"""[plugin]
name = "{name}"
version = "0.1.0"
description = "A custom Towel plugin"
author = ""
license = "MIT"
min_towel = "0.5.0"
tags = []

[plugin.dependencies]
# httpx = ">=0.28"
"""

    skill_code = f'''"""Plugin: {name}"""

from __future__ import annotations
from typing import Any
from towel.skills.base import Skill, ToolDefinition


class {class_name}(Skill):
    @property
    def name(self) -> str:
        return "{name}"

    @property
    def description(self) -> str:
        return "Description of {name} plugin"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="{name.replace("-", "_")}_example",
                description="An example tool",
                parameters={{"type": "object", "properties": {{
                    "input": {{"type": "string", "description": "Input text"}},
                }}, "required": ["input"]}},
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return f"Got: {{arguments.get('input', '')}}"
'''

    (target / "towel-plugin.toml").write_text(manifest, encoding="utf-8")
    (target / "skill.py").write_text(skill_code, encoding="utf-8")

    return target
