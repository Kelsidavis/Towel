"""NPM/package.json skill — inspect dependencies, scripts, and outdated packages."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class NpmSkill(Skill):
    @property
    def name(self) -> str: return "npm"
    @property
    def description(self) -> str: return "Inspect package.json — dependencies, scripts, and project info"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="npm_info", description="Show package.json summary (name, version, deps, scripts)",
                parameters={"type":"object","properties":{
                    "path":{"type":"string","description":"Path to package.json or project dir (default: cwd)"},
                }}),
            ToolDefinition(name="npm_deps", description="List all dependencies with versions",
                parameters={"type":"object","properties":{
                    "path":{"type":"string","description":"Path to package.json or project dir"},
                    "dev":{"type":"boolean","description":"Include devDependencies (default: true)"},
                }}),
            ToolDefinition(name="npm_scripts", description="List available npm scripts",
                parameters={"type":"object","properties":{
                    "path":{"type":"string","description":"Path to package.json or project dir"},
                }}),
            ToolDefinition(name="npm_audit_check", description="Check for known issues in dependencies (reads package.json only, no network)",
                parameters={"type":"object","properties":{
                    "path":{"type":"string","description":"Path to package.json or project dir"},
                }}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "npm_info": return self._info(arguments.get("path", "."))
            case "npm_deps": return self._deps(arguments.get("path", "."), arguments.get("dev", True))
            case "npm_scripts": return self._scripts(arguments.get("path", "."))
            case "npm_audit_check": return self._audit(arguments.get("path", "."))
            case _: return f"Unknown tool: {tool_name}"

    def _load(self, path: str) -> dict | str:
        p = Path(path).expanduser()
        if p.is_dir(): p = p / "package.json"
        if not p.is_file(): return f"Not found: {p}"
        try: return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e: return f"Error reading: {e}"

    def _info(self, path: str) -> str:
        pkg = self._load(path)
        if isinstance(pkg, str): return pkg
        deps = len(pkg.get("dependencies", {}))
        devdeps = len(pkg.get("devDependencies", {}))
        scripts = len(pkg.get("scripts", {}))
        lines = [f"Package: {pkg.get('name', '?')} v{pkg.get('version', '?')}"]
        if pkg.get("description"): lines.append(f"  {pkg['description']}")
        lines.append(f"  Dependencies: {deps} + {devdeps} dev")
        lines.append(f"  Scripts: {scripts}")
        if pkg.get("license"): lines.append(f"  License: {pkg['license']}")
        if pkg.get("engines"): lines.append(f"  Engines: {json.dumps(pkg['engines'])}")
        return "\n".join(lines)

    def _deps(self, path: str, dev: bool) -> str:
        pkg = self._load(path)
        if isinstance(pkg, str): return pkg
        lines = []
        deps = pkg.get("dependencies", {})
        if deps:
            lines.append(f"Dependencies ({len(deps)}):")
            for name, ver in sorted(deps.items()):
                lines.append(f"  {name}: {ver}")
        devdeps = pkg.get("devDependencies", {})
        if dev and devdeps:
            lines.append(f"\nDev dependencies ({len(devdeps)}):")
            for name, ver in sorted(devdeps.items()):
                lines.append(f"  {name}: {ver}")
        return "\n".join(lines) or "No dependencies."

    def _scripts(self, path: str) -> str:
        pkg = self._load(path)
        if isinstance(pkg, str): return pkg
        scripts = pkg.get("scripts", {})
        if not scripts: return "No scripts defined."
        lines = [f"Scripts ({len(scripts)}):"]
        for name, cmd in scripts.items():
            lines.append(f"  {name}: {cmd[:80]}")
        return "\n".join(lines)

    def _audit(self, path: str) -> str:
        pkg = self._load(path)
        if isinstance(pkg, str): return pkg
        all_deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
        issues = []
        for name, ver in all_deps.items():
            if ver in ("*", "latest"): issues.append(f"  [!] {name}: {ver} (unpinned)")
            elif ver.startswith("git") or ver.startswith("http"): issues.append(f"  [!] {name}: {ver} (URL dependency)")
            elif "file:" in ver: issues.append(f"  [!] {name}: {ver} (local file)")
        if not issues: return f"No issues found in {len(all_deps)} dependencies."
        return f"Found {len(issues)} issue(s):\n" + "\n".join(issues)
