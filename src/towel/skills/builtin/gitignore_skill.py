"""Gitignore skill — generate and manage .gitignore files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition

_TEMPLATES: dict[str, list[str]] = {
    "python": ["__pycache__/","*.pyc","*.pyo",".venv/","venv/","dist/","build/","*.egg-info/",".pytest_cache/",".mypy_cache/",".ruff_cache/",".env","*.db",".tox/"],
    "node": ["node_modules/","dist/","build/",".env","*.log","coverage/",".next/",".nuxt/",".cache/","*.tsbuildinfo"],
    "rust": ["target/","Cargo.lock","**/*.rs.bk"],
    "go": ["bin/","vendor/","*.exe","*.test","*.out"],
    "java": ["*.class","*.jar","*.war","target/","build/",".gradle/","out/"],
    "swift": [".build/","Packages/","*.xcodeproj/","*.xcworkspace/","DerivedData/"],
    "c": ["*.o","*.a","*.so","*.dylib","*.out","build/","cmake-build-*/"],
    "general": [".DS_Store","Thumbs.db","*.swp","*.swo","*~",".idea/",".vscode/","*.log",".env"],
    "docker": [".dockerignore","docker-compose.override.yml","*.tar"],
    "terraform": [".terraform/","*.tfstate","*.tfstate.*","crash.log","*.tfvars"],
}


class GitignoreSkill(Skill):
    @property
    def name(self) -> str: return "gitignore"
    @property
    def description(self) -> str: return "Generate and manage .gitignore files"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="gitignore_generate", description="Generate a .gitignore for a language/framework",
                parameters={"type":"object","properties":{
                    "languages":{"type":"array","items":{"type":"string"},
                                 "description":f"Languages: {', '.join(_TEMPLATES.keys())}"},
                    "extras":{"type":"array","items":{"type":"string"},"description":"Extra patterns to add"},
                },"required":["languages"]}),
            ToolDefinition(name="gitignore_check", description="Check which files would be ignored by current .gitignore",
                parameters={"type":"object","properties":{
                    "path":{"type":"string","description":"Project directory (default: cwd)"},
                }}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "gitignore_generate": return self._generate(arguments["languages"], arguments.get("extras",[]))
            case "gitignore_check": return self._check(arguments.get("path","."))
            case _: return f"Unknown tool: {tool_name}"

    def _generate(self, languages: list[str], extras: list[str]) -> str:
        patterns: list[str] = []
        for lang in languages:
            tmpl = _TEMPLATES.get(lang.lower())
            if tmpl:
                patterns.append(f"# {lang}")
                patterns.extend(tmpl)
                patterns.append("")
            else:
                patterns.append(f"# Unknown: {lang}")
        if extras:
            patterns.append("# Custom")
            patterns.extend(extras)
        seen: set[str] = set()
        deduped = []
        for p in patterns:
            if p not in seen or p == "" or p.startswith("#"):
                seen.add(p)
                deduped.append(p)
        return "\n".join(deduped)

    def _check(self, path: str) -> str:
        root = Path(path).expanduser().resolve()
        gi = root / ".gitignore"
        if not gi.exists(): return "No .gitignore found."
        patterns = [l.strip() for l in gi.read_text().splitlines() if l.strip() and not l.startswith("#")]
        return f".gitignore patterns ({len(patterns)}):\n" + "\n".join(f"  {p}" for p in patterns)
