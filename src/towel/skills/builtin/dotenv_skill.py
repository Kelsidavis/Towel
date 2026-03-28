"""Dotenv skill — read, validate, and compare .env files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition


def _parse_env(text: str) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        if "=" not in line: continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'\"")
        result[key] = val
    return result


class DotenvSkill(Skill):
    @property
    def name(self) -> str: return "dotenv"
    @property
    def description(self) -> str: return "Read, validate, and compare .env files"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="dotenv_read", description="Read and parse a .env file",
                parameters={"type":"object","properties":{
                    "path":{"type":"string","description":"Path to .env file (default: .env)"},
                },"required":[]}),
            ToolDefinition(name="dotenv_validate", description="Check a .env file against an .env.example template",
                parameters={"type":"object","properties":{
                    "env_path":{"type":"string","description":"Path to .env (default: .env)"},
                    "template_path":{"type":"string","description":"Path to template (default: .env.example)"},
                },"required":[]}),
            ToolDefinition(name="dotenv_diff", description="Compare two .env files and show differences",
                parameters={"type":"object","properties":{
                    "path_a":{"type":"string","description":"First .env file"},
                    "path_b":{"type":"string","description":"Second .env file"},
                },"required":["path_a","path_b"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "dotenv_read": return self._read(arguments.get("path", ".env"))
            case "dotenv_validate": return self._validate(arguments.get("env_path",".env"), arguments.get("template_path",".env.example"))
            case "dotenv_diff": return self._diff(arguments["path_a"], arguments["path_b"])
            case _: return f"Unknown tool: {tool_name}"

    def _read(self, path: str) -> str:
        p = Path(path).expanduser()
        if not p.is_file(): return f"Not found: {path}"
        env = _parse_env(p.read_text(encoding="utf-8", errors="replace"))
        if not env: return "Empty or no valid entries."
        sensitive = {"key","secret","token","password","passwd","auth","credential"}
        lines = [f".env ({len(env)} variables):"]
        for k, v in sorted(env.items()):
            if any(s in k.lower() for s in sensitive):
                lines.append(f"  {k}=****")
            else:
                display = v[:40] + "..." if len(v) > 40 else v
                lines.append(f"  {k}={display}")
        return "\n".join(lines)

    def _validate(self, env_path: str, template_path: str) -> str:
        ep, tp = Path(env_path).expanduser(), Path(template_path).expanduser()
        if not ep.is_file(): return f"Not found: {env_path}"
        if not tp.is_file(): return f"Not found: {template_path}"
        env = _parse_env(ep.read_text())
        tmpl = _parse_env(tp.read_text())
        missing = [k for k in tmpl if k not in env]
        extra = [k for k in env if k not in tmpl]
        empty = [k for k in tmpl if k in env and not env[k]]
        lines = [f"Validation: {env_path} vs {template_path}"]
        if missing: lines.append(f"\n  Missing ({len(missing)}): {', '.join(missing)}")
        if empty: lines.append(f"\n  Empty ({len(empty)}): {', '.join(empty)}")
        if extra: lines.append(f"\n  Extra ({len(extra)}): {', '.join(extra)}")
        if not missing and not empty: lines.append("\n  All required variables are set.")
        return "\n".join(lines)

    def _diff(self, path_a: str, path_b: str) -> str:
        pa, pb = Path(path_a).expanduser(), Path(path_b).expanduser()
        if not pa.is_file(): return f"Not found: {path_a}"
        if not pb.is_file(): return f"Not found: {path_b}"
        a, b = _parse_env(pa.read_text()), _parse_env(pb.read_text())
        all_keys = sorted(set(a) | set(b))
        lines = [f"Diff: {path_a} vs {path_b}"]
        diffs = 0
        for k in all_keys:
            if k not in a: lines.append(f"  + {k} (only in {path_b})"); diffs += 1
            elif k not in b: lines.append(f"  - {k} (only in {path_a})"); diffs += 1
            elif a[k] != b[k]: lines.append(f"  ~ {k} (different values)"); diffs += 1
        if diffs == 0: lines.append("  Files are identical.")
        else: lines.insert(1, f"  {diffs} difference(s)")
        return "\n".join(lines)
