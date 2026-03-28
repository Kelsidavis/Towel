"""Semver skill — parse, compare, and bump semantic versions."""

from __future__ import annotations

import re
from typing import Any

from towel.skills.base import Skill, ToolDefinition


def _parse_semver(v: str) -> tuple[int,int,int,str] | None:
    v = v.strip().lstrip("v")
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$", v)
    if not m: return None
    return int(m[1]), int(m[2]), int(m[3]), m[4] or ""


class SemverSkill(Skill):
    @property
    def name(self) -> str: return "semver"
    @property
    def description(self) -> str: return "Parse, compare, and bump semantic versions"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="semver_parse", description="Parse a semantic version string into components",
                parameters={"type":"object","properties":{
                    "version":{"type":"string","description":"Version string (e.g., 1.2.3, v2.0.0-beta)"},
                },"required":["version"]}),
            ToolDefinition(name="semver_bump", description="Bump a version (major, minor, or patch)",
                parameters={"type":"object","properties":{
                    "version":{"type":"string","description":"Current version"},
                    "bump":{"type":"string","enum":["major","minor","patch"],"description":"What to bump"},
                    "prerelease":{"type":"string","description":"Prerelease tag (e.g., beta, rc.1)"},
                },"required":["version","bump"]}),
            ToolDefinition(name="semver_compare", description="Compare two versions (which is newer?)",
                parameters={"type":"object","properties":{
                    "a":{"type":"string","description":"First version"},
                    "b":{"type":"string","description":"Second version"},
                },"required":["a","b"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "semver_parse": return self._parse(arguments["version"])
            case "semver_bump": return self._bump(arguments["version"], arguments["bump"], arguments.get("prerelease"))
            case "semver_compare": return self._compare(arguments["a"], arguments["b"])
            case _: return f"Unknown tool: {tool_name}"

    def _parse(self, v: str) -> str:
        p = _parse_semver(v)
        if not p: return f"Invalid semver: {v}"
        major, minor, patch, pre = p
        lines = [f"Version: {major}.{minor}.{patch}" + (f"-{pre}" if pre else "")]
        lines.append(f"  Major: {major}")
        lines.append(f"  Minor: {minor}")
        lines.append(f"  Patch: {patch}")
        if pre: lines.append(f"  Prerelease: {pre}")
        return "\n".join(lines)

    def _bump(self, v: str, bump: str, pre: str|None) -> str:
        p = _parse_semver(v)
        if not p: return f"Invalid semver: {v}"
        major, minor, patch, _ = p
        match bump:
            case "major": major += 1; minor = 0; patch = 0
            case "minor": minor += 1; patch = 0
            case "patch": patch += 1
        new = f"{major}.{minor}.{patch}"
        if pre: new += f"-{pre}"
        return f"{v} -> {new}"

    def _compare(self, a: str, b: str) -> str:
        pa, pb = _parse_semver(a), _parse_semver(b)
        if not pa: return f"Invalid: {a}"
        if not pb: return f"Invalid: {b}"
        ta = (pa[0], pa[1], pa[2])
        tb = (pb[0], pb[1], pb[2])
        if ta > tb: return f"{a} is NEWER than {b}"
        if ta < tb: return f"{a} is OLDER than {b}"
        return f"{a} and {b} are the SAME version"
