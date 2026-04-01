"""Security skill — check for secrets, scan dependencies, audit permissions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition

# Patterns that indicate hardcoded secrets
_SECRET_PATTERNS = [
    (r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]", "API key"),
    (r"(?i)(secret|password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{8,}['\"]", "Password/Secret"),
    (r"(?i)(token)\s*[:=]\s*['\"][A-Za-z0-9_\-\.]{20,}['\"]", "Token"),
    (r"(?i)Bearer\s+[A-Za-z0-9_\-\.]{20,}", "Bearer token"),
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key"),
    (r"(?i)-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----", "Private key"),
    (r"ghp_[A-Za-z0-9]{36}", "GitHub PAT"),
    (r"sk-[A-Za-z0-9]{32,}", "OpenAI API key"),
    (r"xox[bpras]-[A-Za-z0-9\-]{10,}", "Slack token"),
]

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".tox"}
SKIP_EXT = {
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".o",
    ".exe",
    ".bin",
    ".zip",
    ".tar",
    ".gz",
    ".png",
    ".jpg",
    ".gif",
}


class SecuritySkill(Skill):
    @property
    def name(self) -> str:
        return "security"

    @property
    def description(self) -> str:
        return "Security scanning — find hardcoded secrets, audit file permissions"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="scan_secrets",
                description="Scan files for hardcoded secrets, API keys, and tokens",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory or file to scan (default: cwd)",
                        },
                        "glob": {
                            "type": "string",
                            "description": "File filter (e.g., '*.py', '*.env')",
                        },
                    },
                    "required": [],
                },
            ),
            ToolDefinition(
                name="check_permissions",
                description="Check file permissions for security issues (world-writable, etc.)",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory to check (default: cwd)",
                        },
                    },
                    "required": [],
                },
            ),
            ToolDefinition(
                name="scan_dependencies",
                description=(
                    "Check for known patterns in dependency "
                    "files (requirements.txt, package.json)"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Project directory (default: cwd)",
                        },
                    },
                    "required": [],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "scan_secrets":
                return self._scan_secrets(arguments.get("path", "."), arguments.get("glob"))
            case "check_permissions":
                return self._check_perms(arguments.get("path", "."))
            case "scan_dependencies":
                return self._scan_deps(arguments.get("path", "."))
            case _:
                return f"Unknown tool: {tool_name}"

    def _scan_secrets(self, path: str, file_glob: str | None) -> str:
        root = Path(path).expanduser().resolve()
        findings: list[str] = []
        files_scanned = 0

        if root.is_file():
            files = [root]
        else:
            pattern = file_glob or "*"
            files = sorted(root.rglob(pattern))

        for fp in files:
            if not fp.is_file():
                continue
            if any(s in fp.parts for s in SKIP_DIRS):
                continue
            if fp.suffix in SKIP_EXT:
                continue
            try:
                content = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            files_scanned += 1
            for pattern, label in _SECRET_PATTERNS:
                for m in re.finditer(pattern, content):
                    line_num = content[: m.start()].count("\n") + 1
                    snippet = m.group()[:40] + "..." if len(m.group()) > 40 else m.group()
                    rel = fp.relative_to(root) if fp.is_relative_to(root) else fp
                    findings.append(f"  [{label}] {rel}:{line_num}  {snippet}")

        if not findings:
            return f"No secrets found in {files_scanned} files scanned."
        return f"Found {len(findings)} potential secret(s) in {files_scanned} files:\n" + "\n".join(
            findings[:50]
        )

    def _check_perms(self, path: str) -> str:
        import stat

        root = Path(path).expanduser().resolve()
        issues: list[str] = []
        for fp in sorted(root.rglob("*")):
            if any(s in fp.parts for s in SKIP_DIRS):
                continue
            if not fp.is_file():
                continue
            try:
                mode = fp.stat().st_mode
                if mode & stat.S_IWOTH:
                    issues.append(f"  WORLD-WRITABLE: {fp.relative_to(root)}")
                if fp.suffix in (".env", ".pem", ".key", ".p12") and mode & stat.S_IROTH:
                    issues.append(f"  WORLD-READABLE (sensitive): {fp.relative_to(root)}")
            except Exception:
                continue
        if not issues:
            return "No permission issues found."
        return f"Found {len(issues)} permission issue(s):\n" + "\n".join(issues[:50])

    def _scan_deps(self, path: str) -> str:
        root = Path(path).expanduser().resolve()
        lines: list[str] = []
        # Check requirements.txt
        req = root / "requirements.txt"
        if req.exists():
            content = req.read_text()
            unpinned = [
                line.strip()
                for line in content.splitlines()
                if line.strip()
                and not line.startswith("#")
                and "==" not in line
                and ">=" not in line
            ]
            if unpinned:
                lines.append(f"requirements.txt: {len(unpinned)} unpinned dependencies")
                for d in unpinned[:10]:
                    lines.append(f"    {d}")
        # Check package.json
        pkg = root / "package.json"
        if pkg.exists():
            try:
                data = __import__("json").loads(pkg.read_text())
                deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
                stars = [f"{k}: {v}" for k, v in deps.items() if v.startswith("*") or v == "latest"]
                if stars:
                    lines.append(f"package.json: {len(stars)} wildcard/latest versions")
                    for s in stars[:10]:
                        lines.append(f"    {s}")
            except Exception:
                pass
        if not lines:
            return "No dependency issues found."
        return "\n".join(lines)
