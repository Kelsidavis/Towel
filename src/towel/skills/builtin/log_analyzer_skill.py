"""Log analyzer skill — parse, filter, and summarize log files."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition

_LOG_PATTERNS = {
    "timestamp": re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"),
    "level": re.compile(r"\b(DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\b", re.IGNORECASE),
    "ip": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "http_status": re.compile(r"\b[1-5]\d{2}\b"),
    "url": re.compile(r"(?:GET|POST|PUT|DELETE|PATCH)\s+(\S+)"),
}


class LogAnalyzerSkill(Skill):
    @property
    def name(self) -> str: return "logs"
    @property
    def description(self) -> str: return "Parse, filter, and summarize log files"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="log_summary", description="Summarize a log file — line count, error rate, time range, top errors",
                parameters={"type":"object","properties":{
                    "path":{"type":"string","description":"Path to log file"},
                    "lines":{"type":"integer","description":"Max lines to analyze (default: 10000)"},
                },"required":["path"]}),
            ToolDefinition(name="log_filter", description="Filter log lines by level, pattern, or time range",
                parameters={"type":"object","properties":{
                    "path":{"type":"string","description":"Path to log file"},
                    "level":{"type":"string","description":"Log level filter (ERROR, WARN, etc.)"},
                    "pattern":{"type":"string","description":"Regex pattern to match"},
                    "tail":{"type":"integer","description":"Only last N lines (default: all)"},
                },"required":["path"]}),
            ToolDefinition(name="log_errors", description="Extract and group error messages from a log file",
                parameters={"type":"object","properties":{
                    "path":{"type":"string","description":"Path to log file"},
                    "top":{"type":"integer","description":"Number of top errors (default: 10)"},
                },"required":["path"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "log_summary": return self._summary(arguments["path"], arguments.get("lines", 10000))
            case "log_filter": return self._filter(arguments["path"], arguments.get("level"), arguments.get("pattern"), arguments.get("tail"))
            case "log_errors": return self._errors(arguments["path"], arguments.get("top", 10))
            case _: return f"Unknown tool: {tool_name}"

    def _read_lines(self, path: str, max_lines: int = 10000) -> list[str] | str:
        p = Path(path).expanduser()
        if not p.is_file(): return f"Not found: {path}"
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            return lines[-max_lines:] if len(lines) > max_lines else lines
        except Exception as e: return f"Error: {e}"

    def _summary(self, path: str, max_lines: int) -> str:
        lines = self._read_lines(path, max_lines)
        if isinstance(lines, str): return lines

        levels: Counter[str] = Counter()
        ips: Counter[str] = Counter()
        timestamps: list[str] = []

        for line in lines:
            lm = _LOG_PATTERNS["level"].search(line)
            if lm: levels[lm.group(1).upper()] += 1
            im = _LOG_PATTERNS["ip"].search(line)
            if im: ips[im.group()] += 1
            tm = _LOG_PATTERNS["timestamp"].search(line)
            if tm: timestamps.append(tm.group())

        total = len(lines)
        errors = levels.get("ERROR", 0) + levels.get("FATAL", 0) + levels.get("CRITICAL", 0)
        error_rate = (errors / total * 100) if total else 0

        parts = [f"Log summary: {Path(path).name} ({total:,} lines)"]
        if timestamps:
            parts.append(f"  Time range: {timestamps[0]} to {timestamps[-1]}")
        parts.append(f"  Error rate: {error_rate:.1f}% ({errors:,} errors)")
        if levels:
            parts.append("  Levels: " + ", ".join(f"{k}={v}" for k, v in levels.most_common()))
        if ips:
            parts.append("  Top IPs: " + ", ".join(f"{ip}({c})" for ip, c in ips.most_common(5)))
        return "\n".join(parts)

    def _filter(self, path: str, level: str|None, pattern: str|None, tail: int|None) -> str:
        lines = self._read_lines(path)
        if isinstance(lines, str): return lines
        if tail: lines = lines[-tail:]
        filtered = []
        for line in lines:
            if level:
                lm = _LOG_PATTERNS["level"].search(line)
                if not lm or lm.group(1).upper() != level.upper(): continue
            if pattern:
                if not re.search(pattern, line, re.IGNORECASE): continue
            filtered.append(line)
        if not filtered: return "No matching lines."
        result = "\n".join(filtered[:200])
        if len(filtered) > 200: result += f"\n\n... ({len(filtered)} total matches)"
        return result

    def _errors(self, path: str, top: int) -> str:
        lines = self._read_lines(path)
        if isinstance(lines, str): return lines
        errors: list[str] = []
        for line in lines:
            lm = _LOG_PATTERNS["level"].search(line)
            if lm and lm.group(1).upper() in ("ERROR", "FATAL", "CRITICAL"):
                # Extract just the message part after the level
                msg = line[lm.end():].strip().lstrip(":- ")[:120]
                errors.append(msg)
        if not errors: return "No errors found."
        counts = Counter(errors)
        parts = [f"Top {min(top, len(counts))} errors ({len(errors)} total):"]
        for msg, count in counts.most_common(top):
            parts.append(f"  [{count}x] {msg}")
        return "\n".join(parts)
