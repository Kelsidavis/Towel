"""Typo/spelling skill — find and fix common typos in text and code."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition

# Common programming typos: wrong -> right
_CODE_TYPOS: dict[str, str] = {
    "fucntion": "function",
    "funciton": "function",
    "funtion": "function",
    "retrun": "return",
    "reutrn": "return",
    "retunr": "return",
    "pritn": "print",
    "pirnt": "print",
    "improt": "import",
    "ipmort": "import",
    "defintion": "definition",
    "defniition": "definition",
    "lenght": "length",
    "lnegth": "length",
    "widht": "width",
    "heigth": "height",
    "ture": "true",
    "flase": "false",
    "fasle": "false",
    "nulll": "null",
    "nul": "null",
    "strign": "string",
    "stirng": "string",
    "interger": "integer",
    "integre": "integer",
    "booelan": "boolean",
    "aray": "array",
    "arry": "array",
    "arrray": "array",
    "obejct": "object",
    "objcet": "object",
    "calss": "class",
    "clsas": "class",
    "pubilc": "public",
    "priavte": "private",
    "cosnt": "const",
    "ocnst": "const",
    "async": "async",
    "awiat": "await",
    "tempalte": "template",
    "templte": "template",
    "reponse": "response",
    "respnose": "response",
    "reqeust": "request",
    "reuqest": "request",
    "databse": "database",
    "databaes": "database",
    "pasword": "password",
    "passowrd": "password",
    "enviroment": "environment",
    "enviornment": "environment",
    "configration": "configuration",
    "configuraiton": "configuration",
    "initalize": "initialize",
    "initialze": "initialize",
    "destory": "destroy",
    "destry": "destroy",
    "udpate": "update",
    "upadte": "update",
    "delte": "delete",
    "dleet": "delete",
    "craete": "create",
    "cretae": "create",
    "recieve": "receive",
    "recevie": "receive",
    "occured": "occurred",
    "occurrred": "occurred",
    "seperate": "separate",
    "separete": "separate",
    "neccessary": "necessary",
    "necesary": "necessary",
    "succesfull": "successful",
    "succesful": "successful",
    "immediatly": "immediately",
    "imediately": "immediately",
}


class TypoSkill(Skill):
    @property
    def name(self) -> str:
        return "typo"

    @property
    def description(self) -> str:
        return "Find and fix common typos in code and text"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="typo_check",
                description="Check text or code for common typos",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to check"},
                    },
                    "required": ["text"],
                },
            ),
            ToolDefinition(
                name="typo_fix",
                description="Fix all found typos and return corrected text",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to fix"},
                    },
                    "required": ["text"],
                },
            ),
            ToolDefinition(
                name="typo_check_file",
                description="Check a file for typos",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                    },
                    "required": ["path"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "typo_check":
                return self._check(arguments["text"])
            case "typo_fix":
                return self._fix(arguments["text"])
            case "typo_check_file":
                return self._check_file(arguments["path"])
            case _:
                return f"Unknown tool: {tool_name}"

    def _find_typos(self, text: str) -> list[tuple[int, str, str, str]]:
        """Returns [(line, typo, correction, context)]"""
        findings = []
        for i, line in enumerate(text.splitlines()):
            words = re.findall(r"\b[a-zA-Z]+\b", line)
            for word in words:
                lower = word.lower()
                if lower in _CODE_TYPOS:
                    fix = _CODE_TYPOS[lower]
                    if word[0].isupper():
                        fix = fix.capitalize()
                    if word.isupper():
                        fix = fix.upper()
                    findings.append((i + 1, word, fix, line.strip()[:80]))
        return findings

    def _check(self, text: str) -> str:
        findings = self._find_typos(text)
        if not findings:
            return "No typos found."
        lines = [f"Found {len(findings)} typo(s):"]
        for line_num, typo, fix, ctx in findings:
            lines.append(f"  Line {line_num}: '{typo}' → '{fix}'")
            lines.append(f"    {ctx}")
        return "\n".join(lines)

    def _fix(self, text: str) -> str:
        result = text
        count = 0
        for typo, fix in _CODE_TYPOS.items():
            pattern = re.compile(r"\b" + re.escape(typo) + r"\b", re.IGNORECASE)

            def replacer(m):
                w = m.group()
                if w.isupper():
                    return fix.upper()
                if w[0].isupper():
                    return fix.capitalize()
                return fix

            new_result = pattern.sub(replacer, result)
            if new_result != result:
                count += (
                    result.count(typo)
                    + result.count(typo.capitalize())
                    + result.count(typo.upper())
                )
                result = new_result
        return result

    def _check_file(self, path: str) -> str:
        p = Path(path).expanduser()
        if not p.is_file():
            return f"Not found: {path}"
        text = p.read_text(encoding="utf-8", errors="replace")
        findings = self._find_typos(text)
        if not findings:
            return f"No typos in {p.name}."
        lines = [f"Found {len(findings)} typo(s) in {p.name}:"]
        for line_num, typo, fix, ctx in findings[:30]:
            lines.append(f"  {p.name}:{line_num}: '{typo}' → '{fix}'")
        if len(findings) > 30:
            lines.append(f"  ... and {len(findings) - 30} more")
        return "\n".join(lines)
