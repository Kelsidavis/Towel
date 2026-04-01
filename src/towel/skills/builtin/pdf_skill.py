"""PDF skill — extract text and metadata from PDF files (stdlib only)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition


def _extract_pdf_text(data: bytes) -> str:
    """Basic PDF text extraction without external libraries.
    Handles simple text streams — won't work for all PDFs."""
    text_parts = []
    # Find text between BT and ET operators
    for match in re.finditer(rb"BT\s(.*?)\sET", data, re.DOTALL):
        block = match.group(1)
        # Extract text from Tj and TJ operators
        for tj in re.finditer(rb"\(([^)]*)\)\s*Tj", block):
            text_parts.append(tj.group(1).decode("latin-1", errors="replace"))
        for tj in re.finditer(rb"\[(.*?)\]\s*TJ", block):
            for part in re.finditer(rb"\(([^)]*)\)", tj.group(1)):
                text_parts.append(part.group(1).decode("latin-1", errors="replace"))
    return " ".join(text_parts)


def _extract_pdf_metadata(data: bytes) -> dict[str, str]:
    """Extract basic PDF metadata."""
    meta = {}
    # PDF version
    if data[:5] == b"%PDF-":
        meta["version"] = data[5:8].decode("ascii", errors="replace")
    # Page count (approximate)
    pages = len(re.findall(rb"/Type\s*/Page\b", data))
    if pages:
        meta["pages"] = str(pages)
    # Info dict entries
    for key in [b"Title", b"Author", b"Subject", b"Creator", b"Producer"]:
        m = re.search(rb"/" + key + rb"\s*\(([^)]*)\)", data)
        if m:
            meta[key.decode()] = m.group(1).decode("latin-1", errors="replace")
    return meta


class PdfSkill(Skill):
    @property
    def name(self) -> str:
        return "pdf"

    @property
    def description(self) -> str:
        return "Extract text and metadata from PDF files (basic, no dependencies)"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="pdf_info",
                description="Get PDF metadata — pages, title, author, version",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to PDF file"},
                    },
                    "required": ["path"],
                },
            ),
            ToolDefinition(
                name="pdf_text",
                description="Extract text content from a PDF (basic extraction)",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to PDF file"},
                        "max_chars": {
                            "type": "integer",
                            "description": "Max characters to extract (default: 10000)",
                        },
                    },
                    "required": ["path"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "pdf_info":
                return self._info(arguments["path"])
            case "pdf_text":
                return self._text(arguments["path"], arguments.get("max_chars", 10000))
            case _:
                return f"Unknown tool: {tool_name}"

    def _info(self, path: str) -> str:
        p = Path(path).expanduser()
        if not p.is_file():
            return f"Not found: {path}"
        data = p.read_bytes()
        meta = _extract_pdf_metadata(data)
        lines = [f"PDF: {p.name} ({p.stat().st_size:,} bytes)"]
        for k, v in meta.items():
            lines.append(f"  {k}: {v}")
        if not meta:
            lines.append("  (no metadata extracted)")
        return "\n".join(lines)

    def _text(self, path: str, max_chars: int) -> str:
        p = Path(path).expanduser()
        if not p.is_file():
            return f"Not found: {path}"
        data = p.read_bytes()
        text = _extract_pdf_text(data)
        if not text:
            return f"No text extracted from {p.name} (may use advanced encoding)"
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[truncated at {max_chars} chars]"
        return text
