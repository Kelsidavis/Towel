"""Image skill — inspect image metadata and dimensions."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition


def _read_png_size(data: bytes) -> tuple[int, int] | None:
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        return None
    if len(data) < 24:
        return None
    w, h = struct.unpack('>II', data[16:24])
    return w, h


def _read_jpeg_size(data: bytes) -> tuple[int, int] | None:
    if data[:2] != b'\xff\xd8':
        return None
    i = 2
    while i < len(data) - 1:
        if data[i] != 0xff:
            break
        marker = data[i + 1]
        if marker == 0xc0 or marker == 0xc2:  # SOF0 or SOF2
            if i + 9 < len(data):
                h, w = struct.unpack('>HH', data[i+5:i+9])
                return w, h
        if marker == 0xd9:
            break
        if i + 3 < len(data):
            length = struct.unpack('>H', data[i+2:i+4])[0]
            i += 2 + length
        else:
            break
    return None


def _read_gif_size(data: bytes) -> tuple[int, int] | None:
    if data[:4] not in (b'GIF8', b'GIF9'):
        return None
    if len(data) < 10:
        return None
    w, h = struct.unpack('<HH', data[6:10])
    return w, h


class ImageSkill(Skill):
    @property
    def name(self) -> str:
        return "image"

    @property
    def description(self) -> str:
        return "Inspect image files — dimensions, format, file size"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="image_info",
                description="Get image dimensions, format, and file size (supports PNG, JPEG, GIF)",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to image file"},
                    },
                    "required": ["path"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "image_info":
            return f"Unknown tool: {tool_name}"

        p = Path(arguments["path"]).expanduser()
        if not p.is_file():
            return f"Not found: {arguments['path']}"

        try:
            data = p.read_bytes()[:1024]  # only need header
            size = p.stat().st_size

            dims = _read_png_size(data) or _read_jpeg_size(data) or _read_gif_size(data)

            ext = p.suffix.lower()
            fmt_map = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".gif": "GIF",
                       ".webp": "WebP", ".bmp": "BMP", ".svg": "SVG"}
            fmt = fmt_map.get(ext, ext.upper())

            lines = [f"Image: {p.name}"]
            lines.append(f"  Format: {fmt}")
            if dims:
                lines.append(f"  Dimensions: {dims[0]}x{dims[1]} pixels")
            lines.append(f"  File size: {size:,} bytes ({size/1024:.1f} KB)")

            return "\n".join(lines)
        except Exception as e:
            return f"Error reading image: {e}"
