"""Conversation export — renders conversations to markdown, text, JSON, or HTML."""

from __future__ import annotations

import html
import json
from typing import Any

from towel.agent.conversation import Conversation, Role


def export_markdown(conv: Conversation, include_metadata: bool = False) -> str:
    """Export a conversation to Markdown format."""
    lines: list[str] = []

    # Header
    lines.append(f"# {conv.summary}")
    lines.append("")
    lines.append(f"**Session:** `{conv.id}`  ")
    lines.append(f"**Channel:** {conv.channel}  ")
    lines.append(f"**Created:** {conv.created_at.strftime('%Y-%m-%d %H:%M UTC')}  ")
    lines.append(f"**Messages:** {len(conv)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for msg in conv.messages:
        match msg.role:
            case Role.USER:
                lines.append(f"### You")
                if include_metadata:
                    lines.append(f"*{msg.timestamp.strftime('%H:%M:%S')}*")
                lines.append("")
                lines.append(msg.content)
                lines.append("")

            case Role.ASSISTANT:
                lines.append(f"### Towel")
                if include_metadata:
                    ts = msg.timestamp.strftime("%H:%M:%S")
                    meta_parts = [ts]
                    if msg.metadata.get("tps"):
                        meta_parts.append(f"{msg.metadata['tps']:.1f} tok/s")
                    if msg.metadata.get("tokens"):
                        meta_parts.append(f"{msg.metadata['tokens']} tokens")
                    lines.append(f"*{' | '.join(meta_parts)}*")
                lines.append("")
                lines.append(msg.content)
                lines.append("")

            case Role.TOOL:
                lines.append("<details>")
                # Extract tool name from "[tool_name] result..." format
                content = msg.content
                if content.startswith("[") and "]" in content:
                    bracket_end = content.index("]")
                    tool_name = content[1:bracket_end]
                    result = content[bracket_end + 2:]
                    lines.append(f"<summary>Tool: {tool_name}</summary>")
                    lines.append("")
                    lines.append("```")
                    lines.append(result)
                    lines.append("```")
                else:
                    lines.append("<summary>Tool result</summary>")
                    lines.append("")
                    lines.append("```")
                    lines.append(content)
                    lines.append("```")
                lines.append("</details>")
                lines.append("")

            case Role.SYSTEM:
                if include_metadata:
                    lines.append(f"> **System:** {msg.content}")
                    lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Exported from [Towel](https://github.com/towel-ai/towel) — Don't Panic.*")

    return "\n".join(lines)


def export_text(conv: Conversation) -> str:
    """Export a conversation to plain text format."""
    lines: list[str] = []
    lines.append(f"Conversation: {conv.id}")
    lines.append(f"Created: {conv.created_at.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"Channel: {conv.channel}")
    lines.append("=" * 60)
    lines.append("")

    for msg in conv.messages:
        match msg.role:
            case Role.USER:
                lines.append(f"[you] {msg.content}")
            case Role.ASSISTANT:
                lines.append(f"[towel] {msg.content}")
            case Role.TOOL:
                content = msg.content
                if len(content) > 300:
                    content = content[:300] + "..."
                lines.append(f"[tool] {content}")
            case Role.SYSTEM:
                lines.append(f"[system] {msg.content}")
        lines.append("")

    return "\n".join(lines)


def export_json(conv: Conversation, pretty: bool = True) -> str:
    """Export a conversation to JSON format."""
    indent = 2 if pretty else None
    return json.dumps(conv.to_dict(), indent=indent, ensure_ascii=False)


_HTML_STYLE = """\
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#1a1a2e;color:#e0e0e0;max-width:800px;margin:0 auto;padding:24px}
h1{font-size:1.4rem;color:#7fdbca;margin-bottom:4px}
.meta{color:#888;font-size:.85rem;margin-bottom:24px}
.msg{margin-bottom:20px;padding:14px 18px;border-radius:10px;line-height:1.6}
.msg.user{background:#16213e;border-left:3px solid #7fdbca}
.msg.assistant{background:#1a1a2e;border-left:3px solid #c792ea}
.msg.tool{background:#0f0f1a;border-left:3px solid #ffcb6b;font-size:.9rem}
.msg.system{background:#1a1a2e;border-left:3px solid #546e7a;font-size:.85rem;color:#888}
.role{font-weight:600;font-size:.8rem;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
.role.user{color:#7fdbca}
.role.assistant{color:#c792ea}
.role.tool{color:#ffcb6b}
.ts{color:#666;font-size:.75rem;float:right}
pre{background:#0d0d1a;padding:12px;border-radius:6px;overflow-x:auto;margin:8px 0;font-size:.85rem}
code{font-family:"SF Mono",Menlo,Consolas,monospace}
p{margin:6px 0}
.footer{margin-top:32px;text-align:center;color:#555;font-size:.8rem}
.footer a{color:#7fdbca;text-decoration:none}
"""


def _html_escape(text: str) -> str:
    """Escape HTML and convert markdown-style code blocks to <pre><code>."""
    escaped = html.escape(text)
    # Convert ```lang\n...\n``` to <pre><code>
    import re
    def _code_block(m: re.Match) -> str:
        code = m.group(2)
        return f"<pre><code>{code}</code></pre>"
    escaped = re.sub(r"```(\w*)\n(.*?)```", _code_block, escaped, flags=re.DOTALL)
    # Convert inline `code` to <code>
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    # Convert newlines to <br> (outside <pre> blocks handled above)
    escaped = escaped.replace("\n", "<br>\n")
    return escaped


def export_html(conv: Conversation, include_metadata: bool = True) -> str:
    """Export a conversation to a standalone HTML page with dark theme styling."""
    parts: list[str] = []

    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append(f"<title>{html.escape(conv.summary)}</title>")
    parts.append(f"<style>{_HTML_STYLE}</style>")
    parts.append("</head><body>")

    # Header
    parts.append(f"<h1>{html.escape(conv.summary)}</h1>")
    created = conv.created_at.strftime("%Y-%m-%d %H:%M UTC")
    parts.append(f'<div class="meta">{len(conv)} messages &middot; {created}</div>')

    for msg in conv.messages:
        role = msg.role.value
        css_class = role

        if msg.role == Role.SYSTEM and not include_metadata:
            continue

        ts = msg.timestamp.strftime("%H:%M:%S")
        parts.append(f'<div class="msg {css_class}">')
        parts.append(f'<div class="role {css_class}">')

        label = {"user": "You", "assistant": "Towel", "tool": "Tool", "system": "System"}
        parts.append(f'{label.get(role, role)}')
        if include_metadata:
            parts.append(f'<span class="ts">{ts}</span>')
        parts.append("</div>")

        if msg.role == Role.TOOL:
            content = msg.content
            if content.startswith("[") and "]" in content:
                bracket_end = content.index("]")
                tool_name = content[1:bracket_end]
                result = content[bracket_end + 2:]
                parts.append(f"<div><strong>{html.escape(tool_name)}</strong></div>")
                parts.append(f"<pre><code>{html.escape(result)}</code></pre>")
            else:
                parts.append(f"<pre><code>{html.escape(content)}</code></pre>")
        else:
            parts.append(f"<div>{_html_escape(msg.content)}</div>")

        parts.append("</div>")

    # Footer
    parts.append('<div class="footer">')
    parts.append('Exported from <a href="https://github.com/towel-ai/towel">Towel</a> &mdash; Don\'t Panic.')
    parts.append("</div>")
    parts.append("</body></html>")

    return "\n".join(parts)
