"""Diagram skill — generate ASCII flowcharts, trees, and sequence diagrams."""

from __future__ import annotations
from typing import Any
from towel.skills.base import Skill, ToolDefinition


class DiagramSkill(Skill):
    @property
    def name(self) -> str: return "diagram"
    @property
    def description(self) -> str: return "Generate ASCII diagrams — flowcharts, trees, sequence diagrams"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="diagram_tree", description="Draw a tree/hierarchy diagram from nested data",
                parameters={"type":"object","properties":{
                    "root":{"type":"string","description":"Root node label"},
                    "children":{"type":"string","description":"Children as indented text (2-space indent per level)"},
                },"required":["root","children"]}),
            ToolDefinition(name="diagram_flow", description="Draw an ASCII flowchart from a list of steps",
                parameters={"type":"object","properties":{
                    "steps":{"type":"array","items":{"type":"string"},"description":"Flowchart steps in order"},
                    "direction":{"type":"string","enum":["vertical","horizontal"],"description":"Direction (default: vertical)"},
                },"required":["steps"]}),
            ToolDefinition(name="diagram_sequence", description="Draw an ASCII sequence diagram from actors and messages",
                parameters={"type":"object","properties":{
                    "actors":{"type":"array","items":{"type":"string"},"description":"Actor names"},
                    "messages":{"type":"array","items":{"type":"string"},"description":"Messages as 'from->to: text'"},
                },"required":["actors","messages"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "diagram_tree": return self._tree(arguments["root"], arguments["children"])
            case "diagram_flow": return self._flow(arguments["steps"], arguments.get("direction","vertical"))
            case "diagram_sequence": return self._sequence(arguments["actors"], arguments["messages"])
            case _: return f"Unknown tool: {tool_name}"

    def _tree(self, root: str, children_text: str) -> str:
        lines = [root]
        children = children_text.strip().splitlines()
        for i, child in enumerate(children):
            stripped = child.lstrip()
            indent = (len(child) - len(stripped)) // 2
            is_last = (i == len(children) - 1) or (i + 1 < len(children) and (len(children[i+1]) - len(children[i+1].lstrip())) // 2 <= indent)
            prefix = ""
            for d in range(indent):
                prefix += "│   " if d < indent - 1 else ("└── " if is_last else "├── ")
            if indent == 0:
                is_last_root = (i == len(children) - 1) or (i + 1 < len(children) and children[i+1].startswith("  "))
                prefix = "└── " if i == len(children) - 1 else "├── "
            lines.append(prefix + stripped)
        return "\n".join(lines)

    def _flow(self, steps: list[str], direction: str) -> str:
        if direction == "horizontal":
            boxes = [f"[{s}]" for s in steps]
            return " --> ".join(boxes)
        lines = []
        max_len = max(len(s) for s in steps) + 4
        for i, step in enumerate(steps):
            box_w = max(len(step) + 4, 10)
            pad = (box_w - len(step) - 2) // 2
            lines.append("+" + "-" * (box_w - 2) + "+")
            lines.append("|" + " " * pad + step + " " * (box_w - 2 - pad - len(step)) + "|")
            lines.append("+" + "-" * (box_w - 2) + "+")
            if i < len(steps) - 1:
                center = box_w // 2
                lines.append(" " * center + "|")
                lines.append(" " * center + "v")
        return "\n".join(lines)

    def _sequence(self, actors: list[str], messages: list[str]) -> str:
        import re
        col_width = max(len(a) for a in actors) + 6
        total_w = col_width * len(actors)

        # Actor positions
        positions = {a: i * col_width + col_width // 2 for i, a in enumerate(actors)}

        # Header
        lines = []
        header = ""
        for a in actors:
            header += a.center(col_width)
        lines.append(header)
        lines.append("".join("|".center(col_width) for _ in actors))

        # Messages
        for msg in messages:
            m = re.match(r"(\w+)\s*->\s*(\w+)\s*:\s*(.+)", msg)
            if not m: continue
            src, dst, text = m.group(1), m.group(2), m.group(3)
            if src not in positions or dst not in positions: continue

            src_pos = positions[src]
            dst_pos = positions[dst]
            left = min(src_pos, dst_pos)
            right = max(src_pos, dst_pos)
            arrow_len = right - left

            line = list(" " * total_w)
            # Draw lifelines
            for a in actors:
                p = positions[a]
                if p < len(line): line[p] = "|"

            # Draw arrow
            if arrow_len > 0:
                for j in range(left + 1, right):
                    if j < len(line): line[j] = "-"
                if dst_pos > src_pos:
                    if right < len(line): line[right] = ">"
                else:
                    if left < len(line): line[left] = "<"

            # Label
            mid = (left + right) // 2 - len(text) // 2
            label_line = list(" " * total_w)
            for a in actors:
                p = positions[a]
                if p < len(label_line): label_line[p] = "|"
            for j, c in enumerate(text):
                if mid + j < len(label_line): label_line[mid + j] = c

            lines.append("".join(label_line))
            lines.append("".join(line))

        # Footer lifelines
        lines.append("".join("|".center(col_width) for _ in actors))
        return "\n".join(lines)
