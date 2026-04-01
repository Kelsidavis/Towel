"""Metrics skill — track custom counters, gauges, and timers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from towel.skills.base import Skill, ToolDefinition

_counters: dict[str, float] = defaultdict(float)
_timers: dict[str, list[float]] = defaultdict(list)
_gauges: dict[str, float] = {}


class MetricsSkill(Skill):
    @property
    def name(self) -> str: return "metrics"
    @property
    def description(self) -> str: return "Track custom counters, gauges, and timers for monitoring"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="metric_increment", description="Increment a counter",
                parameters={"type":"object","properties":{
                    "name":{"type":"string","description":"Counter name"},
                    "value":{"type":"number","description":"Amount (default: 1)"},
                },"required":["name"]}),
            ToolDefinition(name="metric_gauge", description="Set a gauge value",
                parameters={"type":"object","properties":{
                    "name":{"type":"string","description":"Gauge name"},
                    "value":{"type":"number","description":"Value to set"},
                },"required":["name","value"]}),
            ToolDefinition(name="metric_timer", description="Record a duration measurement",
                parameters={"type":"object","properties":{
                    "name":{"type":"string","description":"Timer name"},
                    "duration_ms":{"type":"number","description":"Duration in milliseconds"},
                },"required":["name","duration_ms"]}),
            ToolDefinition(name="metric_report", description="Show all tracked metrics",
                parameters={"type":"object","properties":{
                    "name":{"type":"string","description":"Filter by metric name (optional)"},
                }}),
            ToolDefinition(name="metric_reset", description="Reset all or specific metrics",
                parameters={"type":"object","properties":{
                    "name":{"type":"string","description":"Metric to reset (or omit for all)"},
                }}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "metric_increment":
                name = arguments["name"]
                _counters[name] += arguments.get("value", 1)
                return f"Counter {name}: {_counters[name]}"
            case "metric_gauge":
                name = arguments["name"]
                _gauges[name] = arguments["value"]
                return f"Gauge {name}: {_gauges[name]}"
            case "metric_timer":
                name = arguments["name"]
                _timers[name].append(arguments["duration_ms"])
                vals = _timers[name]
                avg = sum(vals) / len(vals)
                return f"Timer {name}: {arguments['duration_ms']}ms (avg: {avg:.1f}ms, count: {len(vals)})"
            case "metric_report": return self._report(arguments.get("name"))
            case "metric_reset":
                name = arguments.get("name")
                if name:
                    _counters.pop(name, None); _gauges.pop(name, None); _timers.pop(name, None)
                    return f"Reset: {name}"
                else:
                    _counters.clear(); _gauges.clear(); _timers.clear()
                    return "All metrics reset."
            case _: return f"Unknown tool: {tool_name}"

    def _report(self, name: str|None) -> str:
        lines = []
        for k, v in sorted(_counters.items()):
            if name and name not in k: continue
            lines.append(f"  counter/{k}: {v}")
        for k, v in sorted(_gauges.items()):
            if name and name not in k: continue
            lines.append(f"  gauge/{k}: {v}")
        for k, vals in sorted(_timers.items()):
            if name and name not in k: continue
            avg = sum(vals) / len(vals)
            mn, mx = min(vals), max(vals)
            lines.append(f"  timer/{k}: avg={avg:.1f}ms min={mn:.1f}ms max={mx:.1f}ms count={len(vals)}")
        return "Metrics:\n" + "\n".join(lines) if lines else "No metrics tracked."
