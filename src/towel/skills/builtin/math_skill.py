"""Math skill — statistics, number formatting, and unit-aware calculations."""

from __future__ import annotations
import math
import statistics
from typing import Any
from towel.skills.base import Skill, ToolDefinition


class MathSkill(Skill):
    @property
    def name(self) -> str: return "math"
    @property
    def description(self) -> str: return "Statistics, number formatting, and mathematical functions"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="math_stats", description="Calculate statistics for a list of numbers (mean, median, stdev, min, max, sum)",
                parameters={"type":"object","properties":{
                    "numbers":{"type":"array","items":{"type":"number"},"description":"List of numbers"},
                },"required":["numbers"]}),
            ToolDefinition(name="math_format", description="Format a number (comma separators, scientific notation, percentages, bytes)",
                parameters={"type":"object","properties":{
                    "number":{"type":"number","description":"Number to format"},
                    "format":{"type":"string","enum":["commas","scientific","percent","bytes","binary","hex","roman"],
                              "description":"Format type"},
                },"required":["number","format"]}),
            ToolDefinition(name="math_sequence", description="Generate number sequences (range, fibonacci, primes)",
                parameters={"type":"object","properties":{
                    "type":{"type":"string","enum":["range","fibonacci","primes"],"description":"Sequence type"},
                    "count":{"type":"integer","description":"How many numbers (default: 10)"},
                    "start":{"type":"number","description":"Start value (for range)"},
                    "step":{"type":"number","description":"Step value (for range)"},
                },"required":["type"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "math_stats": return self._stats(arguments["numbers"])
            case "math_format": return self._format(arguments["number"], arguments["format"])
            case "math_sequence": return self._sequence(arguments["type"], arguments.get("count",10), arguments.get("start",0), arguments.get("step",1))
            case _: return f"Unknown tool: {tool_name}"

    def _stats(self, numbers: list[float]) -> str:
        if not numbers: return "Empty list."
        n = numbers
        lines = [
            f"Count:    {len(n)}",
            f"Sum:      {sum(n):,.4g}",
            f"Mean:     {statistics.mean(n):,.4g}",
            f"Median:   {statistics.median(n):,.4g}",
            f"Min:      {min(n):,.4g}",
            f"Max:      {max(n):,.4g}",
            f"Range:    {max(n) - min(n):,.4g}",
        ]
        if len(n) >= 2:
            lines.append(f"Stdev:    {statistics.stdev(n):,.4g}")
            lines.append(f"Variance: {statistics.variance(n):,.4g}")
        return "\n".join(lines)

    def _format(self, number: float, fmt: str) -> str:
        match fmt:
            case "commas": return f"{number:,.2f}"
            case "scientific": return f"{number:.6e}"
            case "percent": return f"{number:.2%}"
            case "bytes":
                n = abs(number)
                for unit in ["B","KB","MB","GB","TB","PB"]:
                    if n < 1024: return f"{n:.1f} {unit}"
                    n /= 1024
                return f"{n:.1f} EB"
            case "binary": return bin(int(number))
            case "hex": return hex(int(number))
            case "roman": return self._to_roman(int(number))
            case _: return str(number)

    def _to_roman(self, n: int) -> str:
        if n <= 0 or n > 3999: return f"{n} (out of range for Roman numerals)"
        vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
                (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
        result = ""
        for val, sym in vals:
            while n >= val:
                result += sym
                n -= val
        return result

    def _sequence(self, seq_type: str, count: int, start: float, step: float) -> str:
        count = min(count, 1000)
        match seq_type:
            case "range":
                nums = [start + i * step for i in range(count)]
            case "fibonacci":
                nums = []
                a, b = 0, 1
                for _ in range(count):
                    nums.append(a)
                    a, b = b, a + b
            case "primes":
                nums = []
                candidate = 2
                while len(nums) < count:
                    if all(candidate % p != 0 for p in nums if p * p <= candidate):
                        nums.append(candidate)
                    candidate += 1
            case _:
                return f"Unknown sequence: {seq_type}"
        return ", ".join(str(int(n) if n == int(n) else n) for n in nums)
