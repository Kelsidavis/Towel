"""Unit conversion skill — convert between common units."""

from __future__ import annotations

from typing import Any

from towel.skills.base import Skill, ToolDefinition

# Conversion factors to base units
_LENGTH = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "km": 1000.0,
    "in": 0.0254,
    "ft": 0.3048,
    "yd": 0.9144,
    "mi": 1609.344,
    "nm": 1852.0,
}
_WEIGHT = {
    "mg": 0.001,
    "g": 1.0,
    "kg": 1000.0,
    "t": 1_000_000.0,
    "oz": 28.3495,
    "lb": 453.592,
    "st": 6350.29,
}
_VOLUME = {
    "ml": 1.0,
    "l": 1000.0,
    "gal": 3785.41,
    "qt": 946.353,
    "pt": 473.176,
    "cup": 236.588,
    "floz": 29.5735,
    "tbsp": 14.7868,
    "tsp": 4.92892,
}
_SPEED = {
    "m/s": 1.0,
    "km/h": 0.277778,
    "mph": 0.44704,
    "kn": 0.514444,
    "ft/s": 0.3048,
}
_DATA = {
    "b": 1,
    "kb": 1024,
    "mb": 1024**2,
    "gb": 1024**3,
    "tb": 1024**4,
    "pb": 1024**5,
}
_TIME = {
    "ms": 0.001,
    "s": 1.0,
    "min": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "wk": 604800.0,
    "yr": 31557600.0,
}

_CATEGORIES: dict[str, dict[str, float]] = {
    "length": _LENGTH,
    "weight": _WEIGHT,
    "volume": _VOLUME,
    "speed": _SPEED,
    "data": _DATA,
    "time": _TIME,
}

_UNIT_TO_CATEGORY: dict[str, str] = {}
for cat, units in _CATEGORIES.items():
    for u in units:
        _UNIT_TO_CATEGORY[u] = cat


def _convert(value: float, from_unit: str, to_unit: str) -> tuple[float, str] | str:
    f = from_unit.lower().strip()
    t = to_unit.lower().strip()

    # Temperature special case
    if f in ("c", "f", "k") or t in ("c", "f", "k"):
        return _convert_temp(value, f, t)

    cat_from = _UNIT_TO_CATEGORY.get(f)
    cat_to = _UNIT_TO_CATEGORY.get(t)

    if not cat_from:
        return f"Unknown unit: {from_unit}"
    if not cat_to:
        return f"Unknown unit: {to_unit}"
    if cat_from != cat_to:
        return f"Cannot convert {from_unit} ({cat_from}) to {to_unit} ({cat_to})"

    units = _CATEGORIES[cat_from]
    base = value * units[f]
    result = base / units[t]
    return result, cat_from


def _convert_temp(value: float, f: str, t: str) -> tuple[float, str] | str:
    # Normalize to Celsius first
    if f == "c":
        c = value
    elif f == "f":
        c = (value - 32) * 5 / 9
    elif f == "k":
        c = value - 273.15
    else:
        return f"Unknown temperature unit: {f}"

    # Convert from Celsius to target
    if t == "c":
        result = c
    elif t == "f":
        result = c * 9 / 5 + 32
    elif t == "k":
        result = c + 273.15
    else:
        return f"Unknown temperature unit: {t}"

    return result, "temperature"


class ConvertSkill(Skill):
    @property
    def name(self) -> str:
        return "convert"

    @property
    def description(self) -> str:
        return "Convert between units (length, weight, volume, temperature, speed, data, time)"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="convert_units",
                description=(
                    "Convert a value from one unit to another. "
                    "Supports length, weight, volume, temperature, "
                    "speed, data size, and time."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "value": {"type": "number", "description": "The numeric value to convert"},
                        "from_unit": {
                            "type": "string",
                            "description": "Source unit (e.g., km, lb, F, GB, min)",
                        },
                        "to_unit": {
                            "type": "string",
                            "description": "Target unit (e.g., mi, kg, C, MB, h)",
                        },
                    },
                    "required": ["value", "from_unit", "to_unit"],
                },
            ),
            ToolDefinition(
                name="list_units",
                description="List all supported units, optionally filtered by category",
                parameters={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": (
                                "Filter: length, weight, volume, "
                                "temperature, speed, data, time"
                            ),
                        },
                    },
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "convert_units":
                return self._convert(
                    arguments["value"], arguments["from_unit"], arguments["to_unit"]
                )
            case "list_units":
                return self._list_units(arguments.get("category"))
            case _:
                return f"Unknown tool: {tool_name}"

    def _convert(self, value: float, from_unit: str, to_unit: str) -> str:
        result = _convert(value, from_unit, to_unit)
        if isinstance(result, str):
            return result
        converted, category = result
        # Smart formatting
        if abs(converted) >= 1000 or abs(converted) < 0.01:
            formatted = f"{converted:.6g}"
        else:
            formatted = f"{converted:.4f}".rstrip("0").rstrip(".")
        return f"{value} {from_unit} = {formatted} {to_unit}"

    def _list_units(self, category: str | None) -> str:
        lines = []
        cats = (
            {category: _CATEGORIES[category]}
            if category and category in _CATEGORIES
            else _CATEGORIES
        )
        if category == "temperature":
            cats = {"temperature": {}}

        for cat, units in cats.items():
            lines.append(f"{cat.title()}: {', '.join(sorted(units.keys()))}")
        lines.append("Temperature: c, f, k")
        return "\n".join(lines)
