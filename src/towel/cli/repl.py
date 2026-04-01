"""REPL — interactive skill execution without the full agent.

Run skill tools directly from the command line, like a REPL.
Faster than chat because it skips model inference entirely.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from typing import Any

from rich.console import Console

console = Console()


def run_repl(skills_registry: Any) -> None:
    """Interactive skill tool REPL."""
    console.print("[bold green]Towel REPL[/bold green] — run tools directly (no AI inference)")
    console.print("[dim]Type 'help' for commands, 'exit' to quit.[/dim]\n")

    tool_defs = skills_registry.tool_definitions()
    tool_names = {t["name"] for t in tool_defs}

    while True:
        try:
            raw = console.input("[bold cyan]tool>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not raw:
            continue

        if raw in ("exit", "quit", "q"):
            break

        if raw == "help":
            console.print("[bold]REPL commands:[/bold]")
            console.print("  [green]list[/green]              List all available tools")
            console.print("  [green]list <skill>[/green]      List tools for a skill")
            console.print("  [green]<tool> <json>[/green]     Run a tool with JSON arguments")
            console.print("  [green]<tool> key=val[/green]    Run with key=value arguments")
            console.print("  [green]exit[/green]              Quit")
            continue

        if raw == "list":
            skills = sorted(skills_registry.list_skills())
            console.print(f"[bold]{len(skills)} skills, {len(tool_defs)} tools:[/bold]")
            for s in skills:
                skill = skills_registry.get_skill(s)
                if skill:
                    tools = [t.name for t in skill.tools()]
                    console.print(f"  [green]{s}[/green]: {', '.join(tools)}")
            continue

        if raw.startswith("list "):
            name = raw[5:].strip()
            skill = skills_registry.get_skill(name)
            if not skill:
                console.print(f"[red]Skill not found:[/red] {name}")
                continue
            for t in skill.tools():
                params = t.parameters.get("properties", {})
                param_str = ", ".join(f"{k}:{v.get('type', '?')}" for k, v in params.items())
                console.print(f"  [green]{t.name}[/green]({param_str})")
                console.print(f"    [dim]{t.description}[/dim]")
            continue

        # Parse: tool_name {json} OR tool_name key=val key=val
        parts = raw.split(None, 1)
        tool_name = parts[0]
        arg_str = parts[1] if len(parts) > 1 else ""

        if tool_name not in tool_names:
            console.print(f"[red]Unknown tool:[/red] {tool_name}")
            console.print("[dim]Type 'list' to see available tools.[/dim]")
            continue

        # Parse arguments
        args: dict[str, Any] = {}
        if arg_str.strip().startswith("{"):
            try:
                args = json.loads(arg_str)
            except json.JSONDecodeError as e:
                console.print(f"[red]Invalid JSON:[/red] {e}")
                continue
        elif "=" in arg_str:
            for pair in shlex.split(arg_str):
                if "=" in pair:
                    k, _, v = pair.partition("=")
                    # Try to parse as JSON value
                    try:
                        args[k] = json.loads(v)
                    except json.JSONDecodeError:
                        args[k] = v
        elif arg_str.strip():
            # Single unnamed arg — guess the first required parameter
            tool_def = next((t for t in tool_defs if t["name"] == tool_name), None)
            if tool_def:
                required = tool_def.get("parameters", {}).get("required", [])
                props = tool_def.get("parameters", {}).get("properties", {})
                if required:
                    args[required[0]] = arg_str
                elif props:
                    args[next(iter(props))] = arg_str

        # Execute
        try:
            result = asyncio.run(skills_registry.execute_tool(tool_name, args))
            console.print(result)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")

        console.print()
