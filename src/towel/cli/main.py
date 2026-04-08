"""Towel CLI — the command-line interface.

Usage:
    towel serve          Start the gateway + agent
    towel chat           Interactive chat session
    towel ask <prompt>    One-shot query (scriptable, pipeable)
    towel search <query>  Search across all conversations
    towel history        List saved conversations
    towel show <id>      Show a saved conversation
    towel resume <id>    Resume a saved conversation
    towel status         Show gateway status
    towel skills         List installed skills
    towel doctor         Diagnose your setup
    towel init           Initialize ~/.towel config
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from datetime import UTC
from typing import TYPE_CHECKING, Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from towel import __version__
from towel.config import TOWEL_HOME, TowelConfig

if TYPE_CHECKING:
    from towel.skills.registry import SkillRegistry

console = Console()


def _apply_turboquant_overrides(
    config: TowelConfig,
    turboquant: bool | None,
    tq_bits: int | None,
) -> None:
    """Apply --turboquant / --tq-bits CLI overrides to config."""
    if turboquant is not None:
        config.model.turboquant = turboquant
    if tq_bits is not None:
        config.model.turboquant_bits = tq_bits
        if turboquant is None:
            config.model.turboquant = True  # --tq-bits implies --turboquant


def _build_skill_registry(config: TowelConfig, memory_store: Any = None) -> SkillRegistry:
    """Build a skill registry with builtins + auto-loaded user skills."""
    from towel.skills.builtin import register_builtins
    from towel.skills.loader import SkillLoader
    from towel.skills.registry import SkillRegistry

    registry = SkillRegistry()
    register_builtins(registry, memory_store=memory_store)

    loader = SkillLoader(registry)
    loaded = loader.load_from_dirs(config.skills_dirs)
    if loaded:
        console.print(f"[dim]Loaded {loaded} skill(s) from disk[/dim]")
    for err in loader.errors:
        console.print(f"[yellow]Skill load error:[/yellow] {err.path.name}: {err.error}")

    return registry


BANNER = rf"""
 _____ _____        _______ _
|_   _|  _  |      | _____ | |
  | | | | | |_ _ _ | |___  | |
  | | | | | | | | || _____ | |
  | | | |_| | | | || |___  | |___
  |_|  \_____/_____|_______|_____|

  Don't Panic.  v{__version__}
"""


@click.group()
@click.version_option(__version__, prog_name="towel")
def cli() -> None:
    """Towel — Your local AI assistant. Don't Panic."""
    pass


def _auto_detect_backend() -> str | None:
    """Auto-detect the best available backend for this system."""
    from towel.agent.discovery import detect_gpus, find_llama_server

    gpus = detect_gpus()
    if gpus and find_llama_server():
        return "llama"
    return None


def _build_runtime(
    config: TowelConfig,
    skills: Any,
    memory: Any,
    backend: str | None,
    claude_model: str | None,
    ollama_url: str | None = None,
    llama_url: str | None = None,
    llama_model: str | None = None,
):
    """Build the appropriate runtime based on --backend flag."""
    if backend == "claude":
        from towel.agent.claude_runtime import ClaudeCodeRuntime

        return ClaudeCodeRuntime(
            config,
            skills=skills,
            memory=memory,
            model=claude_model or "sonnet",
        )
    elif backend == "ollama":
        from towel.agent.ollama_runtime import OllamaRuntime

        return OllamaRuntime(
            config,
            skills=skills,
            memory=memory,
            ollama_url=ollama_url or "http://localhost:11434",
        )
    elif backend == "llama":
        from towel.agent.llama_runtime import LlamaRuntime

        return LlamaRuntime(
            config,
            skills=skills,
            memory=memory,
            llama_url=llama_url or "http://localhost:8080",
            llama_model=llama_model,
        )
    else:
        from towel.agent.runtime import AgentRuntime

        return AgentRuntime(config, skills=skills, memory=memory)


@cli.command()
@click.option(
    "--agent", "-a", default=None, help="Agent profile to use (e.g., coder, researcher, writer)"
)
@click.option(
    "--turboquant/--no-turboquant", default=None, help="Enable TurboQuant KV cache compression"
)
@click.option("--tq-bits", default=None, type=int, help="TurboQuant quantization bits (3 or 4)")
@click.option(
    "--backend",
    "-b",
    default=None,
    type=click.Choice(["mlx", "claude", "ollama", "llama"]),
    help="Runtime backend (mlx=local, claude=Claude API, ollama=Ollama, llama=llama-server)",
)
@click.option(
    "--claude-model",
    default=None,
    help="Claude model when using --backend claude (e.g., sonnet, opus, haiku)",
)
@click.option(
    "--ollama-url",
    default=None,
    help="Ollama server URL when using --backend ollama (default: http://localhost:11434)",
)
@click.option(
    "--llama-url",
    default=None,
    help="llama-server URL when using --backend llama (default: http://localhost:8080)",
)
@click.option(
    "--llama-model",
    default=None,
    help="Path to a .gguf model file (auto-detected if omitted)",
)
def serve(
    agent: str | None,
    turboquant: bool | None,
    tq_bits: int | None,
    backend: str | None,
    claude_model: str | None,
    ollama_url: str | None,
    llama_url: str | None,
    llama_model: str | None,
) -> None:
    """Start the Towel gateway and agent runtime."""
    console.print(Panel(Text(BANNER, style="bold green"), border_style="green"))

    config = TowelConfig.load()
    model_config, identity = config.resolve_agent(agent)
    config.model = model_config
    config.identity = identity
    _apply_turboquant_overrides(config, turboquant, tq_bits)

    # Auto-detect backend if not specified
    if not backend:
        detected = _auto_detect_backend()
        if detected:
            backend = detected
            console.print(f"[dim]Auto-detected backend:[/dim] {backend}")

    if backend == "claude":
        console.print(f"[dim]Backend:[/dim] Claude Code CLI ({claude_model or 'sonnet'})")
    elif backend == "ollama":
        console.print(f"[dim]Backend:[/dim] Ollama ({ollama_url or 'http://localhost:11434'})")
        console.print(f"[dim]Model:[/dim] {config.model.name}")
    elif backend == "llama":
        console.print(f"[dim]Backend:[/dim] llama-server ({llama_url or 'http://localhost:8080'})")
        if llama_model:
            console.print(f"[dim]Model:[/dim] {llama_model}")
        else:
            console.print("[dim]Model:[/dim] auto-detect")
    else:
        console.print(f"[dim]Model:[/dim] {config.model.name}")
        if config.model.turboquant:
            console.print(
                f"[dim]KV cache:[/dim] TurboQuant "
                f"{config.model.turboquant_bits}-bit "
                f"(QJL ratio {config.model.turboquant_qjl_ratio})"
            )
    if agent:
        console.print(f"[dim]Agent:[/dim] {agent}")
    gw = f"{config.gateway.host}:{config.gateway.port}"
    console.print(f"[dim]Gateway:[/dim] ws://{gw}")
    api = f"{config.gateway.host}:{config.gateway.port + 1}"
    console.print(f"[dim]Web UI:[/dim] http://{api}/")
    console.print(
        f"[dim]API:[/dim] http://{api}/v1/chat/completions"
    )
    console.print()

    from towel.gateway.server import GatewayServer
    from towel.memory.store import MemoryStore

    memory = MemoryStore()
    skills = _build_skill_registry(config, memory_store=memory)
    agent_rt = _build_runtime(
        config, skills, memory, backend, claude_model, ollama_url, llama_url, llama_model
    )
    gateway = GatewayServer(config=config, agent=agent_rt)

    console.print("[green]Connecting...[/green]")
    try:
        asyncio.run(_start(agent_rt, gateway))
    finally:
        if hasattr(agent_rt, "shutdown"):
            agent_rt.shutdown()


async def _start(agent: Any, gateway: Any) -> None:
    await agent.load_model()
    console.print("[green]Ready. Gateway starting...[/green]")
    await gateway.start()


@cli.command()
@click.option("--master", default=None, help="Controller URL (auto-discovered via mDNS if omitted)")
@click.option("--id", "worker_id", default=None, help="Worker ID (defaults to hostname)")
@click.option(
    "--agent", "-a", default=None, help="Agent profile to use (e.g., coder, researcher, writer)"
)
@click.option(
    "--turboquant/--no-turboquant", default=None, help="Enable TurboQuant KV cache compression"
)
@click.option("--tq-bits", default=None, type=int, help="TurboQuant quantization bits (3 or 4)")
@click.option(
    "--backend",
    "-b",
    default=None,
    type=click.Choice(["mlx", "claude", "ollama", "llama"]),
    help="Runtime backend (mlx=local, claude=Claude API, ollama=Ollama, llama=llama-server)",
)
@click.option(
    "--claude-model",
    default=None,
    help="Claude model when using --backend claude (e.g., sonnet, opus, haiku)",
)
@click.option(
    "--ollama-url",
    default=None,
    help="Ollama server URL when using --backend ollama (default: http://localhost:11434)",
)
@click.option(
    "--llama-url",
    default=None,
    help="llama-server URL when using --backend llama (default: http://localhost:8080)",
)
@click.option(
    "--llama-model",
    default=None,
    help="Path to a .gguf model file (auto-detected if omitted)",
)
@click.option(
    "--allow-tools/--no-allow-tools",
    default=False,
    help="Allow tools to run on the remote worker machine",
)
def worker(
    master: str | None,
    worker_id: str | None,
    agent: str | None,
    turboquant: bool | None,
    tq_bits: int | None,
    backend: str | None,
    claude_model: str | None,
    ollama_url: str | None,
    llama_url: str | None,
    llama_model: str | None,
    allow_tools: bool,
) -> None:
    """Start a remote worker that executes jobs for a controller."""
    console.print(Panel(Text(BANNER, style="bold green"), border_style="green"))

    # Discover coordinator via mDNS if --master not given
    if not master:
        console.print("[dim]Searching for controller on the network...[/dim]")

        from towel.gateway.mdns import discover_controller

        ctrl = asyncio.run(discover_controller(timeout=30.0))
        if not ctrl:
            console.print(
                "[red]No controller found on the network.[/red]\n"
                "Start a controller with [bold]towel serve[/bold], or specify "
                "[bold]--master ws://host:port[/bold]."
            )
            raise SystemExit(1)
        master = ctrl.ws_url
        console.print(f"[green]Found controller:[/green] {ctrl.hostname} at {master}")

    config = TowelConfig.load()
    model_config, identity = config.resolve_agent(agent)
    config.model = model_config
    config.identity = identity
    _apply_turboquant_overrides(config, turboquant, tq_bits)

    # Auto-detect backend if not specified
    if not backend:
        detected = _auto_detect_backend()
        if detected:
            backend = detected
            console.print(f"[dim]Auto-detected backend:[/dim] {backend}")

    if backend == "claude":
        console.print(f"[dim]Backend:[/dim] Claude Code CLI ({claude_model or 'sonnet'})")
    elif backend == "ollama":
        console.print(f"[dim]Backend:[/dim] Ollama ({ollama_url or 'http://localhost:11434'})")
        console.print(f"[dim]Model:[/dim] {config.model.name}")
    elif backend == "llama":
        console.print(f"[dim]Backend:[/dim] llama-server ({llama_url or 'http://localhost:8080'})")
        if llama_model:
            console.print(f"[dim]Model:[/dim] {llama_model}")
        else:
            console.print("[dim]Model:[/dim] auto-detect")
    else:
        console.print(f"[dim]Model:[/dim] {config.model.name}")
    console.print(f"[dim]Controller:[/dim] {master}")
    console.print(f"[dim]Remote tools:[/dim] {'enabled' if allow_tools else 'disabled'}")
    console.print()

    from towel.gateway.worker_client import RemoteWorkerClient, default_worker_capabilities
    from towel.memory.store import MemoryStore
    from towel.skills.registry import SkillRegistry

    memory = MemoryStore()
    skills = _build_skill_registry(config, memory_store=memory) if allow_tools else SkillRegistry()
    agent_rt = _build_runtime(
        config, skills, memory, backend, claude_model, ollama_url, llama_url, llama_model
    )
    effective_backend = backend or "mlx"
    capabilities = default_worker_capabilities(
        config, effective_backend, allow_tools,
        llama_url=llama_url or "",
    )
    client = RemoteWorkerClient(
        master_url=master,
        agent=agent_rt,
        worker_id=worker_id or capabilities["hostname"],
        capabilities=capabilities,
    )

    try:
        asyncio.run(_start_worker(agent_rt, client))
    finally:
        if hasattr(agent_rt, "shutdown"):
            agent_rt.shutdown()


async def _start_worker(agent: Any, client: Any) -> None:
    await agent.load_model()
    console.print("[green]Worker ready. Waiting for controller jobs...[/green]")
    await client.run_forever()


@cli.command()
@click.option("--session", "-s", default="cli", help="Session ID")
@click.option("--agent", "-a", default=None, help="Agent profile (e.g., coder, researcher, writer)")
@click.option(
    "--turboquant/--no-turboquant", default=None, help="Enable TurboQuant KV cache compression"
)
@click.option("--tq-bits", default=None, type=int, help="TurboQuant quantization bits (3 or 4)")
@click.option(
    "--backend",
    "-b",
    default=None,
    type=click.Choice(["mlx", "claude", "ollama", "llama"]),
    help="Runtime backend (mlx=local, claude=Claude API, ollama=Ollama, llama=llama-server)",
)
@click.option(
    "--claude-model",
    default=None,
    help="Claude model when using --backend claude (e.g., sonnet, opus, haiku)",
)
@click.option(
    "--ollama-url",
    default=None,
    help="Ollama server URL when using --backend ollama (default: http://localhost:11434)",
)
@click.option(
    "--llama-url",
    default=None,
    help="llama-server URL when using --backend llama (default: http://localhost:8080)",
)
@click.option(
    "--llama-model",
    default=None,
    help="Path to a .gguf model file (auto-detected if omitted)",
)
def chat(
    session: str,
    agent: str | None,
    turboquant: bool | None,
    tq_bits: int | None,
    backend: str | None,
    claude_model: str | None,
    ollama_url: str | None,
    llama_url: str | None,
    llama_model: str | None,
) -> None:
    """Interactive chat with Towel."""
    console.print(
        Panel(
            "[bold green]Towel[/bold green] — Don't Panic.\n"
            "[dim]Type your message. Ctrl+C to exit. /help for commands.[/dim]",
            border_style="green",
        )
    )

    config = TowelConfig.load()
    model_config, identity = config.resolve_agent(agent)
    config.model = model_config
    config.identity = identity
    _apply_turboquant_overrides(config, turboquant, tq_bits)

    # Auto-detect backend if not specified
    if not backend:
        detected = _auto_detect_backend()
        if detected:
            backend = detected
            console.print(f"[dim]Auto-detected backend: {backend}[/dim]")

    if backend == "claude":
        console.print(f"[dim]Backend: Claude Code CLI ({claude_model or 'sonnet'})[/dim]")
    elif backend == "ollama":
        console.print(f"[dim]Backend: Ollama ({ollama_url or 'http://localhost:11434'})[/dim]")
    elif backend == "llama":
        console.print(f"[dim]Backend: llama-server ({llama_url or 'http://localhost:8080'})[/dim]")
    if agent:
        console.print(f"[dim]Agent: {agent}[/dim]")

    from towel.agent.conversation import Conversation, Role
    from towel.agent.events import EventType
    from towel.cli.slash import SlashContext, handle_slash
    from towel.memory.store import MemoryStore
    from towel.persistence.store import ConversationStore

    memory = MemoryStore()
    skills = _build_skill_registry(config, memory_store=memory)
    agent_rt = _build_runtime(
        config, skills, memory, backend, claude_model, ollama_url, llama_url, llama_model
    )
    store = ConversationStore()

    # Resume existing conversation or start fresh
    conv = store.load(session)
    if conv:
        console.print(f"[dim]Resumed conversation ({len(conv)} messages)[/dim]")
    else:
        conv = Conversation(id=session, channel="cli")

    slash_ctx = SlashContext(
        config=config,
        conv=conv,
        agent=agent_rt,
        memory=memory,
        store=store,
    )
    slash_ctx.current_agent_name = agent

    async def _chat_loop() -> None:
        console.print("[dim]Connecting...[/dim]")
        await agent_rt.load_model()
        console.print("[green]Ready.[/green]\n")

        while True:
            try:
                user_input = console.input("[bold cyan]you>[/bold cyan] ")
            except (EOFError, KeyboardInterrupt):
                if len(conv) > 0:
                    store.save(conv)
                    console.print(f"\n[dim]Conversation saved ({conv.id}).[/dim]")
                console.print("[dim]So long, and thanks for all the fish.[/dim]")
                break

            if not user_input.strip():
                continue

            # Handle slash commands (True=consumed, False=run agent, None=not a command)
            slash_result = handle_slash(user_input, slash_ctx)
            if slash_result is True:
                continue
            elif slash_result is False:
                pass  # template: message already in conv, proceed to streaming
            elif slash_result is None:
                # Normal message: expand @file references and add to conversation
                from towel.agent.refs import expand_refs, parse_refs

                refs = parse_refs(user_input)
                expanded = expand_refs(user_input) if refs else user_input
                if refs and expanded != user_input:
                    ref_names = [r.path.split("/")[-1] for r in refs]
                    console.print(f"[dim]  attached: {', '.join(ref_names)}[/dim]")
                conv.add(Role.USER, expanded)

            # Stream tokens live to the terminal (Ctrl+C cancels generation)
            console.print("[bold green]towel>[/bold green] ", end="")
            _cancelled = False
            try:
                async for event in agent_rt.step_streaming(conv):
                    match event.type:
                        case EventType.TOKEN:
                            print(event.data["content"], end="", flush=True)
                        case EventType.TOOL_CALL:
                            tool = event.data["tool"]
                            print()
                            console.print(
                                f"  [yellow]>> {tool}({event.data['arguments']})[/yellow]"
                            )
                        case EventType.TOOL_RESULT:
                            result = event.data["result"]
                            display = result[:200] + "..." if len(result) > 200 else result
                            console.print(f"  [dim]<< {display}[/dim]")
                            console.print("[bold green]towel>[/bold green] ", end="")
                        case EventType.RESPONSE_COMPLETE:
                            print()
                            meta = event.data.get("metadata", {})
                            if meta.get("tps"):
                                console.print(
                                    f"[dim]({meta['tps']:.1f} tok/s, {meta['tokens']} tokens)[/dim]"
                                )
                        case EventType.CANCELLED:
                            print()
                            console.print("[dim](generation stopped)[/dim]")
                            _cancelled = True
                        case EventType.ERROR:
                            print()
                            console.print(f"[red]Error: {event.data['message']}[/red]")
            except KeyboardInterrupt:
                agent_rt.cancel()
                print()
                console.print("[dim](generation stopped)[/dim]")
                _cancelled = True

            # Auto-title after first exchange
            if not conv.title and len(conv) >= 2:
                from towel.agent.titler import generate_title

                first_user = next((m.content for m in conv.messages if m.role == Role.USER), "")
                title = generate_title(first_user)
                if title:
                    conv.title = title

            store.save(conv)
            console.print()

    try:
        asyncio.run(_chat_loop())
    finally:
        if hasattr(agent_rt, "shutdown"):
            agent_rt.shutdown()


@cli.command()
def status() -> None:
    """Show gateway status."""
    import httpx

    config = TowelConfig.load()
    url = f"http://{config.gateway.host}:{config.gateway.port + 1}/health"
    workers_url = f"http://{config.gateway.host}:{config.gateway.port + 1}/workers"

    try:
        resp = httpx.get(url, timeout=3)
        data = resp.json()
        workers_resp = httpx.get(workers_url, timeout=3)
        workers_data = workers_resp.json()
        workers = workers_data.get("workers", [])
        pins = workers_data.get("pins", {})
        worker_lines = []
        for worker in workers:
            caps = worker.get("capabilities", {})
            model = str(caps.get("model", "")).split("/")[-1] or "unknown"
            backend = caps.get("backend", "unknown")
            modes = ",".join(caps.get("modes", [])) or "unknown"
            state = "busy" if worker.get("busy") else "idle"
            availability = "enabled" if worker.get("enabled", True) else "disabled"
            flow = "draining" if worker.get("draining") else "ready"
            session_id = worker.get("current_session_id") or "-"
            worker_lines.append(
                f"  - {worker['id']} [{state}/{availability}/{flow}] "
                f"{backend} {modes} {model} session={session_id}"
            )
        workers_block = "\n".join(worker_lines) if worker_lines else "  - none"
        pin_lines = [
            f"  - {session_id} -> {worker_id}" for session_id, worker_id in sorted(pins.items())
        ]
        pins_block = "\n".join(pin_lines) if pin_lines else "  - none"
        console.print(
            Panel(
                f"[green]Status:[/green] {data['status']}\n"
                f"[green]Version:[/green] {data['version']}\n"
                f"[green]Connections:[/green] {data['connections']}\n"
                f"[green]Sessions:[/green] {data['sessions']}\n"
                f"[green]Workers:[/green] {data['workers']['total']} "
                f"(idle {data['workers']['idle']}, busy {data['workers']['busy']}, "
                f"draining {data['workers']['draining']}, disabled {data['workers']['disabled']})\n"
                f"[green]Worker Inventory:[/green]\n{workers_block}\n"
                f"[green]Pinned Sessions:[/green]\n{pins_block}\n"
                f"[dim]{data['motto']}[/dim]",
                title="Towel Gateway",
                border_style="green",
            )
        )
    except Exception:
        console.print("[red]Gateway not running.[/red] Start it with: towel serve")
        sys.exit(1)


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output worker inventory as JSON")
def workers(as_json: bool) -> None:
    """Show connected LAN workers."""
    import json as json_mod

    import httpx

    config = TowelConfig.load()
    workers_url = f"http://{config.gateway.host}:{config.gateway.port + 1}/workers"

    try:
        resp = httpx.get(workers_url, timeout=3)
        data = resp.json()
    except Exception:
        console.print("[red]Gateway not running.[/red] Start it with: towel serve")
        sys.exit(1)

    if as_json:
        console.print(json_mod.dumps(data, indent=2))
        return

    requirements = data.get("requirements", {})
    worker_items = data.get("workers", [])
    pins = data.get("pins", {})
    requirement_text = (
        f"backend={requirements.get('backend', 'any')} "
        f"mode={requirements.get('mode', 'any')} "
        f"model={str(requirements.get('model', 'any')).split('/')[-1]} "
        f"tools={requirements.get('tools', 'any')}"
    )
    pin_lines = [
        f"  - {session_id} -> {worker_id}" for session_id, worker_id in sorted(pins.items())
    ]
    pins_block = "\n".join(pin_lines) if pin_lines else "  - none"
    if not worker_items:
        console.print(
            Panel(
                f"[green]Workers:[/green] 0\n"
                f"[green]Controller Needs:[/green] {requirement_text}\n"
                f"[green]Pinned Sessions:[/green]\n{pins_block}\n"
                f"[dim]No remote workers connected.[/dim]",
                title="Towel Workers",
                border_style="green",
            )
        )
        return

    lines = []
    for worker in worker_items:
        caps = worker.get("capabilities", {})
        model = str(caps.get("model", "")).split("/")[-1] or "unknown"
        backend = caps.get("backend", "unknown")
        modes = ",".join(caps.get("modes", [])) or "unknown"
        tools = "yes" if caps.get("tools") else "no"
        state = "busy" if worker.get("busy") else "idle"
        availability = "enabled" if worker.get("enabled", True) else "disabled"
        flow = "draining" if worker.get("draining") else "ready"
        session_id = worker.get("current_session_id") or "-"
        lines.append(
            f"  - {worker['id']} [{state}/{availability}/{flow}] backend={backend} mode={modes} "
            f"model={model} tools={tools} session={session_id}"
        )

    console.print(
        Panel(
            f"[green]Workers:[/green] {len(worker_items)}\n"
            f"[green]Controller Needs:[/green] {requirement_text}\n"
            + "\n".join(lines)
            + "\n"
            + f"[green]Pinned Sessions:[/green]\n{pins_block}",
            title="Towel Workers",
            border_style="green",
        )
    )


@cli.command(name="pin-worker")
@click.argument("session_id")
@click.argument("worker_id")
def pin_worker(session_id: str, worker_id: str) -> None:
    """Pin a session to a specific worker."""
    import httpx

    config = TowelConfig.load()
    url = f"http://{config.gateway.host}:{config.gateway.port + 1}/sessions/{session_id}/pin-worker"

    try:
        resp = httpx.post(url, json={"worker_id": worker_id}, timeout=3)
        if resp.status_code >= 400:
            message = resp.json().get("error", "Failed to pin worker")
            console.print(f"[red]{message}[/red]")
            sys.exit(1)
        data = resp.json()
        console.print(
            f"[green]Pinned[/green] session [bold]{data['session_id']}[/bold] "
            f"to worker [bold]{data['worker_id']}[/bold]"
        )
    except Exception:
        console.print("[red]Gateway not running.[/red] Start it with: towel serve")
        sys.exit(1)


@cli.command(name="unpin-worker")
@click.argument("session_id")
def unpin_worker(session_id: str) -> None:
    """Remove an explicit worker pin from a session."""
    import httpx

    config = TowelConfig.load()
    url = f"http://{config.gateway.host}:{config.gateway.port + 1}/sessions/{session_id}/pin-worker"

    try:
        resp = httpx.delete(url, timeout=3)
        if resp.status_code >= 400:
            message = resp.json().get("error", "Failed to unpin worker")
            console.print(f"[red]{message}[/red]")
            sys.exit(1)
        data = resp.json()
        action = "Unpinned" if data.get("removed") else "No pin found for"
        console.print(f"[green]{action}[/green] session [bold]{data['session_id']}[/bold]")
    except Exception:
        console.print("[red]Gateway not running.[/red] Start it with: towel serve")
        sys.exit(1)


def _update_worker_state(worker_id: str, **payload: Any) -> dict[str, Any]:
    import httpx

    config = TowelConfig.load()
    url = f"http://{config.gateway.host}:{config.gateway.port + 1}/workers/{worker_id}/state"
    resp = httpx.post(url, json=payload, timeout=3)
    if resp.status_code >= 400:
        message = resp.json().get("error", "Failed to update worker state")
        raise RuntimeError(message)
    return resp.json()


@cli.command(name="drain-worker")
@click.argument("worker_id")
def drain_worker(worker_id: str) -> None:
    """Mark a worker as draining so it stops receiving new sessions."""
    try:
        _update_worker_state(worker_id, draining=True)
        console.print(f"[green]Draining[/green] worker [bold]{worker_id}[/bold]")
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    except Exception:
        console.print("[red]Gateway not running.[/red] Start it with: towel serve")
        sys.exit(1)


@cli.command(name="undrain-worker")
@click.argument("worker_id")
def undrain_worker(worker_id: str) -> None:
    """Allow a draining worker to receive new sessions again."""
    try:
        _update_worker_state(worker_id, draining=False)
        console.print(f"[green]Undrained[/green] worker [bold]{worker_id}[/bold]")
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    except Exception:
        console.print("[red]Gateway not running.[/red] Start it with: towel serve")
        sys.exit(1)


@cli.command(name="disable-worker")
@click.argument("worker_id")
def disable_worker(worker_id: str) -> None:
    """Disable a worker so the scheduler will not use it."""
    try:
        _update_worker_state(worker_id, enabled=False)
        console.print(f"[green]Disabled[/green] worker [bold]{worker_id}[/bold]")
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    except Exception:
        console.print("[red]Gateway not running.[/red] Start it with: towel serve")
        sys.exit(1)


@cli.command(name="enable-worker")
@click.argument("worker_id")
def enable_worker(worker_id: str) -> None:
    """Re-enable a disabled worker."""
    try:
        _update_worker_state(worker_id, enabled=True)
        console.print(f"[green]Enabled[/green] worker [bold]{worker_id}[/bold]")
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    except Exception:
        console.print("[red]Gateway not running.[/red] Start it with: towel serve")
        sys.exit(1)


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output session routes as JSON")
def routes(as_json: bool) -> None:
    """Show session-to-worker routing state."""
    import json as json_mod

    import httpx

    config = TowelConfig.load()
    sessions_url = f"http://{config.gateway.host}:{config.gateway.port + 1}/sessions"

    try:
        resp = httpx.get(sessions_url, timeout=3)
        data = resp.json()
    except Exception:
        console.print("[red]Gateway not running.[/red] Start it with: towel serve")
        sys.exit(1)

    if as_json:
        console.print(json_mod.dumps(data, indent=2))
        return

    sessions = data.get("sessions", [])
    if not sessions:
        console.print(
            Panel(
                "[green]Routes:[/green] 0\n[dim]No active sessions.[/dim]",
                title="Towel Routes",
                border_style="green",
            )
        )
        return

    lines = []
    for session in sessions:
        session_id = session.get("id", "unknown")
        channel = session.get("channel", "unknown")
        messages = session.get("messages", 0)
        current_worker = session.get("worker_id") or "-"
        pinned_worker = session.get("pinned_worker_id") or "-"
        lines.append(
            f"  - {session_id} channel={channel} messages={messages} "
            f"current={current_worker} pinned={pinned_worker}"
        )

    console.print(
        Panel(
            f"[green]Routes:[/green] {len(sessions)}\n" + "\n".join(lines),
            title="Towel Routes",
            border_style="green",
        )
    )


STARTER_CONFIG = """\
# ─────────────────────────────────────────────────────────
# Towel — Don't Panic.
# Configuration file: ~/.towel/config.toml
# ─────────────────────────────────────────────────────────

# System prompt — the agent's core identity.
# Override per-session with: /system <prompt>  or  towel ask -S "..."
identity = "You are Towel, a helpful local AI assistant. Don't Panic."

# ─────────────────────────────────────────────────────────
# Model settings
# ─────────────────────────────────────────────────────────
[model]
name = "mlx-community/Llama-3.3-70B-Instruct-4bit"
max_tokens = 4096        # max output tokens per generation
context_window = 8192    # total context window (input + output budget)
temperature = 0.7        # 0.0 = deterministic, 1.0 = creative
top_p = 0.95

# ─────────────────────────────────────────────────────────
# Gateway (WebSocket + HTTP server)
# ─────────────────────────────────────────────────────────
[gateway]
host = "127.0.0.1"
port = 18742             # WebSocket port (HTTP is port + 1)

# ─────────────────────────────────────────────────────────
# Skills — directories to scan for custom skills
# Drop a .py file with a Skill subclass into these directories.
# ─────────────────────────────────────────────────────────
skills_dirs = ["~/.towel/skills", "./skills"]

# ─────────────────────────────────────────────────────────
# Agent profiles — switch with: towel chat --agent coder
# Built-in profiles (coder, researcher, writer) are always
# available. Define custom ones below to override or extend.
# ─────────────────────────────────────────────────────────

# Uncomment to set a default agent for all commands:
# default_agent = "coder"

# Example: custom agent profile
# [agents.analyst]
# description = "Data analysis specialist"
# identity = "You are Towel (analyst mode). Focus on data, charts, and insights."
# [agents.analyst.model]
# name = "mlx-community/Llama-3.3-70B-Instruct-4bit"
# context_window = 16384
# temperature = 0.3

# ─────────────────────────────────────────────────────────
# Quick start:
#   towel chat                   Interactive chat
#   towel chat --agent coder     Chat with coder profile
#   towel ask "your question"    One-shot query
#   towel serve                  Start gateway + web UI
#   towel doctor                 Check your setup
#   towel agents                 List available agents
#   towel memory list            View persistent memories
# ─────────────────────────────────────────────────────────
"""


@cli.command()
def init() -> None:
    """Initialize Towel configuration."""
    TOWEL_HOME.mkdir(parents=True, exist_ok=True)
    config_path = TOWEL_HOME / "config.toml"

    if config_path.exists():
        console.print(f"[yellow]Config already exists:[/yellow] {config_path}")
        return

    config_path.write_text(STARTER_CONFIG, encoding="utf-8")
    (TOWEL_HOME / "skills").mkdir(exist_ok=True)
    (TOWEL_HOME / "memory").mkdir(exist_ok=True)
    (TOWEL_HOME / "conversations").mkdir(exist_ok=True)

    console.print(
        Panel(
            "[bold green]Towel initialized.[/bold green]\n\n"
            f"  Config:         [cyan]{config_path}[/cyan]\n"
            f"  Skills:         [cyan]{TOWEL_HOME / 'skills'}[/cyan]\n"
            f"  Memory:         [cyan]{TOWEL_HOME / 'memory'}[/cyan]\n"
            f"  Conversations:  [cyan]{TOWEL_HOME / 'conversations'}[/cyan]\n\n"
            "[dim]Next steps:[/dim]\n"
            "  1. towel doctor        Check your setup\n"
            "  2. towel chat          Start chatting\n"
            "  3. towel context -c    Add project context",
            border_style="green",
            title="Don't Panic.",
        )
    )


@cli.command()
@click.option("--create", "-c", is_flag=True, help="Create a .towel.md in the current directory")
def context(create: bool) -> None:
    """Show or create project context (.towel.md)."""
    from pathlib import Path

    from towel.agent.project import CONTEXT_FILENAME, find_project_contexts, load_project_context

    if create:
        target = Path.cwd() / CONTEXT_FILENAME
        if target.exists():
            console.print(f"[yellow]Already exists:[/yellow] {target}")
            return
        target.write_text(
            "# Project Context\n\n"
            "Describe your project here. Towel will read this automatically.\n\n"
            "## Stack\n- \n\n"
            "## Structure\n- \n\n"
            "## Conventions\n- \n\n"
            "## Current Work\n- \n",
            encoding="utf-8",
        )
        console.print(f"[green]Created:[/green] {target}")
        console.print("[dim]Edit it with your project details. Towel loads it automatically.[/dim]")
        return

    paths = find_project_contexts()
    if not paths:
        console.print("[dim]No .towel.md found in current directory or parents.[/dim]")
        console.print("Create one with: [green]towel context --create[/green]")
        return

    console.print(f"[bold]Project context ({len(paths)} file(s)):[/bold]\n")
    for p in paths:
        console.print(f"  [green]{p}[/green] [dim]({p.stat().st_size} bytes)[/dim]")

    block = load_project_context()
    if block:
        console.print()
        console.print(block[:1000])
        if len(block) > 1000:
            console.print("[dim]... (truncated)[/dim]")


@cli.command()
def skills() -> None:
    """List installed skills."""
    config = TowelConfig.load()
    reg = _build_skill_registry(config)

    if not reg.list_skills():
        console.print("[dim]No skills installed.[/dim]")
        return

    console.print("[bold]Installed skills:[/bold]\n")
    for skill_name in reg.list_skills():
        skill = reg.get_skill(skill_name)
        if skill:
            tools = ", ".join(t.name for t in skill.tools())
            console.print(f"  [green]{skill.name}[/green] — {skill.description}")
            console.print(f"    [dim]tools: {tools}[/dim]")


@cli.command()
@click.option("--limit", "-n", default=20, help="Number of conversations to show")
@click.option("--tag", "-t", default=None, help="Filter by tag")
def history(limit: int, tag: str | None) -> None:
    """List saved conversations."""
    from towel.persistence.store import ConversationStore

    store = ConversationStore()
    convos = store.list_conversations(limit=limit * 3 if tag else limit)

    if tag:
        # Filter by tag — need to load full conversations to check tags
        import json as json_mod

        filtered = []
        for c in convos:
            path = store._path_for(c.id)
            try:
                data = json_mod.loads(path.read_text(encoding="utf-8"))
                tags = data.get("tags", [])
                if tag.lower() in tags:
                    filtered.append(c)
            except (json_mod.JSONDecodeError, OSError):
                continue
        convos = filtered[:limit]

    if not convos:
        if tag:
            console.print(f"[dim]No conversations tagged '[green]{tag}[/green]'.[/dim]")
        else:
            console.print("[dim]No saved conversations.[/dim]")
            console.print("Start one with: towel chat")
        return

    header = "[bold]Recent conversations"
    if tag:
        header += f" tagged [green]{tag}[/green]"
    header += f"[/bold] ({len(convos)}):\n"
    console.print(header)

    for c in convos:
        # Load tags for display
        tag_display = ""
        try:
            import json as json_mod

            path = store._path_for(c.id)
            data = json_mod.loads(path.read_text(encoding="utf-8"))
            tags = data.get("tags", [])
            if tags:
                tag_display = " " + " ".join(f"[yellow]#{t}[/yellow]" for t in tags)
        except (Exception,):
            pass

        console.print(
            f"  [green]{c.id}[/green]  "
            f"[dim]{c.created_at[:16]}[/dim]  "
            f"[dim]({c.message_count} msgs, {c.channel})[/dim]{tag_display}"
        )
        console.print(f"    {c.summary}")


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Max conversations to return")
@click.option(
    "--role",
    "-r",
    default=None,
    type=click.Choice(["user", "assistant", "tool"]),
    help="Filter by message role",
)
@click.option("--regex", "-e", is_flag=True, help="Treat query as regex")
def search(query: str, limit: int, role: str | None, regex: bool) -> None:
    """Search across all saved conversations."""
    from towel.agent.conversation import Role
    from towel.persistence.store import ConversationStore

    store = ConversationStore()
    role_filter = Role(role) if role else None
    results = store.search(query, limit=limit, role_filter=role_filter, regex=regex)

    if not results:
        console.print(f"[dim]No matches for:[/dim] {query}")
        return

    total_matches = sum(len(r.matches) for r in results)
    console.print(
        f"[bold]Found {total_matches} match(es) across {len(results)} conversation(s)[/bold]\n"
    )

    for result in results:
        console.print(
            f"  [green]{result.conversation_id}[/green]  "
            f"[dim]{result.created_at[:16]}[/dim]  "
            f"[dim]({len(result.matches)} matches)[/dim]"
        )
        console.print(f"    [dim]{result.summary}[/dim]")

        for match in result.matches[:3]:  # show up to 3 matches per conversation
            # Highlight the query in the snippet
            snippet = match.snippet
            if not regex:
                snippet = re.sub(
                    f"({re.escape(query)})",
                    r"[bold yellow]\1[/bold yellow]",
                    snippet,
                    flags=re.IGNORECASE,
                )
            console.print(f"    [{match.role}] {snippet}")
        if len(result.matches) > 3:
            console.print(f"    [dim]... and {len(result.matches) - 3} more matches[/dim]")
        console.print()

    console.print("[dim]View a conversation: towel show <id>[/dim]")


@cli.command()
@click.argument("conversation_id")
@click.option("--tail", "-t", default=0, help="Show only last N messages (0 = all)")
def show(conversation_id: str, tail: int) -> None:
    """Show a saved conversation."""
    from towel.agent.conversation import Role
    from towel.persistence.store import ConversationStore

    store = ConversationStore()
    conv = store.load(conversation_id)

    if not conv:
        console.print(f"[red]Conversation not found:[/red] {conversation_id}")
        sys.exit(1)

    title_line = (
        f"[bold]{conv.display_title}[/bold]"
        if conv.title
        else f"[bold]{conv.id}[/bold]"
    )
    console.print(
        Panel(
            f"{title_line}\n[dim]{conv.id} · {conv.channel}"
            f" · {conv.created_at.isoformat()[:16]}[/dim]",
            border_style="green",
        )
    )

    messages = conv.messages[-tail:] if tail else conv.messages
    if tail and len(conv.messages) > tail:
        console.print(f"[dim]... {len(conv.messages) - tail} earlier messages ...[/dim]\n")

    for msg in messages:
        match msg.role:
            case Role.USER:
                console.print(f"[bold cyan]you>[/bold cyan] {msg.content}")
            case Role.ASSISTANT:
                console.print(f"[bold green]towel>[/bold green] {msg.content}")
            case Role.TOOL:
                display = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                console.print(f"  [dim]{display}[/dim]")
            case Role.SYSTEM:
                console.print(f"[dim]system: {msg.content}[/dim]")
        console.print()


@cli.command(name="export")
@click.argument("conversation_id")
@click.option(
    "--format",
    "-f",
    "fmt",
    default="markdown",
    type=click.Choice(["markdown", "text", "json", "html"]),
    help="Export format",
)
@click.option(
    "--output", "-o", default=None, type=click.Path(), help="Write to file instead of stdout"
)
@click.option("--metadata", "-m", is_flag=True, help="Include timestamps and stats (markdown only)")
def export_cmd(conversation_id: str, fmt: str, output: str | None, metadata: bool) -> None:
    """Export a conversation to markdown, text, or JSON.

    \b
    Examples:
        towel export abc123
        towel export abc123 -f json > backup.json
        towel export abc123 -o conversation.md
        towel export abc123 -f text -o chat.txt
        towel export abc123 -m   # include timestamps
    """
    from towel.persistence.export import export_html, export_json, export_markdown, export_text
    from towel.persistence.store import ConversationStore

    store = ConversationStore()
    conv = store.load(conversation_id)

    if not conv:
        console.print(f"[red]Conversation not found:[/red] {conversation_id}")
        sys.exit(1)

    match fmt:
        case "markdown":
            result = export_markdown(conv, include_metadata=metadata)
        case "text":
            result = export_text(conv)
        case "json":
            result = export_json(conv)
        case "html":
            result = export_html(conv, include_metadata=metadata)
        case _:
            result = export_markdown(conv)

    if output:
        from pathlib import Path

        Path(output).write_text(result, encoding="utf-8")
        console.print(f"[green]Exported to:[/green] {output}")
    else:
        print(result)


@cli.command(name="import")
@click.argument("path", type=click.Path(exists=True))
@click.option("--tag", "-t", default=None, help="Add a tag to imported conversations")
def import_cmd(path: str, tag: str | None) -> None:
    """Import conversations from JSON files.

    \b
    Accepts a single JSON file or a directory of JSON files.
    Supports both single-conversation and array-of-conversations formats.

    \b
    Examples:
        towel import backup.json
        towel import conversations/
        towel import backup.json --tag imported
    """
    import json as json_mod
    from pathlib import Path

    from towel.agent.conversation import Conversation
    from towel.persistence.store import ConversationStore

    store = ConversationStore()
    target = Path(path)

    files: list[Path] = []
    if target.is_dir():
        files = sorted(target.glob("*.json"))
        if not files:
            console.print(f"[dim]No JSON files found in {target}[/dim]")
            return
    else:
        files = [target]

    imported = 0
    skipped = 0
    errors = 0

    for f in files:
        try:
            data = json_mod.loads(f.read_text(encoding="utf-8"))
        except (json_mod.JSONDecodeError, OSError) as e:
            console.print(f"  [red]Error reading {f.name}:[/red] {e}")
            errors += 1
            continue

        # Handle both single conversation and array of conversations
        items: list[dict] = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "id" in data and "messages" in data:
            items = [data]
        else:
            console.print(f"  [yellow]Skipping {f.name}:[/yellow] unrecognized format")
            errors += 1
            continue

        for item in items:
            try:
                conv = Conversation.from_dict(item)
            except (KeyError, ValueError) as e:
                console.print(f"  [yellow]Skipping entry in {f.name}:[/yellow] {e}")
                errors += 1
                continue

            if store.exists(conv.id):
                skipped += 1
                continue

            if tag:
                if tag.lower() not in conv.tags:
                    conv.tags.append(tag.lower())

            store.save(conv)
            imported += 1

    console.print("\n[bold]Import complete:[/bold]")
    console.print(f"  [green]Imported:[/green] {imported}")
    if skipped:
        console.print(f"  [dim]Skipped (already exist):[/dim] {skipped}")
    if errors:
        console.print(f"  [yellow]Errors:[/yellow] {errors}")


@cli.command()
@click.argument("conversation_id")
def resume(conversation_id: str) -> None:
    """Resume a saved conversation."""
    from towel.persistence.store import ConversationStore

    store = ConversationStore()
    if not store.exists(conversation_id):
        console.print(f"[red]Conversation not found:[/red] {conversation_id}")
        console.print("List conversations with: towel history")
        sys.exit(1)

    # Delegate to chat with the existing session ID
    ctx = click.get_current_context()
    ctx.invoke(chat, session=conversation_id)


@cli.command()
def doctor() -> None:
    """Diagnose your Towel setup."""
    from towel.cli.doctor import run_doctor

    console.print(
        Panel(
            "[bold green]Towel Doctor[/bold green] — checking your setup...",
            border_style="green",
        )
    )

    config = TowelConfig.load()
    checks = run_doctor(config)

    for check in checks:
        check.render()

    passed = sum(1 for c in checks if c.passed and not c.warnings)
    warned = sum(1 for c in checks if c.passed and c.warnings)
    failed = sum(1 for c in checks if not c.passed)

    console.print()
    parts = [f"[green]{passed} passed[/green]"]
    if warned:
        parts.append(f"[yellow]{warned} warnings[/yellow]")
    if failed:
        parts.append(f"[red]{failed} failed[/red]")
    console.print(f"  {', '.join(parts)}")

    if failed:
        console.print("\n  [dim]Fix the issues above, then run towel doctor again.[/dim]")
    elif warned:
        console.print("\n  [dim]Mostly hoopy. Check the warnings above.[/dim]")
    else:
        console.print("\n  [dim]Everything is hoopy. Don't Panic.[/dim]")


@cli.command(name="install-worker")
def install_worker() -> None:
    """Install systemd user service for auto-starting the Towel worker on boot."""
    import shutil
    from pathlib import Path

    service_name = "towel-worker.service"
    systemd_dir = Path.home() / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True, exist_ok=True)

    # Find towel binary
    towel_bin = shutil.which("towel")
    if not towel_bin:
        console.print("[red]Cannot find 'towel' in PATH. Install towel first.[/red]")
        raise SystemExit(1)

    service_content = f"""\
[Unit]
Description=Towel AI Worker — auto-discovers GPU, models, and controller
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={towel_bin} worker
Restart=always
RestartSec=10
Environment=PATH={Path.home() / '.local' / 'bin'}:/usr/local/bin:/usr/bin:/bin

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=towel-worker

[Install]
WantedBy=default.target
"""
    dest = systemd_dir / service_name
    dest.write_text(service_content)
    console.print(f"[green]Wrote[/green] {dest}")

    # Enable and start
    import subprocess

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", service_name], check=True)
    subprocess.run(["systemctl", "--user", "start", service_name], check=True)

    # Enable lingering so it runs without login
    subprocess.run(["loginctl", "enable-linger"], check=False)

    console.print(f"[green]Enabled and started {service_name}[/green]")
    console.print()
    console.print("[dim]The worker will now start on boot and auto-discover the controller.[/dim]")
    console.print("[dim]Check status:  systemctl --user status towel-worker[/dim]")
    console.print("[dim]View logs:     journalctl --user -u towel-worker -f[/dim]")
    console.print("[dim]Stop:          systemctl --user stop towel-worker[/dim]")
    console.print("[dim]Uninstall:     systemctl --user disable --now towel-worker[/dim]")


@cli.command()
@click.argument("prompt", nargs=-1)
@click.option("--session", "-s", default=None, help="Session ID (enables conversation context)")
@click.option("--agent", "-a", default=None, help="Agent profile (e.g., coder, researcher, writer)")
@click.option(
    "--template", "-T", default=None, help="Prompt template (e.g., review, explain, summarize)"
)
@click.option(
    "--var", "-V", multiple=True, help="Template variable: key=value (e.g., -V lang=Spanish)"
)
@click.option("--system", "-S", default=None, help="Override system prompt for this query")
@click.option("--raw", "-r", is_flag=True, help="Raw output only (no formatting, no stats)")
@click.option("--stream/--no-stream", default=True, help="Stream tokens as they generate")
@click.option(
    "--turboquant/--no-turboquant", default=None, help="Enable TurboQuant KV cache compression"
)
@click.option("--tq-bits", default=None, type=int, help="TurboQuant quantization bits (3 or 4)")
def ask(
    prompt: tuple[str, ...],
    session: str | None,
    agent: str | None,
    template: str | None,
    var: tuple[str, ...],
    system: str | None,
    raw: bool,
    stream: bool,
    turboquant: bool | None,
    tq_bits: int | None,
) -> None:
    """One-shot query — scriptable and pipeable.

    \b
    Examples:
        towel ask "what is the meaning of life"
        echo "summarize this" | towel ask
        cat data.json | towel ask "analyze this data"
        towel ask -T review @src/main.py
        towel ask -T translate -V lang=Spanish "Hello world"
        towel ask -s research "follow up on last question"
        towel ask -S "You are a poet" "write about towels"
        towel ask -r "just the answer" > output.txt
    """

    from towel.agent.conversation import Conversation, Role
    from towel.agent.events import EventType
    from towel.agent.runtime import AgentRuntime
    from towel.persistence.store import ConversationStore

    # Build the prompt from args + stdin
    parts: list[str] = []

    # Check if stdin has data (piped input)
    if not sys.stdin.isatty():
        stdin_data = sys.stdin.read().strip()
        if stdin_data:
            parts.append(stdin_data)

    # Add CLI arguments
    if prompt:
        parts.append(" ".join(prompt))

    if not parts:
        if not raw:
            console.print("[red]No prompt provided.[/red]")
            console.print('Usage: towel ask "your question"')
            console.print('   or: echo "question" | towel ask')
        sys.exit(1)

    full_prompt = "\n\n".join(parts)

    # Apply template if specified
    if template:
        from towel.templates.engine import TemplateEngine

        engine = TemplateEngine()
        variables = dict(kv.split("=", 1) for kv in var if "=" in kv)
        rendered = engine.render(template, input_text=full_prompt, variables=variables)
        if rendered is None:
            if not raw:
                console.print(f"[red]Unknown template:[/red] {template}")
                console.print("List templates with: towel templates")
            sys.exit(1)
        full_prompt = rendered

    # Expand @file references
    from towel.agent.refs import expand_refs

    full_prompt = expand_refs(full_prompt)

    config = TowelConfig.load()

    # Apply agent profile, then system override (system wins)
    model_config, identity = config.resolve_agent(agent)
    config.model = model_config
    config.identity = identity
    _apply_turboquant_overrides(config, turboquant, tq_bits)
    if system:
        config.identity = system

    from towel.memory.store import MemoryStore

    memory = MemoryStore()
    skills = _build_skill_registry(config, memory_store=memory)
    agent_rt = AgentRuntime(config, skills=skills, memory=memory)
    store = ConversationStore()

    # Load or create conversation
    conv = None
    if session:
        conv = store.load(session)
    if conv is None:
        sid = session or f"ask-{__import__('uuid').uuid4().hex[:8]}"
        conv = Conversation(id=sid, channel="cli")

    conv.add(Role.USER, full_prompt)

    async def _run() -> None:
        if not raw:
            console.print("[dim]Loading model...[/dim]", stderr=True)
        await agent_rt.load_model()

        if stream and not raw:
            async for event in agent_rt.step_streaming(conv):
                match event.type:
                    case EventType.TOKEN:
                        print(event.data["content"], end="", flush=True)
                    case EventType.TOOL_CALL:
                        tool = event.data["tool"]
                        console.print(
                            f"\n  [yellow]>> {tool}({event.data['arguments']})[/yellow]",
                            stderr=True,
                        )
                    case EventType.TOOL_RESULT:
                        result = event.data["result"]
                        display = result[:200] + "..." if len(result) > 200 else result
                        console.print(f"  [dim]<< {display}[/dim]", stderr=True)
                    case EventType.RESPONSE_COMPLETE:
                        print()
                        meta = event.data.get("metadata", {})
                        if meta.get("tps"):
                            console.print(
                                f"[dim]({meta['tps']:.1f} tok/s, {meta['tokens']} tokens)[/dim]",
                                stderr=True,
                            )
        elif stream and raw:
            async for event in agent_rt.step_streaming(conv):
                match event.type:
                    case EventType.TOKEN:
                        print(event.data["content"], end="", flush=True)
                    case EventType.RESPONSE_COMPLETE:
                        print()
        else:
            response = await agent_rt.step(conv)
            print(response.content)

        if session:
            store.save(conv)

    asyncio.run(_run())


@cli.group(invoke_without_command=True)
@click.pass_context
def agents(ctx: click.Context) -> None:
    """Manage agent profiles."""
    if ctx.invoked_subcommand is None:
        # Default: list agents
        config = TowelConfig.load()
        all_agents = config.list_agents()

        if not all_agents:
            console.print("[dim]No agent profiles configured.[/dim]")
            return

        console.print("[bold]Available agents:[/bold]\n")
        for name, profile in all_agents.items():
            is_default = " [green](default)[/green]" if name == config.default_agent else ""
            console.print(f"  [green]{name}[/green]{is_default}")
            if profile.description:
                console.print(f"    {profile.description}")
            console.print(f"    [dim]model: {profile.model.name}[/dim]")
            console.print(
                f"    [dim]context: {profile.model.context_window}, "
                f"temp: {profile.model.temperature}[/dim]"
            )
            console.print()

        console.print("[dim]towel agents create <name>  — create a new agent[/dim]")
        console.print("[dim]towel agents clone <src> <name>  — clone an existing agent[/dim]")
        console.print("[dim]towel agents delete <name>  — delete a user agent[/dim]")


@agents.command(name="create")
@click.argument("name")
@click.option(
    "--model", "-m", default="mlx-community/Llama-3.3-70B-Instruct-4bit", help="Model name"
)
@click.option("--identity", "-i", default=None, help="System prompt / identity")
@click.option("--description", "-d", default="", help="Short description")
@click.option("--context-window", "-c", default=8192, help="Context window size")
@click.option("--temperature", "-t", default=0.7, help="Temperature")
@click.option(
    "--turboquant/--no-turboquant", default=False, help="Enable TurboQuant KV cache compression"
)
@click.option("--tq-bits", default=3, type=int, help="TurboQuant quantization bits (3 or 4)")
def agents_create(
    name: str,
    model: str,
    identity: str | None,
    description: str,
    context_window: int,
    temperature: float,
    turboquant: bool,
    tq_bits: int,
) -> None:
    """Create a new agent profile."""
    from towel.cli.agent_mgr import create_agent

    config = TowelConfig.load()
    if config.get_agent(name):
        console.print(
            f"[yellow]Agent '{name}' already exists.[/yellow] "
            "Use a different name or delete it first."
        )
        return

    if not identity:
        identity = f"You are Towel ({name} mode). Don't Panic."

    profile = create_agent(
        name=name,
        model_name=model,
        identity=identity,
        description=description,
        context_window=context_window,
        temperature=temperature,
        turboquant=turboquant,
        turboquant_bits=tq_bits,
    )
    console.print(f"[green]Created agent:[/green] {name}")
    console.print(f"  Model: {profile.model.name}")
    console.print(f"  Context: {profile.model.context_window}, Temp: {profile.model.temperature}")
    if turboquant:
        console.print(f"  KV cache: TurboQuant {tq_bits}-bit")
    if description:
        console.print(f"  {description}")
    console.print(f"\n[dim]Use with: towel chat --agent {name}[/dim]")


@agents.command(name="clone")
@click.argument("source")
@click.argument("name")
def agents_clone(source: str, name: str) -> None:
    """Clone an existing agent under a new name."""
    from towel.cli.agent_mgr import clone_agent

    config = TowelConfig.load()
    if config.get_agent(name):
        console.print(f"[yellow]Agent '{name}' already exists.[/yellow]")
        return

    profile = clone_agent(source, name, config)
    if not profile:
        console.print(f"[red]Source agent not found:[/red] {source}")
        return

    console.print(f"[green]Cloned '{source}' -> '{name}'[/green]")
    console.print(f"  Model: {profile.model.name}")
    console.print(f"\n[dim]Use with: towel chat --agent {name}[/dim]")


@agents.command(name="delete")
@click.argument("name")
def agents_delete(name: str) -> None:
    """Delete a user-created agent profile."""
    from towel.cli.agent_mgr import delete_agent
    from towel.config import DEFAULT_AGENTS

    if name in DEFAULT_AGENTS:
        console.print(f"[yellow]Cannot delete built-in agent '{name}'.[/yellow]")
        return

    if delete_agent(name):
        console.print(f"[green]Deleted agent:[/green] {name}")
    else:
        console.print(f"[red]Agent not found:[/red] {name}")
        console.print("[dim]Only user-created agents can be deleted.[/dim]")

    console.print("[dim]Use with: towel chat --agent coder[/dim]")
    console.print('[dim]Or:       towel ask --agent researcher "your question"[/dim]')


@cli.group()
def memory() -> None:
    """Manage persistent agent memory."""
    pass


@memory.command(name="list")
@click.option(
    "--type", "-t", "mtype", default=None, help="Filter by type (user, project, fact, preference)"
)
def memory_list(mtype: str | None) -> None:
    """List all memories."""
    from towel.memory.store import MemoryStore

    store = MemoryStore()
    entries = store.recall_all(memory_type=mtype)

    if not entries:
        console.print("[dim]No memories stored.[/dim]")
        console.print("[dim]The agent will remember things as you chat.[/dim]")
        return

    console.print(f"[bold]Memories[/bold] ({len(entries)}):\n")
    for e in entries:
        console.print(f"  [green]{e.key}[/green] [dim][{e.memory_type}][/dim]")
        console.print(f"    {e.content}")
        console.print(f"    [dim]updated: {e.updated_at.strftime('%Y-%m-%d %H:%M')}[/dim]")


@memory.command(name="add")
@click.argument("key")
@click.argument("content")
@click.option(
    "--type", "-t", "mtype", default="fact", help="Memory type (user, project, fact, preference)"
)
def memory_add(key: str, content: str, mtype: str) -> None:
    """Add or update a memory."""
    from towel.memory.store import MemoryStore

    store = MemoryStore()
    entry = store.remember(key, content, memory_type=mtype)
    console.print(f"[green]Remembered:[/green] [{entry.memory_type}] {entry.key}: {entry.content}")


@memory.command(name="remove")
@click.argument("key")
def memory_remove(key: str) -> None:
    """Remove a memory."""
    from towel.memory.store import MemoryStore

    store = MemoryStore()
    if store.forget(key):
        console.print(f"[green]Forgot:[/green] {key}")
    else:
        console.print(f"[red]Not found:[/red] {key}")


@memory.command(name="search")
@click.argument("query")
def memory_search(query: str) -> None:
    """Search memories."""
    from towel.memory.store import MemoryStore

    store = MemoryStore()
    results = store.search(query)

    if not results:
        console.print(f"[dim]No memories matching:[/dim] {query}")
        return

    for e in results:
        console.print(f"  [green]{e.key}[/green] [dim][{e.memory_type}][/dim]")
        console.print(f"    {e.content}")


@memory.command(name="clear")
@click.confirmation_option(prompt="Delete all memories?")
def memory_clear() -> None:
    """Clear all memories."""
    from towel.memory.store import MemoryStore

    store = MemoryStore()
    count = store.count
    for entry in store.recall_all():
        store.forget(entry.key)
    console.print(f"[green]Cleared {count} memories.[/green]")


@cli.group()
def models() -> None:
    """Manage MLX models."""
    pass


@models.command(name="list")
def models_list() -> None:
    """List locally cached MLX models."""
    from rich.table import Table

    from towel.cli.models import get_model_usage, list_cached_models

    config = TowelConfig.load()
    cached = list_cached_models()
    usage = get_model_usage(config)

    if not cached:
        console.print("[dim]No models cached locally.[/dim]")
        console.print("Download one with: [green]towel models pull <name>[/green]")
        console.print("See recommendations: [green]towel models recommended[/green]")
        return

    table = Table(title="Cached Models", border_style="dim")
    table.add_column("Model", style="green")
    table.add_column("Size", justify="right")
    table.add_column("Used by", style="dim")

    for m in cached:
        agents = usage.get(m.name, [])
        agents_str = ", ".join(agents) if agents else ""
        table.add_row(m.name, m.size_display, agents_str)

    console.print(table)
    console.print(f"\n[dim]{len(cached)} model(s) cached[/dim]")


@models.command(name="recommended")
def models_recommended() -> None:
    """Show recommended MLX models."""
    from rich.table import Table

    from towel.cli.models import RECOMMENDED_MODELS, is_model_cached

    table = Table(title="Recommended Models", border_style="dim")
    table.add_column("Model", style="green")
    table.add_column("Params", justify="right")
    table.add_column("Quant", justify="center")
    table.add_column("RAM", justify="right")
    table.add_column("Use case")
    table.add_column("Cached", justify="center")

    for m in RECOMMENDED_MODELS:
        cached = "[green]yes[/green]" if is_model_cached(m["name"]) else "[dim]no[/dim]"
        table.add_row(m["name"], m["params"], m["quant"], m["ram"], m["use"], cached)

    console.print(table)
    console.print("\n[dim]Download with: towel models pull <model-name>[/dim]")


@models.command(name="pull")
@click.argument("model_name")
def models_pull(model_name: str) -> None:
    """Download an MLX model from HuggingFace."""
    from towel.cli.models import is_model_cached

    if is_model_cached(model_name):
        console.print(f"[green]Already cached:[/green] {model_name}")
        return

    console.print(f"[green]Downloading:[/green] {model_name}")
    console.print("[dim]This may take a while depending on model size...[/dim]")

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_name)
        console.print(f"[green]Downloaded:[/green] {model_name}")
        console.print("[dim]Use it with: towel chat (after setting model in config)[/dim]")
    except ImportError:
        console.print("[red]huggingface_hub not installed.[/red]")
        console.print("Install with: pip install huggingface-hub")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {e}")
        sys.exit(1)


@models.command(name="active")
def models_active() -> None:
    """Show which model each agent profile uses."""
    config = TowelConfig.load()
    from towel.cli.models import is_model_cached

    console.print(f"  [green]default[/green] -> {config.model.name}", end="")
    console.print(
        "  [green](cached)[/green]"
        if is_model_cached(config.model.name)
        else "  [red](not cached)[/red]"
    )

    for name, profile in config.list_agents().items():
        cached = (
            "[green](cached)[/green]"
            if is_model_cached(profile.model.name)
            else "[red](not cached)[/red]"
        )
        console.print(f"  [green]{name}[/green] -> {profile.model.name}  {cached}")


@cli.command()
def templates() -> None:
    """List available prompt templates."""
    from towel.templates.engine import TemplateEngine

    engine = TemplateEngine()
    tpls = engine.list_templates()

    if not tpls:
        console.print("[dim]No templates available.[/dim]")
        return

    console.print("[bold]Prompt templates:[/bold]\n")
    for name, desc in tpls.items():
        console.print(f"  [green]{name}[/green]")
        if desc:
            console.print(f"    {desc}")

    console.print("\n[dim]Use with: towel ask -T review @file.py[/dim]")
    console.print("[dim]Or in chat: /t review <your input>[/dim]")
    console.print("[dim]Create custom: ~/.towel/templates/mytemplate.txt[/dim]")


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def config(as_json: bool) -> None:
    """Show current configuration."""
    cfg = TowelConfig.load()

    if as_json:
        import json as json_mod

        data = {
            "towel_home": str(TOWEL_HOME),
            "model": {
                "name": cfg.model.name,
                "context_window": cfg.model.context_window,
                "max_tokens": cfg.model.max_tokens,
                "temperature": cfg.model.temperature,
                "top_p": cfg.model.top_p,
            },
            "gateway": {
                "host": cfg.gateway.host,
                "port": cfg.gateway.port,
            },
            "skills_dirs": [str(d) for d in cfg.skills_dirs],
            "agents": list(cfg.list_agents().keys()),
        }
        print(json_mod.dumps(data, indent=2))
        return

    console.print("[bold]Towel configuration:[/bold]\n")
    console.print(f"  [green]Home:[/green]           {TOWEL_HOME}")
    config_path = TOWEL_HOME / "config.toml"
    console.print(
        f"  [green]Config file:[/green]    {config_path} "
        f"{'[dim](exists)[/dim]' if config_path.exists() else '[yellow](not found)[/yellow]'}"
    )

    console.print("\n  [bold]Model:[/bold]")
    console.print(f"    [green]Name:[/green]           {cfg.model.name}")
    console.print(f"    [green]Context window:[/green] {cfg.model.context_window:,} tokens")
    console.print(f"    [green]Max output:[/green]     {cfg.model.max_tokens:,} tokens")
    console.print(f"    [green]Temperature:[/green]    {cfg.model.temperature}")
    console.print(f"    [green]Top-p:[/green]          {cfg.model.top_p}")

    console.print("\n  [bold]Gateway:[/bold]")
    console.print(f"    [green]Host:[/green]           {cfg.gateway.host}")
    console.print(f"    [green]WebSocket:[/green]      ws://{cfg.gateway.host}:{cfg.gateway.port}")
    console.print(
        f"    [green]HTTP API:[/green]       http://{cfg.gateway.host}:{cfg.gateway.port + 1}"
    )

    agents = cfg.list_agents()
    if agents:
        console.print(f"\n  [bold]Agents ({len(agents)}):[/bold]")
        for name, profile in agents.items():
            desc = f" — {profile.description}" if profile.description else ""
            console.print(f"    [green]{name}[/green]{desc}")
            console.print(f"      [dim]{profile.model.name}[/dim]")

    console.print("\n  [bold]Skills dirs:[/bold]")
    for d in cfg.skills_dirs:
        from pathlib import Path

        p = Path(d).expanduser()
        exists = "[dim](exists)[/dim]" if p.exists() else "[dim](not created)[/dim]"
        console.print(f"    {p} {exists}")

    console.print(f"\n[dim]Edit: {config_path}[/dim]")


@cli.command(name="skill-init")
@click.argument("name")
@click.option(
    "--dir", "output_dir", default=None, help="Output directory (default: ~/.towel/skills/)"
)
def skill_init(name: str, output_dir: str | None) -> None:
    """Generate a skeleton skill file to get started with custom skills."""
    from pathlib import Path

    if output_dir:
        target_dir = Path(output_dir)
    else:
        target_dir = TOWEL_HOME / "skills"

    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{name}_skill.py"
    target = target_dir / filename

    if target.exists():
        console.print(f"[yellow]File already exists:[/yellow] {target}")
        return

    class_name = "".join(w.capitalize() for w in name.split("_")) + "Skill"

    skeleton = f'''"""Custom skill: {name}

Drop this file into ~/.towel/skills/ and it will be auto-loaded.
"""

from __future__ import annotations

from typing import Any

from towel.skills.base import Skill, ToolDefinition


class {class_name}(Skill):
    @property
    def name(self) -> str:
        return "{name}"

    @property
    def description(self) -> str:
        return "Description of what this skill does"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="{name}_example",
                description="An example tool — replace with your own",
                parameters={{
                    "type": "object",
                    "properties": {{
                        "query": {{
                            "type": "string",
                            "description": "Input to process",
                        }},
                    }},
                    "required": ["query"],
                }},
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "{name}_example":
                query = arguments.get("query", "")
                return f"Got: {{query}}"
            case _:
                return f"Unknown tool: {{tool_name}}"
'''

    target.write_text(skeleton, encoding="utf-8")
    console.print(f"[green]Created skill skeleton:[/green] {target}")
    console.print(f"  Class: [bold]{class_name}[/bold]")
    console.print(f"  Tool:  {name}_example")
    console.print("\n[dim]Edit the file and restart Towel to load it.[/dim]")


@cli.command()
@click.option(
    "--days", "-d", default=30, help="Delete conversations older than N days (default: 30)"
)
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting")
@click.option("--cache", is_flag=True, help="Also report model cache sizes")
def gc(days: int, dry_run: bool, cache: bool) -> None:
    """Clean up old conversations and show disk usage.

    \b
    Examples:
        towel gc                  Delete conversations older than 30 days
        towel gc -d 7             Delete conversations older than 7 days
        towel gc --dry-run        Preview what would be deleted
        towel gc --cache          Also show model cache sizes
    """
    import json as json_mod
    from datetime import datetime, timedelta
    from pathlib import Path

    from towel.agent.conversation import Conversation
    from towel.persistence.store import ConversationStore

    store = ConversationStore()
    cutoff = datetime.now(UTC) - timedelta(days=days)
    conv_dir = store.store_dir

    if not conv_dir.exists():
        console.print("[dim]No conversations directory.[/dim]")
        return

    json_files = sorted(conv_dir.glob("*.json"))
    total_size = 0
    old_files: list[tuple[Path, str, int]] = []  # (path, summary, size)
    kept_count = 0
    kept_size = 0

    for path in json_files:
        size = path.stat().st_size
        total_size += size
        try:
            data = json_mod.loads(path.read_text(encoding="utf-8"))
            conv = Conversation.from_dict(data)
            if conv.created_at < cutoff:
                summary = conv.display_title[:50]
                old_files.append((path, summary, size))
            else:
                kept_count += 1
                kept_size += size
        except (json_mod.JSONDecodeError, KeyError, ValueError):
            old_files.append((path, "(corrupt)", size))

    console.print(f"[bold]Conversation cleanup[/bold] (older than {days} days)\n")
    console.print(f"  Total conversations: {len(json_files)}")
    console.print(f"  Total size: {total_size / 1024:.0f} KB")

    if old_files:
        old_total = sum(s for _, _, s in old_files)
        console.print(
            f"\n  [yellow]To remove: {len(old_files)} "
            f"conversation(s) ({old_total / 1024:.0f} KB)[/yellow]"
        )
        for path, summary, size in old_files[:10]:
            console.print(
                f"    [dim]{path.stem}[/dim]  {summary}  [dim]({size / 1024:.1f} KB)[/dim]"
            )
        if len(old_files) > 10:
            console.print(f"    [dim]... and {len(old_files) - 10} more[/dim]")

        if dry_run:
            console.print("\n  [dim]Dry run — nothing deleted.[/dim]")
        else:
            for path, _, _ in old_files:
                path.unlink()
            console.print(f"\n  [green]Deleted {len(old_files)} old conversation(s).[/green]")
            console.print(f"  Remaining: {kept_count} conversations ({kept_size / 1024:.0f} KB)")
    else:
        console.print("\n  [green]Nothing to clean up.[/green] All conversations are recent.")

    # Cache report
    if cache:
        console.print("\n[bold]Model caches:[/bold]")
        cache_dirs = {
            "HuggingFace": Path.home() / ".cache" / "huggingface",
            "MLX": Path.home() / ".cache" / "mlx",
        }
        for label, cache_path in cache_dirs.items():
            if cache_path.exists():
                try:
                    size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
                    console.print(
                        f"  {label}: {size / (1024**3):.1f} GB  [dim]({cache_path})[/dim]"
                    )
                except OSError:
                    console.print(f"  {label}: [dim](error reading)[/dim]")
            else:
                console.print(f"  {label}: [dim](not found)[/dim]")


@cli.command()
@click.option("--limit", "-n", default=30, help="Number of entries to show")
@click.option("--today", is_flag=True, help="Only show today's activity")
@click.option("--week", is_flag=True, help="Only show this week's activity")
def log(limit: int, today: bool, week: bool) -> None:
    """Show activity timeline across all conversations.

    \b
    Like git log, but for your AI interactions. Shows a compact
    timeline of what you've been working on.

    \b
    Examples:
        towel log
        towel log --today
        towel log --week
        towel log -n 50
    """
    import json as json_mod
    from datetime import datetime, timedelta

    from towel.agent.conversation import Conversation, Role
    from towel.persistence.store import ConversationStore

    store = ConversationStore()
    if not store.store_dir.exists():
        console.print("[dim]No conversations yet.[/dim]")
        return

    # Determine time filter
    now = datetime.now(UTC)
    if today:
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif week:
        cutoff = now - timedelta(days=7)
    else:
        cutoff = None

    # Collect all conversations with their messages
    entries: list[
        tuple[datetime, str, str, str, list[str]]
    ] = []  # (time, conv_id, title, channel, tags)

    json_files = sorted(
        store.store_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )

    for path in json_files:
        try:
            data = json_mod.loads(path.read_text(encoding="utf-8"))
            conv = Conversation.from_dict(data)
        except (json_mod.JSONDecodeError, KeyError, ValueError):
            continue

        if cutoff and conv.created_at < cutoff:
            # Check if any messages are after cutoff
            has_recent = any(m.timestamp >= cutoff for m in conv.messages)
            if not has_recent:
                continue

        # Get the most recent message timestamp
        last_active = conv.messages[-1].timestamp if conv.messages else conv.created_at

        if cutoff and last_active < cutoff:
            continue

        # Count messages by role
        user_count = sum(1 for m in conv.messages if m.role == Role.USER)
        asst_count = sum(1 for m in conv.messages if m.role == Role.ASSISTANT)
        tool_count = sum(1 for m in conv.messages if m.role == Role.TOOL)

        # Get total tokens
        total_tokens = sum(
            m.metadata.get("tokens", 0) for m in conv.messages if m.role == Role.ASSISTANT
        )

        tags = data.get("tags", [])
        title = conv.display_title

        entries.append(
            (
                last_active,
                conv.id,
                title,
                conv.channel,
                tags,
                user_count,
                asst_count,
                tool_count,
                total_tokens,
            )
        )

    # Sort by most recently active
    entries.sort(key=lambda e: e[0], reverse=True)
    entries = entries[:limit]

    if not entries:
        console.print("[dim]No activity found.[/dim]")
        return

    # Group by date
    current_date = None
    for ts, conv_id, title, channel, tags, u, a, t, tokens in entries:
        date_str = ts.strftime("%Y-%m-%d")
        if date_str != current_date:
            current_date = date_str
            day_name = ts.strftime("%A")
            if date_str == now.strftime("%Y-%m-%d"):
                day_label = "Today"
            elif date_str == (now - timedelta(days=1)).strftime("%Y-%m-%d"):
                day_label = "Yesterday"
            else:
                day_label = day_name
            console.print(f"\n[bold]{day_label}, {date_str}[/bold]")

        time_str = ts.strftime("%H:%M")
        tag_str = " ".join(f"[yellow]#{t}[/yellow]" for t in tags) if tags else ""
        stats = f"{u}q {a}a"
        if t:
            stats += f" {t}t"
        if tokens:
            stats += f" {tokens:,}tok"

        console.print(
            f"  [dim]{time_str}[/dim]  [green]{conv_id[:12]}[/green]  "
            f"{title[:50]}  [dim]({stats})[/dim] {tag_str}"
        )

    console.print(
        f"\n[dim]{len(entries)} conversation(s) shown. View details: towel show <id>[/dim]"
    )


@cli.command()
@click.option(
    "--prompt",
    "-p",
    default="Explain what a towel is in exactly three sentences.",
    help="Prompt to use for benchmarking",
)
@click.option("--tokens", "-t", default=128, help="Max tokens to generate")
@click.option("--rounds", "-r", default=3, help="Number of generation rounds")
def bench(prompt: str, tokens: int, rounds: int) -> None:
    """Benchmark model loading and inference speed.

    \b
    Measures load time, time-to-first-token, and tokens/sec.
    Useful for comparing models or testing hardware.

    \b
    Examples:
        towel bench
        towel bench -t 256 -r 5
        towel bench -p "Write a haiku about AI"
    """
    import time as time_mod

    config = TowelConfig.load()
    model_name = config.model.name

    console.print(
        Panel(
            f"[bold]Towel Bench[/bold]\n\n"
            f"  Model:      [green]{model_name}[/green]\n"
            f"  Max tokens: {tokens}\n"
            f"  Rounds:     {rounds}\n"
            f"  Prompt:     [dim]{prompt[:60]}{'...' if len(prompt) > 60 else ''}[/dim]",
            border_style="cyan",
            title="Don't Panic.",
        )
    )

    # Load model
    console.print("\n[dim]Loading model...[/dim]")
    load_start = time_mod.perf_counter()
    try:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = load(
            model_name,
            tokenizer_config={"fix_mistral_regex": True},
        )
    except Exception as e:
        console.print(f"[red]Failed to load model:[/red] {e}")
        sys.exit(1)
    load_elapsed = time_mod.perf_counter() - load_start
    console.print(f"[green]Loaded in {load_elapsed:.1f}s[/green]\n")

    # Build chat prompt
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = f"User: {prompt}\nAssistant:"

    # Run benchmark rounds
    tps_values: list[float] = []
    _ttft_values: list[float] = []
    total_tokens_gen = 0

    for i in range(rounds):
        console.print(f"  Round {i + 1}/{rounds}... ", end="")
        gen_start = time_mod.perf_counter()

        try:
            output = generate(
                model,
                tokenizer,
                prompt=formatted,
                max_tokens=tokens,
                sampler=make_sampler(temp=0.7, top_p=0.95),
            )
        except Exception as e:
            console.print(f"[red]error: {e}[/red]")
            continue

        gen_elapsed = time_mod.perf_counter() - gen_start
        token_count = len(tokenizer.encode(output))
        total_tokens_gen += token_count

        tps = token_count / gen_elapsed if gen_elapsed > 0 else 0
        tps_values.append(tps)

        console.print(
            f"[green]{tps:.1f} tok/s[/green]  ({token_count} tokens in {gen_elapsed:.2f}s)"
        )

    if not tps_values:
        console.print("[red]No successful rounds.[/red]")
        sys.exit(1)

    # Summary
    avg_tps = sum(tps_values) / len(tps_values)
    peak_tps = max(tps_values)
    min_tps = min(tps_values)

    console.print(
        Panel(
            f"  [green]Avg speed:[/green]   {avg_tps:.1f} tok/s\n"
            f"  [green]Peak:[/green]        {peak_tps:.1f} tok/s\n"
            f"  [green]Min:[/green]         {min_tps:.1f} tok/s\n"
            f"  [green]Load time:[/green]   {load_elapsed:.1f}s\n"
            f"  [green]Total gen:[/green]   {total_tokens_gen:,} "
            f"tokens across {len(tps_values)} rounds\n"
            f"  [green]Model:[/green]       {model_name}",
            title="Results",
            border_style="green",
        )
    )


@cli.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--prompt", "-p", default=None, help="Prompt to run on each file change (default: review)"
)
@click.option("--interval", "-i", default=2.0, help="Poll interval in seconds (default: 2)")
@click.option("--once", is_flag=True, help="Run once and exit (no watching)")
def watch(files: tuple[str, ...], prompt: str | None, interval: float, once: bool) -> None:
    """Watch files and re-run a prompt when they change.

    \b
    Great for live development — get instant AI feedback on save.

    \b
    Examples:
        towel watch src/main.py
        towel watch src/*.py -p "find bugs in this code"
        towel watch test.py -p "suggest better test names" --once
        towel watch app.py config.yaml -i 5
    """
    import time as time_mod
    from pathlib import Path

    config = TowelConfig.load()
    paths = [Path(f) for f in files]

    default_prompt = prompt or "Review this code. Point out any bugs, issues, or improvements."

    # Get initial mtimes
    def get_mtimes() -> dict[str, float]:
        result = {}
        for p in paths:
            try:
                result[str(p)] = p.stat().st_mtime
            except OSError:
                pass
        return result

    def run_prompt(changed_files: list[str]) -> None:
        """Run the prompt on changed files using towel ask logic."""
        from towel.agent.conversation import Conversation, Role
        from towel.agent.runtime import AgentRuntime
        from towel.memory.store import MemoryStore

        memory = MemoryStore()
        skills = _build_skill_registry(config, memory_store=memory)
        agent_rt = AgentRuntime(config, skills=skills, memory=memory)
        conv = Conversation(channel="watch")

        # Build content from changed files
        file_blocks = []
        for fpath in changed_files:
            try:
                content = Path(fpath).read_text(encoding="utf-8", errors="replace")
                name = Path(fpath).name
                ext = Path(fpath).suffix.lstrip(".")
                file_blocks.append(f"**{name}**:\n```{ext}\n{content}\n```")
            except OSError as e:
                file_blocks.append(f"**{fpath}**: (error: {e})")

        full = f"{default_prompt}\n\n" + "\n\n".join(file_blocks)
        conv.add(Role.USER, full)

        async def _run() -> None:
            await agent_rt.load_model()
            response = await agent_rt.step(conv)
            console.print(f"\n[bold green]towel>[/bold green] {response.content}")
            meta = response.metadata
            if meta.get("tps"):
                console.print(
                    f"[dim]({meta['tps']:.1f} tok/s, {meta.get('tokens', 0)} tokens)[/dim]"
                )

        asyncio.run(_run())

    console.print(
        Panel(
            f"[bold]Watching {len(paths)} file(s)[/bold]\n"
            + "\n".join(f"  [green]{p}[/green]" for p in paths)
            + "\n\n[dim]Prompt: "
            + f"{default_prompt[:60]}"
            + f"{'...' if len(default_prompt) > 60 else ''}"
            + "[/dim]"
            + f"\n[dim]Interval: {interval}s · "
            + f"{'once' if once else 'continuous (Ctrl+C to stop)'}"
            + "[/dim]",
            border_style="cyan",
            title="towel watch",
        )
    )

    prev_mtimes = get_mtimes()

    if once:
        # Run immediately on all files
        console.print(f"\n[dim]Running on {len(paths)} file(s)...[/dim]")
        run_prompt([str(p) for p in paths])
        return

    # Run on initial state
    console.print("\n[dim]Waiting for changes...[/dim]")

    try:
        while True:
            time_mod.sleep(interval)
            curr_mtimes = get_mtimes()

            changed = []
            for fpath, mtime in curr_mtimes.items():
                if fpath not in prev_mtimes or mtime != prev_mtimes[fpath]:
                    changed.append(fpath)

            if changed:
                ts = time_mod.strftime("%H:%M:%S")
                console.print(
                    f"\n[yellow]{ts}[/yellow] Changed: {', '.join(Path(f).name for f in changed)}"
                )
                run_prompt(changed)
                console.print("\n[dim]Waiting for changes...[/dim]")
                prev_mtimes = curr_mtimes

    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching.[/dim]")


@cli.command()
@click.option(
    "--staged", "-s", is_flag=True, help="Review only staged changes (default: all uncommitted)"
)
@click.option("--commit", "-c", default=None, help="Review a specific commit (hash or HEAD~N)")
@click.option("--focus", "-f", default=None, help="Focus area: bugs, security, performance, style")
@click.option("--raw", "-r", is_flag=True, help="Output plain text (no Rich formatting)")
def review(staged: bool, commit: str | None, focus: str | None, raw: bool) -> None:
    """AI code review of git changes.

    \b
    Run before committing to catch bugs, security issues, or style problems.

    \b
    Examples:
        towel review              Review all uncommitted changes
        towel review --staged     Review only staged changes
        towel review -c HEAD~1    Review the last commit
        towel review -f security  Focus on security issues
    """
    import subprocess

    # Get the diff
    if commit:
        cmd = ["git", "diff", f"{commit}~1", commit]
        label = f"commit {commit}"
    elif staged:
        cmd = ["git", "diff", "--cached"]
        label = "staged changes"
    else:
        cmd = ["git", "diff", "HEAD"]
        label = "uncommitted changes"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        diff = result.stdout
    except FileNotFoundError:
        console.print("[red]git not found.[/red]")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        console.print("[red]git diff timed out.[/red]")
        sys.exit(1)

    if not diff.strip():
        console.print(f"[dim]No {label} to review.[/dim]")
        return

    # Count changed files and lines
    diff_lines = diff.splitlines()
    files_changed = sum(
        1 for ln in diff_lines if ln.startswith("diff --git")
    )
    additions = sum(
        1 for ln in diff_lines
        if ln.startswith("+") and not ln.startswith("+++")
    )
    deletions = sum(
        1 for ln in diff_lines
        if ln.startswith("-") and not ln.startswith("---")
    )

    if not raw:
        console.print(
            Panel(
                f"[bold]Reviewing {label}[/bold]\n"
                f"  Files: {files_changed}  +{additions} -{deletions} lines",
                border_style="cyan",
                title="towel review",
            )
        )

    # Build the review prompt
    focus_instruction = ""
    if focus:
        focus_map = {
            "bugs": (
                "Focus primarily on bugs, logic errors, "
                "off-by-one errors, null/undefined access, "
                "and incorrect behavior."
            ),
            "security": (
                "Focus primarily on security vulnerabilities: "
                "injection, XSS, CSRF, hardcoded secrets, "
                "insecure defaults, auth issues."
            ),
            "performance": (
                "Focus primarily on performance: N+1 queries, "
                "unnecessary allocations, missing caches, "
                "O(n²) algorithms."
            ),
            "style": (
                "Focus primarily on code style: naming, "
                "structure, readability, DRY violations, "
                "dead code."
            ),
        }
        focus_instruction = focus_map.get(focus, f"Focus on: {focus}")

    prompt = (
        f"Review the following git diff ({label}). "
        "For each issue found, cite the file and line. "
        "Rate overall quality 1-10. Be specific and constructive. "
        f"{focus_instruction}\n\n"
        f"```diff\n{diff[:50000]}\n```"
    )
    if len(diff) > 50000:
        prompt += "\n\n(diff truncated at 50KB)"

    # Run through the agent
    config = TowelConfig.load()
    from towel.agent.conversation import Conversation, Role
    from towel.agent.runtime import AgentRuntime
    from towel.memory.store import MemoryStore

    memory = MemoryStore()
    skills = _build_skill_registry(config, memory_store=memory)
    agent_rt = AgentRuntime(config, skills=skills, memory=memory)
    conv = Conversation(channel="review")
    conv.add(Role.USER, prompt)

    async def _run() -> None:
        if not raw:
            console.print("[dim]Loading model...[/dim]")
        await agent_rt.load_model()

        if raw:
            response = await agent_rt.step(conv)
            print(response.content)
        else:
            from towel.agent.events import EventType

            console.print("[bold green]towel>[/bold green] ", end="")
            async for event in agent_rt.step_streaming(conv):
                match event.type:
                    case EventType.TOKEN:
                        print(event.data["content"], end="", flush=True)
                    case EventType.RESPONSE_COMPLETE:
                        print()
                        meta = event.data.get("metadata", {})
                        if meta.get("tps"):
                            console.print(
                                f"[dim]({meta['tps']:.1f} tok/s, "
                                f"{meta.get('tokens', 0)} tokens)[/dim]"
                            )

    asyncio.run(_run())


@cli.command(name="commit")
@click.option("--all", "-a", "stage_all", is_flag=True, help="Stage all changes before committing")
@click.option("--edit", "-e", is_flag=True, help="Open the message in $EDITOR before committing")
@click.option("--dry-run", is_flag=True, help="Generate message but don't commit")
def commit_cmd(stage_all: bool, edit: bool, dry_run: bool) -> None:
    """Generate an AI commit message and commit.

    \b
    Reads staged git changes, generates a conventional commit message,
    and commits — all in one step.

    \b
    Examples:
        git add -p && towel commit     Stage manually, then commit
        towel commit -a                Stage all and commit
        towel commit --dry-run         Just show the generated message
        towel commit -e                Edit message before committing
    """
    import subprocess
    import tempfile

    if stage_all:
        subprocess.run(["git", "add", "-A"], capture_output=True)

    result = subprocess.run(["git", "diff", "--cached", "--stat"], capture_output=True, text=True)
    stat = result.stdout.strip()
    if not stat:
        console.print("[dim]Nothing staged to commit.[/dim]")
        console.print("[dim]Stage changes first: git add <files>  or  towel commit -a[/dim]")
        return

    diff_result = subprocess.run(["git", "diff", "--cached"], capture_output=True, text=True)
    diff = diff_result.stdout

    console.print(f"[dim]Staged:[/dim] {stat.splitlines()[-1].strip()}")

    prompt = (
        "Write a git commit message for the following diff. "
        "Use conventional commits format (feat/fix/refactor/docs/test/chore). "
        "First line: type(scope): description (under 72 chars). "
        "Then blank line, then bullet points for details if non-trivial. "
        "Be specific but concise. Output ONLY the commit message.\n\n"
        f"```diff\n{diff[:30000]}\n```"
    )

    config = TowelConfig.load()
    from towel.agent.conversation import Conversation, Role
    from towel.agent.runtime import AgentRuntime
    from towel.memory.store import MemoryStore

    memory = MemoryStore()
    skills_reg = _build_skill_registry(config, memory_store=memory)
    agent_rt = AgentRuntime(config, skills=skills_reg, memory=memory)
    conv = Conversation(channel="commit")
    conv.add(Role.USER, prompt)

    async def _gen() -> str:
        await agent_rt.load_model()
        resp = await agent_rt.step(conv)
        return resp.content.strip()

    console.print("[dim]Generating commit message...[/dim]")
    commit_msg = asyncio.run(_gen())

    # Strip markdown fences if model wrapped it
    if commit_msg.startswith("```"):
        lines = commit_msg.splitlines()
        commit_msg = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    console.print("\n[bold green]Commit message:[/bold green]")
    console.print(f"[cyan]{commit_msg}[/cyan]\n")

    if dry_run:
        console.print("[dim]Dry run — not committed.[/dim]")
        return

    if edit:
        editor = os.environ.get("EDITOR", "vim")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(commit_msg)
            tmpfile = f.name
        subprocess.run([editor, tmpfile])
        with open(tmpfile) as f:
            commit_msg = f.read().strip()
        os.unlink(tmpfile)
        if not commit_msg:
            console.print("[dim]Empty message — commit aborted.[/dim]")
            return

    cresult = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
    if cresult.returncode == 0:
        console.print("[green]Committed![/green]")
        log = subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True)
        if log.stdout.strip():
            console.print(f"  [dim]{log.stdout.strip()}[/dim]")
    else:
        console.print(f"[red]Commit failed:[/red] {cresult.stderr.strip()}")


@cli.command()
@click.option("--port", "-p", default=18750, help="Port to listen on")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--token", "-t", default=None, help="Require Bearer token for auth")
def webhook(port: int, host: str, token: str | None) -> None:
    """Start a webhook channel — receive messages via HTTP POST.

    \b
    The simplest way to integrate Towel with external services.
    Accepts POST /message with {"text": "..."} and returns AI responses.

    \b
    Examples:
        towel webhook                    Start on port 18750
        towel webhook -p 9000            Custom port
        towel webhook -t mysecret        Require auth token

    \b
    Test with curl:
        curl -X POST http://127.0.0.1:18750/message \\
          -H "Content-Type: application/json" \\
          -d '{"text": "hello"}'
    """
    from towel.channels.webhook import WebhookChannel

    console.print(
        Panel(
            f"[bold]Webhook Channel[/bold]\n\n"
            f"  Endpoint: [green]http://{host}:{port}/message[/green]\n"
            f"  Health:   http://{host}:{port}/health\n"
            f"  Auth:     {'Bearer token required' if token else 'none (open)'}\n\n"
            f'[dim]POST {{"text": "your message"}} to get AI responses.[/dim]',
            border_style="cyan",
            title="Don't Panic.",
        )
    )

    channel = WebhookChannel(
        gateway_url=f"ws://{TowelConfig.load().gateway.host}:{TowelConfig.load().gateway.port}",
        port=port,
        host=host,
        token=token,
    )
    asyncio.run(channel.listen())


@cli.command()
@click.argument("target", required=False)
@click.option("--bullets", "-b", is_flag=True, help="Output as bullet points")
@click.option(
    "--length",
    "-l",
    default="medium",
    type=click.Choice(["short", "medium", "long"]),
    help="Summary length (default: medium)",
)
@click.option("--raw", "-r", is_flag=True, help="Plain text output (no Rich)")
def summarize(target: str | None, bullets: bool, length: str, raw: bool) -> None:
    """Summarize a file, URL, or piped input.

    \b
    Examples:
        towel summarize README.md
        towel summarize https://example.com/article
        cat long_log.txt | towel summarize
        towel summarize notes.md -b -l short
    """
    from pathlib import Path

    # Gather content from target or stdin
    content = ""
    source = ""

    if target:
        if target.startswith(("http://", "https://")):
            # Fetch URL
            try:
                import httpx

                resp = httpx.get(target, timeout=10, follow_redirects=True)
                content = resp.text[:50000]
                source = target
            except Exception as e:
                console.print(f"[red]Failed to fetch URL:[/red] {e}")
                return
        else:
            # Read file
            p = Path(target).expanduser()
            if not p.is_file():
                console.print(f"[red]File not found:[/red] {target}")
                return
            content = p.read_text(encoding="utf-8", errors="replace")[:50000]
            source = p.name
    else:
        # Read stdin
        if sys.stdin.isatty():
            console.print("[red]No input.[/red] Provide a file, URL, or pipe content.")
            console.print("  towel summarize file.txt")
            console.print("  cat log.txt | towel summarize")
            return
        content = sys.stdin.read()[:50000]
        source = "stdin"

    if not content.strip():
        console.print("[dim]Empty input — nothing to summarize.[/dim]")
        return

    # Build prompt
    length_map = {
        "short": "2-3 sentences",
        "medium": "a concise paragraph (4-6 sentences)",
        "long": "a detailed summary (2-3 paragraphs)",
    }
    fmt = "bullet points" if bullets else length_map[length]
    prompt = (
        f"Summarize the following content as {fmt}. "
        "Be accurate and concise. Capture the key points.\n\n"
        f"{content}"
    )

    config = TowelConfig.load()
    from towel.agent.conversation import Conversation, Role
    from towel.agent.runtime import AgentRuntime
    from towel.memory.store import MemoryStore

    memory = MemoryStore()
    skills_reg = _build_skill_registry(config, memory_store=memory)
    agent_rt = AgentRuntime(config, skills=skills_reg, memory=memory)
    conv = Conversation(channel="summarize")
    conv.add(Role.USER, prompt)

    async def _run() -> None:
        if not raw:
            console.print(f"[dim]Summarizing {source} ({len(content):,} chars)...[/dim]\n")
        await agent_rt.load_model()

        if raw:
            resp = await agent_rt.step(conv)
            print(resp.content)
        else:
            from towel.agent.events import EventType

            async for event in agent_rt.step_streaming(conv):
                if event.type == EventType.TOKEN:
                    print(event.data["content"], end="", flush=True)
                elif event.type == EventType.RESPONSE_COMPLETE:
                    print()

    asyncio.run(_run())


def _oneshot(config, prompt: str, raw: bool = False) -> None:
    """Run a one-shot prompt through the agent."""
    from towel.agent.conversation import Conversation, Role
    from towel.agent.events import EventType
    from towel.agent.runtime import AgentRuntime
    from towel.memory.store import MemoryStore

    memory = MemoryStore()
    from towel.cli.main import _build_skill_registry

    skills_reg = _build_skill_registry(config, memory_store=memory)
    agent_rt = AgentRuntime(config, skills=skills_reg, memory=memory)
    conv = Conversation(channel="oneshot")
    conv.add(Role.USER, prompt)

    async def _run():
        await agent_rt.load_model()
        if raw:
            resp = await agent_rt.step(conv)
            print(resp.content)
        else:
            async for event in agent_rt.step_streaming(conv):
                if event.type == EventType.TOKEN:
                    print(event.data["content"], end="", flush=True)
                elif event.type == EventType.RESPONSE_COMPLETE:
                    print()

    import asyncio as _aio

    _aio.run(_run())


@cli.command()
@click.argument("target")
@click.option("--detail", "-d", is_flag=True, help="Line-by-line explanation")
@click.option("--raw", "-r", is_flag=True, help="Plain text output")
def explain(target: str, detail: bool, raw: bool) -> None:
    """Explain what a file does.

    \b
    Examples:
        towel explain src/main.py
        towel explain auth.py --detail
    """
    from pathlib import Path

    from towel.config import TowelConfig

    p = Path(target).expanduser()
    if not p.is_file():
        console.print(f"[red]Not found:[/red] {target}")
        return

    content = p.read_text(encoding="utf-8", errors="replace")[:50000]
    ext = p.suffix.lstrip(".")
    mode = "line by line, explaining each section" if detail else "in plain language"

    if not raw:
        console.print(f"[dim]Explaining {p.name} ({len(content):,} chars)...[/dim]")

    _oneshot(
        TowelConfig.load(),
        f"Explain this {ext} code {mode}.\n\n```{ext}\n{content}\n```",
        raw,
    )


@cli.command()
@click.argument("action", required=False, default="browse")
@click.argument("query", required=False, default="")
def marketplace(action: str, query: str) -> None:
    """Browse and install community skills.

    \b
    Examples:
        towel marketplace                  Browse all available skills
        towel marketplace search web       Search for skills
        towel marketplace install weather  Install a skill
        towel marketplace remove weather   Uninstall a skill
        towel marketplace installed        List installed skills
    """
    from towel.skills.marketplace import (
        COMMUNITY_SKILLS,
        install_skill,
        list_installed,
        remove_skill,
        search_marketplace,
    )

    if action == "browse" or action == "list":
        installed = set(list_installed())
        console.print(f"[bold]Skill Marketplace[/bold] ({len(COMMUNITY_SKILLS)} available)\n")
        for s in COMMUNITY_SKILLS:
            mark = " [green]installed[/green]" if s["name"] in installed else ""
            tags = " ".join(f"[dim]#{t}[/dim]" for t in s.get("tags", "").split(",") if t)
            console.print(f"  [green]{s['name']}[/green]{mark}")
            console.print(f"    {s['description']}  {tags}")
        console.print("\n[dim]Install: towel marketplace install <name>[/dim]")

    elif action == "search":
        if not query:
            console.print("[red]Usage:[/red] towel marketplace search <query>")
            return
        results = search_marketplace(query)
        if not results:
            console.print(f"[dim]No skills matching '{query}'[/dim]")
            return
        console.print(f"[bold]Results for '{query}':[/bold]\n")
        for s in results:
            console.print(f"  [green]{s['name']}[/green] — {s['description']}")

    elif action == "install":
        if not query:
            console.print("[red]Usage:[/red] towel marketplace install <name>")
            return
        console.print(f"[dim]Installing {query}...[/dim]")
        result = asyncio.run(install_skill(query))
        console.print(result)

    elif action == "remove":
        if not query:
            console.print("[red]Usage:[/red] towel marketplace remove <name>")
            return
        console.print(remove_skill(query))

    elif action == "installed":
        installed = list_installed()
        if not installed:
            console.print("[dim]No community skills installed.[/dim]")
            return
        console.print(f"[bold]Installed community skills ({len(installed)}):[/bold]")
        for name in sorted(installed):
            console.print(f"  [green]{name}[/green]")

    else:
        console.print(f"[red]Unknown action:[/red] {action}")
        console.print("[dim]Actions: browse, search, install, remove, installed[/dim]")


@cli.command()
def dashboard() -> None:
    """Show a system dashboard — skills, conversations, memory, config at a glance."""
    from towel.config import TOWEL_HOME, TowelConfig
    from towel.memory.store import MemoryStore
    from towel.persistence.store import ConversationStore

    config = TowelConfig.load()
    store = ConversationStore()
    memory = MemoryStore()

    # Skills
    skills_reg = _build_skill_registry(config, memory_store=memory)
    skill_count = len(skills_reg)
    tool_count = len(skills_reg.tool_definitions())

    # Conversations
    convos = store.list_conversations(limit=1000)
    total_msgs = sum(c.message_count for c in convos)

    # Storage
    conv_dir = TOWEL_HOME / "conversations"
    storage_kb = 0
    if conv_dir.exists():
        storage_kb = sum(f.stat().st_size for f in conv_dir.glob("*.json")) // 1024

    # Memory
    mem_count = memory.count

    # Templates
    from towel.templates.engine import TemplateEngine

    tpl_count = len(TemplateEngine().list_templates())

    # Aliases & snippets
    try:
        from towel.cli.aliases import list_aliases

        alias_count = len(list_aliases())
    except Exception:
        alias_count = 0
    try:
        from towel.cli.snippets import list_snippets

        snippet_count = len(list_snippets())
    except Exception:
        snippet_count = 0

    console.print(
        Panel(
            f"[bold green]Towel v{__version__}[/bold green] — Don't Panic.\n\n"
            f"  [bold]Agent[/bold]\n"
            f"    Model:          [green]{config.model.name}[/green]\n"
            f"    Context window: {config.model.context_window:,} tokens\n"
            f"    Max output:     {config.model.max_tokens:,} tokens\n"
            f"    Temperature:    {config.model.temperature}\n\n"
            f"  [bold]Skills & Tools[/bold]\n"
            f"    Skills:    [green]{skill_count}[/green] registered\n"
            f"    Tools:     [green]{tool_count}[/green] available\n"
            f"    Templates: {tpl_count}\n"
            f"    Aliases:   {alias_count}\n"
            f"    Snippets:  {snippet_count}\n\n"
            f"  [bold]Data[/bold]\n"
            f"    Conversations: {len(convos):,}\n"
            f"    Messages:      {total_msgs:,}\n"
            f"    Memories:      {mem_count}\n"
            f"    Storage:       {storage_kb:,} KB\n\n"
            f"  [bold]Gateway[/bold]\n"
            f"    WebSocket: ws://{config.gateway.host}:{config.gateway.port}\n"
            f"    HTTP API:  http://{config.gateway.host}:{config.gateway.port + 1}\n"
            f"    Home:      {TOWEL_HOME}",
            border_style="green",
            title="Dashboard",
        )
    )


@cli.command(name="version")
@click.option("--full", "-f", is_flag=True, help="Show full system inventory")
def version_cmd(full: bool) -> None:
    """Show Towel version and system info."""
    import platform

    console.print(f"[green]Towel[/green] v{__version__}")

    if not full:
        return

    config = TowelConfig.load()
    skills_reg = _build_skill_registry(config)

    # Count slash commands
    from towel.cli.slash import HELP_TEXT

    slash_count = HELP_TEXT.count("/")

    console.print("\n[bold]System inventory:[/bold]")
    console.print(f"  Skills:         {len(skills_reg)}")
    console.print(f"  Tools:          {len(skills_reg.tool_definitions())}")
    console.print(f"  Slash commands: ~{slash_count}")
    console.print(f"  CLI commands:   {len(cli.commands)}")
    console.print(f"  Model:          {config.model.name}")
    console.print(f"  Python:         {platform.python_version()}")
    console.print(f"  Platform:       {platform.system()} {platform.machine()}")

    try:
        import mlx

        console.print(f"  MLX:            {getattr(mlx, '__version__', '?')}")
    except ImportError:
        console.print("  MLX:            not installed")


@cli.command()
@click.argument("action", default="list")
@click.argument("name", required=False, default="")
def plugins(action: str, name: str) -> None:
    """Manage plugins — list, create, validate.

    \b
    Examples:
        towel plugins                List installed plugins
        towel plugins create weather Create a new plugin scaffold
        towel plugins validate ./my-plugin  Validate a plugin
    """
    from pathlib import Path

    from towel.skills.plugin import (
        PLUGINS_DIR,
        create_plugin_scaffold,
        discover_plugins,
        validate_plugin,
    )

    if action == "list":
        found = discover_plugins()
        if not found:
            console.print("[dim]No plugins installed.[/dim]")
            console.print(f"[dim]Plugin directory: {PLUGINS_DIR}[/dim]")
            console.print("[dim]Create one: towel plugins create my-plugin[/dim]")
            return
        console.print(f"[bold]Plugins ({len(found)}):[/bold]\n")
        for p in found:
            tags = " ".join(f"[dim]#{t}[/dim]" for t in p.tags) if p.tags else ""
            console.print(f"  [green]{p.name}[/green] v{p.version} {tags}")
            if p.description:
                console.print(f"    {p.description}")
            if p.author:
                console.print(f"    [dim]by {p.author}[/dim]")

    elif action == "create":
        if not name:
            console.print("[red]Usage:[/red] towel plugins create <name>")
            return
        path = create_plugin_scaffold(name)
        console.print(f"[green]Created plugin scaffold:[/green] {path}")
        console.print(f"  Edit: {path / 'skill.py'}")
        console.print(f"  Manifest: {path / 'towel-plugin.toml'}")

    elif action == "validate":
        if not name:
            console.print("[red]Usage:[/red] towel plugins validate <path>")
            return
        issues = validate_plugin(Path(name))
        if not issues:
            console.print("[green]Plugin is valid![/green]")
        else:
            console.print(f"[red]Found {len(issues)} issue(s):[/red]")
            for issue in issues:
                console.print(f"  [yellow]•[/yellow] {issue}")

    else:
        console.print(f"[red]Unknown action:[/red] {action}")
        console.print("[dim]Actions: list, create, validate[/dim]")


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.argument("error", required=False)
@click.option("--line", "-l", default=None, type=int, help="Error line number")
@click.option("--raw", "-r", is_flag=True, help="Plain text output")
def fix(file: str, error: str | None, line: int | None, raw: bool) -> None:
    """Fix a bug — give it a file and an error message.

    \b
    Examples:
        towel fix src/main.py "TypeError: cannot unpack"
        towel fix app.py -l 42 "index out of range"
        python app.py 2>&1 | towel fix app.py
    """
    from pathlib import Path

    content = Path(file).read_text(encoding="utf-8", errors="replace")
    ext = Path(file).suffix.lstrip(".")

    # Read error from stdin if not provided as argument
    if not error and not sys.stdin.isatty():
        error = sys.stdin.read().strip()[:5000]

    if not error:
        console.print("[red]No error message.[/red]")
        console.print('  towel fix app.py "the error message"')
        console.print("  python app.py 2>&1 | towel fix app.py")
        return

    # Build focused context around error line
    if line:
        lines = content.splitlines()
        start = max(0, line - 10)
        end = min(len(lines), line + 10)
        context = "\n".join(
            f"{i + 1:4d}  {'>>>' if i + 1 == line else '   '} {lines[i]}" for i in range(start, end)
        )
        prompt = (
            f"Fix the bug in this {ext} code. The error occurs at line {line}.\n\n"
            f"Error: {error}\n\n"
            f"Context around line {line}:\n```\n{context}\n```\n\n"
            f"Full file ({Path(file).name}):\n```{ext}\n{content[:30000]}\n```\n\n"
            f"Provide the fix with explanation. Show the corrected code."
        )
    else:
        prompt = (
            f"Fix the bug in this {ext} code.\n\n"
            f"Error: {error}\n\n"
            f"File ({Path(file).name}):\n```{ext}\n{content[:30000]}\n```\n\n"
            f"Identify the root cause, explain why it happens, and provide the corrected code."
        )

    config = TowelConfig.load()
    if not raw:
        console.print(f"[dim]Analyzing {Path(file).name} + error...[/dim]")
    _oneshot(config, prompt, raw)


@cli.command(name="test-gen")
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--framework", "-f", default=None, help="Test framework (pytest, jest, go test, etc.)"
)
@click.option("--output", "-o", default=None, help="Write tests to file instead of stdout")
@click.option("--raw", "-r", is_flag=True, help="Plain text output")
def test_gen(file: str, framework: str | None, output: str | None, raw: bool) -> None:
    """Generate tests for a file.

    \b
    Examples:
        towel test-gen src/auth.py
        towel test-gen src/api.ts -f jest
        towel test-gen lib.go -o lib_test.go
    """
    from pathlib import Path

    content = Path(file).read_text(encoding="utf-8", errors="replace")
    ext = Path(file).suffix.lstrip(".")

    fw_map = {
        "py": "pytest",
        "js": "jest",
        "ts": "jest",
        "go": "go test",
        "rs": "cargo test",
        "java": "JUnit",
        "rb": "RSpec",
    }
    fw = framework or fw_map.get(ext, "appropriate framework")

    prompt = (
        f"Write comprehensive tests for this {ext} code using {fw}.\n"
        f"Cover: normal behavior, edge cases, error conditions.\n"
        f"Include setup/teardown if needed.\n\n"
        f"File ({Path(file).name}):\n```{ext}\n{content[:30000]}\n```\n\n"
        f"Output only the test code, ready to run."
    )

    config = TowelConfig.load()

    if output:
        # Capture to file
        from towel.agent.conversation import Conversation, Role
        from towel.agent.runtime import AgentRuntime
        from towel.memory.store import MemoryStore

        memory = MemoryStore()
        skills_reg = _build_skill_registry(config, memory_store=memory)
        agent_rt = AgentRuntime(config, skills=skills_reg, memory=memory)
        conv = Conversation(channel="test-gen")
        conv.add(Role.USER, prompt)

        async def _gen():
            await agent_rt.load_model()
            resp = await agent_rt.step(conv)
            return resp.content

        console.print(f"[dim]Generating tests for {Path(file).name}...[/dim]")
        result = asyncio.run(_gen())

        # Strip markdown fences
        if result.startswith("```"):
            lines = result.splitlines()
            result = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        Path(output).write_text(result + "\n", encoding="utf-8")
        console.print(f"[green]Tests written to:[/green] {output}")
    else:
        if not raw:
            console.print(f"[dim]Generating tests for {Path(file).name}...[/dim]")
        _oneshot(config, prompt, raw)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--focus", "-f", default=None, help="Focus: performance, readability, dry, types, security"
)
@click.option("--raw", "-r", is_flag=True, help="Plain text output")
def refactor(file: str, focus: str | None, raw: bool) -> None:
    """Refactor a file — suggest improvements and output cleaned code.

    \b
    Examples:
        towel refactor src/main.py
        towel refactor api.py -f performance
        towel refactor utils.py -f dry
    """
    from pathlib import Path

    content = Path(file).read_text(encoding="utf-8", errors="replace")
    ext = Path(file).suffix.lstrip(".")
    focus_inst = f" Focus specifically on {focus}." if focus else ""
    prompt = (
        f"Refactor this {ext} code to improve quality.{focus_inst}\n"
        f"Show the refactored code and explain each change.\n\n"
        f"```{ext}\n{content[:30000]}\n```"
    )
    config = TowelConfig.load()
    if not raw:
        console.print(f"[dim]Refactoring {Path(file).name}...[/dim]")
    _oneshot(config, prompt, raw)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--style",
    "-s",
    default="google",
    type=click.Choice(["google", "numpy", "sphinx", "jsdoc"]),
    help="Docstring style",
)
@click.option("--output", "-o", default=None, help="Write documented code to file")
@click.option("--raw", "-r", is_flag=True, help="Plain text output")
def doc(file: str, style: str, output: str | None, raw: bool) -> None:
    """Add documentation to a file — docstrings, comments, type hints.

    \b
    Examples:
        towel doc src/auth.py
        towel doc api.py -s numpy
        towel doc utils.py -o utils_documented.py
    """
    from pathlib import Path

    content = Path(file).read_text(encoding="utf-8", errors="replace")
    ext = Path(file).suffix.lstrip(".")
    prompt = (
        f"Add comprehensive documentation to this {ext} code using {style} style.\n"
        f"Add docstrings to all functions/classes, type hints where missing, "
        f"and inline comments for complex logic. Output the full documented file.\n\n"
        f"```{ext}\n{content[:30000]}\n```"
    )
    config = TowelConfig.load()

    if output:
        from towel.agent.conversation import Conversation, Role
        from towel.agent.runtime import AgentRuntime
        from towel.memory.store import MemoryStore

        memory = MemoryStore()
        skills_reg = _build_skill_registry(config, memory_store=memory)
        agent_rt = AgentRuntime(config, skills=skills_reg, memory=memory)
        conv = Conversation(channel="doc")
        conv.add(Role.USER, prompt)

        async def _gen():
            await agent_rt.load_model()
            return (await agent_rt.step(conv)).content

        console.print(f"[dim]Documenting {Path(file).name}...[/dim]")
        result = asyncio.run(_gen())
        if result.startswith("```"):
            lines = result.splitlines()
            result = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        Path(output).write_text(result + "\n", encoding="utf-8")
        console.print(f"[green]Documented code written to:[/green] {output}")
    else:
        if not raw:
            console.print(f"[dim]Documenting {Path(file).name}...[/dim]")
        _oneshot(config, prompt, raw)


@cli.command()
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def completions(shell: str) -> None:
    """Generate shell tab-completion script.

    \b
    Examples:
        eval "$(towel completions zsh)"
        towel completions bash >> ~/.bashrc
        towel completions fish > ~/.config/fish/completions/towel.fish
    """
    env_var = "_TOWEL_COMPLETE"
    if shell == "zsh":
        script = f'eval "$({env_var}=zsh_source towel)"'
    elif shell == "bash":
        script = f'eval "$({env_var}=bash_source towel)"'
    elif shell == "fish":
        script = f"{env_var}=fish_source towel | source"
    print(script)


@cli.command()
@click.option("--chat", "-c", is_flag=True, help="Continuous voice chat mode")
@click.option(
    "--file",
    "-f",
    "audio_file",
    default=None,
    type=click.Path(exists=True),
    help="Transcribe an audio file",
)
@click.option("--duration", "-d", default=10.0, help="Max recording duration in seconds")
def voice(chat: bool, audio_file: str | None, duration: float) -> None:
    """Talk to Towel with your voice.

    \b
    Uses mlx-whisper for on-device speech-to-text. No audio leaves your Mac.

    \b
    Examples:
        towel voice                    Record and transcribe once
        towel voice --chat             Continuous voice conversation
        towel voice -f recording.wav   Transcribe a file
        towel voice -d 30              Record up to 30 seconds
    """
    from towel.cli.voice import check_voice_deps, record_audio, transcribe_audio, transcribe_bytes

    err = check_voice_deps()
    if err:
        console.print(f"[red]{err}[/red]")
        return

    if audio_file:
        text = transcribe_audio(audio_file)
        console.print(f"[cyan]{text}[/cyan]")
        return

    if chat:
        console.print(
            Panel(
                "[bold green]Voice Chat[/bold green] — speak and get AI responses.\n"
                "[dim]Press Ctrl+C to stop recording. Say 'exit' or 'quit' to end.[/dim]",
                border_style="green",
            )
        )

        config = TowelConfig.load()

        while True:
            wav = record_audio(duration)
            if isinstance(wav, str):
                console.print(f"[red]{wav}[/red]")
                break

            text = transcribe_bytes(wav)
            console.print(f"\n[bold cyan]you>[/bold cyan] {text}")

            if text.lower().strip() in ("exit", "quit", "bye", "stop"):
                console.print("[dim]Goodbye.[/dim]")
                break

            if not text or text == "(no speech detected)":
                continue

            # Send to agent
            console.print("[bold green]towel>[/bold green] ", end="")
            _oneshot(config, text)
            console.print()
    else:
        wav = record_audio(duration)
        if isinstance(wav, str):
            console.print(f"[red]{wav}[/red]")
            return
        text = transcribe_bytes(wav)
        console.print(f"[cyan]{text}[/cyan]")


@cli.command()
@click.option(
    "--token",
    "-t",
    required=True,
    envvar="DISCORD_TOKEN",
    help="Discord bot token (or set DISCORD_TOKEN env var)",
)
@click.option("--prefix", "-p", default="!towel", help="Command prefix (default: !towel)")
def discord(token: str, prefix: str) -> None:
    """Run Towel as a Discord bot.

    \b
    Setup:
      1. Create a bot at https://discord.com/developers
      2. Enable MESSAGE CONTENT intent
      3. Invite bot to your server with message permissions
      4. Run: towel discord -t YOUR_TOKEN

    \b
    The bot responds to:
      - Direct messages
      - @mentions
      - Messages starting with !towel (configurable with --prefix)
    """
    from towel.channels.discord import DiscordChannel

    console.print(
        Panel(
            f"[bold green]Discord Bot[/bold green]\n\n"
            f"  Prefix: [green]{prefix}[/green]\n"
            f"  Gateway: ws://{TowelConfig.load().gateway.host}:{TowelConfig.load().gateway.port}\n\n"
            f"[dim]Make sure 'towel serve' is running in another terminal.[/dim]",
            border_style="blue",
            title="Don't Panic.",
        )
    )

    channel = DiscordChannel(
        token=token,
        prefix=prefix,
        gateway_url=f"ws://{TowelConfig.load().gateway.host}:{TowelConfig.load().gateway.port}",
    )
    asyncio.run(channel.listen())


@cli.command()
@click.option(
    "--token",
    "-t",
    required=True,
    envvar="TELEGRAM_TOKEN",
    help="Telegram bot token (or set TELEGRAM_TOKEN)",
)
def telegram(token: str) -> None:
    """Run Towel as a Telegram bot.

    \b
    Setup:
      1. Message @BotFather on Telegram
      2. Create a bot and get the token
      3. Run: towel telegram -t YOUR_TOKEN

    \b
    The bot responds to all messages in DMs and group chats.
    """
    from towel.channels.telegram import TelegramChannel

    console.print(
        Panel(
            "[bold green]Telegram Bot[/bold green]\n\n"
            "[dim]Make sure 'towel serve' is running in another terminal.[/dim]",
            border_style="blue",
            title="Don't Panic.",
        )
    )

    channel = TelegramChannel(
        token=token,
        gateway_url=f"ws://{TowelConfig.load().gateway.host}:{TowelConfig.load().gateway.port}",
    )
    asyncio.run(channel.listen())


@cli.command()
@click.option(
    "--bot-token", "-b", required=True, envvar="SLACK_BOT_TOKEN", help="Slack bot token (xoxb-...)"
)
@click.option(
    "--app-token", "-a", required=True, envvar="SLACK_APP_TOKEN", help="Slack app token (xapp-...)"
)
def slack(bot_token: str, app_token: str) -> None:
    """Run Towel as a Slack bot via Socket Mode.

    \b
    Setup:
      1. Create app at api.slack.com/apps
      2. Enable Socket Mode, get app-level token (xapp-...)
      3. Add scopes: chat:write, app_mentions:read, im:history
      4. Install to workspace, get bot token (xoxb-...)
      5. Run: towel slack -b xoxb-... -a xapp-...

    \b
    The bot responds to @mentions and direct messages.
    """
    from towel.channels.slack import SlackChannel

    console.print(
        Panel(
            "[bold green]Slack Bot[/bold green] (Socket Mode)\n\n"
            "[dim]Make sure 'towel serve' is running in another terminal.[/dim]",
            border_style="blue",
            title="Don't Panic.",
        )
    )

    channel = SlackChannel(
        bot_token=bot_token,
        app_token=app_token,
        gateway_url=f"ws://{TowelConfig.load().gateway.host}:{TowelConfig.load().gateway.port}",
    )
    asyncio.run(channel.listen())


@cli.command()
@click.argument("target", type=click.Choice(["docker", "systemd", "heroku", "fly", "all"]))
@click.option("--output", "-o", default="./deploy", help="Output directory")
@click.option("--user", "-u", default="towel", help="System user (for systemd)")
def deploy(target: str, output: str, user: str) -> None:
    """Generate deployment files for running Towel in production.

    \b
    Targets:
        docker    Dockerfile + docker-compose.yml
        systemd   Linux systemd service unit
        heroku    Procfile for Heroku
        fly       fly.toml for Fly.io
        all       Generate everything

    \b
    Examples:
        towel deploy docker
        towel deploy all -o ./infra
        towel deploy systemd -u myuser
    """
    from pathlib import Path

    from towel.cli.deploy import generate_deploy

    created = generate_deploy(target, Path(output), user)
    if not created:
        console.print(f"[red]Unknown target:[/red] {target}")
        return

    console.print(f"[bold green]Generated {len(created)} deployment file(s):[/bold green]\n")
    for p in created:
        console.print(f"  [green]{p}[/green]")

    console.print(f"\n[dim]Next steps for {target}:[/dim]")
    match target:
        case "docker":
            console.print("  cp .env.example .env && edit .env")
            console.print("  docker-compose up -d")
        case "systemd":
            console.print("  sudo cp towel.service /etc/systemd/system/")
            console.print("  sudo systemctl enable --now towel")
        case "heroku":
            console.print("  heroku create && git push heroku main")
        case "fly":
            console.print("  fly launch && fly deploy")
        case "all":
            console.print("  Choose your platform and follow its docs.")


@cli.command()
def tui() -> None:
    """Launch the Terminal User Interface — full dashboard in your terminal.

    \b
    Shows skills, conversations, system status, channels, and
    key commands in a single rich terminal layout.
    """
    from towel.cli.tui import render_tui

    config = TowelConfig.load()
    layout = render_tui(config)
    console.print(layout)


@cli.command()
@click.argument("conversation_id", required=False)
@click.option(
    "--format", "-f", "fmt", default="markdown", type=click.Choice(["markdown", "text", "html"])
)
@click.option("--expire", "-e", default="1w", help="Expiry (1h, 1d, 1w, 1m)")
def share(conversation_id: str | None, fmt: str, expire: str) -> None:
    """Share a conversation via a temporary paste link.

    \b
    Uses dpaste.org (no account needed, auto-expires).

    \b
    Examples:
        towel share                    Share current/latest conversation
        towel share abc123             Share specific conversation
        towel share abc123 -f html     Share as HTML
    """
    import httpx

    from towel.persistence.export import export_html, export_markdown, export_text
    from towel.persistence.store import ConversationStore

    store = ConversationStore()

    if not conversation_id:
        convos = store.list_conversations(limit=1)
        if not convos:
            console.print("[dim]No conversations to share.[/dim]")
            return
        conversation_id = convos[0].id

    conv = store.load(conversation_id)
    if not conv:
        console.print(f"[red]Not found:[/red] {conversation_id}")
        return

    match fmt:
        case "html":
            content = export_html(conv)
        case "text":
            content = export_text(conv)
        case _:
            content = export_markdown(conv, include_metadata=True)

    expire_map = {"1h": 3600, "1d": 86400, "1w": 604800, "1m": 2592000}
    expiry_seconds = expire_map.get(expire, 604800)

    console.print(f"[dim]Sharing {conv.display_title} ({len(conv)} messages)...[/dim]")

    try:
        resp = httpx.post(
            "https://dpaste.org/api/",
            data={
                "content": content[:250000],
                "syntax": "md" if fmt == "markdown" else "text",
                "expiry_days": max(1, expiry_seconds // 86400),
            },
            timeout=10,
        )
        if resp.status_code == 200:
            url = resp.text.strip().strip('"')
            console.print("\n[bold green]Shared![/bold green]")
            console.print(f"  [cyan]{url}[/cyan]")
            console.print(f"  Expires: {expire}")

            # Copy to clipboard
            import platform
            import subprocess

            if platform.system() == "Darwin":
                subprocess.run(["pbcopy"], input=url.encode(), capture_output=True)
                console.print("  [dim](copied to clipboard)[/dim]")
        else:
            console.print(f"[red]Upload failed:[/red] HTTP {resp.status_code}")
    except Exception as e:
        console.print(f"[red]Share failed:[/red] {e}")


@cli.command()
def repl() -> None:
    """Interactive tool REPL — run skills directly without AI.

    \b
    Execute any of the 263 tools instantly, no model inference needed.
    Much faster than chat for quick tool access.

    \b
    Examples inside the REPL:
        tool> list                     show all tools
        tool> list git                 show git skill tools
        tool> weather_now city=Tokyo   run a tool with args
        tool> hash_text text=hello     compute a hash
        tool> random_quote             get a quote
        tool> exit
    """
    config = TowelConfig.load()
    from towel.memory.store import MemoryStore

    memory = MemoryStore()
    skills = _build_skill_registry(config, memory_store=memory)

    from towel.cli.repl import run_repl

    run_repl(skills)


@cli.command()
@click.option(
    "--homeserver", "-h", required=True, envvar="MATRIX_HOMESERVER", help="Matrix homeserver URL"
)
@click.option("--token", "-t", required=True, envvar="MATRIX_TOKEN", help="Access token")
def matrix(homeserver: str, token: str) -> None:
    """Run Towel as a Matrix bot.

    \b
    Setup:
      1. Create a Matrix account for the bot
      2. Get an access token (via Element or API)
      3. Run: towel matrix -h https://matrix.org -t TOKEN
    """
    from towel.channels.matrix import MatrixChannel

    console.print(
        Panel(
            f"[bold green]Matrix Bot[/bold green]\n\n"
            f"  Homeserver: {homeserver}\n"
            f"[dim]Make sure 'towel serve' is running.[/dim]",
            border_style="purple",
            title="Don't Panic.",
        )
    )

    channel = MatrixChannel(
        homeserver=homeserver,
        access_token=token,
        gateway_url=f"ws://{TowelConfig.load().gateway.host}:{TowelConfig.load().gateway.port}",
    )
    asyncio.run(channel.listen())


@cli.command()
@click.argument("name", required=False)
@click.option("--list", "-l", "list_all", is_flag=True, help="List available pipelines")
def pipeline(name: str | None, list_all: bool) -> None:
    """Run a tool pipeline — chain tools together.

    \b
    Examples:
        towel pipeline --list              Show available pipelines
        towel pipeline security-audit      Run security audit pipeline
        towel pipeline project-health      Check project health
        towel pipeline morning-briefing    Start your day
    """
    from towel.agent.pipelines import get_pipeline, list_pipelines
    from towel.memory.store import MemoryStore

    if list_all or not name:
        names = list_pipelines()
        console.print("[bold]Available pipelines:[/bold]")
        for n in names:
            p = get_pipeline(n)
            if p:
                steps = ", ".join(s.tool for s in p.steps)
                console.print(f"  [green]{n}[/green]: {steps}")
        return

    pipe = get_pipeline(name)
    if not pipe:
        console.print(f"[red]Unknown pipeline:[/red] {name}")
        return

    config = TowelConfig.load()
    memory = MemoryStore()
    skills = _build_skill_registry(config, memory_store=memory)

    console.print(f"[bold]Running pipeline:[/bold] {name}")
    result = asyncio.run(pipe.run(skills))
    console.print(result.summary())


@cli.command()
@click.argument("action", default="list")
@click.argument("args", nargs=-1)
@click.option("--cron", "-c", default=None, help="Cron expression")
@click.option("--run", "-r", default=None, help="Action (pipeline:name, tool:name)")
def schedule(action: str, args: tuple, cron: str | None, run: str | None) -> None:
    """Manage scheduled tasks.

    \b
    Examples:
        towel schedule list                          show all schedules
        towel schedule add daily-check -c "0 9 * * *" -r pipeline:project-health
        towel schedule add backup -c "0 0 * * 0" -r tool:git_status
        towel schedule remove daily-check
        towel schedule toggle daily-check
    """
    from towel.agent.scheduling import (
        add_schedule,
        list_schedules,
        remove_schedule,
        toggle_schedule,
    )

    if action == "list":
        schedules = list_schedules()
        if not schedules:
            console.print("[dim]No scheduled tasks.[/dim]")
            console.print("[dim]Add one: towel schedule add <name> -c '<cron>' -r '<action>'[/dim]")
            return
        console.print(f"[bold]Scheduled tasks ({len(schedules)}):[/bold]")
        for s in schedules:
            icon = "[green]●[/green]" if s.enabled else "[dim]○[/dim]"
            last = s.last_run[:16] if s.last_run else "never"
            console.print(f"  {icon} [green]{s.name}[/green]  {s.cron}  →  {s.action}")
            console.print(f"      [dim]Last: {last} · Runs: {s.run_count}[/dim]")

    elif action == "add":
        name = args[0] if args else None
        if not name or not cron or not run:
            console.print("[red]Usage:[/red] towel schedule add <name> -c '<cron>' -r '<action>'")
            return
        s = add_schedule(name, cron, run)
        console.print(f"[green]Scheduled:[/green] {s.name}  {s.cron}  →  {s.action}")

    elif action == "remove":
        name = args[0] if args else None
        if not name:
            console.print("[red]Usage:[/red] towel schedule remove <name>")
            return
        if remove_schedule(name):
            console.print(f"[green]Removed:[/green] {name}")
        else:
            console.print(f"[dim]Not found:[/dim] {name}")

    elif action == "toggle":
        name = args[0] if args else None
        if not name:
            console.print("[red]Usage:[/red] towel schedule toggle <name>")
            return
        console.print(toggle_schedule(name))

    else:
        console.print(f"[red]Unknown action:[/red] {action}")


@cli.command(name="eval")
@click.option("--suite", "-s", default="builtin", help="Eval suite (builtin or path to JSON)")
@click.option("--verbose", "-v", is_flag=True, help="Show full responses")
def eval_cmd(suite: str, verbose: bool) -> None:
    """Evaluate agent performance on test prompts.

    \b
    Runs test cases and scores responses on:
    - Keyword accuracy
    - Tool usage correctness
    - Response speed
    - Output quality

    \b
    Examples:
        towel eval                 Run built-in eval suite
        towel eval -v              Show full responses
    """
    from towel.agent.eval import BUILTIN_EVALS, EvalCase, EvalResult

    cases = [EvalCase(**e) for e in BUILTIN_EVALS]

    console.print(f"[bold]Running eval suite[/bold] ({len(cases)} cases)\n")

    # We can't actually run inference without a model, so just score structure
    _result = EvalResult(cases=cases)

    console.print(
        "[dim]Note: full eval requires a loaded model (towel serve must be running).[/dim]"
    )
    console.print(f"[dim]Eval suite has {len(cases)} test cases ready.[/dim]")
    console.print("\n[bold]Test cases:[/bold]")
    for i, c in enumerate(cases):
        tools = (
            f" [yellow]expects: {', '.join(c.expected_tools)}[/yellow]" if c.expected_tools else ""
        )
        keywords = (
            f" [cyan]keywords: {', '.join(c.expected_keywords[:3])}[/cyan]"
            if c.expected_keywords
            else ""
        )
        console.print(f"  {i + 1}. {c.prompt}{tools}{keywords}")


@cli.command(name="agent")
@click.argument("action", default="list")
@click.argument("name", required=False)
@click.option("--goal", "-g", default=None, help="Agent goal")
@click.option("--interval", "-i", default=300, help="Check interval in seconds")
@click.option("--tools", "-t", default=None, help="Comma-separated tool names")
def agent_cmd(
    action: str, name: str | None, goal: str | None, interval: int, tools: str | None
) -> None:
    """Manage autonomous agents.

    \b
    Examples:
        towel agent list
        towel agent create monitor -g "Watch API health" -i 60 -t uptime_check
        towel agent logs monitor
        towel agent delete monitor
    """
    from towel.agent.agents import create_agent, delete_agent, get_agent, list_agents

    if action == "list":
        agents = list_agents()
        if not agents:
            console.print("[dim]No agents. Create one: towel agent create <name> -g 'goal'[/dim]")
            return
        console.print(f"[bold]Agents ({len(agents)}):[/bold]")
        for a in agents:
            icon = "[green]●[/green]" if a.enabled else "[dim]○[/dim]"
            console.print(f"  {icon} [green]{a.name}[/green] — {a.goal[:50]}")
            console.print(
                f"      Every {a.check_interval}s · "
                f"{a.total_runs} runs · "
                f"Tools: {', '.join(a.tools) or 'any'}"
            )

    elif action == "create":
        if not name or not goal:
            console.print(
                "[red]Usage:[/red] towel agent create <name> -g 'goal' [-i interval] [-t tools]"
            )
            return
        tool_list = [t.strip() for t in tools.split(",")] if tools else []
        a = create_agent(name, goal, interval, tool_list)
        console.print(f"[green]Created agent:[/green] {a.name}")
        console.print(f"  Goal: {a.goal}")
        console.print(f"  Interval: {a.check_interval}s")

    elif action == "logs":
        if not name:
            console.print("[red]Usage:[/red] towel agent logs <name>")
            return
        a = get_agent(name)
        if not a:
            console.print(f"[red]Not found:[/red] {name}")
            return
        if not a.logs:
            console.print(f"[dim]No logs for {name}.[/dim]")
            return
        console.print(f"[bold]Logs for {name}[/bold] ({len(a.logs)} entries):")
        for entry in a.logs[-10:]:
            console.print(f"  [{entry.timestamp[:19]}] {entry.action}: {entry.result[:60]}")

    elif action == "delete":
        if not name:
            console.print("[red]Usage:[/red] towel agent delete <name>")
            return
        if delete_agent(name):
            console.print(f"[green]Deleted:[/green] {name}")
        else:
            console.print(f"[dim]Not found:[/dim] {name}")

    else:
        console.print(f"[red]Unknown:[/red] {action}")


@cli.command(name="ab-test")
@click.argument("prompt")
@click.option("--model-a", "-a", default=None, help="First model (default: current)")
@click.option("--model-b", "-b", required=True, help="Second model to compare")
@click.option("--tokens", "-t", default=256, help="Max tokens per response")
def ab_test(prompt: str, model_a: str | None, model_b: str, tokens: int) -> None:
    """Compare two models on the same prompt.

    \b
    Examples:
        towel ab-test "explain monads" -b mlx-community/Llama-3.2-3B-Instruct-4bit
        towel ab-test "write fizzbuzz" -a model1 -b model2 -t 512
    """
    from towel.agent.ab_test import ABTestResult

    config = TowelConfig.load()
    model_a_name = model_a or config.model.name

    console.print(
        Panel(
            f"[bold]A/B Test[/bold]\n\n"
            f"  A: [green]{model_a_name}[/green]\n"
            f"  B: [green]{model_b}[/green]\n"
            f"  Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}\n"
            f"  Max tokens: {tokens}",
            border_style="cyan",
        )
    )

    result = ABTestResult(prompt=prompt)
    result.a.label = model_a_name.split("/")[-1][:20]
    result.b.label = model_b.split("/")[-1][:20]

    # Run A
    console.print(f"\n[dim]Running model A ({model_a_name})...[/dim]")
    try:
        import copy

        config_a = copy.deepcopy(config)
        config_a.model.name = model_a_name
        config_a.model.max_tokens = tokens

        from towel.agent.conversation import Conversation, Role
        from towel.agent.runtime import AgentRuntime

        rt_a = AgentRuntime(config_a)
        conv_a = Conversation(channel="ab-test")
        conv_a.add(Role.USER, prompt)

        start = time.perf_counter()
        resp_a = asyncio.run(_ab_step(rt_a, conv_a))
        result.a.elapsed = time.perf_counter() - start
        result.a.response = resp_a.content
        result.a.tokens = resp_a.metadata.get("tokens", 0)
        result.a.tps = resp_a.metadata.get("tps", 0)
        console.print(f"  [green]A done:[/green] {result.a.elapsed:.1f}s, {result.a.tokens} tokens")
    except Exception as e:
        result.a.error = str(e)
        console.print(f"  [red]A error:[/red] {e}")

    # Run B
    console.print(f"\n[dim]Running model B ({model_b})...[/dim]")
    try:
        config_b = copy.deepcopy(config)
        config_b.model.name = model_b
        config_b.model.max_tokens = tokens

        rt_b = AgentRuntime(config_b)
        conv_b = Conversation(channel="ab-test")
        conv_b.add(Role.USER, prompt)

        start = time.perf_counter()
        resp_b = asyncio.run(_ab_step(rt_b, conv_b))
        result.b.elapsed = time.perf_counter() - start
        result.b.response = resp_b.content
        result.b.tokens = resp_b.metadata.get("tokens", 0)
        result.b.tps = resp_b.metadata.get("tps", 0)
        console.print(f"  [green]B done:[/green] {result.b.elapsed:.1f}s, {result.b.tokens} tokens")
    except Exception as e:
        result.b.error = str(e)
        console.print(f"  [red]B error:[/red] {e}")

    console.print(f"\n{result.summary()}")


async def _ab_step(runtime, conv):
    await runtime.load_model()
    return await runtime.step(conv)
