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
import re
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from towel import __version__
from towel.config import TowelConfig, TOWEL_HOME

console = Console()


def _build_skill_registry(config: TowelConfig, memory_store: Any = None) -> "SkillRegistry":
    """Build a skill registry with builtins + auto-loaded user skills."""
    from towel.skills.registry import SkillRegistry
    from towel.skills.builtin import register_builtins
    from towel.skills.loader import SkillLoader

    registry = SkillRegistry()
    register_builtins(registry, memory_store=memory_store)

    loader = SkillLoader(registry)
    loaded = loader.load_from_dirs(config.skills_dirs)
    if loaded:
        console.print(f"[dim]Loaded {loaded} skill(s) from disk[/dim]")
    for err in loader.errors:
        console.print(f"[yellow]Skill load error:[/yellow] {err.path.name}: {err.error}")

    return registry

BANNER = r"""
 _____ _____        _______ _
|_   _|  _  |      | _____ | |
  | | | | | |_ _ _ | |___  | |
  | | | | | | | | || _____ | |
  | | | |_| | | | || |___  | |___
  |_|  \_____/_____|_______|_____|

  Don't Panic.  v{}
""".format(__version__)


@click.group()
@click.version_option(__version__, prog_name="towel")
def cli() -> None:
    """Towel — Your local AI assistant. Don't Panic."""
    pass


@cli.command()
@click.option("--agent", "-a", default=None, help="Agent profile to use (e.g., coder, researcher, writer)")
def serve(agent: str | None) -> None:
    """Start the Towel gateway and agent runtime."""
    console.print(Panel(Text(BANNER, style="bold green"), border_style="green"))

    config = TowelConfig.load()
    model_config, identity = config.resolve_agent(agent)
    config.model = model_config
    config.identity = identity

    console.print(f"[dim]Model:[/dim] {config.model.name}")
    if agent:
        console.print(f"[dim]Agent:[/dim] {agent}")
    console.print(f"[dim]Gateway:[/dim] ws://{config.gateway.host}:{config.gateway.port}")
    console.print(f"[dim]Web UI:[/dim] http://{config.gateway.host}:{config.gateway.port + 1}/")
    console.print(f"[dim]API:[/dim] http://{config.gateway.host}:{config.gateway.port + 1}/v1/chat/completions")
    console.print()

    from towel.agent.runtime import AgentRuntime
    from towel.gateway.server import GatewayServer
    from towel.memory.store import MemoryStore

    memory = MemoryStore()
    skills = _build_skill_registry(config, memory_store=memory)
    agent_rt = AgentRuntime(config, skills=skills, memory=memory)
    gateway = GatewayServer(config=config, agent=agent_rt)

    console.print("[green]Loading model...[/green]")
    asyncio.run(_start(agent_rt, gateway))


async def _start(agent: "AgentRuntime", gateway: "GatewayServer") -> None:
    await agent.load_model()
    console.print("[green]Model loaded. Gateway starting...[/green]")
    await gateway.start()


@cli.command()
@click.option("--session", "-s", default="cli", help="Session ID")
@click.option("--agent", "-a", default=None, help="Agent profile (e.g., coder, researcher, writer)")
def chat(session: str, agent: str | None) -> None:
    """Interactive chat with Towel."""
    console.print(Panel(
        "[bold green]Towel[/bold green] — Don't Panic.\n"
        "[dim]Type your message. Ctrl+C to exit. /help for commands.[/dim]",
        border_style="green",
    ))

    config = TowelConfig.load()
    model_config, identity = config.resolve_agent(agent)
    config.model = model_config
    config.identity = identity

    if agent:
        console.print(f"[dim]Agent: {agent}[/dim]")

    from towel.agent.runtime import AgentRuntime
    from towel.agent.conversation import Conversation, Role
    from towel.agent.events import EventType
    from towel.persistence.store import ConversationStore
    from towel.memory.store import MemoryStore

    from towel.cli.slash import SlashContext, handle_slash

    memory = MemoryStore()
    skills = _build_skill_registry(config, memory_store=memory)
    agent_rt = AgentRuntime(config, skills=skills, memory=memory)
    store = ConversationStore()

    # Resume existing conversation or start fresh
    conv = store.load(session)
    if conv:
        console.print(f"[dim]Resumed conversation ({len(conv)} messages)[/dim]")
    else:
        conv = Conversation(id=session, channel="cli")

    slash_ctx = SlashContext(
        config=config, conv=conv, agent=agent_rt,
        memory=memory, store=store,
    )
    slash_ctx.current_agent_name = agent

    async def _chat_loop() -> None:
        console.print("[dim]Loading model...[/dim]")
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
            cancelled = False
            try:
                async for event in agent_rt.step_streaming(conv):
                    match event.type:
                        case EventType.TOKEN:
                            print(event.data["content"], end="", flush=True)
                        case EventType.TOOL_CALL:
                            tool = event.data["tool"]
                            print()
                            console.print(f"  [yellow]>> {tool}({event.data['arguments']})[/yellow]")
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
                                    f"[dim]({meta['tps']:.1f} tok/s, "
                                    f"{meta['tokens']} tokens)[/dim]"
                                )
                        case EventType.CANCELLED:
                            print()
                            console.print("[dim](generation stopped)[/dim]")
                            cancelled = True
                        case EventType.ERROR:
                            print()
                            console.print(f"[red]Error: {event.data['message']}[/red]")
            except KeyboardInterrupt:
                agent_rt.cancel()
                print()
                console.print("[dim](generation stopped)[/dim]")
                cancelled = True

            # Auto-title after first exchange
            if not conv.title and len(conv) >= 2:
                from towel.agent.titler import generate_title
                first_user = next((m.content for m in conv.messages if m.role == Role.USER), "")
                title = generate_title(first_user)
                if title:
                    conv.title = title

            store.save(conv)
            console.print()

    asyncio.run(_chat_loop())


@cli.command()
def status() -> None:
    """Show gateway status."""
    import httpx

    config = TowelConfig.load()
    url = f"http://{config.gateway.host}:{config.gateway.port + 1}/health"

    try:
        resp = httpx.get(url, timeout=3)
        data = resp.json()
        console.print(Panel(
            f"[green]Status:[/green] {data['status']}\n"
            f"[green]Version:[/green] {data['version']}\n"
            f"[green]Connections:[/green] {data['connections']}\n"
            f"[green]Sessions:[/green] {data['sessions']}\n"
            f"[dim]{data['motto']}[/dim]",
            title="Towel Gateway",
            border_style="green",
        ))
    except Exception:
        console.print("[red]Gateway not running.[/red] Start it with: towel serve")
        sys.exit(1)


STARTER_CONFIG = '''\
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
'''


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

    console.print(Panel(
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
    ))


@cli.command()
@click.option("--create", "-c", is_flag=True, help="Create a .towel.md in the current directory")
def context(create: bool) -> None:
    """Show or create project context (.towel.md)."""
    from pathlib import Path
    from towel.agent.project import find_project_contexts, load_project_context, CONTEXT_FILENAME

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
        from towel.agent.conversation import Conversation
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

    header = f"[bold]Recent conversations"
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
@click.option("--role", "-r", default=None, type=click.Choice(["user", "assistant", "tool"]), help="Filter by message role")
@click.option("--regex", "-e", is_flag=True, help="Treat query as regex")
def search(query: str, limit: int, role: str | None, regex: bool) -> None:
    """Search across all saved conversations."""
    from towel.persistence.store import ConversationStore
    from towel.agent.conversation import Role

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

    console.print(f"[dim]View a conversation: towel show <id>[/dim]")


@cli.command()
@click.argument("conversation_id")
@click.option("--tail", "-t", default=0, help="Show only last N messages (0 = all)")
def show(conversation_id: str, tail: int) -> None:
    """Show a saved conversation."""
    from towel.persistence.store import ConversationStore
    from towel.agent.conversation import Role

    store = ConversationStore()
    conv = store.load(conversation_id)

    if not conv:
        console.print(f"[red]Conversation not found:[/red] {conversation_id}")
        sys.exit(1)

    title_line = f"[bold]{conv.display_title}[/bold]" if conv.title else f"[bold]{conv.id}[/bold]"
    console.print(Panel(
        f"{title_line}\n[dim]{conv.id} · {conv.channel} · {conv.created_at.isoformat()[:16]}[/dim]",
        border_style="green",
    ))

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
@click.option("--format", "-f", "fmt", default="markdown", type=click.Choice(["markdown", "text", "json"]), help="Export format")
@click.option("--output", "-o", default=None, type=click.Path(), help="Write to file instead of stdout")
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
    from towel.persistence.store import ConversationStore
    from towel.persistence.export import export_markdown, export_text, export_json

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
        case _:
            result = export_markdown(conv)

    if output:
        from pathlib import Path
        Path(output).write_text(result, encoding="utf-8")
        console.print(f"[green]Exported to:[/green] {output}")
    else:
        print(result)


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
    from click import Context
    ctx = click.get_current_context()
    ctx.invoke(chat, session=conversation_id)


@cli.command()
def doctor() -> None:
    """Diagnose your Towel setup."""
    from towel.cli.doctor import run_doctor

    console.print(Panel(
        "[bold green]Towel Doctor[/bold green] — checking your setup...",
        border_style="green",
    ))

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


@cli.command()
@click.argument("prompt", nargs=-1)
@click.option("--session", "-s", default=None, help="Session ID (enables conversation context)")
@click.option("--agent", "-a", default=None, help="Agent profile (e.g., coder, researcher, writer)")
@click.option("--template", "-T", default=None, help="Prompt template (e.g., review, explain, summarize)")
@click.option("--var", "-V", multiple=True, help="Template variable: key=value (e.g., -V lang=Spanish)")
@click.option("--system", "-S", default=None, help="Override system prompt for this query")
@click.option("--raw", "-r", is_flag=True, help="Raw output only (no formatting, no stats)")
@click.option("--stream/--no-stream", default=True, help="Stream tokens as they generate")
def ask(
    prompt: tuple[str, ...],
    session: str | None,
    agent: str | None,
    template: str | None,
    var: tuple[str, ...],
    system: str | None,
    raw: bool,
    stream: bool,
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
    import select

    from towel.agent.runtime import AgentRuntime
    from towel.agent.conversation import Conversation, Role
    from towel.agent.events import EventType
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
            console.print("Usage: towel ask \"your question\"")
            console.print("   or: echo \"question\" | towel ask")
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
                                f"[dim]({meta['tps']:.1f} tok/s, "
                                f"{meta['tokens']} tokens)[/dim]",
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
            console.print(f"    [dim]context: {profile.model.context_window}, temp: {profile.model.temperature}[/dim]")
            console.print()

        console.print("[dim]towel agents create <name>  — create a new agent[/dim]")
        console.print("[dim]towel agents clone <src> <name>  — clone an existing agent[/dim]")
        console.print("[dim]towel agents delete <name>  — delete a user agent[/dim]")


@agents.command(name="create")
@click.argument("name")
@click.option("--model", "-m", default="mlx-community/Llama-3.3-70B-Instruct-4bit", help="Model name")
@click.option("--identity", "-i", default=None, help="System prompt / identity")
@click.option("--description", "-d", default="", help="Short description")
@click.option("--context-window", "-c", default=8192, help="Context window size")
@click.option("--temperature", "-t", default=0.7, help="Temperature")
def agents_create(
    name: str, model: str, identity: str | None,
    description: str, context_window: int, temperature: float,
) -> None:
    """Create a new agent profile."""
    from towel.cli.agent_mgr import create_agent, load_user_agents
    from towel.config import DEFAULT_AGENTS

    config = TowelConfig.load()
    if config.get_agent(name):
        console.print(f"[yellow]Agent '{name}' already exists.[/yellow] Use a different name or delete it first.")
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
    )
    console.print(f"[green]Created agent:[/green] {name}")
    console.print(f"  Model: {profile.model.name}")
    console.print(f"  Context: {profile.model.context_window}, Temp: {profile.model.temperature}")
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
    console.print("[dim]Or:       towel ask --agent researcher \"your question\"[/dim]")


@cli.group()
def memory() -> None:
    """Manage persistent agent memory."""
    pass


@memory.command(name="list")
@click.option("--type", "-t", "mtype", default=None, help="Filter by type (user, project, fact, preference)")
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
@click.option("--type", "-t", "mtype", default="fact", help="Memory type (user, project, fact, preference)")
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
    from towel.cli.models import list_cached_models, get_model_usage

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
        console.print(f"[dim]Use it with: towel chat (after setting model in config)[/dim]")
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
    console.print("  [green](cached)[/green]" if is_model_cached(config.model.name) else "  [red](not cached)[/red]")

    for name, profile in config.list_agents().items():
        cached = "[green](cached)[/green]" if is_model_cached(profile.model.name) else "[red](not cached)[/red]"
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

    console.print(f"\n[dim]Use with: towel ask -T review @file.py[/dim]")
    console.print(f"[dim]Or in chat: /t review <your input>[/dim]")
    console.print(f"[dim]Create custom: ~/.towel/templates/mytemplate.txt[/dim]")


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
    console.print(f"  [green]Config file:[/green]    {config_path} {'[dim](exists)[/dim]' if config_path.exists() else '[yellow](not found)[/yellow]'}")

    console.print(f"\n  [bold]Model:[/bold]")
    console.print(f"    [green]Name:[/green]           {cfg.model.name}")
    console.print(f"    [green]Context window:[/green] {cfg.model.context_window:,} tokens")
    console.print(f"    [green]Max output:[/green]     {cfg.model.max_tokens:,} tokens")
    console.print(f"    [green]Temperature:[/green]    {cfg.model.temperature}")
    console.print(f"    [green]Top-p:[/green]          {cfg.model.top_p}")

    console.print(f"\n  [bold]Gateway:[/bold]")
    console.print(f"    [green]Host:[/green]           {cfg.gateway.host}")
    console.print(f"    [green]WebSocket:[/green]      ws://{cfg.gateway.host}:{cfg.gateway.port}")
    console.print(f"    [green]HTTP API:[/green]       http://{cfg.gateway.host}:{cfg.gateway.port + 1}")

    agents = cfg.list_agents()
    if agents:
        console.print(f"\n  [bold]Agents ({len(agents)}):[/bold]")
        for name, profile in agents.items():
            desc = f" — {profile.description}" if profile.description else ""
            console.print(f"    [green]{name}[/green]{desc}")
            console.print(f"      [dim]{profile.model.name}[/dim]")

    console.print(f"\n  [bold]Skills dirs:[/bold]")
    for d in cfg.skills_dirs:
        from pathlib import Path
        p = Path(d).expanduser()
        exists = "[dim](exists)[/dim]" if p.exists() else "[dim](not created)[/dim]"
        console.print(f"    {p} {exists}")

    console.print(f"\n[dim]Edit: {config_path}[/dim]")


@cli.command(name="skill-init")
@click.argument("name")
@click.option("--dir", "output_dir", default=None, help="Output directory (default: ~/.towel/skills/)")
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
    console.print(f"\n[dim]Edit the file and restart Towel to load it.[/dim]")


@cli.command()
@click.option("--days", "-d", default=30, help="Delete conversations older than N days (default: 30)")
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
    from datetime import datetime, timezone, timedelta
    from pathlib import Path
    from towel.persistence.store import ConversationStore
    from towel.agent.conversation import Conversation

    store = ConversationStore()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
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
        console.print(f"\n  [yellow]To remove: {len(old_files)} conversation(s) ({old_total / 1024:.0f} KB)[/yellow]")
        for path, summary, size in old_files[:10]:
            console.print(f"    [dim]{path.stem}[/dim]  {summary}  [dim]({size / 1024:.1f} KB)[/dim]")
        if len(old_files) > 10:
            console.print(f"    [dim]... and {len(old_files) - 10} more[/dim]")

        if dry_run:
            console.print(f"\n  [dim]Dry run — nothing deleted.[/dim]")
        else:
            for path, _, _ in old_files:
                path.unlink()
            console.print(f"\n  [green]Deleted {len(old_files)} old conversation(s).[/green]")
            console.print(f"  Remaining: {kept_count} conversations ({kept_size / 1024:.0f} KB)")
    else:
        console.print(f"\n  [green]Nothing to clean up.[/green] All conversations are recent.")

    # Cache report
    if cache:
        console.print(f"\n[bold]Model caches:[/bold]")
        cache_dirs = {
            "HuggingFace": Path.home() / ".cache" / "huggingface",
            "MLX": Path.home() / ".cache" / "mlx",
        }
        for label, cache_path in cache_dirs.items():
            if cache_path.exists():
                try:
                    size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
                    console.print(f"  {label}: {size / (1024**3):.1f} GB  [dim]({cache_path})[/dim]")
                except OSError:
                    console.print(f"  {label}: [dim](error reading)[/dim]")
            else:
                console.print(f"  {label}: [dim](not found)[/dim]")
